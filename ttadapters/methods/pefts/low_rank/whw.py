"""
WHW (What, How, and Where to adapt) Engine - PEFT Style

Parameter-Efficient Fine-Tuning approach using parallel adapters.
Framework-agnostic implementation that works like LoRA.
"""

import re
import copy
import warnings
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from ...base import AdaptationEngine, AdaptationConfig
from ...framework_adapters import FrameworkAdapter, create_adapter


class ParallelAdapter(nn.Module):
    """
    Parallel Adapter for PEFT

    Similar to LoRA but with non-linearity:
    output = original(x) + adapter(x)
    adapter(x) = up_proj(relu(down_proj(x)))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        layer_type: str = "conv2d"  # "conv2d" or "linear"
    ):
        super().__init__()
        self.layer_type = layer_type
        bottleneck_features = max(1, in_features // rank)

        if layer_type == "conv2d":
            self.down_proj = nn.Conv2d(in_features, bottleneck_features, 1, bias=True)
            self.up_proj = nn.Conv2d(bottleneck_features, out_features, 1, bias=True)
        elif layer_type == "linear":
            self.down_proj = nn.Linear(in_features, bottleneck_features, bias=True)
            self.up_proj = nn.Linear(bottleneck_features, out_features, bias=True)
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")

        self.activation = nn.ReLU()

        # Zero initialization for identity at start
        nn.init.kaiming_uniform_(self.down_proj.weight, a=np.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return self.up_proj(self.activation(self.down_proj(x)))


@dataclass
class WHWConfig(AdaptationConfig):
    """Configuration for WHW engine"""
    adaptation_name: str = "WHW"

    # PEFT-style adapter configuration
    target_modules: List[str] = field(default_factory=lambda: [".*backbone.*"])  # Regex patterns
    adapter_rank: int = 32  # Bottleneck ratio

    # Optimization settings
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer_type: str = "SGD"

    # Feature alignment
    enable_global_align: bool = True
    enable_foreground_align: bool = False  # Detectron2 only
    alpha_global: float = 1.0
    alpha_foreground: float = 1.0
    ema_gamma: int = 128  # EMA update rate for statistics

    # Source statistics (for alignment)
    source_statistics_path: Optional[Path] = None
    clean_dataset: Optional[Dataset] = None
    clean_batch_size: int = 32

    # Redundancy skipping
    skip_redundant: Optional[str] = None  # "ema+", "period+stat", etc.
    skip_period: int = 10
    skip_beta: float = 1.2
    skip_tau: float = 1.0

    # Device
    device: str = "cuda"


class WHWEngine(AdaptationEngine):
    """
    WHW: What, How, and Where to adapt

    PEFT-style implementation using parallel adapters.
    Works with any framework (Detectron2, Transformers, Ultralytics).

    Key features:
    - Pattern-based adapter injection (like LoRA's target_modules)
    - Framework-agnostic parallel adapters
    - Optional feature alignment

    Example:
        config = WHWConfig(
            target_modules=[".*backbone.*conv.*", ".*encoder.*"],
            adapter_rank=32,
            enable_global_align=True
        )
        engine = WHWEngine(model, config)
    """

    model_name = "WHW"

    def __init__(
        self,
        basemodel: nn.Module,
        config: WHWConfig,
        adapter: Optional[FrameworkAdapter] = None
    ):
        super().__init__(basemodel, config)

        # Auto-detect framework adapter if not provided
        if adapter is None:
            adapter = create_adapter(basemodel)

        self.adapter = adapter
        self.cfg = config
        self.device = torch.device(config.device)

        # Track injected adapters
        self.injected_adapters = nn.ModuleDict()

        # Optimizer
        self.optimizer = None

        # Feature alignment
        self.source_stats: Optional[Dict[str, Any]] = None
        self.target_stats: Dict[str, Any] = {}
        self.ema_n: Dict[int, int] = {}  # EMA counters per class

        # Adaptation tracking
        self.adaptation_steps = 0
        self.used_steps = 0

        # Loss EMA for skipping
        self.loss_ema99 = 0.0
        self.loss_ema95 = 0.0
        self.loss_ema90 = 0.0

        # Setup
        self._setup()

    def _setup(self):
        """Initialize the WHW engine"""
        # Move model to device
        self.basemodel.to(self.device)

        # Freeze all parameters initially
        for param in self.basemodel.parameters():
            param.requires_grad = False

        # Inject adapters (PEFT-style)
        self._inject_adapters()

        # Load or extract source statistics if alignment is enabled
        if self.cfg.enable_global_align or self.cfg.enable_foreground_align:
            self._load_or_extract_statistics()

        # Setup optimizer
        self._setup_optimizer()

        print(f"WHWEngine: Injected {len(self.injected_adapters)} adapters")
        print(f"WHWEngine: Framework detected: {self.adapter.framework_name}")

    def _inject_adapters(self):
        """Inject parallel adapters to target modules (PEFT-style)"""
        adapter_count = 0

        for name, module in self.basemodel.named_modules():
            # Check if module matches any target pattern
            if self._matches_target_modules(name) and self._is_supported_layer(module):
                # Inject adapter
                self._wrap_module_with_adapter(name, module)
                adapter_count += 1

        if adapter_count == 0:
            warnings.warn(
                f"No adapters injected! Check your target_modules patterns: {self.cfg.target_modules}"
            )

    def _matches_target_modules(self, module_name: str) -> bool:
        """Check if module name matches any target pattern"""
        for pattern in self.cfg.target_modules:
            if re.search(pattern, module_name):
                return True
        return False

    def _is_supported_layer(self, module: nn.Module) -> bool:
        """Check if module is supported for adapter injection"""
        return isinstance(module, (nn.Conv2d, nn.Linear))

    def _wrap_module_with_adapter(self, name: str, module: nn.Module):
        """Wrap a module with parallel adapter"""
        # Determine layer type and dimensions
        if isinstance(module, nn.Conv2d):
            layer_type = "conv2d"
            in_features = module.in_channels
            out_features = module.out_channels
        elif isinstance(module, nn.Linear):
            layer_type = "linear"
            in_features = module.in_features
            out_features = module.out_features
        else:
            return

        # Create parallel adapter
        parallel_adapter = ParallelAdapter(
            in_features=in_features,
            out_features=out_features,
            rank=self.cfg.adapter_rank,
            layer_type=layer_type
        )
        parallel_adapter.to(self.device)

        # Store adapter
        safe_name = name.replace('.', '_')
        self.injected_adapters[safe_name] = parallel_adapter

        # Patch forward method
        original_forward = module.forward

        def forward_with_adapter(x):
            # Original layer output
            out = original_forward(x)
            # Add adapter output
            adapter_out = parallel_adapter(x)
            return out + adapter_out

        # Bind the new forward method
        module.forward = forward_with_adapter.__get__(module, module.__class__)
        module._whw_adapter = parallel_adapter  # Store reference

    def _load_or_extract_statistics(self):
        """Load or extract source domain statistics"""
        # Determine save path
        if self.cfg.source_statistics_path is None:
            stats_dir = Path("./whw_statistics")
            stats_dir.mkdir(parents=True, exist_ok=True)
            adapter_name = self.adapter.framework_name
            self.cfg.source_statistics_path = stats_dir / f"whw_{adapter_name}_source.pt"

        # Load existing statistics if available
        if self.cfg.source_statistics_path.exists():
            print(f"Loading WHW statistics from {self.cfg.source_statistics_path}")
            self.source_stats = torch.load(self.cfg.source_statistics_path)
            self._initialize_target_stats()
        else:
            # Extract statistics if dataset provided
            if self.cfg.clean_dataset is not None:
                print("Extracting WHW statistics from clean dataset...")
                self.source_stats = self._extract_source_statistics()

                # Save statistics
                print(f"Saving WHW statistics to {self.cfg.source_statistics_path}")
                torch.save(self.source_stats, self.cfg.source_statistics_path)

                self._initialize_target_stats()
            else:
                warnings.warn(
                    "No source statistics available. Feature alignment disabled. "
                    "Provide clean_dataset or source_statistics_path for alignment."
                )
                self.cfg.enable_global_align = False
                self.cfg.enable_foreground_align = False

    def _extract_source_statistics(self) -> Dict[str, Any]:
        """Extract source domain statistics"""
        dataset = self.cfg.clean_dataset
        collate_fn = getattr(dataset, 'collate_fn', None)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.clean_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Collect global features (backbone outputs)
        global_features = {}

        with torch.no_grad():
            self.basemodel.eval()

            for batch in tqdm(loader, desc="Extracting source statistics"):
                # Get backbone features (framework-specific)
                features = self._extract_backbone_features(batch)

                # Accumulate global features
                for key, feat in features.items():
                    # Compute mean over spatial dimensions
                    if len(feat.shape) == 4:  # [B, C, H, W]
                        feat_mean = feat.mean(dim=[2, 3])  # [B, C]
                    else:
                        feat_mean = feat

                    if key not in global_features:
                        global_features[key] = feat_mean.cpu()
                    else:
                        global_features[key] = torch.cat([
                            global_features[key], feat_mean.cpu()
                        ], dim=0)

        # Compute statistics
        stats = {"global": {}}

        for key, feats in global_features.items():
            mean = feats.mean(dim=0)
            cov = torch.from_numpy(np.cov(feats.T.numpy())).float()
            # Regularize covariance
            cov = cov + torch.eye(cov.shape[0]) * 1e-4
            stats["global"][key] = (mean, cov)
            print(f"  Global[{key}]: mean shape {mean.shape}, cov shape {cov.shape}")

        return stats

    def _extract_backbone_features(self, batch) -> Dict[str, torch.Tensor]:
        """Extract backbone features (framework-specific)"""
        features = {}

        if self.adapter.framework_name == "detectron2":
            # Detectron2: extract FPN features
            images = self.basemodel.preprocess_image(batch)
            backbone_features = self.basemodel.backbone(images.tensor)
            if isinstance(backbone_features, tuple):
                backbone_features = backbone_features[0]
            features = backbone_features

        elif self.adapter.framework_name == "transformers":
            # Transformers (RT-DETR): extract encoder features
            # This is simplified - full implementation would hook into encoder
            pixel_values = batch['pixel_values'].to(self.device)
            # For now, just use backbone output (would need to hook encoder)
            with torch.no_grad():
                outputs = self.basemodel.model.backbone(pixel_values)
                if hasattr(outputs, 'feature_maps'):
                    for i, feat in enumerate(outputs.feature_maps):
                        features[f"layer{i}"] = feat
                elif isinstance(outputs, (list, tuple)):
                    for i, feat in enumerate(outputs):
                        features[f"layer{i}"] = feat

        elif self.adapter.framework_name == "ultralytics":
            # YOLO: extract backbone features
            # Simplified implementation
            images = self.adapter.prepare_batch(batch, self.device)
            # Would need to hook into backbone layers
            features["backbone"] = images  # Placeholder

        return features

    def _initialize_target_stats(self):
        """Initialize target statistics from source"""
        if self.source_stats is None:
            return

        self.target_stats = {"global": {}}

        if "global" in self.source_stats:
            for key, (mean, cov) in self.source_stats["global"].items():
                self.target_stats["global"][key] = (mean.clone(), cov.clone())

    def _setup_optimizer(self):
        """Setup optimizer for adapter parameters"""
        # Collect adapter parameters
        params = list(self.injected_adapters.parameters())

        if len(params) == 0:
            warnings.warn("No trainable parameters found in adapters!")
            self.optimizer = None
            return

        if self.cfg.optimizer_type == "SGD":
            self.optimizer = optim.SGD(
                params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer_type == "AdamW":
            self.optimizer = optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.optimizer_type}")

        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"WHWEngine: Training {trainable_params:,} adapter parameters")

    def forward(self, *args, **kwargs):
        """
        Forward pass with WHW adaptation

        Args:
            batch: Input batch (framework-specific format)

        Returns:
            Model outputs
        """
        # Get batch from args or kwargs
        if len(args) > 0:
            batch = args[0]
        else:
            batch = kwargs

        # If no optimizer, just run inference
        if self.optimizer is None:
            self.basemodel.eval()
            return self.adapter.execute_forward(self.basemodel, batch, self.device)

        # Adaptation step
        self.optimizer.zero_grad()

        # Forward pass
        self.basemodel.train()  # Adapters in train mode
        outputs = self.adapter.execute_forward(self.basemodel, batch, self.device)

        # Compute adaptation loss
        adapt_loss = {}

        if self.cfg.enable_global_align and self.source_stats is not None:
            global_loss = self._compute_global_alignment_loss(batch)
            if global_loss is not None:
                adapt_loss["global_align"] = global_loss

        # Decide whether to adapt
        if adapt_loss and self._should_adapt(adapt_loss):
            total_loss = sum(adapt_loss.values()) * 1.0  # Scale if needed

            if total_loss > 0:
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.injected_adapters.parameters(), 1.0)

                self.optimizer.step()
                self.used_steps += 1

                # Update loss EMA
                if "global_align" in adapt_loss:
                    self._update_loss_ema(adapt_loss["global_align"].item())

        self.adaptation_steps += 1

        return outputs

    def _compute_global_alignment_loss(self, batch) -> Optional[torch.Tensor]:
        """Compute global feature alignment loss"""
        if self.source_stats is None or "global" not in self.source_stats:
            return None

        # Extract current features
        features = self._extract_backbone_features(batch)

        loss = torch.tensor(0.0, device=self.device)
        loss_count = 0

        for key, feat in features.items():
            if key not in self.source_stats["global"]:
                continue

            # Compute feature mean
            if len(feat.shape) == 4:  # [B, C, H, W]
                cur_mean = feat.mean(dim=[2, 3]).mean(dim=0)  # [C]
            else:
                cur_mean = feat.mean(dim=0)

            # Update target statistics with EMA
            source_mean, source_cov = self.source_stats["global"][key]
            target_mean, _ = self.target_stats["global"][key]

            # EMA update
            diff = cur_mean.cpu() - target_mean
            delta = (1 / self.cfg.ema_gamma) * diff
            new_target_mean = target_mean + delta

            # Compute KL divergence
            try:
                # Create template covariance
                template_cov = torch.eye(source_cov.shape[0]) * (source_cov.max().item() / 30)

                # Source and target distributions
                source_dist = torch.distributions.MultivariateNormal(
                    source_mean.to(self.device),
                    (source_cov + template_cov).to(self.device)
                )
                target_dist = torch.distributions.MultivariateNormal(
                    new_target_mean.to(self.device),
                    (source_cov + template_cov).to(self.device)
                )

                # Symmetric KL divergence
                kl_loss = (
                    torch.distributions.kl.kl_divergence(source_dist, target_dist) +
                    torch.distributions.kl.kl_divergence(target_dist, source_dist)
                ) / 2

                if kl_loss < 1e5:  # Sanity check
                    loss = loss + kl_loss
                    loss_count += 1

                    # Update target stats
                    self.target_stats["global"][key] = (new_target_mean.detach(), None)

            except Exception as e:
                warnings.warn(f"KL divergence computation failed for {key}: {e}")
                continue

        if loss_count > 0:
            return self.cfg.alpha_global * loss
        return None

    def _should_adapt(self, adapt_loss: Dict[str, torch.Tensor]) -> bool:
        """Determine whether to perform adaptation"""
        if self.cfg.skip_redundant is None:
            return True

        if "global_align" not in adapt_loss:
            return True

        loss_value = adapt_loss["global_align"].item()

        # Period-based skipping
        if "period" in self.cfg.skip_redundant:
            if self.adaptation_steps % self.cfg.skip_period == 0:
                return True

        # EMA-based skipping
        if "ema" in self.cfg.skip_redundant and self.loss_ema99 > 0:
            ema_ratio = loss_value / (self.loss_ema99 + 1e-7)
            if ema_ratio > self.cfg.skip_beta:
                return True

        return False

    def _update_loss_ema(self, loss_value: float):
        """Update loss EMA for redundancy skipping"""
        self.loss_ema99 = 0.99 * self.loss_ema99 + 0.01 * loss_value
        self.loss_ema95 = 0.95 * self.loss_ema95 + 0.05 * loss_value
        self.loss_ema90 = 0.9 * self.loss_ema90 + 0.1 * loss_value

    def online(self, mode=True):
        """Enable/disable online adaptation mode"""
        self.adapting = mode
        return self

    def offline(self):
        """Disable online adaptation mode"""
        return self.online(False)
