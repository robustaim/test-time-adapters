"""
ActMAD (Activation Mean Alignment and Discrepancy) Engine

Framework-agnostic implementation using adapter pattern.
"""

import math
import copy
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from ..base import AdaptationEngine, AdaptationConfig
from ..framework_adapters import FrameworkAdapter, create_adapter


class SaveOutput:
    """Hook handler to save layer outputs for statistics computation"""

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

    def compute_mean(self) -> torch.Tensor:
        """Compute mean across batch dimension"""
        if len(self.outputs) == 0:
            return None
        out = self.outputs[0]
        if len(out.shape) == 4:  # Conv2d: [B, C, H, W]
            return out.mean(dim=[0, 2, 3])
        elif len(out.shape) == 3:  # LayerNorm: [B, L, D]
            return out.mean(dim=[0, 1])
        elif len(out.shape) == 2:  # FC: [B, D]
            return out.mean(dim=0)
        return out.mean(dim=0)

    def compute_var(self) -> torch.Tensor:
        """Compute variance across batch dimension"""
        if len(self.outputs) == 0:
            return None
        out = self.outputs[0]
        if len(out.shape) == 4:  # Conv2d: [B, C, H, W]
            return out.var(dim=[0, 2, 3])
        elif len(out.shape) == 3:  # LayerNorm: [B, L, D]
            return out.var(dim=[0, 1])
        elif len(out.shape) == 2:  # FC: [B, D]
            return out.var(dim=0)
        return out.var(dim=0)


class StatisticsAccumulator:
    """Accumulate statistics across batches"""

    def __init__(self):
        self.sum = None
        self.count = 0

    def update(self, value: torch.Tensor):
        """Update running statistics"""
        if value is None:
            return
        if self.sum is None:
            self.sum = value.detach().cpu()
        else:
            self.sum += value.detach().cpu()
        self.count += 1

    @property
    def avg(self) -> Optional[torch.Tensor]:
        """Get average"""
        if self.sum is None or self.count == 0:
            return None
        return self.sum / self.count


@dataclass
class ActMADConfig(AdaptationConfig):
    """Configuration for ActMAD engine"""
    adaptation_name: str = "ActMAD"

    # Adaptation settings
    adaptation_layers: str = "backbone+encoder"  # "backbone", "encoder", "backbone+encoder"

    # Optimization settings
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer_type: str = "SGD"  # "SGD" or "AdamW"

    # Loss function
    loss_type: str = "L1"  # "L1" or "MSE"

    # Clean dataset for statistics extraction
    clean_dataset: Optional[Dataset] = None
    clean_batch_size: int = 32

    # Statistics save path
    statistics_path: Optional[Path] = None

    # Device
    device: str = "cuda"


class ActMADEngine(AdaptationEngine):
    """
    ActMAD: Activation Mean Alignment and Discrepancy

    Framework-agnostic implementation that works with:
    - Detectron2 (Faster R-CNN, Mask R-CNN, etc.)
    - Transformers (RT-DETR, DETR, etc.)
    - Ultralytics (YOLO v8, v11, etc.)
    """

    model_name = "ActMAD"

    def __init__(
        self,
        basemodel: nn.Module,
        config: ActMADConfig,
        adapter: Optional[FrameworkAdapter] = None
    ):
        super().__init__(basemodel, config)

        # Auto-detect framework adapter if not provided
        if adapter is None:
            adapter = create_adapter(basemodel)

        self.adapter = adapter
        self.cfg = config
        self.device = torch.device(config.device)

        # Statistics storage
        self.clean_mean_list: Optional[List[torch.Tensor]] = None
        self.clean_var_list: Optional[List[torch.Tensor]] = None
        self.layer_names: Optional[List[str]] = None
        self.chosen_layers: Optional[List[nn.Module]] = None

        # Optimizer
        self.optimizer = None

        # Loss function
        if config.loss_type == "L1":
            self.loss_fn = nn.L1Loss(reduction="mean")
        elif config.loss_type == "MSE":
            self.loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")

        # Setup
        self._setup()

    def _setup(self):
        """Initialize the ActMAD engine"""
        # Move model to device
        self.basemodel.to(self.device)

        # Freeze all parameters initially
        for param in self.basemodel.parameters():
            param.requires_grad = False

        # Unfreeze normalization layer parameters that will be adapted
        params = self._setup_trainable_params()

        # Setup optimizer
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

        # Extract or load clean statistics
        self._load_or_extract_statistics()

    def _setup_trainable_params(self) -> List[torch.nn.Parameter]:
        """Setup trainable parameters based on adaptation layers"""
        params = []

        # Create layer filter based on adaptation_layers setting
        layer_filter = self._create_layer_filter()

        # Identify normalization layers using adapter
        norm_layers = self.adapter.identify_normalization_layers(
            self.basemodel,
            layer_filter=layer_filter
        )

        # Enable gradient for weight and bias of these layers
        for name, module in norm_layers:
            if hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module.weight, nn.Parameter):
                    module.weight.requires_grad = True
                    params.append(module.weight)

            if hasattr(module, 'bias') and module.bias is not None:
                if isinstance(module.bias, nn.Parameter):
                    module.bias.requires_grad = True
                    params.append(module.bias)

        print(f"ActMADEngine: Identified {len(params)} trainable parameters")
        return params

    def _create_layer_filter(self):
        """Create layer filter based on adaptation_layers config"""
        if self.cfg.adaptation_layers == "backbone":
            if hasattr(self.adapter, 'create_layer_filter_backbone'):
                return self.adapter.create_layer_filter_backbone()
            else:
                # Generic backbone filter
                return lambda name, module: 'backbone' in name.lower()

        elif self.cfg.adaptation_layers == "encoder":
            if hasattr(self.adapter, 'create_layer_filter_encoder'):
                return self.adapter.create_layer_filter_encoder()
            else:
                # Generic encoder filter
                return lambda name, module: 'encoder' in name.lower()

        elif self.cfg.adaptation_layers == "backbone+encoder":
            if hasattr(self.adapter, 'create_layer_filter_backbone_and_encoder'):
                return self.adapter.create_layer_filter_backbone_and_encoder()
            else:
                # Generic filter: exclude decoder
                return lambda name, module: 'decoder' not in name.lower()

        else:
            # Default: no filter (all normalization layers)
            return None

    def _load_or_extract_statistics(self):
        """Load existing statistics or extract from clean dataset"""
        # Determine save path
        if self.cfg.statistics_path is None:
            stats_dir = Path("./actmad_statistics")
            stats_dir.mkdir(parents=True, exist_ok=True)
            adapter_name = self.adapter.framework_name
            layers_name = self.cfg.adaptation_layers.replace('+', '_')
            self.cfg.statistics_path = stats_dir / f"actmad_{adapter_name}_{layers_name}.pt"

        # Load existing statistics if available
        if self.cfg.statistics_path.exists():
            print(f"Loading ActMAD statistics from {self.cfg.statistics_path}")
            stats = torch.load(self.cfg.statistics_path)
            self.clean_mean_list = stats["clean_mean_list"]
            self.clean_var_list = stats["clean_var_list"]
            self.layer_names = stats["layer_names"]
        else:
            # Extract statistics from clean dataset
            if self.cfg.clean_dataset is None:
                raise ValueError(
                    "No statistics file found and no clean_dataset provided. "
                    "Please provide clean_dataset for statistics extraction."
                )

            print("Extracting ActMAD statistics from clean dataset...")
            self.clean_mean_list, self.clean_var_list, self.layer_names = \
                self._extract_statistics()

            # Save statistics
            print(f"Saving ActMAD statistics to {self.cfg.statistics_path}")
            torch.save({
                "clean_mean_list": self.clean_mean_list,
                "clean_var_list": self.clean_var_list,
                "layer_names": self.layer_names
            }, self.cfg.statistics_path)

        # Setup chosen layers
        self._setup_chosen_layers()

    def _extract_statistics(self):
        """Extract activation statistics from clean dataset"""
        dataset = self.cfg.clean_dataset
        collate_fn = getattr(dataset, 'collate_fn', None)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.clean_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        loader_len = math.ceil(len(dataset) / self.cfg.clean_batch_size)

        # Create layer filter
        layer_filter = self._create_layer_filter()

        # Identify layers to track
        chosen_layers_info = self.adapter.identify_normalization_layers(
            self.basemodel,
            layer_filter=layer_filter
        )

        # Select only the second half of layers (as in original ActMAD)
        cutoff = len(chosen_layers_info) // 2
        chosen_layers_info = chosen_layers_info[cutoff:]

        layer_names = [name for name, _ in chosen_layers_info]
        chosen_layers = [module for _, module in chosen_layers_info]
        n_layers = len(chosen_layers)

        print(f"ActMAD: Tracking {n_layers} normalization layers")

        # Create hook handlers
        save_outputs = [SaveOutput() for _ in range(n_layers)]
        mean_accumulators = [StatisticsAccumulator() for _ in range(n_layers)]
        var_accumulators = [StatisticsAccumulator() for _ in range(n_layers)]

        # Extract statistics
        with torch.no_grad():
            self.basemodel.eval()
            for batch in tqdm(loader, total=loader_len, desc="Extracting statistics"):
                # Register hooks
                hooks = [
                    self.adapter.register_feature_hook(chosen_layers[i], save_outputs[i])
                    for i in range(n_layers)
                ]

                # Forward pass
                _ = self.adapter.execute_forward(self.basemodel, batch, self.device)

                # Collect statistics
                for i in range(n_layers):
                    mean_accumulators[i].update(save_outputs[i].compute_mean())
                    var_accumulators[i].update(save_outputs[i].compute_var())
                    save_outputs[i].clear()
                    hooks[i].remove()

        # Get final statistics
        clean_mean_list = [acc.avg for acc in mean_accumulators]
        clean_var_list = [acc.avg for acc in var_accumulators]

        return clean_mean_list, clean_var_list, layer_names

    def _setup_chosen_layers(self):
        """Setup chosen layers from layer names"""
        # Create a mapping of layer names to modules
        layer_dict = {name: module for name, module in self.basemodel.named_modules()}

        self.chosen_layers = []
        for layer_name in self.layer_names:
            if layer_name in layer_dict:
                self.chosen_layers.append(layer_dict[layer_name])
            else:
                warnings.warn(f"Layer {layer_name} not found in current model!")

    def forward(self, *args, **kwargs):
        """
        Forward pass with ActMAD adaptation

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

        # Set model to train mode for adaptation
        self.basemodel.train()

        # Keep normalization layers in eval mode (use running stats, not batch stats)
        for module in self.basemodel.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()

        self.optimizer.zero_grad()

        n_layers = len(self.chosen_layers)
        save_outputs = [SaveOutput() for _ in range(n_layers)]

        # Register hooks
        hooks = [
            self.adapter.register_feature_hook(self.chosen_layers[i], save_outputs[i])
            for i in range(n_layers)
        ]

        # Forward pass
        outputs = self.adapter.execute_forward(self.basemodel, batch, self.device)

        # Extract current batch statistics
        batch_means = [save_outputs[i].compute_mean() for i in range(n_layers)]
        batch_vars = [save_outputs[i].compute_var() for i in range(n_layers)]

        # Compute ActMAD loss
        loss_mean = torch.tensor(0.0, requires_grad=True, device=self.device)
        loss_var = torch.tensor(0.0, requires_grad=True, device=self.device)

        for i in range(n_layers):
            if batch_means[i] is not None and self.clean_mean_list[i] is not None:
                loss_mean = loss_mean + self.loss_fn(
                    batch_means[i].to(self.device),
                    self.clean_mean_list[i].to(self.device)
                )

            if batch_vars[i] is not None and self.clean_var_list[i] is not None:
                loss_var = loss_var + self.loss_fn(
                    batch_vars[i].to(self.device),
                    self.clean_var_list[i].to(self.device)
                )

        loss = loss_mean + loss_var

        # Backward and update
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.basemodel.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Clean up hooks
        for i in range(n_layers):
            save_outputs[i].clear()
            hooks[i].remove()

        return outputs

    def online(self, mode=True):
        """Enable/disable online adaptation mode"""
        self.adapting = mode
        return self

    def offline(self):
        """Disable online adaptation mode"""
        return self.online(False)
