import re
import os
from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

from ....base import AdaptationEngine, BaseModel
from .....models.base import ModelProvider, DataPreparation, ObjectDetectionMixin
from .....utils.validator import DetectionEvaluator
from .configuration_sgp import SGPConfig


class RunningStats:
    """Welford-style online accumulator for channel-wise mean and variance."""

    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sum_sq = 0

    def update(self, x: torch.Tensor):
        x = x.detach().cpu()
        self.n += x.shape[0]
        self.sum += x.sum(dim=0)
        self.sum_sq += (x ** 2).sum(dim=0)

    def mean(self):
        return self.sum / self.n

    def var(self):
        return (self.sum_sq / self.n) - (self.mean() ** 2)

    def std(self):
        return torch.sqrt(self.var().clamp(min=1e-8))


class SGPEngine(AdaptationEngine):
    """Sensitivity-Guided Pruning Engine for Test-Time Adaptation.

    Implements the SGP algorithm (CVPR 2025) which prunes domain-sensitive BN
    channels via weighted sparsity regularisation while adapting the remaining
    domain-invariant channels through feature-distribution alignment.

    Currently supports **Detectron2** models only (FasterRCNN / SwinRCNN).

    Reference:
        Wang et al., "Efficient Test-time Adaptive Object Detection via
        Sensitivity-Guided Pruning", CVPR 2025.
    """

    model_name = "SGPEngine"
    config_class = SGPConfig

    def __init__(self, config: SGPConfig, base_model: BaseModel):
        self.config: SGPConfig
        super().__init__(config, base_model)

    def _pre_init(self):
        """Initialise state variables before base model registration."""
        self.current_bn_feats: dict[str, torch.Tensor] = {}
        self.pruning_masks: dict[str, torch.Tensor] = {}
        self.source_stats: dict | None = None
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._bn_params: list[nn.Parameter] = []
        # Instance-level: stage-output features captured by hooks
        self._stage_features: dict[str, torch.Tensor] = {}
        self._stage_hooks: list[torch.utils.hooks.RemovableHook] = []
        self._stage_strides: dict[str, int] = {}  # stage_name → output stride
        # Category-wise RoI statistics (for instance-level alignment)
        self._category_roi_stats: dict[int, dict] = {}  # category_id → {mean, std}

    def _post_init(self):
        """Initialise components after base model registration."""
        self._validate_provider()
        self._convert_frozen_bn()     # FrozenBatchNorm2d → nn.BatchNorm2d
        self._compile_layer_filter()
        self._setup_bn_hooks()
        if self.config.use_instance_sensitivity:
            self._setup_stage_hooks()
        self._collect_bn_params()
        self._load_source_stats()
        self._init_pruning_masks()

        self.to(self.device, dtype=self.dtype)

        self._reset_stats()
        self._base_bn_state = self._snapshot_bn_state()

        if self.config.verbose:
            n_target = len(self.pruning_masks)
            print("=" * 60)
            print(f"[{self.model_name}] Initialisation Summary")
            print("-" * 60)
            print(f"  Target BN layers   : {n_target}")
            print(f"  Pruning rate target : {self.config.pruning_rate:.0%}")
            print(f"  Instance sensitivity: {self.config.use_instance_sensitivity}")
            print(f"  Reactivation prob   : {self.config.reactivation_prob}")
            print("=" * 60)

    def _validate_provider(self):
        if self.model_provider != ModelProvider.Detectron2:
            raise NotImplementedError(
                f"SGPEngine only supports Detectron2 models. "
                f"Got provider: {self.model_provider}"
            )

    @staticmethod
    def _frozen_bn_to_bn(frozen: "FrozenBatchNorm2d") -> nn.BatchNorm2d:
        """Convert a single FrozenBatchNorm2d to a trainable nn.BatchNorm2d."""
        num_features = frozen.weight.shape[0]
        bn = nn.BatchNorm2d(num_features, eps=frozen.eps)
        bn.weight.data.copy_(frozen.weight)
        bn.bias.data.copy_(frozen.bias)
        bn.running_mean.copy_(frozen.running_mean)
        bn.running_var.copy_(frozen.running_var)
        bn.num_batches_tracked.zero_()
        return bn

    def _convert_frozen_bn(self):
        """Replace all FrozenBatchNorm2d layers in the backbone with nn.BatchNorm2d.

        SGP requires learnable scaling factors (γ, β) in BatchNorm layers for
        channel pruning.  Detectron2's default ResNet uses FrozenBatchNorm2d
        whose weight/bias are plain buffers, not Parameters.
        """
        from detectron2.layers import FrozenBatchNorm2d

        count = 0
        for name, module in list(self.base_model.named_modules()):
            if not isinstance(module, FrozenBatchNorm2d):
                continue

            # Navigate to the parent module and replace the child
            parts = name.split(".")
            parent = self.base_model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            new_bn = self._frozen_bn_to_bn(module)
            setattr(parent, parts[-1], new_bn)
            count += 1

        if self.config.verbose:
            print(f"[{self.model_name}] Converted {count} FrozenBatchNorm2d → nn.BatchNorm2d")

    def _compile_layer_filter(self):
        """Compile exclude-layer regex patterns."""
        if self.config.exclude_layers:
            self._exclude_re = re.compile(
                "|".join(f"({p})" for p in self.config.exclude_layers),
                flags=re.IGNORECASE,
            )
        else:
            self._exclude_re = None

    def _is_target_bn(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.BatchNorm2d):
            return False
        if self._exclude_re and self._exclude_re.search(name):
            return False
        return True

    def _setup_bn_hooks(self):
        """Register forward hooks on target BN layers to capture input features."""
        for name, module in self.base_model.named_modules():
            if self._is_target_bn(name, module):
                handle = module.register_forward_hook(self._make_bn_hook(name))
                self._hooks.append(handle)

    def _make_bn_hook(self, name: str):
        def hook(module, input, output):
            x = input[0]
            if x.dim() == 4:
                self.current_bn_feats[name] = x.mean(dim=(2, 3))  # (B, C)
            else:
                raise ValueError(f"Unexpected BN input dim={x.dim()} at {name}")
        return hook

    def _setup_stage_hooks(self):
        """Register hooks on ResNet backbone stage outputs for instance-level features."""
        backbone = self.base_model.backbone
        bottom_up = getattr(backbone, "bottom_up", backbone)

        # ResNet-FPN strides: res2=4, res3=8, res4=16, res5=32
        resnet_strides = [4, 8, 16, 32]

        if not hasattr(bottom_up, "stages"):
            raise NotImplementedError(
                "SGP instance-level sensitivity requires a ResNet backbone with .stages attribute."
            )

        for idx, stage in enumerate(bottom_up.stages):
            stage_name = f"stage_{idx}"
            self._stage_strides[stage_name] = resnet_strides[idx]
            handle = stage.register_forward_hook(self._make_stage_hook(stage_name))
            self._stage_hooks.append(handle)

    def _make_stage_hook(self, stage_name: str):
        def hook(module, input, output):
            feat = output
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat.dim() == 3:                    # (B, L, C) → skip
                return
            self._stage_features[stage_name] = feat
        return hook

    def _collect_bn_params(self):
        self._bn_params = []
        for name, module in self.base_model.named_modules():
            if self._is_target_bn(name, module):
                self._bn_params.extend([module.weight, module.bias])

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self._bn_params)

    def _load_source_stats(self):
        path = self.config.source_stats_path
        if path and os.path.exists(path):
            self.source_stats = torch.load(path, map_location=self.device, weights_only=False)
            if self.config.verbose:
                print(f"[{self.model_name}] Loaded source stats from {path}")
        else:
            self.source_stats = None

    def fit(
        self,
        source_dataset: DataPreparation,
        batch_size: int = 1,
        max_samples: int = 50000,
        dtype=torch.float32,
        **kwargs,
    ):
        """Collect source-domain BN feature statistics and category-wise RoI statistics."""
        if self.source_stats is not None:
            if self.config.verbose:
                print(f"[{self.model_name}] Source stats already loaded. Skipping fit().")
            return

        if not isinstance(self.base_model, ObjectDetectionMixin):
            raise NotImplementedError("SGP fit() requires an ObjectDetectionMixin model.")

        print(f"[{self.model_name}] Collecting source statistics ({max_samples} samples)...")
        self.base_model.eval()

        bn_runners: dict[str, RunningStats] = {}
        category_roi_runners: dict[int, RunningStats] = {}  # category_id → RunningStats

        loader = DataLoader(
            source_dataset, batch_size=batch_size,
            collate_fn=source_dataset.collate_fn, **kwargs
        )

        sample_count = 0
        total_batches = min(len(loader), (max_samples + batch_size - 1) // batch_size)

        from tqdm.auto import tqdm
        with torch.no_grad():
            with tqdm(total=total_batches, desc="Collecting source stats") as pbar:
                for batch in loader:
                    if sample_count >= max_samples:
                        break

                    # Forward: triggers BN hooks
                    _ = self.base_model(batch)

                    # BN features (image-level)
                    for name, feats in self.current_bn_feats.items():
                        if name not in bn_runners:
                            bn_runners[name] = RunningStats()
                        bn_runners[name].update(feats)

                    # Instance-level: collect category-wise RoI features
                    if self.config.use_instance_sensitivity and self._stage_features:
                        self._collect_category_roi_stats_for_fit(batch, category_roi_runners)

                    sample_count += feats.shape[0] if self.current_bn_feats else batch_size
                    pbar.update(1)

        # Build stats dict
        stats: dict = {"bn": {}, "category_roi": {}}

        # BN statistics (with both mean and std)
        for name, runner in bn_runners.items():
            stats["bn"][name] = {
                "mean": runner.mean().to(self.device),
                "std": runner.std().to(self.device)
            }

        # Category-wise RoI statistics
        for category_id, runner in category_roi_runners.items():
            stats["category_roi"][category_id] = {
                "mean": runner.mean().to(self.device),
                "std": runner.std().to(self.device)
            }

        self.source_stats = stats

        # Optionally persist
        if self.config.source_stats_path:
            os.makedirs(os.path.dirname(self.config.source_stats_path) or ".", exist_ok=True)
            torch.save(stats, self.config.source_stats_path)

        print(f"[{self.model_name}] Source stats collected: "
              f"{len(stats['bn'])} BN layers, {len(stats['category_roi'])} categories.")

    def _collect_category_roi_stats_for_fit(self, batch, category_roi_runners: dict):
        """Extract category-wise RoI features from stage outputs during fit() and accumulate.

        This implements the instance-level statistics collection for Eq. 10 in the paper.
        """
        images = self.base_model.preprocess_image(batch)
        features = self.base_model.backbone(images.tensor)

        proposals, _ = self.base_model.proposal_generator(images, features, None)

        # Get GT boxes and labels for category-wise grouping
        gt_instances = [x["instances"].to(self.device) for x in batch]

        for img_idx, (proposals_per_img, gt_per_img) in enumerate(zip(proposals, gt_instances)):
            if len(gt_per_img) == 0:
                continue

            gt_boxes = gt_per_img.gt_boxes.tensor
            gt_classes = gt_per_img.gt_classes

            # Use the last stage for instance-level features (highest semantic level)
            stage_name = f"stage_{len(self._stage_strides) - 1}"
            if stage_name not in self._stage_features:
                continue

            stage_feat = self._stage_features[stage_name]
            stride = self._stage_strides[stage_name]

            # Extract RoI features for GT boxes
            batch_boxes = [gt_boxes]
            roi_feats = self._extract_roi_features(
                stage_feat[img_idx:img_idx+1],
                batch_boxes,
                stride
            )

            if roi_feats is None or roi_feats.shape[0] == 0:
                continue

            # Pool to (M, C)
            pooled = roi_feats.mean(dim=(2, 3))

            # Group by category
            for category_id in gt_classes.unique():
                category_mask = gt_classes == category_id
                category_feats = pooled[category_mask]

                if category_feats.shape[0] == 0:
                    continue

                cat_id = category_id.item()
                if cat_id not in category_roi_runners:
                    category_roi_runners[cat_id] = RunningStats()
                category_roi_runners[cat_id].update(category_feats)

    def _init_pruning_masks(self):
        self.pruning_masks = {}
        for name, module in self.base_model.named_modules():
            if self._is_target_bn(name, module):
                self.pruning_masks[name] = torch.ones_like(module.weight.data).to(self.device)

    def _apply_pruning_mask(self):
        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name in self.pruning_masks:
                    mask = self.pruning_masks[name]
                    module.weight.data *= mask
                    module.bias.data *= mask

    def _mask_gradients(self):
        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name in self.pruning_masks:
                    mask = self.pruning_masks[name]
                    if module.weight.grad is not None:
                        module.weight.grad *= mask
                    if module.bias.grad is not None:
                        module.bias.grad *= mask

    def _prune_parameters(self):
        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name in self.pruning_masks:
                    dead = torch.abs(module.weight) < self.config.pruning_threshold
                    if dead.any():
                        self.pruning_masks[name][dead] = 0

    def _stochastic_reactivation(self):
        """Bernoulli sampling to restore pruned channels to source pre-trained values."""
        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name not in self.pruning_masks:
                    continue
                mask = self.pruning_masks[name]
                pruned_idx = (mask == 0)
                if not pruned_idx.any():
                    continue

                # Bernoulli: reactivate each pruned channel with probability p
                reactivate = torch.bernoulli(
                    torch.full_like(mask[pruned_idx], self.config.reactivation_prob)
                ).bool()

                if reactivate.any():
                    # Restore to source pre-trained values
                    src_w = self._base_bn_state[f"{name}.weight"]
                    src_b = self._base_bn_state[f"{name}.bias"]
                    pruned_positions = torch.where(pruned_idx)[0]
                    restore_positions = pruned_positions[reactivate]
                    pos_cpu = restore_positions.cpu()
                    module.weight.data[restore_positions] = src_w[pos_cpu].to(module.weight.device)
                    module.bias.data[restore_positions] = src_b[pos_cpu].to(module.bias.device)
                    mask[restore_positions] = 1

    def get_pruning_rate(self) -> float:
        total = 0
        pruned = 0
        with torch.no_grad():
            for mask in self.pruning_masks.values():
                total += mask.numel()
                pruned += (mask == 0).sum().item()
        return pruned / total if total > 0 else 0.0

    def _snapshot_bn_state(self) -> dict[str, torch.Tensor]:
        state = {}
        for name, module in self.base_model.named_modules():
            if name in self.pruning_masks:
                state[f"{name}.weight"] = module.weight.data.cpu().clone()
                state[f"{name}.bias"] = module.bias.data.cpu().clone()
        return state

    def _compute_alignment_loss(self) -> torch.Tensor:
        """L_adp: Image-level and Instance-level feature alignment.

        Image-level (Eq. 9): BN feature alignment via KL divergence
        Instance-level (Eq. 10): Category-wise RoI feature alignment
        """
        if self.source_stats is None:
            return torch.tensor(0.0, device=self.device)

        # --- Image-level alignment (L_img) ---
        img_losses = []
        for name, feats in self.current_bn_feats.items():
            if name not in self.source_stats["bn"]:
                continue

            curr_mean = feats.mean(dim=0)
            curr_std = torch.sqrt(feats.var(dim=0, unbiased=False) + 1e-8)

            # Load source statistics
            src_stats = self.source_stats["bn"][name]
            if isinstance(src_stats, dict):
                src_mean = src_stats["mean"].detach()
                src_std = src_stats.get("std", None)
            else:
                # Backward compatibility: if only mean is stored
                src_mean = src_stats.detach()
                src_std = None

            # Mean alignment (essential part of KL divergence)
            loss = F.mse_loss(curr_mean, src_mean)

            # Std alignment (if available and enabled)
            if self.config.lambda_mu_std > 0 and src_std is not None:
                src_std = src_std.detach()
                loss = loss + self.config.lambda_mu_std * F.mse_loss(curr_std, src_std)

            img_losses.append(loss)

        loss_img = torch.stack(img_losses).sum() if img_losses else torch.tensor(0.0, device=self.device)

        # --- Instance-level alignment (L_ins) ---
        loss_ins = self._compute_instance_alignment_loss()

        return loss_img + loss_ins

    def _compute_instance_alignment_loss(self) -> torch.Tensor:
        """Compute instance-level (category-wise RoI) alignment loss (Eq. 10).

        L_ins = Σ_k w_k · D_KL(N(μ_s^k, Σ_s^k), N(μ_t^k, Σ_s^k))

        where k is the category index, w_k is the category weight.
        """
        if not self.config.use_instance_sensitivity:
            return torch.tensor(0.0, device=self.device)

        if not hasattr(self, "_current_category_feats") or not self._current_category_feats:
            return torch.tensor(0.0, device=self.device)

        if "category_roi" not in self.source_stats or not self.source_stats["category_roi"]:
            return torch.tensor(0.0, device=self.device)

        ins_losses = []
        category_weights = []

        for category_id, curr_feats in self._current_category_feats.items():
            if category_id not in self.source_stats["category_roi"]:
                continue

            if curr_feats.shape[0] == 0:
                continue

            # Current statistics
            curr_mean = curr_feats.mean(dim=0)
            curr_std = torch.sqrt(curr_feats.var(dim=0, unbiased=False) + 1e-8)

            # Source statistics
            src_stats = self.source_stats["category_roi"][category_id]
            src_mean = src_stats["mean"].detach()
            src_std = src_stats.get("std", None)

            # KL divergence approximation (mean alignment + optional std)
            loss = F.mse_loss(curr_mean, src_mean)
            if self.config.lambda_mu_std > 0 and src_std is not None:
                src_std = src_std.detach()
                loss = loss + self.config.lambda_mu_std * F.mse_loss(curr_std, src_std)

            ins_losses.append(loss)

            # Category weight: inverse frequency (rare classes get higher weight)
            category_count = curr_feats.shape[0]
            category_weights.append(1.0 / (category_count + 1e-6))

        if not ins_losses:
            return torch.tensor(0.0, device=self.device)

        # Normalize weights
        weights = torch.tensor(category_weights, device=self.device)
        weights = weights / weights.sum()

        # Weighted sum
        losses = torch.stack(ins_losses)
        return (losses * weights).sum()

    def _compute_sensitivity_weights(self) -> dict[str, torch.Tensor]:
        """Compute per-BN-layer sensitivity ω = normalize(S_img [+ S_ins]).

        Image-level (Eq. 5-6): |curr_channel_mean - src_channel_mean|
        Instance-level (Eq. 7-8): RoI feature discrepancy (if enabled)
        """
        sensitivity: dict[str, torch.Tensor] = {}

        # --- Image-level sensitivity (S_img) ---
        for name, feats in self.current_bn_feats.items():
            if name not in self.source_stats["bn"]:
                continue

            curr_mean = feats.mean(dim=0)

            src_stats = self.source_stats["bn"][name]
            if isinstance(src_stats, dict):
                src_mean = src_stats["mean"].detach()
            else:
                src_mean = src_stats.detach()

            sensitivity[name] = torch.abs(curr_mean - src_mean)

        # --- Instance-level sensitivity (S_ins) ---
        if self.config.use_instance_sensitivity:
            instance_sens = self._compute_instance_sensitivity()

            # Add instance-level to matching BN layers
            for name in list(sensitivity.keys()):
                matched_stage = self._match_bn_to_stage(name)
                if matched_stage and matched_stage in instance_sens:
                    s_ins = instance_sens[matched_stage]
                    s_img = sensitivity[name]
                    # Channel counts must match
                    if s_ins.shape == s_img.shape:
                        sensitivity[name] = s_img + s_ins

        # --- Normalize across all active channels ---
        all_active = []
        for name, sens in sensitivity.items():
            mask = self.pruning_masks.get(name, torch.ones_like(sens))
            active = sens[mask == 1]
            if active.numel() > 0:
                all_active.append(active)

        if all_active:
            all_cat = torch.cat(all_active)
            s_min = all_cat.min()
            s_max = all_cat.max()
            denom = s_max - s_min + 1e-6
            sensitivity = {
                name: (sens - s_min) / denom
                for name, sens in sensitivity.items()
            }

        return sensitivity

    def _compute_instance_sensitivity(self) -> dict[str, torch.Tensor]:
        """Compute instance-level (RoI) sensitivity per backbone stage."""
        if not self._stage_features:
            return {}

        instance_sens: dict[str, torch.Tensor] = {}

        for stage_name, stage_feat in self._stage_features.items():
            roi_feats = self._extract_roi_features_from_stored(stage_feat, stage_name)
            if roi_feats is None or roi_feats.shape[0] == 0:
                continue

            pooled = roi_feats.mean(dim=(2, 3))  # (M, C)
            curr_mean = pooled.mean(dim=0)        # (C,)

            # For sensitivity, we use aggregated stats across all categories
            # (unlike alignment loss which is category-specific)
            if "category_roi" in self.source_stats and self.source_stats["category_roi"]:
                # Average source mean across all categories
                src_means = [stats["mean"] for stats in self.source_stats["category_roi"].values()]
                if src_means:
                    src_mean = torch.stack(src_means).mean(dim=0).detach()
                    instance_sens[stage_name] = torch.abs(curr_mean - src_mean)

        return instance_sens

    def _extract_roi_features_from_stored(
        self, stage_feat: torch.Tensor, stage_name: str
    ) -> torch.Tensor | None:
        """Extract RoI features from a stage feature map using stored proposals."""
        if not hasattr(self, "_current_proposals") or self._current_proposals is None:
            return None
        proposal_boxes = [p.proposal_boxes.tensor for p in self._current_proposals]
        stride = self._stage_strides.get(stage_name)
        if stride is None:
            return None
        return self._extract_roi_features(stage_feat, proposal_boxes, stride)

    def _extract_roi_features(
        self,
        feature_map: torch.Tensor,
        proposal_boxes: list[torch.Tensor],
        stride: int,
    ) -> torch.Tensor | None:
        """Apply RoI-Align on a feature map using proposal boxes.

        Args:
            feature_map: (B, C, H, W) feature tensor
            proposal_boxes: list of (N_i, 4) box tensors per image (in input-image coords)
            stride: backbone stride for this feature level (e.g. 4, 8, 16, 32)

        Returns:
            (M, C, roi_size, roi_size) pooled features or None
        """
        if feature_map.dim() != 4:
            return None

        spatial_scale = 1.0 / stride

        # Build ROI list: [(batch_idx, x1, y1, x2, y2), ...]
        rois_list = []
        for batch_idx, boxes in enumerate(proposal_boxes):
            if boxes.numel() == 0:
                continue
            batch_col = torch.full((boxes.shape[0], 1), batch_idx,
                                   device=boxes.device, dtype=boxes.dtype)
            rois_list.append(torch.cat([batch_col, boxes], dim=1))

        if not rois_list:
            return None

        rois = torch.cat(rois_list, dim=0)
        pooled = roi_align(
            feature_map, rois,
            output_size=self.config.roi_output_size,
            spatial_scale=spatial_scale,
            aligned=True,
        )
        return pooled

    def _match_bn_to_stage(self, bn_name: str) -> str | None:
        """Map a BN layer name to its ResNet backbone stage for instance-level sensitivity."""
        for i, stage_key in enumerate(["res2", "res3", "res4", "res5"]):
            if stage_key in bn_name:
                return f"stage_{i}"
        return None

    def _compute_sparse_loss(self, sensitivity: dict[str, torch.Tensor]) -> torch.Tensor:
        """L_wreg = Σ ‖ω_i · γ_i‖₁  (weighted sparsity on BN scaling factors, Eq. 3)."""
        total = torch.tensor(0.0, device=self.device)
        for name, module in self.base_model.named_modules():
            if name in sensitivity and name in self.pruning_masks:
                w = sensitivity[name]
                layer_loss = (w * torch.abs(module.weight)).sum()
                total = total + layer_loss
        return total

    def _reset_stats(self):
        self._stats = {
            "losses": [],
            "pruning_rates": [],
            "config": vars(self.config),
        }

    def _extract_category_feats_from_proposals(self, proposals, features):
        """Extract category-wise RoI features from current proposals for instance-level alignment.

        This is used during test-time adaptation to compute L_ins.
        """
        self._current_category_feats = {}

        if not proposals:
            return

        # Use the last stage for highest semantic level
        stage_name = f"stage_{len(self._stage_strides) - 1}"
        if stage_name not in self._stage_features:
            return

        stage_feat = self._stage_features[stage_name]
        stride = self._stage_strides[stage_name]

        # Get proposals with high confidence
        for img_idx, proposals_per_img in enumerate(proposals):
            if not hasattr(proposals_per_img, 'objectness_logits'):
                continue

            # Filter foreground proposals
            scores = proposals_per_img.objectness_logits.sigmoid()
            fg_mask = scores > self.config.fg_confidence_threshold

            if fg_mask.sum() == 0:
                continue

            fg_boxes = proposals_per_img.proposal_boxes.tensor[fg_mask]

            # Extract RoI features
            batch_boxes = [fg_boxes]
            roi_feats = self._extract_roi_features(
                stage_feat[img_idx:img_idx+1],
                batch_boxes,
                stride
            )

            if roi_feats is None or roi_feats.shape[0] == 0:
                continue

            pooled = roi_feats.mean(dim=(2, 3))  # (M, C)

            # During test-time, we don't have GT labels
            # We use pseudo-labels from the model's predictions
            # For simplicity, we can aggregate all foreground proposals
            # or use predicted class scores to group them

            # Simple approach: treat all foreground as one "category"
            # (More sophisticated: use argmax of classification head)
            pseudo_category = 0  # Aggregate category
            if pseudo_category not in self._current_category_feats:
                self._current_category_feats[pseudo_category] = []
            self._current_category_feats[pseudo_category].append(pooled)

        # Concatenate features for each category
        for cat_id in self._current_category_feats:
            if self._current_category_feats[cat_id]:
                self._current_category_feats[cat_id] = torch.cat(
                    self._current_category_feats[cat_id], dim=0
                )

    def forward(self, batched_inputs):
        # Only supports Detectron2
        from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

        if not self.adapting:
            return self.base_model(batched_inputs)

        # 1. Apply pruning mask
        self._apply_pruning_mask()

        # 2. Decomposed forward (Detectron2 GeneralizedRCNN)
        images = self.base_model.preprocess_image(batched_inputs)
        features = self.base_model.backbone(images.tensor)
        if isinstance(features, tuple):
            features = features[0]

        proposals, _ = self.base_model.proposal_generator(images, features, None)
        self._current_proposals = proposals  # store for instance-level sensitivity

        # Extract category-wise features for instance-level alignment
        if self.config.use_instance_sensitivity:
            self._extract_category_feats_from_proposals(proposals, features)

        # ROI heads inference
        self.base_model.roi_heads.training = False
        self.base_model.proposal_generator.training = False
        results, _ = self.base_model.roi_heads(images, features, proposals, None)

        # 3. Compute alignment loss (L_adp = L_img + L_ins)
        loss_align = self._compute_alignment_loss()

        # 4. Compute pruning loss (conditional on pruning rate)
        current_rate = self.get_pruning_rate()
        if current_rate < self.config.pruning_rate:
            sensitivity = self._compute_sensitivity_weights()
            loss_sparse = self._compute_sparse_loss(sensitivity)
            loss_total = (self.config.lambda_align * loss_align
                          + self.config.lambda_sparse * loss_sparse)
        else:
            loss_total = self.config.lambda_align * loss_align

        # 5. Backward + masked gradient update
        self.optimizer.zero_grad()
        loss_total.backward()
        self._mask_gradients()
        self.optimizer.step()

        # 6. Prune or reactivate
        if current_rate < self.config.pruning_rate:
            self._prune_parameters()
        else:
            self._stochastic_reactivation()

        # 7. Track stats
        self._stats["losses"].append(loss_total.item())
        self._stats["pruning_rates"].append(self.get_pruning_rate())

        # Cleanup
        self._current_proposals = None
        self._stage_features.clear()
        if hasattr(self, "_current_category_feats"):
            self._current_category_feats.clear()

        # 8. Postprocess
        results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def reset(self, reset_stats=False):
        """Reset model to initial state (source pre-trained weights)."""
        # Restore base model
        with torch.no_grad():
            self.base_model.load_state_dict(self.base_state)

        # Re-init masks
        self._init_pruning_masks()

        # Reset optimizer
        self._optimizer = None

        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)

        # Re-snapshot BN state
        self._base_bn_state = self._snapshot_bn_state()

        if reset_stats:
            current_stats = self._stats
            self._reset_stats()
            return current_stats
        return None
