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

    def _post_init(self):
        """Initialise components after base model registration."""
        self._validate_provider()
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
            if feat.dim() == 3:                    # (B, L, C) → (B, C, H, W) proxy not possible; skip
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
        max_samples: int = 2000,
        dtype=torch.float32,
        **kwargs,
    ):
        """Collect source-domain BN feature statistics (and RoI statistics)."""
        if self.source_stats is not None:
            if self.config.verbose:
                print(f"[{self.model_name}] Source stats already loaded. Skipping fit().")
            return

        if not isinstance(self.base_model, ObjectDetectionMixin):
            raise NotImplementedError("SGP fit() requires an ObjectDetectionMixin model.")

        print(f"[{self.model_name}] Collecting source statistics ({max_samples} samples)...")
        self.base_model.eval()

        bn_runners: dict[str, RunningStats] = {}
        roi_runners: dict[str, RunningStats] = {}

        loader = DataLoader(
            source_dataset, batch_size=batch_size,
            collate_fn=source_dataset.collate_fn, **kwargs
        )

        sample_count = 0
        with torch.no_grad():
            for batch in loader:
                if sample_count >= max_samples:
                    break

                # Forward: triggers BN hooks
                _ = self.base_model(batch)

                # BN features
                for name, feats in self.current_bn_feats.items():
                    if name not in bn_runners:
                        bn_runners[name] = RunningStats()
                    bn_runners[name].update(feats)

                # Instance-level: collect RoI features per stage
                if self.config.use_instance_sensitivity and self._stage_features:
                    self._collect_roi_stats_for_fit(batch, roi_runners)

                sample_count += feats.shape[0] if self.current_bn_feats else batch_size

        # Build stats dict
        stats: dict = {"bn": {}, "roi": {}}
        for name, runner in bn_runners.items():
            stats["bn"][name] = runner.mean().to(self.device)
        for stage_name, runner in roi_runners.items():
            stats["roi"][stage_name] = runner.mean().to(self.device)

        self.source_stats = stats

        # Optionally persist
        if self.config.source_stats_path:
            os.makedirs(os.path.dirname(self.config.source_stats_path) or ".", exist_ok=True)
            torch.save(stats, self.config.source_stats_path)

        print(f"[{self.model_name}] Source stats collected: "
              f"{len(stats['bn'])} BN layers, {len(stats['roi'])} stages.")

    def _collect_roi_stats_for_fit(self, batch, roi_runners: dict):
        """Extract RoI features from stage outputs during fit() and accumulate."""
        images = self.base_model.preprocess_image(batch)
        features = self.base_model.backbone(images.tensor)

        proposals, _ = self.base_model.proposal_generator(images, features, None)
        proposal_boxes = [p.proposal_boxes.tensor for p in proposals]

        for stage_name, stage_feat in self._stage_features.items():
            stride = self._stage_strides.get(stage_name)
            if stride is None:
                continue
            roi_feats = self._extract_roi_features(stage_feat, proposal_boxes, stride)
            if roi_feats is not None and roi_feats.shape[0] > 0:
                pooled = roi_feats.mean(dim=(2, 3))  # (M, C)
                if stage_name not in roi_runners:
                    roi_runners[stage_name] = RunningStats()
                roi_runners[stage_name].update(pooled)

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

                    module.weight.data[restore_positions] = src_w[restore_positions].to(self.device)
                    module.bias.data[restore_positions] = src_b[restore_positions].to(self.device)
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
        """L_adp: BN feature alignment between current and source statistics."""
        if self.source_stats is None:
            return torch.tensor(0.0, device=self.device)

        losses = []
        for name, feats in self.current_bn_feats.items():
            if name not in self.source_stats["bn"]:
                continue
            curr_mean = feats.mean(dim=0)
            curr_std = torch.sqrt(feats.var(dim=0, unbiased=False) + 1e-8)
            src_mean = self.source_stats["bn"][name].detach()

            loss = F.mse_loss(curr_mean, src_mean)
            if self.config.lambda_mu_std > 0:
                # Source std can be computed from RunningStats; approximate as 1.0 if not stored
                loss = loss + self.config.lambda_mu_std * curr_std.mean()
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).sum()

    def _compute_sensitivity_weights(self) -> dict[str, torch.Tensor]:
        """Compute per-BN-layer sensitivity ω = normalize(S_img [+ S_ins]).

        Image-level: |curr_channel_mean - src_channel_mean|
        Instance-level: RoI feature discrepancy at corresponding stage
        """
        sensitivity: dict[str, torch.Tensor] = {}

        # --- Image-level sensitivity (S_img) ---
        for name, feats in self.current_bn_feats.items():
            if name not in self.source_stats["bn"]:
                continue
            curr_mean = feats.mean(dim=0)
            src_mean = self.source_stats["bn"][name].detach()
            sensitivity[name] = torch.abs(curr_mean - src_mean)

        # --- Instance-level sensitivity (S_ins) ---
        if self.config.use_instance_sensitivity and self.source_stats.get("roi"):
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

        # Get proposals from the most recent forward
        # We run a lightweight inference to get proposals
        instance_sens: dict[str, torch.Tensor] = {}

        for stage_name, stage_feat in self._stage_features.items():
            if stage_name not in self.source_stats.get("roi", {}):
                continue

            roi_feats = self._extract_roi_features_from_stored(stage_feat, stage_name)
            if roi_feats is None or roi_feats.shape[0] == 0:
                continue

            pooled = roi_feats.mean(dim=(2, 3))  # (M, C)
            curr_mean = pooled.mean(dim=0)        # (C,)
            src_mean = self.source_stats["roi"][stage_name].detach()

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

        # spatial_scale = 1 / stride maps input-image coordinates to feature-map coordinates
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
        """L_wreg = Σ ‖ω_i · γ_i‖₁  (weighted sparsity on BN scaling factors)."""
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

        # ROI heads inference
        self.base_model.roi_heads.training = False
        self.base_model.proposal_generator.training = False
        results, _ = self.base_model.roi_heads(images, features, proposals, None)

        # 3. Compute alignment loss
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
