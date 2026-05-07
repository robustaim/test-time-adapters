import re
import os
from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align, box_iou

from ....base import AdaptationEngine, BaseModel
from .....models.base import ModelProvider, DataPreparation, ObjectDetectionMixin
from .....utils.validator import DetectionEvaluator
from .configuration_sgp import SGPConfig


class WelfordStats:
    """GPU-resident online accumulator. Stays on device until .mean()/.std() called."""

    def __init__(self):
        self.n = 0
        self.sum = None
        self.sum_sq = None

    def update(self, x: torch.Tensor):
        """x: (N, C) tensor. Stays on its current device."""
        x = x.detach()
        if x.dtype != torch.float32:
            x = x.float()
        s = x.sum(dim=0)
        sq = (x * x).sum(dim=0)
        if self.sum is None:
            self.sum = s.clone()
            self.sum_sq = sq.clone()
        else:
            self.sum += s
            self.sum_sq += sq
        self.n += x.shape[0]

    def mean(self) -> torch.Tensor:
        return self.sum / max(self.n, 1)

    def var(self) -> torch.Tensor:
        return (self.sum_sq / max(self.n, 1)) - (self.mean() ** 2)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var().clamp(min=1e-8))


class SpatialFeatureAccumulator:
    """Accumulates per-channel spatially-flattened features for Eq. 5 (image-level).

    Stores running |F_t - F_s|_1 reference statistic.
    F_s is the per-channel mean over (N, H, W) of source feature maps.
    """

    def __init__(self):
        self.n_total = 0
        self.sum = None  # (C,)

    def update(self, x: torch.Tensor):
        """x: (B, C, H, W). Stays on its current device."""
        x = x.detach()
        if x.dtype != torch.float32:
            x = x.float()
        # Sum over batch + spatial directly — no reshape needed
        s = x.sum(dim=(0, 2, 3))  # (C,)
        if self.sum is None:
            self.sum = s.clone()
        else:
            self.sum += s
        B, C, H, W = x.shape
        self.n_total += B * H * W

    def mean(self) -> torch.Tensor:
        return self.sum / max(self.n_total, 1)


class SGPEngine(AdaptationEngine):
    """Sensitivity-Guided Pruning Engine for Test-Time Adaptation (CVPR 2025).

    Faithful re-implementation of Wang et al. 2025:
      • Eq. 5/6 image-level sensitivity with spatial preservation + per-layer normalization.
      • Eq. 7/8 instance-level sensitivity using foreground-filtered RoIs.
      • Eq. 9 image-level KL alignment with EMA target tracking.
      • Eq. 10 intra-class instance-level KL alignment with rare-class weighting.
      • Eq. 13 conditional sparsity loss + Eq. 14 stochastic channel reactivation.
    """

    model_name = "SGPEngine"
    config_class = SGPConfig

    def __init__(self, config: SGPConfig, base_model: BaseModel):
        super().__init__(config, base_model)
        self.config: SGPConfig
        self.num_classes = self.base_model.num_classes
        if self.num_classes == 0:
            raise ValueError("num_classes must be set in base_model")

    def _pre_init(self):
        # BN forward features (spatial preserved)
        self.current_bn_feats: dict[str, torch.Tensor] = {}
        self.pruning_masks: dict[str, torch.Tensor] = {}
        self.source_stats: dict | None = None
        self._hooks: list = []
        self._bn_params: list[nn.Parameter] = []

        # Backbone stage features
        self._stage_features: dict[str, torch.Tensor] = {}
        self._stage_hooks: list = []
        self._stage_strides: dict[str, int] = {}

        # Per-class fg RoI features for current batch (Eq. 10)
        self._current_class_feats: dict[int, torch.Tensor] = {}
        self._current_fg_roi_feats: torch.Tensor | None = None

        # EMA-tracked target statistics (Sec 3.4)
        self._target_bn_ema: dict[str, dict[str, torch.Tensor]] = {}
        self._target_class_ema: dict[int, dict[str, torch.Tensor]] = {}
        self._target_class_freq: torch.Tensor | None = None

        # Instance-level: stage used for RoI feature extraction (last ResNet stage)
        self._instance_stage_name: str | None = None

        # Source-side proposals captured during fit() for predicted-RoI statistics
        self._current_proposals_source: list | None = None

    def _post_init(self):
        self._validate_provider()
        self._convert_frozen_bn()
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
            print("=" * 60)
            print(f"[{self.model_name}] Initialisation Summary")
            print("-" * 60)
            print(f"  Target BN layers     : {len(self.pruning_masks)}")
            print(f"  Pruning rate target  : {self.config.pruning_rate:.0%}")
            print(f"  Instance sensitivity : {self.config.use_instance_sensitivity}")
            print(f"  Num classes          : {self.num_classes}")
            print(f"  EMA momentum (γ)     : {self.config.ema_momentum}")
            print(f"  Reactivation prob    : {self.config.reactivation_prob}")
            print("=" * 60)

    def _validate_provider(self):
        if self.model_provider != ModelProvider.Detectron2:
            raise NotImplementedError(
                f"SGPEngine only supports Detectron2 models. Got: {self.model_provider}"
            )

    @staticmethod
    def _frozen_bn_to_bn(frozen) -> nn.BatchNorm2d:
        num_features = frozen.weight.shape[0]
        bn = nn.BatchNorm2d(num_features, eps=frozen.eps)
        bn.weight.data.copy_(frozen.weight)
        bn.bias.data.copy_(frozen.bias)
        bn.running_mean.copy_(frozen.running_mean)
        bn.running_var.copy_(frozen.running_var)
        bn.num_batches_tracked.zero_()
        return bn

    def _convert_frozen_bn(self):
        from detectron2.layers import FrozenBatchNorm2d
        count = 0
        for name, module in list(self.base_model.named_modules()):
            if not isinstance(module, FrozenBatchNorm2d):
                continue
            parts = name.split(".")
            parent = self.base_model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], self._frozen_bn_to_bn(module))
            count += 1
        if self.config.verbose:
            print(f"[{self.model_name}] Converted {count} FrozenBatchNorm2d → nn.BatchNorm2d")

    def _compile_layer_filter(self):
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
        """Capture BN inputs with spatial dimensions preserved (for Eq. 5)."""
        for name, module in self.base_model.named_modules():
            if self._is_target_bn(name, module):
                handle = module.register_forward_hook(self._make_bn_hook(name))
                self._hooks.append(handle)

    def _make_bn_hook(self, name: str):
        def hook(module, input, output):
            x = input[0]
            if x.dim() != 4:
                raise ValueError(f"Unexpected BN input dim={x.dim()} at {name}")
            # Store spatial-preserved feature for Eq. 5 / Eq. 9 computations
            self.current_bn_feats[name] = x
        return hook

    def _setup_stage_hooks(self):
        """Hooks on ResNet backbone stages for instance-level RoI features."""
        backbone = self.base_model.backbone
        bottom_up = getattr(backbone, "bottom_up", backbone)
        resnet_strides = [4, 8, 16, 32]

        if not hasattr(bottom_up, "stages"):
            raise NotImplementedError(
                "SGP instance-level requires a ResNet backbone with .stages attribute."
            )

        for idx, stage in enumerate(bottom_up.stages):
            stage_name = f"stage_{idx}"
            self._stage_strides[stage_name] = resnet_strides[idx]
            handle = stage.register_forward_hook(self._make_stage_hook(stage_name))
            self._stage_hooks.append(handle)

        # Use the deepest available (non-excluded) stage for instance features
        # If res5 is excluded from pruning, we still use it for RoI features
        # because instance-level alignment lives in semantic-rich layers.
        self._instance_stage_name = f"stage_{len(self._stage_strides) - 1}"

    def _make_stage_hook(self, stage_name: str):
        def hook(module, input, output):
            feat = output
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat.dim() == 4:
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
        """Collect source-domain statistics:
            - BN-layer per-channel mean (μ_s) and std (Σ_s) over spatial+batch positions
            - Per-class RoI feature mean (μ_s^k) and std (Σ_s^k) — Eq. 10
            - Foreground RoI mean for instance-level sensitivity — Eq. 7-8

        Following Sec 3.3 of the paper, RoI features are extracted from
        *predicted* RoIs filtered by foreground confidence; class labels for
        per-class statistics are assigned by IoU-matching to GT boxes.
        """
        if self.source_stats is not None:
            if self.config.verbose:
                print(f"[{self.model_name}] Source stats already loaded. Skipping fit().")
            return

        if not isinstance(self.base_model, ObjectDetectionMixin):
            raise NotImplementedError("SGP fit() requires an ObjectDetectionMixin model.")

        print(f"[{self.model_name}] Collecting source statistics ({max_samples} samples)...")
        self.base_model.eval()

        # BN spatial accumulators (one per BN layer)
        bn_spatial: dict[str, SpatialFeatureAccumulator] = {}
        bn_running: dict[str, WelfordStats] = {}

        # Per-class RoI accumulators (Eq. 10)
        class_runners: dict[int, WelfordStats] = {}
        # Instance-level fg RoI accumulator (Eq. 7-8)
        fg_roi_runner = WelfordStats()

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

                    # --- Decomposed Detectron2 forward to capture proposals ---
                    images = self.base_model.preprocess_image(batch)
                    features = self.base_model.backbone(images.tensor)
                    proposals, _ = self.base_model.proposal_generator(images, features, None)
                    self._current_proposals_source = proposals

                    # --- BN statistics (spatial preserved) ---
                    for name, feat in self.current_bn_feats.items():
                        if name not in bn_spatial:
                            bn_spatial[name] = SpatialFeatureAccumulator()
                            bn_running[name] = WelfordStats()
                        bn_spatial[name].update(feat)
                        bn_running[name].update(feat.mean(dim=(2, 3)))

                    # --- Instance-level RoI statistics (predicted RoIs + GT-matched labels) ---
                    if (
                        self.config.use_instance_sensitivity
                        and self._instance_stage_name in self._stage_features
                    ):
                        self._accumulate_source_roi_stats(
                            batch, class_runners, fg_roi_runner
                        )

                    sample_count += batch_size
                    pbar.update(1)

                    # Cleanup per-batch state
                    self._current_proposals_source = None
                    self.current_bn_feats.clear()
                    self._stage_features.clear()

        # --- Build stats dict ---
        stats: dict = {"bn": {}, "class_roi": {}, "fg_roi": {}}

        for name in bn_spatial:
            stats["bn"][name] = {
                "channel_mean": bn_spatial[name].mean().to(self.device),  # (C,) for Eq. 5
                "img_mean": bn_running[name].mean().to(self.device),       # (C,) for Eq. 9 mean
                "img_std": bn_running[name].std().to(self.device),         # (C,) for Eq. 9 cov-weighting
            }

        for k, runner in class_runners.items():
            if runner.n > 0:
                stats["class_roi"][k] = {
                    "mean": runner.mean().to(self.device),
                    "std": runner.std().to(self.device),
                    "count": runner.n,
                }

        if fg_roi_runner.n > 0:
            stats["fg_roi"] = {
                "mean": fg_roi_runner.mean().to(self.device),  # (C,) for Eq. 7 reference
            }

        self.source_stats = stats

        if self.config.source_stats_path:
            os.makedirs(os.path.dirname(self.config.source_stats_path) or ".", exist_ok=True)
            torch.save(stats, self.config.source_stats_path)

        print(f"[{self.model_name}] Source stats: "
              f"{len(stats['bn'])} BN layers, "
              f"{len(stats['class_roi'])} classes, "
              f"fg_roi={'yes' if stats['fg_roi'] else 'no'}.")

    def _accumulate_source_roi_stats(
        self, batch, class_runners: dict, fg_roi_runner: WelfordStats
    ):
        """Source RoI extraction following paper Sec 3.3:
            "based on predicted RoIs and their confidence scores"

        Strategy:
            1. Use *predicted* RoIs from RPN (matches target-time distribution).
            2. Filter by foreground confidence (objectness > threshold).
            3. Assign class labels by IoU-matching predicted RoIs to GT boxes
               (clean labels for per-class statistics in Eq. 10).
            4. Accumulate fg-pooled features for instance-level sensitivity (Eq. 7).
        """
        if self._instance_stage_name not in self._stage_features:
            return
        if self._current_proposals_source is None:
            return

        stage_feat = self._stage_features[self._instance_stage_name]
        stride = self._stage_strides[self._instance_stage_name]

        gt_instances = [x["instances"].to(self.device) for x in batch]
        proposals = self._current_proposals_source

        for img_idx, (prop, gt) in enumerate(zip(proposals, gt_instances)):
            if len(prop) == 0:
                continue

            prop_boxes = prop.proposal_boxes.tensor

            # 1. Foreground filter via objectness (paper: bg conf < 0.5)
            if hasattr(prop, "objectness_logits"):
                objectness = prop.objectness_logits.sigmoid()
                fg_mask = objectness > self.config.fg_confidence_threshold
            else:
                fg_mask = torch.ones(
                    len(prop), dtype=torch.bool, device=prop_boxes.device
                )

            if fg_mask.sum() == 0:
                continue

            fg_boxes = prop_boxes[fg_mask]

            # 2. Pool RoI features for foreground proposals
            roi_feats = self._extract_roi_features(
                stage_feat[img_idx:img_idx+1], [fg_boxes], stride
            )
            if roi_feats is None or roi_feats.shape[0] == 0:
                continue

            pooled_1d = roi_feats.mean(dim=(2, 3))  # (M, C) — for Eq. 7 / Eq. 10

            # 3. Foreground-pool accumulation (Sec 3.3 sensitivity reference, Eq. 7)
            fg_roi_runner.update(pooled_1d)

            # 4. Per-class accumulation via IoU matching to GT (Eq. 10)
            if len(gt) == 0:
                continue
            gt_boxes = gt.gt_boxes.tensor
            gt_classes = gt.gt_classes

            iou = box_iou(fg_boxes, gt_boxes)        # (M, N_gt)
            best_iou, best_gt_idx = iou.max(dim=1)
            matched = best_iou > 0.5
            if matched.sum() == 0:
                continue

            matched_classes = gt_classes[best_gt_idx[matched]]
            matched_feats = pooled_1d[matched]

            for cls_id in matched_classes.unique().tolist():
                cls_mask = (matched_classes == cls_id)
                if cls_mask.sum() == 0:
                    continue
                if cls_id not in class_runners:
                    class_runners[cls_id] = WelfordStats()
                class_runners[cls_id].update(matched_feats[cls_mask])

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
        """Eq. 14: Bernoulli sampling restores pruned channels to source γ."""
        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name not in self.pruning_masks:
                    continue
                mask = self.pruning_masks[name]
                pruned_idx = (mask == 0)
                if not pruned_idx.any():
                    continue

                reactivate = torch.bernoulli(
                    torch.full_like(mask[pruned_idx], self.config.reactivation_prob)
                ).bool()

                if reactivate.any():
                    src_w = self._base_bn_state[f"{name}.weight"]
                    src_b = self._base_bn_state[f"{name}.bias"]
                    pruned_positions = torch.where(pruned_idx)[0]
                    restore_positions = pruned_positions[reactivate]
                    pos_cpu = restore_positions.cpu()
                    module.weight.data[restore_positions] = src_w[pos_cpu].to(module.weight.device)
                    module.bias.data[restore_positions] = src_b[pos_cpu].to(module.bias.device)
                    mask[restore_positions] = 1

                    # Reset Adam state for reactivated channels (so they don't re-zero immediately)
                    if self._optimizer is not None:
                        self._reset_optimizer_state(module.weight, restore_positions)
                        self._reset_optimizer_state(module.bias, restore_positions)

    def _reset_optimizer_state(self, param: nn.Parameter, indices: torch.Tensor):
        """Reset Adam/SGD running state at given channel indices."""
        if self._optimizer is None:
            return
        state = self._optimizer.state.get(param, None)
        if state is None:
            return
        for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
            if key in state and state[key] is not None:
                state[key].data[indices] = 0

    def get_pruning_rate(self) -> float:
        total = pruned = 0
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

    def _compute_image_level_sensitivity(self) -> dict[str, torch.Tensor]:
        """Eq. 5: S_img(c) = (1/ND) Σ_n Σ_{h,w} |F^n_t(c,h,w) - F_s(c)|

        Implementation: for each channel c, compute mean over (B, H, W) of
        absolute deviations from source per-channel mean.
        """
        sens: dict[str, torch.Tensor] = {}
        if self.source_stats is None:
            return sens

        for name, feat in self.current_bn_feats.items():
            if name not in self.source_stats["bn"]:
                continue
            # feat: (B, C, H, W)
            src_chan_mean = self.source_stats["bn"][name]["channel_mean"].to(feat.device)
            # Broadcast (1, C, 1, 1) - subtract from feature, take abs, mean over (B,H,W)
            diff = (feat - src_chan_mean.view(1, -1, 1, 1)).abs()
            S_img = diff.mean(dim=(0, 2, 3))  # (C,)
            sens[name] = S_img.detach()  # detach: sensitivity is a weighting, not a learnable signal
        return sens

    def _compute_instance_level_sensitivity(self) -> dict[str, torch.Tensor]:
        """Eq. 7: S_ins(c) = (1/(M·D_RoI)) Σ_m Σ_{h,w} |f^m_t(c,h,w) - f_s(c)|

        Operates on foreground-filtered RoI features at the instance backbone stage.
        Returns a single sensitivity vector for that stage.
        """
        result: dict[str, torch.Tensor] = {}
        if not self.config.use_instance_sensitivity:
            return result
        if self.source_stats is None or "fg_roi" not in self.source_stats:
            return result
        if not self.source_stats["fg_roi"]:
            return result
        if self._instance_stage_name not in self._stage_features:
            return result
        if not hasattr(self, "_current_fg_roi_feats") or self._current_fg_roi_feats is None:
            return result

        # _current_fg_roi_feats: (M, C, H_roi, W_roi)
        roi_feats = self._current_fg_roi_feats
        if roi_feats.shape[0] == 0:
            return result

        src_mean = self.source_stats["fg_roi"]["mean"].to(roi_feats.device)  # (C,)
        diff = (roi_feats - src_mean.view(1, -1, 1, 1)).abs()
        S_ins = diff.mean(dim=(0, 2, 3))  # (C,)

        result[self._instance_stage_name] = S_ins.detach()
        return result

    def _compute_sensitivity_weights(self) -> dict[str, torch.Tensor]:
        """Eq. 4-8: ω = w_img + w_ins, with per-layer normalization (Eq. 6, 8)."""
        # 1. Image-level (one vector per BN layer)
        img_sens = self._compute_image_level_sensitivity()

        # 2. Instance-level (one vector per stage)
        ins_sens = self._compute_instance_level_sensitivity()

        # 3. Per-layer normalization (Eq. 6): w = C * S / sum(S)
        weights: dict[str, torch.Tensor] = {}
        for name, S in img_sens.items():
            C = S.numel()
            denom = S.sum() + 1e-6
            w_img = C * S / denom

            # Add instance-level contribution if BN layer matches the instance stage
            stage = self._match_bn_to_stage(name)
            if stage in ins_sens:
                S_ins = ins_sens[stage]
                if S_ins.shape == S.shape:
                    C_ins = S_ins.numel()
                    denom_ins = S_ins.sum() + 1e-6
                    w_ins = C_ins * S_ins / denom_ins
                    weights[name] = w_img + w_ins
                else:
                    weights[name] = w_img
            else:
                weights[name] = w_img

        return weights

    def _match_bn_to_stage(self, bn_name: str) -> str | None:
        """Map BN layer name to backbone stage for instance-level sensitivity."""
        for i, stage_key in enumerate(["res2", "res3", "res4", "res5"]):
            if stage_key in bn_name:
                return f"stage_{i}"
        return None

    def _compute_image_alignment_loss(self) -> torch.Tensor:
        """Eq. 9: L_img = D_KL(N(μ_s, Σ_s), N(μ_t, Σ_s))
                       = 0.5 (μ_s - μ_t)^T Σ_s^{-1} (μ_s - μ_t)

        With Σ_s diagonal (per-channel variance), this becomes a covariance-weighted
        squared error on per-channel means. μ_t is tracked via EMA (Sec 3.4).
        """
        if self.source_stats is None:
            return torch.tensor(0.0, device=self.device)

        losses = []
        for name, feat in self.current_bn_feats.items():
            if name not in self.source_stats["bn"]:
                continue
            stats = self.source_stats["bn"][name]
            mu_s = stats["img_mean"].to(feat.device).detach()
            std_s = stats["img_std"].to(feat.device).detach().clamp(min=1e-4)

            # Current batch per-channel mean
            curr_mean = feat.mean(dim=(0, 2, 3))  # (C,)

            # EMA update for μ_t
            mu_t = self._update_bn_ema(name, curr_mean.detach(), std_s.shape)

            # Use the EMA-tracked μ_t for the alignment loss, but allow gradient
            # flow through curr_mean by mixing them
            mu_t_grad = (1 - self.config.ema_momentum) * mu_t + self.config.ema_momentum * curr_mean

            # Covariance-weighted MSE (= KL up to constant for fixed Σ_s)
            diff = (mu_s - mu_t_grad) / std_s
            losses.append((diff ** 2).mean())

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).sum()

    def _update_bn_ema(self, name: str, curr_mean: torch.Tensor, shape) -> torch.Tensor:
        """EMA of per-channel target mean (returned as detached reference)."""
        if name not in self._target_bn_ema:
            self._target_bn_ema[name] = {"mean": curr_mean.clone()}
        else:
            ema = self._target_bn_ema[name]["mean"]
            ema.mul_(1 - self.config.ema_momentum).add_(
                curr_mean, alpha=self.config.ema_momentum
            )
        return self._target_bn_ema[name]["mean"].detach()

    def _compute_instance_alignment_loss(self) -> torch.Tensor:
        """Eq. 10: L_ins = Σ_k w_k · D_KL(N(μ_s^k, Σ_s^k), N(μ_t^k, Σ_s^k))

        Per-class alignment with rare-class up-weighting.
        """
        if not self.config.use_instance_sensitivity:
            return torch.tensor(0.0, device=self.device)
        if not self._current_class_feats:
            return torch.tensor(0.0, device=self.device)
        if self.source_stats is None or not self.source_stats["class_roi"]:
            return torch.tensor(0.0, device=self.device)

        # 1. Update class-frequency EMA in target domain (for w_k)
        self._update_class_freq_ema()

        per_class_losses = []
        per_class_weights = []

        for cls_id, curr_feats in self._current_class_feats.items():
            if cls_id not in self.source_stats["class_roi"]:
                continue
            if curr_feats.shape[0] == 0:
                continue

            src = self.source_stats["class_roi"][cls_id]
            mu_s = src["mean"].to(curr_feats.device).detach()
            std_s = src["std"].to(curr_feats.device).detach().clamp(min=1e-4)

            # Current per-class mean
            curr_mean = curr_feats.mean(dim=0)  # (C,)

            # EMA for μ_t^k
            mu_t = self._update_class_ema(cls_id, curr_mean.detach())
            mu_t_grad = (1 - self.config.ema_momentum) * mu_t + self.config.ema_momentum * curr_mean

            # KL with shared Σ_s^k → covariance-weighted MSE
            diff = (mu_s - mu_t_grad) / std_s
            loss_k = (diff ** 2).mean()

            # Rare-class weight: inverse frequency in target domain
            freq_k = self._target_class_freq[cls_id].item() + 1e-6
            w_k = 1.0 / freq_k

            per_class_losses.append(loss_k)
            per_class_weights.append(w_k)

        if not per_class_losses:
            return torch.tensor(0.0, device=self.device)

        weights = torch.tensor(per_class_weights, device=self.device)
        weights = weights / weights.sum()  # normalize
        losses = torch.stack(per_class_losses)
        return (losses * weights).sum()

    def _update_class_ema(self, cls_id: int, curr_mean: torch.Tensor) -> torch.Tensor:
        if cls_id not in self._target_class_ema:
            self._target_class_ema[cls_id] = {"mean": curr_mean.clone()}
        else:
            ema = self._target_class_ema[cls_id]["mean"]
            ema.mul_(1 - self.config.ema_momentum).add_(
                curr_mean, alpha=self.config.ema_momentum
            )
        return self._target_class_ema[cls_id]["mean"].detach()

    def _update_class_freq_ema(self):
        """EMA-tracked per-class frequency in target domain for w_k."""
        K = self.num_classes
        if self._target_class_freq is None:
            self._target_class_freq = torch.ones(K, device=self.device) / K  # uniform prior

        # Current batch counts
        batch_counts = torch.zeros(K, device=self.device)
        for cls_id, feats in self._current_class_feats.items():
            if 0 <= cls_id < K:
                batch_counts[cls_id] = feats.shape[0]

        total = batch_counts.sum()
        if total > 0:
            batch_freq = batch_counts / total
            self._target_class_freq.mul_(1 - self.config.ema_momentum).add_(
                batch_freq, alpha=self.config.ema_momentum
            )

    def _compute_sparse_loss(self, weights: dict[str, torch.Tensor]) -> torch.Tensor:
        """Eq. 3: L_wreg = Σ_i ‖ω_i · γ_i‖_1"""
        total = torch.tensor(0.0, device=self.device)
        for name, module in self.base_model.named_modules():
            if name in weights and name in self.pruning_masks:
                w = weights[name].to(module.weight.device)
                total = total + (w * torch.abs(module.weight)).sum()
        return total

    def _extract_roi_features(
        self,
        feature_map: torch.Tensor,
        proposal_boxes: list[torch.Tensor],
        stride: int,
    ) -> torch.Tensor | None:
        """RoI-Align on a single-level feature map."""
        if feature_map.dim() != 4:
            return None
        spatial_scale = 1.0 / stride
        rois_list = []
        for batch_idx, boxes in enumerate(proposal_boxes):
            if boxes.numel() == 0:
                continue
            batch_col = torch.full(
                (boxes.shape[0], 1), batch_idx,
                device=boxes.device, dtype=boxes.dtype,
            )
            rois_list.append(torch.cat([batch_col, boxes], dim=1))
        if not rois_list:
            return None
        rois = torch.cat(rois_list, dim=0)
        return roi_align(
            feature_map, rois,
            output_size=self.config.roi_output_size,
            spatial_scale=spatial_scale,
            aligned=True,
        )

    def _build_target_roi_features(self, proposals):
        """Extract foreground RoI features and group them by predicted class.

        Sets:
            self._current_fg_roi_feats : (M, C, H, W) for Eq. 7 sensitivity
            self._current_class_feats  : {cls_id: (N_k, C)} for Eq. 10 alignment
        """
        self._current_fg_roi_feats = None
        self._current_class_feats = {}

        if self._instance_stage_name not in self._stage_features:
            return
        stage_feat = self._stage_features[self._instance_stage_name]
        stride = self._stage_strides[self._instance_stage_name]

        all_fg_feats = []        # for fg_roi sensitivity
        per_class: dict[int, list] = {}

        for img_idx, prop in enumerate(proposals):
            num_props = len(prop)
            if num_props == 0:
                continue

            boxes = prop.proposal_boxes.tensor

            # Pool RoI features
            roi_feats = self._extract_roi_features(
                stage_feat[img_idx:img_idx+1], [boxes], stride
            )
            if roi_feats is None or roi_feats.shape[0] == 0:
                continue

            pooled_2d = roi_feats                  # (N, C, H, W)
            pooled_1d = roi_feats.mean(dim=(2, 3)) # (N, C)

            # --- Foreground filter (bg conf < 0.5) using objectness ---
            if hasattr(prop, "objectness_logits"):
                objectness = prop.objectness_logits.sigmoid()
                fg_mask = objectness > self.config.fg_confidence_threshold
            else:
                fg_mask = torch.ones(num_props, dtype=torch.bool, device=boxes.device)

            if fg_mask.sum() == 0:
                continue

            # Sensitivity (Sec 3.3): all fg RoIs pooled together (with spatial dims)
            all_fg_feats.append(pooled_2d[fg_mask])

            # Alignment (Eq. 10): need per-class grouping. Use detector class scores
            # if available on proposals; otherwise we fall back later via roi_heads output.
            if hasattr(prop, "pred_classes"):
                pred_cls = prop.pred_classes
                for cls_id in pred_cls[fg_mask].unique().tolist():
                    cls_mask = fg_mask & (pred_cls == cls_id)
                    if cls_mask.sum() == 0:
                        continue
                    per_class.setdefault(cls_id, []).append(pooled_1d[cls_mask])

        if all_fg_feats:
            self._current_fg_roi_feats = torch.cat(all_fg_feats, dim=0)

        for cls_id, feat_list in per_class.items():
            self._current_class_feats[cls_id] = torch.cat(feat_list, dim=0)

    def _attach_class_predictions_to_proposals(self, proposals, results):
        """Attach roi_heads predictions to proposals so that we can group RoI
        features by predicted class for Eq. 10.

        Detectron2 GeneralizedRCNN returns 'results' as a list of Instances
        (one per image) with .pred_classes and .scores. We match them back to
        proposals by box IoU.
        """
        for prop, res in zip(proposals, results):
            if len(prop) == 0 or len(res) == 0:
                # No predictions: assign default class -1 (will be filtered out)
                prop.pred_classes = torch.full(
                    (len(prop),), -1, dtype=torch.long, device=prop.proposal_boxes.tensor.device
                )
                continue

            prop_boxes = prop.proposal_boxes.tensor
            pred_boxes = res.pred_boxes.tensor
            pred_classes = res.pred_classes

            iou = box_iou(prop_boxes, pred_boxes)  # (N_prop, N_pred)
            best_iou, best_idx = iou.max(dim=1)

            assigned = torch.full(
                (len(prop),), -1, dtype=torch.long, device=prop_boxes.device
            )
            matched = best_iou > 0.5
            assigned[matched] = pred_classes[best_idx[matched]]
            prop.pred_classes = assigned

    def forward(self, batched_inputs):
        from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

        if not self.adapting:
            return self.base_model(batched_inputs)

        # 1. Apply pruning mask
        self._apply_pruning_mask()

        # 2. Decomposed Detectron2 forward
        images = self.base_model.preprocess_image(batched_inputs)
        features = self.base_model.backbone(images.tensor)
        proposals, _ = self.base_model.proposal_generator(images, features, None)

        # 3. RoI heads inference (gives us predicted classes for grouping)
        prev_roi_train = self.base_model.roi_heads.training
        prev_rpn_train = self.base_model.proposal_generator.training
        self.base_model.roi_heads.training = False
        self.base_model.proposal_generator.training = False
        try:
            roi_outputs = self.base_model.roi_heads(images, features, proposals, None)
            results = roi_outputs[0]
        finally:
            self.base_model.roi_heads.training = prev_roi_train
            self.base_model.proposal_generator.training = prev_rpn_train

        # 4. Attach predicted classes to proposals via IoU matching
        if self.config.use_instance_sensitivity:
            self._attach_class_predictions_to_proposals(proposals, results)

        # 5. Build target-side instance features (fg-pool + per-class)
        if self.config.use_instance_sensitivity:
            self._build_target_roi_features(proposals)

        # 6. Adaptation loss (Eq. 11): L_adp = L_img + L_ins
        loss_img = self._compute_image_alignment_loss()
        loss_ins = self._compute_instance_alignment_loss()
        loss_adp = loss_img + loss_ins

        # 7. Conditional sparsity loss (Eq. 13)
        current_rate = self.get_pruning_rate()
        if current_rate < self.config.pruning_rate:
            sensitivity = self._compute_sensitivity_weights()
            loss_sparse = self._compute_sparse_loss(sensitivity)
            loss_total = (
                self.config.lambda_align * loss_adp
                + self.config.lambda_sparse * loss_sparse
            )
        else:
            loss_total = self.config.lambda_align * loss_adp
            loss_sparse = torch.tensor(0.0, device=self.device)

        # 8. Backward + masked update
        self.optimizer.zero_grad()
        loss_total.backward()
        self._mask_gradients()
        self.optimizer.step()

        # 9. Prune or stochastic-reactivate
        if current_rate < self.config.pruning_rate:
            self._prune_parameters()
        else:
            self._stochastic_reactivation()

        # 10. Stats
        self._stats["losses"].append(float(loss_total.item()))
        self._stats["loss_img"].append(float(loss_img.item()) if torch.is_tensor(loss_img) else 0.0)
        self._stats["loss_ins"].append(float(loss_ins.item()) if torch.is_tensor(loss_ins) else 0.0)
        self._stats["loss_sparse"].append(float(loss_sparse.item()) if torch.is_tensor(loss_sparse) else 0.0)
        self._stats["pruning_rates"].append(self.get_pruning_rate())

        # 11. Cleanup
        self.current_bn_feats.clear()
        self._stage_features.clear()
        self._current_class_feats.clear()
        self._current_fg_roi_feats = None

        # 12. Postprocess
        results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def _reset_stats(self):
        self._stats = {
            "losses": [],
            "loss_img": [],
            "loss_ins": [],
            "loss_sparse": [],
            "pruning_rates": [],
            "config": vars(self.config),
        }

    def reset(self, reset_stats=False):
        """Reset model to initial state (source pre-trained weights)."""
        # Restore base model
        with torch.no_grad():
            self.base_model.load_state_dict(self.base_state)

        # Re-init masks
        self._init_pruning_masks()
        self._target_bn_ema.clear()
        self._target_class_ema.clear()
        self._target_class_freq = None

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
