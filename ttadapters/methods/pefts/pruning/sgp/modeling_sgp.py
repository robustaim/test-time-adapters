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

    Currently supports Detectron2 FasterRCNN model only.

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

        # --- Source-domain statistics ---
        # Pre-computed mean/variance of BN-input features (μ_s, Σ_s in Eq.9),
        # per-stage and per-class RoI features. Loaded from cache by
        # ``_load_source_stats`` or collected from the source dataset by
        # ``fit()`` before TTA begins.
        self.source_stats: dict | None = None

        # --- BN forward-hook captures (refreshed every adaptation forward) ---
        # ``current_bn_feats[name]``: (B, C) spatial-mean of the BN input.
        #     Used as the per-batch sample for the EMA target mean μ_t in L_adp
        #     (Eq.9) and as the source of channel features during ``fit()``.
        # ``current_bn_simg[name]``: (C,) per-channel image-level sensitivity
        #     S_img (Eq.5), computed in the hook as mean over batch+spatial of
        #     |F_t − F̄_s|. The in-hook computation matches the paper exactly
        #     and avoids the Jensen lower bound that ``|mean(F_t) − F̄_s|``
        #     would produce if computed from the spatial mean alone.
        # ``current_bn_spatial[name]``: (B, C, H, W) full spatial BN input,
        #     used for per-BN-layer RoI-Align in the instance sensitivity term
        #     S_ins (Eq.7). Cleared at the end of every forward to free memory.
        self.current_bn_feats: dict[str, torch.Tensor] = {}
        self.current_bn_simg: dict[str, torch.Tensor] = {}
        self.current_bn_spatial: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        # --- ResNet stage hooks (for instance-level RoI features) ---
        # ``_stage_features[name]``: (B, C, H, W) output of each backbone stage,
        #     used to compute per-stage RoI features during ``fit()``.
        # ``_stage_strides[name]``: image-coordinate stride of each stage,
        #     required by RoI-Align spatial_scale.
        self._stage_features: dict[str, torch.Tensor] = {}
        self._stage_hooks: list[torch.utils.hooks.RemovableHook] = []
        self._stage_strides: dict[str, int] = {}

        # --- Pruning masks and trainable parameters ---
        # ``pruning_masks[name]``: (C,) {0, 1} mask. Channel is "pruned" when
        #     mask == 0; ``_apply_pruning_mask`` zeros the corresponding γ and
        #     β so the BN output is exactly 0 (paper Sec 3.4 "channel removed").
        # ``_bn_params``: γ parameters that the optimizer updates. β is frozen
        #     at the source value per paper Sec 4.1 ("learnable scaling
        #     factors only").
        # ``_last_reactivated[name]``: indices of channels just restored by
        #     stochastic reactivation (Eq.14); used by
        #     ``_clear_adam_state_for_reactivated`` so the freshly-restored γ
        #     starts adapting from a clean Adam (m, v) state.
        self.pruning_masks: dict[str, torch.Tensor] = {}
        self._bn_params: list[nn.Parameter] = []
        self._last_reactivated: dict[str, torch.Tensor] = {}

        # --- Adaptation loss (L_adp = L_img + L_ins, Eq.11) running estimates ---
        # ``_target_ema[name]``: μ_t per BN layer for the image-level KL
        #     L_img (Eq.9), updated via exponential moving average across
        #     adaptation steps with momentum ``target_ema_momentum``.
        # ``_target_ema_per_class[k]``: μ_t^k per predicted class for the
        #     intra-class instance KL L_ins (Eq.10). Same EMA scheme.
        # ``_target_class_freq[k]``: running count of foreground RoIs
        #     predicted as class k. Used to compute the inverse-frequency
        #     weight w_k that balances L_ins across rare and common classes.
        self._target_ema: dict[str, torch.Tensor] = {}
        self._target_ema_per_class: dict[int, torch.Tensor] = {}
        self._target_class_freq: dict[int, float] = {}

    def _post_init(self):
        """Initialise components after base model registration."""
        self._validate_provider()
        self._convert_frozen_bn()  # FrozenBatchNorm2d → nn.BatchNorm2d
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
    def _frozen_bn_to_bn(frozen) -> nn.BatchNorm2d:
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

    def _set_backbone_bn_eval(self):
        """Force backbone BN modules to inference-mode normalization.

        After ``_convert_frozen_bn`` swaps FrozenBatchNorm2d for the trainable
        nn.BatchNorm2d, the BN modules default to ``training=True``. In that
        mode the BN forward (a) normalises by the current batch statistics —
        which on the online-TTA batch size of 1 degenerates to InstanceNorm —
        and (b) overwrites the layer's ``running_mean`` / ``running_var`` on
        every step. Both deviate from the paper's intent: the source-domain
        running statistics carried over from the FrozenBN snapshot are
        considered "all other parameters" frozen during adaptation
        (paper Sec 4.1: "we adapt the learnable scaling factors in the BN
        layers while freezing all other parameters pre-trained on the source
        domain").

        Toggling the modules to ``eval()`` restores the source-frozen running
        stats while leaving γ and β trainable — ``BN.eval()`` only flips
        ``self.training=False`` and does NOT affect the ``requires_grad`` of
        the learnable parameters.
        """
        for _, m in self.base_model.backbone.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _setup_bn_hooks(self):
        """Register forward hooks on target BN layers to capture input features."""
        for name, module in self.base_model.named_modules():
            if self._is_target_bn(name, module):
                handle = module.register_forward_hook(self._make_bn_hook(name))
                self._hooks.append(handle)

    def _make_bn_hook(self, name: str):
        def hook(module, input, output):
            x = input[0]
            if x.dim() != 4:
                raise ValueError(f"Unexpected BN input dim={x.dim()} at {name}")

            # Spatial-mean BN feature: (B, C). Used both as the per-batch
            # sample for the EMA target mean μ_t in L_adp and as the source
            # of channel features during ``fit()``.
            self.current_bn_feats[name] = x.mean(dim=(2, 3))

            # Image-level sensitivity S_img (paper Eq.5):
            #   S_img = mean over batch+spatial of |F_t − F̄_s|.
            # Computed in-hook to match the paper exactly without retaining
            # the full spatial feature for the loss path. Computing on the
            # spatial mean alone would yield |mean(F_t) − F̄_s|, which is
            # only a Jensen lower bound of the paper's quantity.
            # The result is detached because S_img is used as a fixed weight
            # in the sparsity loss; no gradient needs to flow through it.
            if (
                self.adapting
                and self.source_stats is not None
                and name in self.source_stats.get("bn", {})
            ):
                src_mean = self.source_stats['bn'][name]['mean'].detach()
                self.current_bn_simg[name] = (
                    (x.detach() - src_mean.view(1, -1, 1, 1))
                    .abs()
                    .mean(dim=(0, 2, 3))
                )  # (C,)

            # Full spatial BN input retained for per-BN-layer instance
            # sensitivity (paper Eq.7) which RoI-Aligns this feature map
            # using the current-step proposals. Detached because the
            # resulting S_ins is also a fixed weight, not a backprop target.
            # ``forward`` clears this dictionary after the sensitivity
            # weights are consumed so memory does not accumulate across
            # adaptation steps.
            if self.config.use_instance_sensitivity:
                self.current_bn_spatial[name] = x.detach()

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

            if feat.dim() == 3:  # (B, L, C) → (B, C, H, W) proxy not possible; skip
                return

            self._stage_features[stage_name] = feat

        return hook

    def _collect_bn_params(self):
        """Collect the BN scaling factors γ that the optimizer will update.

        Paper Sec 4.1 specifies that only the BN scaling factors γ are
        adapted at test time: "we adapt the learnable scaling factors in
        the BN layers while freezing all other parameters pre-trained on
        the source domain". β is therefore explicitly excluded from the
        trainable set via ``requires_grad_(False)``.
        """
        self._bn_params = []
        for name, module in self.base_model.named_modules():
            if self._is_target_bn(name, module):
                self._bn_params.append(module.weight)
                module.bias.requires_grad_(False)

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self._bn_params)

    def _load_source_stats(self):
        """Load a cached source-statistics file, re-fitting when its schema is
        out of date or empty.

        Current schema:
            ``stats['bn'][name] = {'mean': Tensor, 'var': Tensor}``
        Older variants seen in the wild:
            * mean-only legacy: ``stats['bn'][name]`` is a Tensor.
            * empty/missing ``stats['bn']`` from corrupted runs.
        Both legacy variants force a re-fit so the engine never operates on
        stale or partial statistics.
        """
        path = self.config.source_stats_path
        if not (path and os.path.exists(path)):
            self.source_stats = None
            return

        loaded = torch.load(path, map_location=self.device, weights_only=False)
        bn_dict = loaded.get("bn") if isinstance(loaded, dict) else None

        # Re-fit on (a) missing/empty bn dict, OR (b) legacy mean-only schema.
        if not bn_dict or isinstance(next(iter(bn_dict.values())), torch.Tensor):
            if self.config.verbose:
                print(
                    f"[{self.model_name}] Legacy or empty source_stats schema at {path}; "
                    f"re-fit required."
                )
            self.source_stats = None
            return

        # Backward compatibility for caches produced before per-class instance
        # alignment (paper Eq.10) and per-BN-layer instance sensitivity
        # (paper Eq.7) were added. Missing keys default to empty dicts so the
        # corresponding loss / sensitivity terms degenerate to a no-op until
        # the user regenerates the cache via ``fit()``.
        if "roi_per_class" not in loaded:
            loaded['roi_per_class'] = {}

        if "bn_roi" not in loaded:
            loaded['bn_roi'] = {}

        self.source_stats = loaded

        if self.config.verbose:
            print(f"[{self.model_name}] Loaded source stats from {path}")

    def _update_target_ema(self, name: str, batch_mean: torch.Tensor) -> torch.Tensor:
        """Update and return the EMA-tracked target mean μ_t for one BN layer.

        Implements ``μ_t ← α·EMA_old + (1−α)·batch_mean`` (paper Eq.9,
        Sec 3.4 "we estimate the mean of the test features μ_t using an
        exponentially moving average") with momentum
        ``self.config.target_ema_momentum``.

        The returned tensor is a *live* μ_t whose gradient flows through
        the current ``batch_mean`` so L_adp can push the current activations
        toward the source mean. Between steps the buffered EMA is kept as a
        detached snapshot to avoid retaining the prior step's autograd graph.

        On first encounter of a layer the buffer is initialised from the
        source mean μ_s, giving a deterministic warm start
        (step-1 μ_t = α·μ_s + (1−α)·batch_mean) so the very first KL is
        meaningful even before the EMA has accumulated history.
        """
        α = self.config.target_ema_momentum
        if name not in self._target_ema:
            self._target_ema[name] = self.source_stats['bn'][name]['mean'].detach().clone()

        # Live μ_t: detached history + (1-α)·batch_mean (graph-attached).
        μ_t_for_loss = α * self._target_ema[name] + (1 - α) * batch_mean

        # Snapshot for next step (detach to break the graph between steps).
        self._target_ema[name] = μ_t_for_loss.detach()

        return μ_t_for_loss

    def _recompute_stateless_mask(self):
        """Recompute the per-step pruning mask from the current γ values.

        Implements paper Algorithm 1 line 2 / Eq.12 verbatim:
        ``M = 1[γ ≥ t]`` evaluated every adaptation step using the *signed* γ
        (``γ < t`` includes negative values, matching the paper's prune
        criterion). The mask is stateless: a channel whose γ later recovers
        above the threshold — for example because stochastic reactivation
        restored it to the source value — automatically returns to alive on
        the next step without any monotonic bookkeeping.
        """
        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name in self.pruning_masks:
                    alive = (module.weight >= self.config.pruning_threshold).float()
                    self.pruning_masks[name] = alive.to(self.device)

    def _clear_adam_state_for_pruned(self):
        """Zero Adam (m, v) for currently-pruned channels.

        Adam's running moments would otherwise persist while the channel is
        pruned. When stochastic reactivation later restores γ to the source
        value, those stale (m, v) would produce an unwarranted large first
        update that immediately overrides the restored γ — defeating the
        purpose of reactivation. Clearing them keeps the post-reactivation
        trajectory close to a freshly-initialised Adam optimizer.

        Bias-correction note: ``1/(1−β₁^step) ≈ 1`` once ``step`` is large,
        so the first non-zero gradient following the re-zero produces an
        update of magnitude ``lr·sign(g)``, identical to a freshly-started
        Adam optimizer (no under-weighting from the cumulative step count).
        """
        if self._optimizer is None:
            return

        for name, module in self.base_model.named_modules():
            if name not in self.pruning_masks:
                continue

            mask = self.pruning_masks[name]
            state = self.optimizer.state.get(module.weight)

            if not state:
                continue

            dead = (mask == 0)
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][dead] = 0

    def _clear_adam_state_for_reactivated(self):
        """Zero Adam (m, v) for channels reactivated this step.

        Counterpart of ``_clear_adam_state_for_pruned``: the paper's
        reactivation rule (Eq.14) explicitly resets γ to its source
        pre-trained value, so the Adam optimizer state for that channel
        should match the same "fresh start" semantics rather than carrying
        moments from before the pruning episode.
        """
        if self._optimizer is None or not self._last_reactivated:
            return

        module_by_name = dict(self.base_model.named_modules())
        for name, indices in self._last_reactivated.items():
            module = module_by_name.get(name)
            if module is None:
                continue

            state = self.optimizer.state.get(module.weight)
            if not state:
                continue

            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][indices] = 0

        self._last_reactivated = {}

    def fit(
        self,
        source_dataset: DataPreparation,
        batch_size: int = 1,
        max_samples: int = 50000,
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

        # Online accumulators for the four kinds of source statistics:
        #   bn_runners[name]            → BN-input feature stats (μ_s, Σ_s for L_img / Eq.9).
        #   roi_runners[stage]          → per-stage RoI feature stats (image-level S_img source).
        #   roi_per_class_runners[k]    → per-class RoI feature stats (μ_s^k, Σ_s^k for L_ins / Eq.10).
        #   bn_roi_runners[name]        → per-BN-layer RoI feature stats (f̄_s for S_ins / Eq.7).
        bn_runners: dict[str, RunningStats] = {}
        roi_runners: dict[str, RunningStats] = {}
        roi_per_class_runners: dict[int, RunningStats] = {}
        bn_roi_runners: dict[str, RunningStats] = {}

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

                    # BN features
                    for name, feats in self.current_bn_feats.items():
                        if name not in bn_runners:
                            bn_runners[name] = RunningStats()
                        bn_runners[name].update(feats)

                    # Instance-level RoI statistics: per-stage source means
                    # (used by image-level S_img weighting), per-class source
                    # means/variances (paper Eq.10), and per-BN-layer source
                    # means (paper Eq.7).
                    if self.config.use_instance_sensitivity and self._stage_features:
                        self._collect_roi_stats_for_fit(
                            batch, roi_runners, roi_per_class_runners, bn_roi_runners,
                        )
                    # Free the per-BN spatial captures held by the hook so
                    # they do not accumulate across fit batches.
                    self.current_bn_spatial.clear()

                    sample_count += feats.shape[0] if self.current_bn_feats else batch_size
                    pbar.update(1)

        # Materialise the source-statistics dict consumed by the alignment
        # losses and sensitivity weights:
        #   "bn"           → mean and variance per BN layer for the closed-form
        #                    KL of L_img (paper Eq.9, requires both moments).
        #   "roi"          → per-stage RoI mean/variance.
        #   "roi_per_class"→ per-class RoI mean/variance for the intra-class
        #                    KL of L_ins (paper Eq.10).
        #   "bn_roi"       → per-BN-layer RoI mean/variance, the f̄_s used by
        #                    instance sensitivity S_ins (paper Eq.7).
        # Variances are clamped at ``source_var_floor`` to prevent the KL
        # term from diverging on rare near-zero σ²_s channels.
        floor = self.config.source_var_floor
        stats: dict = {'bn': {}, 'roi': {}, 'roi_per_class': {}, 'bn_roi': {}}
        for name, runner in bn_runners.items():
            stats['bn'][name] = {
                'mean': runner.mean().to(self.device),
                'var': runner.var().to(self.device).clamp(min=floor),
            }
        for stage_name, runner in roi_runners.items():
            stats['roi'][stage_name] = {
                'mean': runner.mean().to(self.device),
                'var': runner.var().to(self.device).clamp(min=floor),
            }
        for class_k, runner in roi_per_class_runners.items():
            stats['roi_per_class'][int(class_k)] = {
                'mean': runner.mean().to(self.device),
                'var': runner.var().to(self.device).clamp(min=floor),
                'count': int(runner.n),
            }
        for name, runner in bn_roi_runners.items():
            stats['bn_roi'][name] = {
                'mean': runner.mean().to(self.device),
                'var': runner.var().to(self.device).clamp(min=floor),
            }

        self.source_stats = stats

        # Optionally persist
        if self.config.source_stats_path:
            os.makedirs(os.path.dirname(self.config.source_stats_path) or ".", exist_ok=True)
            torch.save(stats, self.config.source_stats_path)

        print(f"[{self.model_name}] Source stats collected: "
              f"{len(stats['bn'])} BN layers, {len(stats['roi'])} stages, "
              f"{len(stats['roi_per_class'])} classes (per-class RoI), "
              f"{len(stats['bn_roi'])} per-BN RoI groups.")

        # γ-source transferability check.
        # The paper's prune threshold t=0.05 was tuned on ResNet-18, whose γ
        # distribution differs from this codebase's ResNet-50-FPN. Channels
        # with γ_source already below the threshold get pruned at step 1, and
        # the stochastic reactivation (Eq.14) cannot rescue them either —
        # restoring γ to a value that is itself below threshold leaves the
        # channel immediately re-pruned. We surface a warning when this
        # fraction is non-trivial so the user can revisit the threshold or
        # the ``exclude_layers`` preset before launching adaptation.
        below = 0
        total = 0
        for name, module in self.base_model.named_modules():
            if self._is_target_bn(name, module):
                below += int((torch.abs(module.weight.detach()) < self.config.pruning_threshold).sum().item())
                total += int(module.weight.numel())
        if total > 0 and (below / total) >= 0.01:
            print(
                f"[{self.model_name}] WARNING: {below}/{total} ({below / total:.1%}) source γ "
                f"values fall below pruning_threshold={self.config.pruning_threshold}. "
                f"These channels will be pruned at step 1 and reactivation per Eq.14 may be "
                f"futile (γ_source < t). Consider raising the threshold or excluding the "
                f"affected layer prefixes."
            )

    def _collect_roi_stats_for_fit(self, batch, roi_runners: dict, roi_per_class_runners: dict, bn_roi_runners: dict):
        """Populate the per-stage, per-BN-layer, and per-class RoI source-stat
        runners for one batch of source data.

        Three RoI feature pools are collected from the same set of foreground
        proposals so that downstream loss / sensitivity computations can
        reuse the source distributions consistently:

        * ``roi_runners[stage_name]``: per-stage RoI features (image-level).
        * ``bn_roi_runners[name]``: per-BN-layer RoI features used as the
          source mean f̄_s in instance sensitivity S_ins (paper Eq.7).
        * ``roi_per_class_runners[class_k]``: per-class RoI features taken
          at the deepest backbone stage (2048-channel res5 output), used as
          μ_s^k / Σ_s^k for the intra-class instance KL (paper Eq.10).

        Per-class collection uses post-NMS detections from the source model
        with confidence above ``fg_confidence_threshold``; per-stage and
        per-BN collections use RPN proposals filtered by foreground
        objectness (paper Sec 3.3: "RoIs with background confidence < 0.5").
        """
        images = self.base_model.preprocess_image(batch)
        features = self.base_model.backbone(images.tensor)

        proposals, _ = self.base_model.proposal_generator(images, features, None)
        # Filter RPN proposals by foreground objectness so subsequent RoI
        # statistics only see likely-foreground regions (paper Sec 3.3).
        conf_thr_rpn = self.config.fg_confidence_threshold
        proposal_boxes = []
        for p in proposals:
            if hasattr(p, "objectness_logits"):
                keep = p.objectness_logits.sigmoid() > conf_thr_rpn
                proposal_boxes.append(p.proposal_boxes.tensor[keep])
            else:
                proposal_boxes.append(p.proposal_boxes.tensor)

        # Per-stage RoI features (image-level source distribution).
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

        # Per-BN-layer RoI features (paper Eq.7). Each target BN layer gets
        # its own f̄_s computed by RoI-Aligning the layer's preceding spatial
        # feature map. Stride is approximated by the BN's containing stage —
        # Detectron2 ResNet stages share a uniform output stride and any
        # within-stage downsampling is internal. BN layers excluded from
        # pruning (for example ``res5`` and ``conv3.norm`` under the
        # RESNET_FPN preset) are skipped.
        for name, spatial in self.current_bn_spatial.items():
            if name not in self.pruning_masks:
                continue

            matched_stage = self._match_bn_to_stage(name)
            stride_bn = self._stage_strides.get(matched_stage) if matched_stage else None
            if stride_bn is None:
                continue

            roi_feats_bn = self._extract_roi_features(spatial, proposal_boxes, stride_bn)
            if roi_feats_bn is None or roi_feats_bn.shape[0] == 0:
                continue

            pooled_bn = roi_feats_bn.mean(dim=(2, 3))  # (M, C)
            if name not in bn_roi_runners:
                bn_roi_runners[name] = RunningStats()

            bn_roi_runners[name].update(pooled_bn)

        # Per-class RoI features at stage_3 (Eq.10 L_ins source stats).
        results, _ = self.base_model.roi_heads(images, features, proposals, None)
        stage3_feat = self._stage_features.get("stage_3")
        stride3 = self._stage_strides.get("stage_3")

        if stage3_feat is None or stride3 is None:
            return

        rois_list = []
        classes_list = []
        conf_thr = self.config.fg_confidence_threshold
        for batch_idx, instances in enumerate(results):
            if len(instances) == 0:
                continue

            scores = instances.scores
            keep = scores > conf_thr
            if not keep.any():
                continue

            kept_boxes = instances.pred_boxes.tensor[keep]
            kept_classes = instances.pred_classes[keep]
            batch_col = torch.full(
                (kept_boxes.shape[0], 1), batch_idx,
                device=kept_boxes.device, dtype=kept_boxes.dtype,
            )
            rois_list.append(torch.cat([batch_col, kept_boxes], dim=1))
            classes_list.append(kept_classes)

        if not rois_list:
            return

        rois = torch.cat(rois_list, dim=0)
        classes = torch.cat(classes_list, dim=0)
        pooled = roi_align(
            stage3_feat, rois,
            output_size=self.config.roi_output_size,
            spatial_scale=1.0 / stride3,
            aligned=True,
        )
        pooled_mean = pooled.mean(dim=(2, 3))  # (M, C)
        for k in classes.unique().tolist():
            mask = (classes == k)
            if not mask.any():
                continue

            class_feats = pooled_mean[mask]  # (m_k, C)
            if k not in roi_per_class_runners:
                roi_per_class_runners[k] = RunningStats()

            roi_per_class_runners[k].update(class_feats)

    def _init_pruning_masks(self):
        self.pruning_masks = {}
        for name, module in self.base_model.named_modules():
            if self._is_target_bn(name, module):
                self.pruning_masks[name] = torch.ones_like(module.weight.data).to(self.device)

    def _apply_pruning_mask(self):
        """Mask γ AND β for pruned channels (paper Sec 3.4: pruned channel +
        preceding conv filter are removed → BN output is 0, not β_source).

        Note: β is frozen in the gradient sense (``requires_grad=False``), but
        we still write 0 to its ``.data`` for pruned channels and restore
        from snapshot on reactivation. Without this, pruned channels output a
        constant β_source which propagates through downstream layers.
        """
        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name in self.pruning_masks:
                    mask = self.pruning_masks[name]
                    module.weight.data *= mask
                    module.bias.data *= mask

    def _mask_gradients(self):
        """Zero γ.grad for currently-pruned channels.

        Only γ is masked because β has ``requires_grad=False`` (paper
        Sec 4.1 freezes everything but the BN scaling factors), so its
        ``.grad`` is always ``None`` and would not have been written by the
        backward pass.
        """
        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name in self.pruning_masks:
                    mask = self.pruning_masks[name]
                    if module.weight.grad is not None:
                        module.weight.grad *= mask

    def _stochastic_reactivation(self):
        """Stochastically restore pruned channels to their source values.

        Implements paper Eq.14: each currently-pruned channel is sampled
        independently from a Bernoulli distribution with probability
        ``reactivation_prob``. Sampled channels have both γ and β reset to
        the source pre-trained snapshot and are marked alive in the mask.
        The indices that were reactivated this step are recorded in
        ``self._last_reactivated`` so the paired Adam-state cleanup
        (``_clear_adam_state_for_reactivated``) can wipe stale optimizer
        moments for those parameters.
        """
        reactivated: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, module in self.base_model.named_modules():
                if name not in self.pruning_masks:
                    continue

                mask = self.pruning_masks[name]
                pruned_idx = (mask == 0)
                if not pruned_idx.any():
                    continue

                # Bernoulli: reactivate each pruned channel with probability r
                reactivate = torch.bernoulli(
                    torch.full_like(mask[pruned_idx], self.config.reactivation_prob)
                ).bool()

                if reactivate.any():
                    # Restore both γ and β to their source pre-trained values.
                    # β is frozen (no gradient), but the forward-time mask in
                    # ``_apply_pruning_mask`` zeros β.data for pruned channels;
                    # reactivation must restore it from snapshot for the channel
                    # to contribute meaningfully again.
                    src_w = self._base_bn_state[f'{name}.weight']
                    src_b = self._base_bn_state[f'{name}.bias']
                    pruned_positions = torch.where(pruned_idx)[0]
                    restore_positions = pruned_positions[reactivate]
                    pos_cpu = restore_positions.cpu()
                    module.weight.data[restore_positions] = src_w[pos_cpu].to(module.weight.device)
                    module.bias.data[restore_positions] = src_b[pos_cpu].to(module.bias.device)
                    mask[restore_positions] = 1
                    reactivated[name] = restore_positions

        self._last_reactivated = reactivated

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
                state[f'{name}.weight'] = module.weight.data.cpu().clone()
                state[f'{name}.bias'] = module.bias.data.cpu().clone()
        return state

    def _compute_alignment_loss(self, results=None) -> torch.Tensor:
        """Composite adaptation loss L_adp = L_img + L_ins (paper Eq.11).

        L_img is the image-level closed-form KL with the shared source
        covariance (Eq.9); L_ins is the per-class intra-class KL with
        inverse-frequency weighting (Eq.10).

        ``results`` (optional) is the list of post-NMS detections from the
        current adapting forward and is required to compute L_ins. When
        ``None`` or when no per-class source statistics are available L_ins
        gracefully degenerates to zero.
        """
        l_img = self._compute_image_alignment_loss()
        l_ins = self._compute_instance_alignment_loss(results)
        return l_img + l_ins

    def _compute_image_alignment_loss(self) -> torch.Tensor:
        """L_img = 0.5 · Σ_c (μ_s − μ_t)² / σ²_s  (arXiv:2506.02462 Eq.9).

        Closed form of D_KL(N(μ_s, Σ_s) ∥ N(μ_t, Σ_s)) for diagonal Σ shared
        between the two Gaussians (the inter-Gaussian terms cancel).
        μ_t is EMA-updated across steps via ``_update_target_ema``.
        """
        if self.source_stats is None:
            return torch.tensor(0.0, device=self.device)
        losses = []
        for name, feats in self.current_bn_feats.items():
            if name not in self.source_stats['bn']:
                continue
            batch_mean = feats.mean(dim=0)
            μ_t = self._update_target_ema(name, batch_mean)
            μ_s = self.source_stats['bn'][name]['mean'].detach()
            σ2_s = self.source_stats['bn'][name]['var'].detach()
            kl = 0.5 * ((μ_s - μ_t) ** 2 / σ2_s).sum()
            losses.append(kl)
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).sum()

    def _update_target_per_class_ema(self, k: int, batch_mean_k: torch.Tensor) -> torch.Tensor:
        """μ_t^k ← α·EMA_old + (1−α)·batch_mean_k for class k (Eq.10).

        Initialised from μ_s^k on first encounter of class k. Returns a *live*
        tensor with gradient through ``batch_mean_k`` (analogous to
        ``_update_target_ema``).
        """
        α = self.config.target_ema_momentum
        if k not in self._target_ema_per_class:
            self._target_ema_per_class[k] = (
                self.source_stats['roi_per_class'][k]['mean'].detach().clone()
            )
        μ_t = α * self._target_ema_per_class[k] + (1 - α) * batch_mean_k
        self._target_ema_per_class[k] = μ_t.detach()
        return μ_t

    def _compute_instance_alignment_loss(self, results) -> torch.Tensor:
        """L_ins = Σ_k w_k · D_KL(N(μ_s^k,Σ_s^k), N(μ_t^k,Σ_s^k))  (arXiv:2506.02462 Eq.10).

        - Per-class KL with shared source covariance (closed form like L_img).
        - w_k = inverse target class frequency (rare classes weighted higher;
          paper Sec 3.4 'dynamically adjust the weight w_k for each category
          based on its frequency in the target domain').
        - RoI features taken from stage_3 (deepest backbone level), RoI-Aligned
          using detector-predicted boxes filtered by ``fg_confidence_threshold``.
        - Source per-class stats live at ``self.source_stats['roi_per_class']``.
        """
        if (
            self.source_stats is None
            or not self.source_stats.get("roi_per_class")
            or results is None
            or not self.config.use_instance_sensitivity
        ):
            return torch.tensor(0.0, device=self.device)

        stage3_feat = self._stage_features.get("stage_3")
        stride3 = self._stage_strides.get("stage_3")
        if stage3_feat is None or stride3 is None:
            return torch.tensor(0.0, device=self.device)

        # Build foreground RoIs from current detections.
        rois_list = []
        classes_list = []
        conf_thr = self.config.fg_confidence_threshold
        for batch_idx, instances in enumerate(results):
            if len(instances) == 0:
                continue

            keep = instances.scores > conf_thr
            if not keep.any():
                continue

            kept_boxes = instances.pred_boxes.tensor[keep]
            kept_classes = instances.pred_classes[keep]
            batch_col = torch.full(
                (kept_boxes.shape[0], 1), batch_idx,
                device=kept_boxes.device, dtype=kept_boxes.dtype,
            )
            rois_list.append(torch.cat([batch_col, kept_boxes], dim=1))
            classes_list.append(kept_classes)

        if not rois_list:
            return torch.tensor(0.0, device=self.device)

        rois = torch.cat(rois_list, dim=0)
        classes = torch.cat(classes_list, dim=0)
        pooled = roi_align(
            stage3_feat, rois,
            output_size=self.config.roi_output_size,
            spatial_scale=1.0 / stride3,
            aligned=True,
        )
        pooled_mean = pooled.mean(dim=(2, 3))  # (M, C) — gradient flows through

        # Update per-class running frequency (used for w_k = 1 / freq_k).
        for k in classes.unique().tolist():
            n_k = int((classes == k).sum().item())
            self._target_class_freq[k] = self._target_class_freq.get(k, 0.0) + n_k

        # Per-class KL, weighted by inverse target frequency.
        total_freq = sum(self._target_class_freq.values()) or 1.0
        per_class_losses = []
        for k in classes.unique().tolist():
            k_int = int(k)
            if k_int not in self.source_stats['roi_per_class']:
                continue  # class never seen in source; skip

            mask_k = (classes == k)
            if not mask_k.any():
                continue

            batch_mean_k = pooled_mean[mask_k].mean(dim=0)  # (C,)
            μ_t = self._update_target_per_class_ema(k_int, batch_mean_k)
            μ_s = self.source_stats['roi_per_class'][k_int]['mean'].detach()
            σ2_s = self.source_stats['roi_per_class'][k_int]['var'].detach()
            kl_k = 0.5 * ((μ_s - μ_t) ** 2 / σ2_s).sum()

            # Inverse frequency weighting (rare classes get higher w_k).
            freq_k = self._target_class_freq[k_int] / total_freq
            w_k = 1.0 / max(freq_k, 1e-6)
            per_class_losses.append(w_k * kl_k)

        if not per_class_losses:
            return torch.tensor(0.0, device=self.device)

        # Paper Eq.10: L_ins = Σ_k w_k · D_KL(...). Direct sum, no per-class
        # normalisation (the prior /num_classes was a magnitude-balancing
        # heuristic not present in the paper).
        return torch.stack(per_class_losses).sum()

    def _compute_sensitivity_weights(self) -> dict[str, torch.Tensor]:
        """Per-BN-layer sensitivity weights ω = w_img + w_ins (paper Eq.4-8).

        Implements:
            * Eq.5  S_img = mean over batch+spatial of |F_t − F̄_s| per channel.
            * Eq.6  w_img = C · S_img / Σ S_img — per-layer normalisation so
                    each layer's weights sum to its channel count C.
            * Eq.7  S_ins = per-BN RoI-feature discrepancy.
            * Eq.8  w_ins = C · S_ins / Σ S_ins — per-layer normalisation.
            * Eq.4  ω = w_img + w_ins, combined per BN layer.

        Per-layer normalisation (rather than a single global min-max) keeps
        each layer's weight budget independent and matches the paper's
        per-layer formulation in Eq.6 and Eq.8.
        """
        sensitivity_img: dict[str, torch.Tensor] = {}

        # --- Image-level S_img per BN layer (paper Eq.5) ---
        # The hook populates ``current_bn_simg[name]`` with the
        # ``mean over batch+spatial of |F_t − F̄_s|`` value while
        # ``self.adapting`` is True. ``_compute_sensitivity_weights`` is
        # only invoked inside the adapting forward, so the hook output is
        # always present at this point.
        for name in self.current_bn_feats:
            if name in self.current_bn_simg and name in self.source_stats['bn']:
                sensitivity_img[name] = self.current_bn_simg[name]

        # --- Instance-level S_ins per BN layer (paper Eq.7) ---
        instance_sens: dict[str, torch.Tensor] = {}
        if self.config.use_instance_sensitivity and self.source_stats.get("bn_roi"):
            instance_sens = self._compute_instance_sensitivity()

        # --- Per-layer normalisation + combination (paper Eq.4, 6, 8) ---
        eps = 1e-8
        sensitivity: dict[str, torch.Tensor] = {}
        for name, s_img in sensitivity_img.items():
            C = float(s_img.numel())
            # w_img: C × S_img / Σ S_img per layer (Eq.6).
            w_img = C * s_img / (s_img.sum() + eps)
            # w_ins: C × S_ins / Σ S_ins per layer when this BN has matching
            # instance sensitivity available (Eq.8).
            if name in instance_sens:
                s_ins = instance_sens[name]
                if s_ins.shape == s_img.shape:
                    w_ins = C * s_ins / (s_ins.sum() + eps)
                    sensitivity[name] = w_img + w_ins
                    continue
            sensitivity[name] = w_img

        return sensitivity

    def _compute_instance_sensitivity(self) -> dict[str, torch.Tensor]:
        """Per-BN-layer instance sensitivity S_ins (paper Eq.7).

        For each target BN layer, RoI-Align its preceding spatial feature
        map (captured by the BN forward hook) using the current batch's
        confidence-filtered proposals, pool over spatial, and compare to
        the per-BN source RoI mean f̄_s. The result is keyed by BN layer
        name so it composes directly with the layer-keyed ``S_img`` in
        ``_compute_sensitivity_weights``.

        Stride is approximated by the BN's containing stage. Detectron2
        ResNet stages share a uniform output stride (within-stage
        downsampling happens internally to the block) so this approximation
        holds for all conv1/conv2/shortcut BN layers within a stage.
        """
        if not self.current_bn_spatial:
            return {}
        if not hasattr(self, "_current_proposals") or self._current_proposals is None:
            return {}
        bn_roi_src = self.source_stats.get("bn_roi", {}) if self.source_stats else {}
        if not bn_roi_src:
            return {}

        # Filter the stored RPN proposals by foreground objectness (paper
        # Sec 3.3) so only likely-foreground regions contribute to S_ins.
        conf_thr = self.config.fg_confidence_threshold
        proposal_boxes = []
        for p in self._current_proposals:
            if hasattr(p, "objectness_logits"):
                keep = p.objectness_logits.sigmoid() > conf_thr
                proposal_boxes.append(p.proposal_boxes.tensor[keep])
            else:
                proposal_boxes.append(p.proposal_boxes.tensor)

        if not any(b.numel() > 0 for b in proposal_boxes):
            return {}

        instance_sens: dict[str, torch.Tensor] = {}
        for name, spatial in self.current_bn_spatial.items():
            if name not in bn_roi_src:
                continue  # BN excluded from pruning or not in source stats

            matched_stage = self._match_bn_to_stage(name)
            stride_bn = self._stage_strides.get(matched_stage) if matched_stage else None
            if stride_bn is None:
                continue

            roi_feats = self._extract_roi_features(spatial, proposal_boxes, stride_bn)
            if roi_feats is None or roi_feats.shape[0] == 0:
                continue

            pooled = roi_feats.mean(dim=(2, 3))  # (M, C)
            curr_mean = pooled.mean(dim=0)       # (C,)
            src_mean = bn_roi_src[name]['mean'].detach()
            instance_sens[name] = torch.abs(curr_mean - src_mean)

        return instance_sens

    def _extract_roi_features_from_stored(
        self, stage_feat: torch.Tensor, stage_name: str
    ) -> torch.Tensor | None:
        """Extract RoI features from a stage feature map using stored proposals.

        Proposals are filtered by RPN foreground objectness above
        ``fg_confidence_threshold`` to satisfy the paper's RoI-selection
        rule (Sec 3.3: "RoIs with background confidence < 0.5"). For the
        Detectron2 RPN, ``proposal.objectness_logits`` is the foreground
        score logit and ``sigmoid > 0.5`` is equivalent to background
        confidence < 0.5.
        """
        if not hasattr(self, "_current_proposals") or self._current_proposals is None:
            return None

        stride = self._stage_strides.get(stage_name)
        if stride is None:
            return None

        conf_thr = self.config.fg_confidence_threshold
        proposal_boxes = []
        for p in self._current_proposals:
            if hasattr(p, "objectness_logits"):
                keep = p.objectness_logits.sigmoid() > conf_thr
                proposal_boxes.append(p.proposal_boxes.tensor[keep])
            else:
                proposal_boxes.append(p.proposal_boxes.tensor)

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
            batch_col = torch.full((boxes.shape[0], 1), batch_idx, device=boxes.device, dtype=boxes.dtype)
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
            # Per-step trajectories (appended every adaptation forward).
            'losses': [],
            'pruning_rates': [],
            # Round-level diagnostic counters. Accumulated each step in
            # ``forward`` and zeroed by ``dump_round_diag`` after a round
            # ends, so each entry summarises one round's adaptation activity.
            'round_diag': {
                'steps': 0,
                'reactivation_calls': 0,
                'reactivated_channels': 0,
                'loss_align_sum': 0.0,
                'loss_sparse_sum': 0.0,
            },
            # Frozen snapshot of the config for downstream serialisation.
            'config': vars(self.config),
        }

    def forward(self, batched_inputs):
        # Only supports Detectron2
        from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

        if not self.adapting:
            return self.base_model(batched_inputs)

        # ---------- pre-forward hygiene ----------
        # Recompute the per-step pruning mask M = 1[γ ≥ t] from the current
        # γ values (paper Algorithm 1 line 2).
        self._recompute_stateless_mask()
        # Zero γ and β data for currently-pruned channels so their BN
        # output is exactly 0 (paper Sec 3.4 "channel removed" semantics).
        self._apply_pruning_mask()
        # Force backbone BN modules into eval-mode normalisation so the
        # source running stats (carried over from the FrozenBN snapshot)
        # are used instead of batch statistics on the online B=1 input.
        self._set_backbone_bn_eval()
        # Wipe Adam (m, v) for currently-pruned channels so a future
        # reactivation does not inherit stale optimizer momentum.
        self._clear_adam_state_for_pruned()

        # ---------- decomposed forward ----------
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

        # ---------- loss ----------
        # Single Adam (paper-faithful): combined L_total drives one optimizer
        # step. Adam's sign-like updates implicitly select per-channel
        # adapt-vs-prune behaviour based on which loss component dominates the
        # combined gradient sign for that channel.
        loss_align = self._compute_alignment_loss(results=results)
        current_rate = self.get_pruning_rate()
        sensitivity = None
        loss_sparse = torch.tensor(0.0, device=self.device)
        if current_rate < self.config.pruning_rate:
            sensitivity = self._compute_sensitivity_weights()
            loss_sparse = self._compute_sparse_loss(sensitivity)

        # ---------- single-Adam combined-loss backward (paper-faithful) ----------
        # Eq.13: L_total = λ_align · L_adp + λ_sparse · L_wreg  (when ρ < p);
        # else L_total = λ_align · L_adp only.
        if sensitivity is not None:
            loss_total = (
                self.config.lambda_align * loss_align
                + self.config.lambda_sparse * loss_sparse
            )
        else:
            loss_total = self.config.lambda_align * loss_align
        self.optimizer.zero_grad()
        loss_total.backward()
        self._mask_gradients()
        self.optimizer.step()

        # ---------- reactivation per Eq.14 only when ρ ≥ p (Algorithm 1 line 9) ----------
        # NOTE: no explicit prune call here — the per-step _recompute_stateless_mask
        # at the top of the next forward IS the prune step (Algorithm 1 line 2).
        if current_rate >= self.config.pruning_rate:
            self._stochastic_reactivation()
            if self._last_reactivated:
                self._stats['round_diag']['reactivation_calls'] += 1
                self._stats['round_diag']['reactivated_channels'] += sum(
                    int(idx.numel()) for idx in self._last_reactivated.values()
                )
            self._clear_adam_state_for_reactivated()

        # Round diagnostic accumulation
        self._stats['round_diag']['steps'] += 1
        self._stats['round_diag']['loss_align_sum'] += float(loss_align.detach().item())
        if current_rate < self.config.pruning_rate:
            self._stats['round_diag']['loss_sparse_sum'] += float(loss_sparse.detach().item())

        # ---------- stats / cleanup ----------
        # Append the scaled total loss being minimised this step
        # (paper Eq.13). When ρ ≥ p the L_wreg term degenerates to zero
        # because the conditional above skipped its computation.
        composite_loss = (
            self.config.lambda_align * loss_align.detach().item()
            + self.config.lambda_sparse * loss_sparse.detach().item()
        )
        self._stats['losses'].append(composite_loss)
        self._stats['pruning_rates'].append(self.get_pruning_rate())
        self._current_proposals = None
        self._stage_features.clear()
        # Release the per-BN spatial captures held by the hook so memory
        # does not accumulate across adaptation steps.
        self.current_bn_spatial.clear()

        # ---------- postprocess ----------
        results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def dump_round_diag(self, round_num: int):
        """Print round-level diagnostics: pruning state, reactivation activity,
        γ distribution, mean losses. Resets the diagnostic counters."""
        d = self._stats['round_diag']
        steps = max(d['steps'], 1)

        # γ distribution across all target BN layers
        all_g = []
        for name, module in self.base_model.named_modules():
            if name in self.pruning_masks:
                all_g.append(module.weight.data.detach().cpu().flatten())

        all_g = torch.cat(all_g) if all_g else torch.zeros(1)
        below = int((all_g < self.config.pruning_threshold).sum().item())
        n_total = all_g.numel()

        # Sorted percentiles
        s, _ = all_g.sort()
        p = lambda f: s[min(int(f * (n_total - 1)), n_total - 1)].item()
        rate = self.get_pruning_rate()

        print(f"[{self.model_name}] === Round {round_num} diagnostics ===")
        print(f"  steps={d['steps']}  pruning_rate={rate:.2%} ({int(rate*n_total)}/{n_total} masked)")
        print(f"  γ<threshold(={self.config.pruning_threshold}): {below}/{n_total} ({below/n_total:.2%})")
        print(f"  γ stats: min={all_g.min().item():.4f}  p1={p(0.01):.4f}  median={p(0.5):.4f}  "
              f"p99={p(0.99):.4f}  max={all_g.max().item():.4f}")
        print(f"  reactivation: calls={d['reactivation_calls']}  total_channels_reactivated={d['reactivated_channels']}")
        print(f"  mean L_adp/step = {d['loss_align_sum']/steps:.4f}  mean L_sparse/step = {d['loss_sparse_sum']/steps:.4f}")

        # Reset counters for next round.
        self._stats['round_diag'] = {
            'steps': 0,
            'reactivation_calls': 0,
            'reactivated_channels': 0,
            'loss_align_sum': 0.0,
            'loss_sparse_sum': 0.0,
        }

    def reset(self, reset_stats=False):
        """Reset model to initial state (source pre-trained weights)."""
        # Restore base model
        with torch.no_grad():
            self.base_model.load_state_dict(self.base_state)

        # Re-init masks
        self._init_pruning_masks()

        # Clear per-episode adaptation buffers so a fresh round starts from
        # a clean slate: the EMA target means (μ_t per layer for L_img and
        # μ_t^k per class for L_ins) and the reactivation/class-frequency
        # bookkeeping must not leak across round boundaries.
        self._target_ema = {}
        self._last_reactivated = {}
        self._target_ema_per_class = {}
        self._target_class_freq = {}

        # Reset optimizer (Adam state cleared)
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
