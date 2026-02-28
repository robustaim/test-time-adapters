from typing import Dict, List, Iterator
import math
import types
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.modeling.backbone.swin import window_partition, window_reverse

from ....base import AdaptationEngine, BaseModel
from .configuration_whw import WHWConfig


# Adapter Modules: ResNet
def conv1x1_func(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class Conv1x1(nn.Module):
    def __init__(self, planes, out_planes=None, stride=1, mode="parallel", bias=False):
        super().__init__()
        self.mode = mode
        if mode == 'series':
            self.conv = nn.Sequential(nn.BatchNorm2d(planes), conv1x1_func(planes))
        elif mode == 'parallel':
            self.conv = conv1x1_func(planes, out_planes, stride, bias=bias)
        else:
            self.conv = conv1x1_func(planes)

    def forward(self, x):
        y = self.conv(x)
        if self.mode == 'series_adapters':
            y += x
        return y


class Adapter(nn.Module):
    """Adapter module for ResNet BottleneckBlock."""

    def __init__(self, in_planes, out_planes, r=32, mode='parallel'):
        super().__init__()
        self.down_proj = Conv1x1(in_planes, in_planes // r, 1, mode=mode, bias=True)
        self.non_linear_func = nn.ReLU()
        self.up_proj = Conv1x1(in_planes // r, out_planes, 1, mode=mode, bias=True)

        # Initialize weights
        nn.init.kaiming_uniform_(self.down_proj.conv.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.conv.weight)
        nn.init.zeros_(self.down_proj.conv.bias)
        nn.init.zeros_(self.up_proj.conv.bias)

    def forward(self, x):
        return self.up_proj(self.non_linear_func(self.down_proj(x)))


# Adapter Modules: SwinT
class SwinAdapter(nn.Module):
    def __init__(self, config=None, d_model=None, bottleneck=None):
        super().__init__()
        self.n_embd = config.D_MODEL if d_model is None else d_model
        self.down_size = config.BOTTLENECK if bottleneck is None else bottleneck
        self.mean_feats = torch.zeros((1, self.n_embd))
        self.adapter_layernorm_option = config.LAYERNORM_OPTION

        self.adapter_layer_norm_before = None
        if config.LAYERNORM_OPTION == "in" or config.LAYERNORM_OPTION == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        self.scalar_type = config.SCALAR
        self.scalar = nn.Parameter(torch.ones(1)) if config.SCALAR == 'learnable_scalar' else torch.ones(1)
        self.cossim = 0

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = config.DROPOUT
        if config.INIT_OPTION == "bert":
            raise NotImplementedError
        elif config.INIT_OPTION == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        if self.scalar_type == 'cosine_sim':
            self.cossim = F.cosine_similarity(self.mean_feats.to(x.device), x.mean(dim=[0, 1]).unsqueeze(0).detach(), dim=1)
            up = up * self.cossim
            self.mean_feats = 0.9 * self.mean_feats.to(x.device) + 0.1 * x.mean(dim=[0, 1]).unsqueeze(0).detach()
        else:
            up = up * self.scalar.to(up.device)

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


# WHW method
class WHWEngine(AdaptationEngine):
    model_name: str = "WHWEngine"

    def __init__(self, config: WHWConfig, base_model: BaseModel):
        super().__init__(config, base_model)
        self.config = config

    def _pre_init(self):
        # Skip state tracking
        self._idx = 0
        self._pause_iter = 0
        self._cur_used = True
        self._loss_ema99 = 0.0
        self._div_thr = None

        # Adapter parameters list
        self._adapter_params = []

    def _post_init(self):
        # Add adapters to backbone
        self._add_adapters()

        # Change model methods
        self._patch_model_methods()

        # Initialize model attributes
        self._init_model_attributes()

        self.to(self.device)

    def _add_adapters(self):
        if self.config.backbone == "rcnn":
            self._add_resnet_adapters()
        elif self.config.backbone == "swinrcnn":
            self._add_swin_adapters()

    def _add_resnet_adapters(self):
        def _bottleneck_forward(self, x):
            if isinstance(x, tuple):
                x, g1, g2 = x
            out = self.conv1(x)
            out = F.relu_(out)

            out_tmp = self.conv2(out)
            out = F.relu_(out_tmp)

            out = self.conv3(out)

            if self.shortcut is not None:
                shortcut = self.shortcut(x)
            else:
                shortcut = x

            out = out + shortcut + self.adapter(out_tmp)
            out = F.relu_(out)

            return out

        for stage in self.base_model.backbone.bottom_up.stages:
            for block in stage:
                bottleneck_channels = block.conv2.out_channels
                out_channels = block.conv3.out_channels

                # Add adapter
                block.adapter = Adapter(
                    bottleneck_channels, out_channels,
                    r=self.config.adapter_ratio
                )
                block.adapter.to(self.device)
                self._adapter_params.extend(block.adapter.parameters())

                # Replace forward method
                block.forward = types.MethodType(_bottleneck_forward, block)

    def _add_swin_adapters(self):
        adapting_config = SimpleNamespace(
            TYPE="parallel",
            LAYERNORM_OPTION=self.config.adapter_layernorm_option,
            SCALAR=self.config.adapter_scalar,
            INIT_OPTION=self.config.adapter_init_option,
            DROPOUT=self.config.adapter_dropout,
        )
        def _block_forword(self, x, mask_matrix):
            """Forward function.
            Args:
                x: Input feature, tensor size (B, H*W, C).
                H, W: Spatial resolution of the input feature.
                mask_matrix: Attention mask for cyclic shift.
            """
            B, L, C = x.shape
            H, W = self.H, self.W
            assert L == H * W, "input feature has wrong size"

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)
            if hasattr(self.norm1, 'out_batch_norm') and self.norm1.out_batch_norm:
                self.norm1.update(x.clone())

            # pad feature maps to multiples of window size
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None

            # partition windows
            x_windows = window_partition(
                shifted_x, self.window_size
            )  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(
                -1, self.window_size * self.window_size, C
            )  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()

            x = x.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            if self.adapter is not None:
                adapt_x = self.adapter(x)
                x = x + self.drop_path(self.mlp(self.norm2(x))) + adapt_x
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x
        
        for layer in self.base_model.backbone.bottom_up.layers:
            for block in layer.blocks:
                dim = block.mlp.fc1.in_features

                # Add adapter
                block.adapter = SwinAdapter(adapting_config, d_model=dim, bottleneck=dim//self.config.adapter_ratio)
                block.adapter.to(self.device)
                self._adapter_params.extend(block.adapter.parameters())

                # Replace forward method
                block.forward = types.MethodType(_block_forword, block)

    def _patch_model_methods(self):
        model = self.base_model

        # Patch ROI heads _forward_box
        def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], outs=False):
            features_list = [features[f] for f in self.box_in_features]
            box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
            box_features, box_features_pre = self.box_head(box_features, out_features=True)
            predictions = self.box_predictor(box_features)

            if not outs:
                del box_features

            if self.training:
                losses = self.box_predictor.losses(predictions, proposals)
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
                        for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                return losses
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                if outs:
                    return pred_instances, predictions, box_features_pre
                else:
                    return pred_instances

        model.roi_heads._forward_box = types.MethodType(_forward_box, model.roi_heads)

        # Patch box_head forward
        def _box_head_forward(self, x, out_features=False):
            x = self.fc2(self.fc_relu1(self.fc1(self.flatten(x))))
            if out_features:
                return self.fc_relu2(x), x
            else:
                return self.fc_relu2(x)

        model.roi_heads.box_head.forward = types.MethodType(_box_head_forward, model.roi_heads.box_head)

    def _init_model_attributes(self):
        model = self.base_model
        config = self.config

        # Set model attributes
        model.num_classes = config.num_classes
        model.alpha_gl = config.alpha_gl
        model.alpha_fg = config.alpha_fg
        model.ema_gamma = config.ema_gamma
        model.freq_weight = config.freq_weight
        model.gl_align = config.gl_align
        model.fg_align = config.fg_align
        model.collect_iou_thr = config.collect_iou_thr

        # Load source statistics
        if config.source_stats_path is not None:
            import os
            if os.path.exists(config.source_stats_path):
                model.s_stats = torch.load(config.source_stats_path, weights_only=False)
            else:
                print(f"[WHWEngine] Source stats file not found: {config.source_stats_path}. Will collect during fit().")
                model.s_stats = None
        else:
            model.s_stats = None

        # Initialize target statistics
        model.t_stats = {}
        model.template_cov = {}
        model.s_div = None

        if model.gl_align:
            model.t_stats["gl"] = {}
            model.template_cov["gl"] = {}

        if model.fg_align:
            model.t_stats["fg"] = {}
            model.template_cov["fg"] = {}
            model.ema_n = {}

        # For source stats collection
        model.gl_features = {}
        model.fg_features = {}
        model.iou_with_gt = {}

        def initialize(self):
            self.gl_features = {}
            self.fg_features = {}

            if self.gl_align == "KL":
                for k in self.s_stats["gl"]:
                    mean, cov = self.s_stats["gl"][k]
                    self.template_cov["gl"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
                    self.t_stats["gl"][k] = (mean, cov)

            if self.fg_align == "KL":
                for k in self.s_stats["fg"]:
                    mean, cov = self.s_stats["fg"][k]
                    self.template_cov["fg"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
                    self.t_stats["fg"][k] = (mean, cov)
                    self.ema_n[k] = 0

            self.s_div = self.s_stats.get("kl_div", None) if self.s_stats is not None else None

        def adapt(self, batched_inputs):
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)

            if isinstance(features, tuple):
                features = features[0]

            adapt_loss = {}
            feature_sim = {}

            self.roi_heads.training = False
            self.proposal_generator.training = False

            proposals, _ = self.proposal_generator(images, features, None)
            pred_instances, predictions, box_features = self.roi_heads._forward_box(features, proposals, outs=True)
            results = self.roi_heads.forward_with_given_boxes(features, pred_instances)

            # Foreground alignment
            if self.fg_align is not None:
                _scores = nn.Softmax(dim=1)(predictions[0])
                bg_scores = _scores[:, -1]
                fg_scores, fg_preds = _scores[:, :-1].max(dim=1)

                valid = fg_scores >= 0.5
                fg_preds[~valid] = torch.ones((~valid).sum()).long().to(valid.device) * self.num_classes
                fg_scores[~valid] = bg_scores[~valid]

                loss_fg_align = 0
                loss_n = 0

                for _k in fg_preds[fg_preds != self.num_classes].unique():
                    k = _k.item()
                    if k not in self.t_stats["fg"]:
                        continue
                    if k not in self.ema_n:
                        self.ema_n[k] = 0

                    if (fg_preds == k).sum() > 0:
                        cur_feats = box_features[fg_preds == k]
                        self.ema_n[k] += cur_feats.shape[0]

                        diff = cur_feats - self.t_stats["fg"][k][0][None, :].to(self.device)
                        delta = 1 / self.ema_gamma * diff.sum(dim=0)
                        cur_target_mean = self.t_stats["fg"][k][0].to(self.device) + delta

                        t_dist = torch.distributions.MultivariateNormal(
                            cur_target_mean,
                            self.s_stats["fg"][k][1].to(self.device) + self.template_cov["fg"][k].to(self.device)
                        )
                        s_dist = torch.distributions.MultivariateNormal(
                            self.s_stats["fg"][k][0].to(self.device),
                            self.s_stats["fg"][k][1].to(self.device) + self.template_cov["fg"][k].to(self.device)
                        )

                        if self.freq_weight:
                            max_n = max([self.ema_n[_k] for _k in self.ema_n.keys()])
                            class_weight = math.log(max_n / self.ema_n[k])
                            cur_loss_fg_align = min(class_weight + 0.01, 10) ** 2 * (
                                (torch.distributions.kl.kl_divergence(s_dist, t_dist) +
                                 torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2
                            )
                        else:
                            cur_loss_fg_align = (
                                torch.distributions.kl.kl_divergence(s_dist, t_dist) +
                                torch.distributions.kl.kl_divergence(t_dist, s_dist)
                            ) / 2

                        if cur_loss_fg_align < 10**5:
                            loss_fg_align += cur_loss_fg_align
                            self.t_stats["fg"][k] = (cur_target_mean.detach(), None)
                            loss_n += 1

                if loss_n > 0:
                    adapt_loss["fg_align"] = self.alpha_fg * loss_fg_align

            # Global alignment
            if self.gl_align == "KL":
                loss_gl_align = 0
                for k in features.keys():
                    if k not in self.t_stats["gl"]:
                        continue

                    cur_feats = features[k].mean(dim=[2, 3])
                    diff = cur_feats - self.t_stats["gl"][k][0][None, :].to(self.device)
                    delta = 1 / self.ema_gamma * diff.sum(dim=0)
                    cur_target_mean = self.t_stats["gl"][k][0].to(self.device) + delta

                    t_dist = torch.distributions.MultivariateNormal(
                        cur_target_mean,
                        self.s_stats["gl"][k][1].to(self.device) + self.template_cov["gl"][k].to(self.device)
                    )
                    s_dist = torch.distributions.MultivariateNormal(
                        self.s_stats["gl"][k][0].to(self.device),
                        self.s_stats["gl"][k][1].to(self.device) + self.template_cov["gl"][k].to(self.device)
                    )

                    cur_loss_gl_align = (
                        torch.distributions.kl.kl_divergence(s_dist, t_dist) +
                        torch.distributions.kl.kl_divergence(t_dist, s_dist)
                    ) / 2

                    if cur_loss_gl_align < 10**5:
                        loss_gl_align += cur_loss_gl_align
                        self.t_stats["gl"][k] = (cur_target_mean.detach(), None)

                adapt_loss["global_align"] = self.alpha_gl * loss_gl_align

            # Postprocess
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

            return results, adapt_loss, feature_sim

        def inference(self, batched_inputs):
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)

            if isinstance(features, tuple):
                features = features[0]

            proposals, _ = self.proposal_generator(images, features, None)
            results, _ = self.roi_heads(images, features, proposals, None)

            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        def collect_feats(self, batched_inputs):
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)

            if isinstance(features, tuple):
                features = features[0]

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            # Collect backbone features
            for k in features.keys():
                cur_feats = features[k].mean(dim=[2, 3]).detach()
                if k not in self.gl_features:
                    self.gl_features[k] = cur_feats
                else:
                    self.gl_features[k] = torch.cat([self.gl_features[k], cur_feats], dim=0)

            # Collect foreground features
            proposals, _ = self.proposal_generator(images, features, gt_instances)
            pred_instances, predictions, box_features = self.roi_heads._forward_box(features, proposals, outs=True)

            cls_probs = nn.Softmax(dim=1)(predictions[0])

            iou_with_gt = torch.Tensor([]).to(cls_probs.device)
            gt_label = torch.Tensor([]).to(cls_probs.device)

            for i, p in enumerate(proposals):
                if len(gt_instances[i]) > 0:
                    _iou_with_gt, _gt_label_idx = pairwise_iou(p.proposal_boxes, gt_instances[i].gt_boxes).max(dim=1)
                    iou_with_gt = torch.cat([iou_with_gt, _iou_with_gt])
                    _gt_label = torch.where(
                        _iou_with_gt > self.collect_iou_thr,
                        gt_instances[i].gt_classes[_gt_label_idx],
                        self.num_classes
                    )
                    gt_label = torch.cat([gt_label, _gt_label])
                else:
                    iou_with_gt = torch.cat([iou_with_gt, torch.zeros(len(p)).to(self.device)])
                    gt_label = torch.cat([gt_label, torch.ones(len(p), dtype=torch.long).to(self.device) * self.num_classes])

            gt_label = gt_label.type(torch.long)
            box_features = box_features.clone().detach()

            for _gt in torch.unique(gt_label):
                gt = _gt.item()
                pick_idx = torch.randperm((gt_label == gt).sum())[:100]
                cur_feats = box_features[gt_label == gt][pick_idx]
                cur_iou = iou_with_gt[gt_label == gt][pick_idx]

                if gt not in self.fg_features:
                    self.fg_features[gt] = cur_feats
                    self.iou_with_gt[gt] = cur_iou
                else:
                    self.fg_features[gt] = torch.cat([self.fg_features[gt], cur_feats])
                    self.iou_with_gt[gt] = torch.cat([self.iou_with_gt[gt], cur_iou])

        # Apply monkey-patches
        model.initialize = types.MethodType(initialize, model)
        model.adapt = types.MethodType(adapt, model)
        model.inference = types.MethodType(inference, model)
        model.collect_feats = types.MethodType(collect_feats, model)

    def initialize(self):
        if self.base_model.s_stats is None:
            print("[WHWEngine] Source statistics not loaded. Skipping initialization (collection mode).")
            return

        self.base_model.initialize()

        # Initialize skip threshold
        if self.base_model.s_div is not None:
            self._div_thr = 2 * sum(self.base_model.s_div.values()) * self.config.skip_tau

    def fit(self, data_preparation=None, batch_size=1, **kwargs):
        if self.base_model.s_stats is None and self.config.source_stats_path and data_preparation is not None:
            from torch.utils.data import DataLoader
            collate_fn = getattr(data_preparation, 'collate_fn', None)
            dataloader = DataLoader(data_preparation, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            self.collect_source_stats(dataloader)
            self.base_model.s_stats = self.save_source_stats(self.config.source_stats_path)
        self.initialize()

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self._adapter_params)

    @property
    def optimizer(self):
        if self._optimizer is None:
            if self.config.optim == "SGD":
                self._optimizer = torch.optim.SGD(
                    self._adapter_params,
                    lr=self.config.adapt_lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    nesterov=False
                )
            elif self.config.optim == "AdamW":
                self._optimizer = torch.optim.AdamW(
                    self._adapter_params,
                    lr=self.config.adapt_lr,
                    betas=(0.9, 0.999),
                    weight_decay=self.config.weight_decay
                )
        return self._optimizer

    def _reset_stats(self):
        """Reset statistics tracking."""
        self._stats = {
            'losses': [],
            'skipped': 0,
            'updated': 0,
        }

    def reset(self, reset_stats=False):
        if self.config.backbone == "rcnn":
            for stage in self.base_model.backbone.bottom_up.stages:
                for block in stage:
                    if hasattr(block, 'adapter'):
                        adapter = block.adapter
                        nn.init.kaiming_uniform_(adapter.down_proj.conv.weight, a=math.sqrt(5))
                        nn.init.zeros_(adapter.up_proj.conv.weight)
                        nn.init.zeros_(adapter.down_proj.conv.bias)
                        nn.init.zeros_(adapter.up_proj.conv.bias)

        elif self.config.backbone == "swinrcnn":
            for layer in self.base_model.backbone.bottom_up.layers:
                for block in layer.blocks:
                    if hasattr(block, 'adapter') and block.adapter is not None:
                        adapter = block.adapter
                        nn.init.kaiming_uniform_(adapter.down_proj.weight, a=math.sqrt(5))
                        nn.init.zeros_(adapter.up_proj.weight)
                        nn.init.zeros_(adapter.down_proj.bias)
                        nn.init.zeros_(adapter.up_proj.bias)

                        adapter.mean_feats = torch.zeros((1, adapter.n_embd))
                        adapter.cossim = 0

                        if hasattr(adapter, 'scalar') and isinstance(adapter.scalar, nn.Parameter):
                            adapter.scalar.data.fill_(1.0)

                        if adapter.adapter_layer_norm_before is not None:
                            adapter.adapter_layer_norm_before.reset_parameters()

        # Reset skip state
        self._idx = 0
        self._pause_iter = 0
        self._cur_used = True
        self._loss_ema99 = 0.0

        # Optimizer reset
        self._optimizer = None

        # Re-initialize model attributes
        self.initialize()

        self.online(self.adapting)
        self.to(self.device)

        if reset_stats:
            current_stats = self._stats
            self._reset_stats()
            return current_stats
        return None

    def forward(self, batched_inputs):
        if not self.adapting:
            return self.base_model(batched_inputs)

        skip_type = self.config.skip_redundant

        should_adapt = (
            self._cur_used or
            (not self._cur_used and self._pause_iter % self.config.skip_period == 0) or
            (skip_type is not None and 'period' in skip_type and self._idx % self.config.skip_period == 0)
        )

        if not should_adapt:
            self._pause_iter += 1
            self._idx += 1
            self._stats['skipped'] += 1
            with torch.no_grad():
                return self.base_model.inference(batched_inputs)

        # Run adaptation
        results, adapt_loss, feature_sim = self.base_model.adapt(batched_inputs)

        if len(adapt_loss) > 0:
            total_loss = sum(adapt_loss.values())
            gl_loss = adapt_loss.get("global_align", None)

            # Determine if we should update
            self._cur_used = False

            if skip_type is None:
                self._cur_used = True
            else:
                # stat: KL divergence exceeds threshold
                if 'stat' in skip_type and gl_loss is not None and self._div_thr is not None:
                    if gl_loss.item() > self._div_thr:
                        self._cur_used = True

                # period: periodic update
                if not self._cur_used and 'period' in skip_type:
                    if self._idx % self.config.skip_period == 0:
                        self._cur_used = True

                # ema: EMA ratio exceeds threshold
                if not self._cur_used and 'ema' in skip_type and gl_loss is not None:
                    if self._loss_ema99 > 0 and gl_loss.item() / (self._loss_ema99 + 1e-7) > self.config.skip_beta:
                        self._cur_used = True

            # Perform update
            if self._cur_used:
                self.optimizer.zero_grad()
                total_loss.backward()

                if self.config.clip_grad_enabled:
                    if self.config.clip_grad_type == "value":
                        torch.nn.utils.clip_grad_value_(
                            self._adapter_params,
                            self.config.clip_grad_value
                        )
                    else:  # "norm"
                        torch.nn.utils.clip_grad_norm_(
                            self._adapter_params,
                            self.config.clip_grad_value
                        )

                self.optimizer.step()
                self._pause_iter = 0
                self._stats['updated'] += 1
                self._stats['losses'].append(total_loss.item())
            else:
                self._pause_iter = 1
                self._stats['skipped'] += 1

            # Update EMA
            if gl_loss is not None:
                self._loss_ema99 = 0.99 * self._loss_ema99 + 0.01 * gl_loss.item()

        self._idx += 1
        return results

    def collect_source_stats(self, dataloader, max_batches=None):
        from tqdm.auto import tqdm

        model = self.base_model
        model.eval()
        model.gl_features = {}
        model.fg_features = {}
        model.iou_with_gt = {}

        print("[WHWEngine] Collecting source statistics...")

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Collecting stats")):
                if max_batches is not None and i >= max_batches:
                    break
                model.collect_feats(batch)

        print(f"[WHWEngine] Collected {len(model.gl_features)} global feature levels")
        print(f"[WHWEngine] Collected {len(model.fg_features)} foreground classes")

    def save_source_stats(self, save_path):
        import os

        model = self.base_model
        feat_stats = {}

        # Global features statistics
        feat_stats["gl"] = {}
        for k in model.gl_features:
            feats = model.gl_features[k].cpu()
            mean = feats.mean(dim=0)
            cov = torch.cov(feats.T)
            feat_stats["gl"][k] = (mean, cov)

        # Foreground features statistics
        feat_stats["fg"] = {}
        for k in model.fg_features:
            if k == model.num_classes:
                continue
            feats = model.fg_features[k].cpu()
            if feats.shape[0] < 2:
                continue
            mean = feats.mean(dim=0)
            cov = torch.cov(feats.T)
            feat_stats["fg"][k] = (mean, cov)

        # KL divergence for skip threshold
        feat_stats["kl_div"] = {}
        for k in model.gl_features:
            feats = model.gl_features[k].cpu()
            if feats.shape[0] < 20:
                continue
            try:
                dist1 = torch.distributions.MultivariateNormal(
                    feats.mean(dim=0),
                    torch.cov(feats.T) + torch.eye(feats.shape[1]) * 1e-6
                )
                dist2 = torch.distributions.MultivariateNormal(
                    feats[:20, :].mean(dim=0),
                    torch.cov(feats.T) + torch.eye(feats.shape[1]) * 1e-6
                )
                kl_div = (
                    torch.distributions.kl.kl_divergence(dist1, dist2) +
                    torch.distributions.kl.kl_divergence(dist2, dist1)
                ) / 2
                feat_stats["kl_div"][k] = kl_div.item()
            except Exception as e:
                print(f"[WHWEngine] KL divergence calculation failed for {k}: {e}")

        # Save
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save(feat_stats, save_path)

        print(f"[WHWEngine] Saved source statistics to {save_path}")
        return feat_stats
