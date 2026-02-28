from typing import Iterator, List
from tqdm import tqdm
from types import SimpleNamespace
import copy
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from detectron2.structures import Instances
from detectron2.utils.events import EventStorage

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    def non_max_suppression(*args, **kwargs):
        raise NotImplementedError("non_max_suppression is not implemented")

from ....base import AdaptationEngine, BaseModel
from .configuration_test import TeSTConfig


class TeSTEngine(AdaptationEngine):
    model_name: str = "TeSTEngine"

    def __init__(self, config: TeSTConfig, base_model: BaseModel):
        self.config: TeSTConfig
        super().__init__(config, base_model)

    def _pre_init(self):
        self._trainable_params: List[nn.Parameter] = []
        self._init_weights: List[torch.Tensor] = []
        self._teacher_optimizer: optim.Optimizer = None
        self._predictor: nn.Module = None
        self._pixel_mask_cache: dict = {}  # (B, H, W) → pixel_mask tensor

    def _post_init(self):
        self._setup_teacher_model()
        self._setup_augmentation()
        self.to(self.device)

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self._trainable_params)

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None:
            if not self._trainable_params:
                return None
            if self.config.optim == "SGD":
                self._optimizer = optim.SGD(
                    self._trainable_params,
                    lr=self.config.adapt_lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                )
            elif self.config.optim == "Adam":
                self._optimizer = optim.Adam(
                    self._trainable_params,
                    lr=self.config.adapt_lr,
                    weight_decay=self.config.weight_decay,
                )
            else:
                self._optimizer = optim.AdamW(
                    self._trainable_params,
                    lr=self.config.adapt_lr,
                    weight_decay=self.config.weight_decay,
                )
        return self._optimizer

    def _reset_stats(self):
        self._stats = {'losses': [], 'num_batches': 0}

    def reset(self, reset_stats=False):
        self.base_model.load_state_dict(self.base_state)

        self._trainable_params = []
        self._init_weights = []
        self._optimizer = None
        self._teacher_optimizer = None
        self._predictor = None

        self._setup_teacher_model()
        self.online(self.adapting)
        self.to(self.device)

        if reset_stats:
            current_stats = self._stats
            self._reset_stats()
            return current_stats
        return None

    def _setup_teacher_model(self):
        self.teacher_model = copy.deepcopy(self.base_model)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        params = []
        for p in self.base_model.parameters():
            p.requires_grad = True
            params.append(p)
        self._trainable_params = params
        self._init_weights = [p.clone().detach() for p in self._trainable_params]

    def _setup_augmentation(self):
        """
        Weak : rotation in [-10°, 10°] + translation ±10% + random resized crop  (paper §5.1)
        Strong: RandAugment
        """
        rand_aug = T.RandAugment(
            num_ops=self.config.augment_strength_n,
            magnitude=9,
        )

        class _RandAugWrapper:
            def __call__(self, img):
                orig_dtype = torch.float32
                if isinstance(img, torch.Tensor):
                    orig_dtype = img.dtype
                    img = T.ToPILImage()(img.float())
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                img = rand_aug(img)
                return T.ToTensor()(img).to(dtype=orig_dtype)

        self.strong_augment = _RandAugWrapper()

    def _apply_augmentation(self, batch):
        def _weak_aug(img: torch.Tensor) -> torch.Tensor:
            c, h, w = img.shape
            # Random rotation in [-10°, 10°] + translation up to ±10%
            angle = random.uniform(-10.0, 10.0)
            tx = random.uniform(-0.1, 0.1) * w
            ty = random.uniform(-0.1, 0.1) * h
            img = TF.affine(img, angle=angle, translate=[tx, ty], scale=1.0, shear=0)

            # Random resized crop: keep 80–100 % of the image
            scale = random.uniform(0.8, 1.0)
            nh = max(1, int(h * scale))
            nw = max(1, int(w * scale))
            top  = random.randint(0, h - nh)
            left = random.randint(0, w - nw)
            img = TF.resized_crop(img, top, left, nh, nw, (h, w))
            return img

        if self.config.base_type == "rtdetr":
            pixel_values = batch['pixel_values']
            n = pixel_values.shape[0]
            weak_pvs, strong_pvs = [], []
            for i in range(n):
                img = pixel_values[i]
                try:
                    weak_img = _weak_aug(img)
                except Exception:
                    weak_img = img
                weak_pvs.append(weak_img)
                try:
                    strong_pvs.append(self.strong_augment(weak_img))
                except Exception:
                    strong_pvs.append(weak_img)
            return (
                {'pixel_values': torch.stack(weak_pvs),  'labels': batch.get('labels', [None] * n)},
                {'pixel_values': torch.stack(strong_pvs), 'labels': batch.get('labels', [None] * n)},
            )

        elif self.config.base_type in ("rcnn", "swinrcnn"):
            weak_batch, strong_batch = [], []
            for item in batch:
                weak_item = copy.deepcopy(item)
                try:
                    weak_item["image"] = _weak_aug(weak_item["image"])
                except Exception:
                    pass
                weak_batch.append(weak_item)
                strong_item = copy.deepcopy(weak_item)
                try:
                    strong_item["strong_aug_image"] = self.strong_augment(strong_item["image"])
                except Exception:
                    strong_item["strong_aug_image"] = strong_item["image"]
                strong_batch.append(strong_item)
            return weak_batch, strong_batch

        elif self.config.base_type == "yolo11":
            img = batch if torch.is_tensor(batch) else batch['img']
            weak_imgs, strong_imgs = [], []
            for i in range(img.shape[0]):
                frame = img[i]
                try:
                    weak_frame = _weak_aug(frame)
                except Exception:
                    weak_frame = frame
                weak_imgs.append(weak_frame)
                try:
                    strong_imgs.append(self.strong_augment(weak_frame))
                except Exception:
                    strong_imgs.append(weak_frame)
            weak_batch  = {'img': torch.stack(weak_imgs)}
            strong_batch = {'img': torch.stack(strong_imgs)}
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if k != 'img':
                        weak_batch[k]  = v
                        strong_batch[k] = v
            return weak_batch, strong_batch

        return batch, batch

    def _inference(self, model, batched_inputs):
        if self.config.base_type == "rtdetr":
            pixel_values = batched_inputs['pixel_values'].to(self.device)
            return model(pixel_values=pixel_values)
        elif self.config.base_type == "yolo11":
            img = batched_inputs if torch.is_tensor(batched_inputs) else batched_inputs['img']
            return model(img.to(self.device))
        else:  # rcnn
            return model(batched_inputs)

    def _compute_student_loss(self, pseudo_labeled_batch):
        """Hard pseudo-label supervised loss."""
        if self.config.base_type == "rtdetr":
            output = self.base_model(
                pixel_values=pseudo_labeled_batch['pixel_values'].to(self.device),
                labels=pseudo_labeled_batch['labels'],
            )
            loss = output.loss
            if self.config.lambda_ent > 0.0 and output.logits is not None:
                p_s = F.softmax(output.logits, dim=-1)
                ent_loss = -(p_s * p_s.log().clamp(min=-100.0)).sum(dim=-1).mean()
                loss = loss + self.config.lambda_ent * ent_loss
            return loss

        elif self.config.base_type in ("rcnn", "swinrcnn"):
            with EventStorage():
                output = self.base_model(pseudo_labeled_batch)
                if isinstance(output, dict):
                    return sum(output[k] for k in output)
            return None

        elif self.config.base_type == "yolo11":
            if not hasattr(self.base_model, 'args'):
                self.base_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in pseudo_labeled_batch.items()}
            output = self.base_model(batch)
            if isinstance(output, tuple):
                return output[0].sum()
            return output.sum() if output.dim() > 0 else output

        return None

    def _set_pseudo_labels(self, batched_inputs, teacher_outputs):
        """Convert teacher outputs to hard pseudo-label training batch."""
        if self.config.base_type == "rtdetr":
            annotation = []
            for bbox, logit in zip(teacher_outputs.pred_boxes, teacher_outputs.logits):
                probs = F.softmax(logit, dim=-1)
                scores, labels = probs.max(dim=-1)
                mask = scores > self.config.conf_threshold
                annotation.append({
                    'class_labels': labels[mask],
                    'boxes': bbox[mask],
                })
            return {
                'pixel_values': batched_inputs['pixel_values'],
                'labels': annotation,
            }

        elif self.config.base_type in ("rcnn", "swinrcnn"):
            pseudo_labels = []
            for img, label in zip(batched_inputs, teacher_outputs):
                inst = label['instances'][label['instances'].scores > self.config.conf_threshold]
                new_inp = {k: img[k] for k in img if k not in ['instances', 'image']}
                new_inp['image'] = img['image']
                new_img_size = img['instances'].image_size
                ori_img_size = inst.image_size
                new_inst = Instances(new_img_size)
                new_inst.gt_classes = inst.pred_classes
                new_inst.gt_boxes = inst.pred_boxes
                if new_img_size != ori_img_size:
                    new_inst.gt_boxes.scale(
                        new_img_size[1] / ori_img_size[1],
                        new_img_size[0] / ori_img_size[0],
                    )
                new_inp['instances'] = new_inst
                pseudo_labels.append(new_inp)
            return pseudo_labels

        elif self.config.base_type == "yolo11":
            preds = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
            nms_results = non_max_suppression(preds, conf_thres=self.config.conf_threshold)
            img = batched_inputs if torch.is_tensor(batched_inputs) else batched_inputs['img']
            _, _, img_h, img_w = img.shape
            all_cls, all_bboxes, all_batch_idx = [], [], []
            for i, det in enumerate(nms_results):
                if det is not None and len(det) > 0:
                    boxes_xyxy = det[:, :4]
                    cls = det[:, 5]
                    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / img_w
                    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / img_h
                    w  = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
                    h  = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
                    all_cls.append(cls)
                    all_bboxes.append(torch.stack([cx, cy, w, h], dim=1))
                    all_batch_idx.append(
                        torch.full((len(det),), i, dtype=torch.float32, device=det.device))
            device = img.device
            if all_cls:
                return {'img': img,
                        'cls': torch.cat(all_cls).unsqueeze(-1),
                        'bboxes': torch.cat(all_bboxes),
                        'batch_idx': torch.cat(all_batch_idx)}
            else:
                return {'img': img,
                        'cls': torch.zeros((0, 1), device=device),
                        'bboxes': torch.zeros((0, 4), device=device),
                        'batch_idx': torch.zeros(0, device=device)}

        return batched_inputs

    def _get_or_create_predictor(self, C: int) -> nn.Module:
        if self._predictor is None:
            self._predictor = nn.Sequential(
                nn.Linear(C, C), nn.ReLU(), nn.Linear(C, C)
            ).to(self.device)
            # Add predictor params to teacher optimizer if it already exists
            if self._teacher_optimizer is not None:
                self._teacher_optimizer.add_param_group(
                    {'params': list(self._predictor.parameters())}
                )
        return self._predictor

    def _extract_pooled_backbone_features(self, model, batch) -> torch.Tensor:
        if self.config.base_type in ("rcnn", "swinrcnn"):
            images = model.preprocess_image(batch)
            feats = model.backbone(images.tensor)           
            vals = list(feats.values())
            feat = vals[len(vals) // 2]
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]

        if self.config.base_type == "rtdetr":
            pixel_values = batch["pixel_values"].to(self.device)
            B, _, H, W = pixel_values.shape
            key = (B, H, W)
            if key not in self._pixel_mask_cache:
                self._pixel_mask_cache[key] = torch.ones(B, H, W, dtype=torch.long, device=self.device)
            pixel_mask = self._pixel_mask_cache[key]
            backbone_out = model.model.backbone(pixel_values, pixel_mask)
            feat = backbone_out[-1][0]                      # deepest stage → (feat, mask)[0]
            return F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]

        # YOLO11: partial forward — run only layers 0-8 (backbone, ends at SPPF)
        # Hook-based approach fires at layer 8 but the full model (neck+head) still
        # runs; iterating layers and breaking at 8 avoids that wasted compute.
        # model is DetectionModel: model.model = Sequential, model.save = save set
        layers = model.model   # nn.Sequential of YOLO layers
        save   = model.save    # set of layer indices whose outputs must be cached
        img_tensor = (batch if torch.is_tensor(batch) else batch["img"]).to(self.device)
        x = img_tensor
        y: list = []
        for layer in layers:
            if layer.f != -1:
                x = (y[layer.f] if isinstance(layer.f, int)
                     else [x if j == -1 else y[j] for j in layer.f])
            x = layer(x)
            y.append(x if layer.i in save else None)
            if layer.i == 8:  # SPPF — end of YOLO11 backbone
                break
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def _teacher_backbone_params(self) -> List[nn.Parameter]:
        """Return only the backbone parameters of teacher_model (for Stage 1 optimizer)."""
        mt = self.config.base_type
        if mt in ("rcnn", "swinrcnn"):
            return list(self.teacher_model.backbone.parameters())
        if mt == "rtdetr":
            return list(self.teacher_model.model.backbone.parameters())
        if mt == "yolo11":
            params = []
            for layer in self.teacher_model.model:
                params.extend(layer.parameters())
                if layer.i == 8:  # SPPF — end of backbone
                    break
            return params
        return list(self.teacher_model.parameters())

    def _enable_teacher_backbone_grad(self):
        """Freeze whole teacher model, then re-enable only backbone params."""
        self.teacher_model.requires_grad_(False)
        for p in self._teacher_backbone_params():
            p.requires_grad_(True)

    def _compute_teacher_loss(self, weak_batch, strong_batch) -> torch.Tensor:
        # For RCNN/SwinRCNN, swap 'image' key to use the strong-augmented pixels.
        if self.config.base_type in ("rcnn", "swinrcnn"):
            strong_batch_for_feat = [
                {**item, "image": item.get("strong_aug_image", item["image"])}
                for item in strong_batch
            ]
        else:
            strong_batch_for_feat = strong_batch

        # Target branch (strong)
        with torch.no_grad():
            feat_s = self._extract_pooled_backbone_features(
                self.teacher_model, strong_batch_for_feat
            ).detach()

        # Online branch (weak)
        feat_w = self._extract_pooled_backbone_features(self.teacher_model, weak_batch)

        predictor = self._get_or_create_predictor(feat_w.shape[-1])

        L_cons = F.mse_loss(feat_s, predictor(feat_w))

        return self.config.lambda_cons * L_cons

    @property
    def teacher_optimizer(self) -> optim.Optimizer:
        if self._teacher_optimizer is None:
            params = self._teacher_backbone_params()
            if self._predictor is not None:
                params += list(self._predictor.parameters())
            if self.config.optim == "SGD":
                self._teacher_optimizer = optim.SGD(
                    params,
                    lr=self.config.adapt_lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                )
            elif self.config.optim == "Adam":
                self._teacher_optimizer = optim.Adam(
                    params,
                    lr=self.config.adapt_lr,
                    weight_decay=self.config.weight_decay,
                )
            else:
                self._teacher_optimizer = optim.AdamW(
                    params,
                    lr=self.config.adapt_lr,
                    weight_decay=self.config.weight_decay,
                )
        return self._teacher_optimizer

    def fit(self, dataloader, n_epochs: int = None):
        n_epochs = n_epochs or self.config.n_teacher_epochs

        self.teacher_model.train()
        self._enable_teacher_backbone_grad()

        for epoch in range(n_epochs):
            pbar = tqdm(dataloader, desc=f"[TeST Stage 1] Epoch {epoch + 1}/{n_epochs}", leave=True)
            for batch in pbar:
                weak_batch, strong_batch = self._apply_augmentation(batch)

                self.teacher_optimizer.zero_grad()
                loss = self._compute_teacher_loss(weak_batch, strong_batch)
                loss.backward()
                clip_params = self._teacher_backbone_params()
                if self._predictor is not None:
                    clip_params += list(self._predictor.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                self.teacher_optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.config.stage = "student"
        self._predictor = None

    def forward(self, batched_inputs=None, **kwargs):
        if batched_inputs is None and kwargs:
            batched_inputs = kwargs

        if not self.adapting:
            return self._inference(self.base_model, batched_inputs)

        if self.config.stage == "teacher":
            return self._teacher_adapt_step(batched_inputs)
        elif self.config.stage == "online":
            return self._online_both_step(batched_inputs)
        else:
            return self._student_distill_step(batched_inputs)

    def _teacher_adapt_step(self, batched_inputs) -> object:
        self.teacher_model.train()
        self._enable_teacher_backbone_grad()

        weak_batch, strong_batch = self._apply_augmentation(batched_inputs)

        self.teacher_optimizer.zero_grad()
        loss = self._compute_teacher_loss(weak_batch, strong_batch)
        loss.backward()

        clip_params = self._teacher_backbone_params()
        if self._predictor is not None:
            clip_params += list(self._predictor.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
        self.teacher_optimizer.step()

        self._stats["losses"].append(loss.item())
        self._stats["num_batches"] = self._stats.get("num_batches", 0) + 1

        self.teacher_model.eval()
        with torch.no_grad():
            return self._inference(self.teacher_model, batched_inputs)

    def _online_both_step(self, batched_inputs) -> object:
        self.teacher_model.train()
        self._enable_teacher_backbone_grad()

        # Stage 1: K gradient steps on current batch
        weak_batch, strong_batch = self._apply_augmentation(batched_inputs)
        loss1 = torch.tensor(0.0, device=self.device)
        clip_params = self._teacher_backbone_params()
        for _ in range(self.config.n_teacher_steps):
            self.teacher_optimizer.zero_grad()
            loss1 = self._compute_teacher_loss(weak_batch, strong_batch)
            if not torch.isfinite(loss1):
                break  # teacher diverged — stop early, use last valid state
            loss1.backward()
            _cp = clip_params + (list(self._predictor.parameters()) if self._predictor is not None else [])
            torch.nn.utils.clip_grad_norm_(_cp, max_norm=1.0)
            self.teacher_optimizer.step()

        self._stats["losses"].append(loss1.item())
        self._stats["num_batches"] = self._stats.get("num_batches", 0) + 1

        # Freeze teacher (predictor kept alive for next batch's Stage 1)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        torch.cuda.empty_cache()

        return self._student_distill_step(batched_inputs)

    def _backbone_entropy(self, batched_inputs) -> torch.Tensor:
        feat = self._extract_pooled_backbone_features(self.base_model, batched_inputs)
        p = F.softmax(feat, dim=-1)
        return -(p * p.log().clamp(min=-100.0)).sum(dim=-1).mean()

    def _rcnn_kd_loss_with_entropy(self, batched_inputs, teacher_outputs) -> torch.Tensor:
        pseudo_labeled_batch = self._set_pseudo_labels(batched_inputs, teacher_outputs)

        captured_logits: List[torch.Tensor] = []
        handle = None
        try:
            def _hook(m, inp, out):
                # FastRCNNOutputLayers.forward returns (class_logits, box_deltas)
                captured_logits.append(out[0])
            handle = self.base_model.roi_heads.box_predictor.register_forward_hook(_hook)
        except AttributeError:
            pass  

        with EventStorage():
            output = self.base_model(pseudo_labeled_batch)
            loss = sum(output[k] for k in output) if isinstance(output, dict) else None

        if handle is not None:
            handle.remove()

        # L_ent
        if self.config.lambda_ent > 0.0:
            if captured_logits:
                # Per-proposal class distribution [N_proposals, num_classes+1]
                p = F.softmax(captured_logits[0], dim=-1)
                ent_loss = -(p * p.log().clamp(min=-100.0)).sum(dim=-1).mean()
            else:
                ent_loss = self._backbone_entropy(batched_inputs)
            base = loss if loss is not None else torch.tensor(0.0, device=self.device)
            loss = base + self.config.lambda_ent * ent_loss

        return loss

    def _compute_student_loss_with_entropy(self, batched_inputs, pseudo_labeled_batch, teacher_outputs) -> torch.Tensor:
        """Compute L_S = L_KD + λ * H for one gradient step."""
        if self.config.base_type == "rtdetr":
            return self._compute_student_loss(pseudo_labeled_batch)

        elif self.config.base_type in ("rcnn", "swinrcnn"):
            return self._rcnn_kd_loss_with_entropy(batched_inputs, teacher_outputs)

        else:  # yolo11
            captured_scores: List[torch.Tensor] = []
            if self.config.lambda_ent > 0.0:
                def _detect_hook(m, inp, out):
                    if isinstance(out, dict):
                        s = out.get('scores', out.get('one2many', {}).get('scores'))
                        if s is not None:
                            captured_scores.append(s)
                _handle = self.base_model.model[-1].register_forward_hook(_detect_hook)
            loss = self._compute_student_loss(pseudo_labeled_batch)
            if self.config.lambda_ent > 0.0:
                _handle.remove()
                if captured_scores:
                    s = captured_scores[0]
                    p = F.softmax(s, dim=1)
                    ent = -(p * p.log().clamp(min=-100.0)).sum(dim=1).mean()
                    base = loss if loss is not None else torch.tensor(0.0, device=self.device)
                    loss = base + self.config.lambda_ent * ent
            return loss

    def _student_distill_step(self, batched_inputs) -> object:
        # Teacher generates pseudo-labels once (frozen)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self._inference(self.teacher_model, batched_inputs)

        pseudo_labeled_batch = self._set_pseudo_labels(batched_inputs, teacher_outputs)

        # NaN in teacher outputs → skip student update entirely
        if self._has_nan_outputs(teacher_outputs):
            self.base_model.eval()
            with torch.no_grad():
                return self._inference(self.base_model, batched_inputs)

        # N student gradient steps on the same pseudo-labeled batch
        self.base_model.train()
        loss = None
        for _ in range(self.config.n_student_steps):
            self.optimizer.zero_grad()
            try:
                loss = self._compute_student_loss_with_entropy(batched_inputs, pseudo_labeled_batch, teacher_outputs)
            except (FloatingPointError, RuntimeError):
                # Model has diverged — reset student to source state
                self.base_model.load_state_dict(self.base_state)
                self._optimizer = None
                break

            if loss is not None and torch.isfinite(loss) and loss > 0:
                if self.config.weight_reg > 0.0:
                    reg_loss = torch.tensor(0.0, device=self.device)
                    for param, init_param in zip(self._trainable_params, self._init_weights):
                        reg_loss += torch.mean((param - init_param.to(param.device)) ** 2)
                    loss = loss + self.config.weight_reg * reg_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # NaN in weights after update -> reset student
                if any(p.isnan().any() for p in self.base_model.parameters()):
                    self.base_model.load_state_dict(self.base_state)
                    self._optimizer = None
                    break

        self._stats["losses"].append(loss.item() if loss is not None else 0.0)
        self._stats["num_batches"] = self._stats.get("num_batches", 0) + 1

        self.base_model.eval()
        with torch.no_grad():
            student_outputs = self._inference(self.base_model, batched_inputs)
        return student_outputs

    def _has_nan_outputs(self, teacher_outputs) -> bool:
        """Return True if teacher outputs contain NaN/Inf (teacher has diverged)."""
        try:
            if hasattr(teacher_outputs, 'logits') and teacher_outputs.logits is not None:
                return not torch.isfinite(teacher_outputs.logits).all()
            if hasattr(teacher_outputs, 'pred_boxes') and teacher_outputs.pred_boxes is not None:
                return not torch.isfinite(teacher_outputs.pred_boxes).all()
            if isinstance(teacher_outputs, list):
                for item in teacher_outputs:
                    if isinstance(item, dict) and 'instances' in item:
                        boxes = item['instances'].pred_boxes.tensor
                        if not torch.isfinite(boxes).all():
                            return True
        except Exception:
            pass
        return False
