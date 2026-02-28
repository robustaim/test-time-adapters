from typing import Iterator, List
from types import SimpleNamespace
import copy
import random

import numpy as np
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from detectron2.structures import Instances
from detectron2.utils.events import EventStorage

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    def non_max_suppression(*args, **kwargs):
        raise NotImplementedError("non_max_suppression is not implemented")

from ....base import AdaptationEngine, BaseModel
from .configuration_mean_teacher import MeanTeacherConfig


class MeanTeacherEngine(AdaptationEngine):
    model_name = "MeanTeacherEngine"
    config_class = MeanTeacherConfig

    def __init__(self, config: MeanTeacherConfig, base_model: BaseModel):
        self.config: MeanTeacherConfig
        super().__init__(config, base_model)

    def _pre_init(self):
        self._trainable_params = []
        self._init_weights = []

    def _post_init(self):
        self._setup_teacher_model()
        self._setup_strong_augmentation()
        self.to(self.device)

    def _setup_teacher_model(self):
        # Create the teacher model

        self.teacher_model = copy.deepcopy(self.base_model)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        params = self._collect_trainable_params()
        self._trainable_params = params
        self._init_weights = [p.clone().detach() for p in params]

    def _collect_trainable_params(self) -> List[nn.Parameter]:
        params = []
        for p in self.base_model.parameters():
            p.requires_grad = True
            params.append(p)
        return params

    def _setup_strong_augmentation(self):
        # FixMatch-style RandAugmentMC

        cutout_size = self.config.cutout_size

        def AutoContrast(img, _):
            return PIL.ImageOps.autocontrast(img)

        def Brightness(img, v):
            return PIL.ImageEnhance.Brightness(img).enhance(v)

        def Color(img, v):
            return PIL.ImageEnhance.Color(img).enhance(v)

        def Contrast(img, v):
            return PIL.ImageEnhance.Contrast(img).enhance(v)

        def Equalize(img, _):
            return PIL.ImageOps.equalize(img)

        def Identity(img, _):
            return img

        def Posterize(img, v):
            return PIL.ImageOps.posterize(img, int(v))

        def Rotate(img, v):
            return img.rotate(v)

        def Sharpness(img, v):
            return PIL.ImageEnhance.Sharpness(img).enhance(v)

        def ShearX(img, v):
            return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

        def ShearY(img, v):
            return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

        def Solarize(img, v):
            return PIL.ImageOps.solarize(img, int(v))

        def TranslateX(img, v):
            return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v * img.size[0], 0, 1, 0))

        def TranslateY(img, v):
            return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v * img.size[1]))

        def CutoutAbs(img, v):
            w, h = img.size
            x0 = np.random.uniform(0, w)
            y0 = np.random.uniform(0, h)
            x0 = int(max(0, x0 - v / 2.))
            y0 = int(max(0, y0 - v / 2.))
            x1 = int(min(w, x0 + v))
            y1 = int(min(h, y0 + v))
            img = img.copy()
            PIL.ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), (128, 128, 128))
            return img

        # FixMatch augmentation pool
        augment_pool = [
            (AutoContrast, 0, 1),
            (Brightness, 0.05, 0.95),
            (Color, 0.05, 0.95),
            (Contrast, 0.05, 0.95),
            (Equalize, 0, 1),
            (Identity, 0, 1),
            (Posterize, 4, 8),
            (Rotate, -30, 30),
            (Sharpness, 0.05, 0.95),
            (ShearX, -0.3, 0.3),
            (ShearY, -0.3, 0.3),
            (Solarize, 0, 256),
            (TranslateX, -0.3, 0.3),
            (TranslateY, -0.3, 0.3),
        ]

        n = self.config.augment_strength_n
        m = self.config.augment_strength_m

        class RandAugmentMC:
            def __init__(self, n, m, pool, cutout_size):
                self.n = n
                self.m = m
                self.augment_pool = pool
                self.cutout_size = cutout_size

            def __call__(self, img):
                orig_dtype = torch.float32
                if isinstance(img, torch.Tensor):
                    orig_dtype = img.dtype
                    img = T.ToPILImage()(img.float())
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                ops = random.choices(self.augment_pool, k=self.n)
                for op, min_val, max_val in ops:
                    val = (float(self.m) / 10) * float(max_val - min_val) + min_val
                    img = op(img, val)

                img = CutoutAbs(img, self.cutout_size)

                return T.ToTensor()(img).to(dtype=orig_dtype)

        self.strong_augment = RandAugmentMC(n, m, augment_pool, cutout_size)

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self._trainable_params)

    @property
    def optimizer(self):
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
            elif self.config.optim == "AdamW":
                self._optimizer = optim.AdamW(
                    self._trainable_params,
                    lr=self.config.adapt_lr,
                    weight_decay=self.config.weight_decay,
                )
        return self._optimizer

    def _reset_stats(self):
        self._stats = {
            'losses': [],
            'num_batches': 0,
        }

    def reset(self, reset_stats=False):
        self.base_model.load_state_dict(self.base_state)
        self._trainable_params = []
        self._init_weights = []
        self._optimizer = None
        self._setup_teacher_model()

        self.online(self.adapting)
        self.to(self.device)

        if reset_stats:
            current_stats = self._stats
            self._reset_stats()
            return current_stats
        return None

    def forward(self, batched_inputs=None, **kwargs):
        if batched_inputs is None and kwargs:
            batched_inputs = kwargs

        if not self.adapting:
            return self._inference(self.base_model, batched_inputs)

        # Augmentation — weak (original) and strong
        weak_batch, strong_batch = self._apply_augmentation(batched_inputs)

        # Teacher generates pseudo-labels on weak (original) input
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self._inference(self.teacher_model, weak_batch)

        # Assign pseudo-labels to strong batch
        pseudo_labeled_batch = self._set_pseudo_labels(strong_batch, teacher_outputs)

        # Student trains on strongly augmented input with pseudo-labels
        self.base_model.train()
        self.optimizer.zero_grad()

        loss = self._compute_student_loss(pseudo_labeled_batch)

        if loss is not None and loss > 0:
            if self.config.weight_reg > 0.0:
                reg_loss = torch.tensor(0.0, device=self.device)
                for param, init_param in zip(self._trainable_params, self._init_weights):
                    reg_loss += torch.mean((param - init_param.to(param.device)) ** 2)
                loss = loss + self.config.weight_reg * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # EMA update teacher
        self._update_teacher_ema()

        self._stats['losses'].append(loss.item() if loss is not None else 0.0)
        self._stats['num_batches'] = self._stats.get('num_batches', 0) + 1

        # Return teacher's prediction (stable model output)
        return teacher_outputs

    def _inference(self, model, batched_inputs):
        if self.config.base_type == "rtdetr":
            pixel_values = batched_inputs['pixel_values'].to(self.device)
            return model(pixel_values=pixel_values)
        elif self.config.base_type == "yolo11":
            img = batched_inputs if torch.is_tensor(batched_inputs) else batched_inputs['img']
            return model(img.to(self.device))
        else:
            return model(batched_inputs)

    def _compute_student_loss(self, pseudo_labeled_batch):
        if self.config.base_type == "rtdetr":
            output = self.base_model(
                pixel_values=pseudo_labeled_batch['pixel_values'].to(self.device),
                labels=pseudo_labeled_batch['labels'],
            )
            return output.loss if output.loss is not None else None

        elif self.config.base_type == "rcnn":
            with EventStorage():
                output = self.base_model(pseudo_labeled_batch)
                if isinstance(output, dict):
                    return sum(output[k] for k in output)
            return None

        elif self.config.base_type == "yolo11":
            # YOLO: passing a dict triggers loss() path internally
            # Ensure model.args exists for v8DetectionLoss initialization
            if not hasattr(self.base_model, 'args'):
                self.base_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in pseudo_labeled_batch.items()}
            
            output = self.base_model(batch)

            if isinstance(output, tuple):
                return output[0].sum()  # (loss_vec[box,cls,dfl], loss_items) -> scalar
            
            return output.sum() if output.dim() > 0 else output

        return None

    def _update_teacher_ema(self):
        alpha = self.config.ema_alpha
        with torch.no_grad():
            for t_p, s_p in zip(self.teacher_model.parameters(), self.base_model.parameters()):
                if s_p.requires_grad:
                    t_p.data = alpha * t_p.data + (1 - alpha) * s_p.data

    def _apply_augmentation(self, batch):
        if self.config.base_type == "rtdetr":
            weak_batch = batch
            pixel_values = batch['pixel_values']
            strong_pixel_values = []
            for i in range(pixel_values.shape[0]):
                try:
                    strong_pixel_values.append(self.strong_augment(pixel_values[i]))
                except Exception:
                    strong_pixel_values.append(pixel_values[i])
            strong_batch = {
                'pixel_values': torch.stack(strong_pixel_values),
                'labels': batch['labels'],
            }
            return weak_batch, strong_batch

        elif self.config.base_type == "rcnn":
            weak_batch = []
            strong_batch = []
            for item in batch:
                weak_batch.append(copy.deepcopy(item))
                strong_item = copy.deepcopy(item)
                try:
                    strong_item["strong_aug_image"] = self.strong_augment(strong_item["image"])
                except Exception:
                    strong_item["strong_aug_image"] = strong_item["image"]
                strong_batch.append(strong_item)
            return weak_batch, strong_batch

        elif self.config.base_type == "yolo11":
            weak_batch = batch
            img = batch if torch.is_tensor(batch) else batch['img']
            strong_imgs = []
            for i in range(img.shape[0]):
                try:
                    strong_imgs.append(self.strong_augment(img[i]))
                except Exception:
                    strong_imgs.append(img[i])
            strong_batch = {'img': torch.stack(strong_imgs)}
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if k != 'img':
                        strong_batch[k] = v
            return weak_batch, strong_batch

        return batch, batch

    def _set_pseudo_labels(self, strong_batch, teacher_outputs):
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
                'pixel_values': strong_batch['pixel_values'],
                'labels': annotation,
            }

        elif self.config.base_type == "rcnn":
            pseudo_labels = []
            for img, label in zip(strong_batch, teacher_outputs):
                inst = label['instances'][label['instances'].scores > self.config.conf_threshold]

                new_inp = {k: img[k] for k in img if k not in ['instances', 'image', 'strong_aug_image']}
                new_inp['image'] = img.get('strong_aug_image', img['image'])

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

            # teacher_outputs is (predictions, features) tuple from predict()
            preds = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
            nms_results = non_max_suppression(preds, conf_thres=self.config.conf_threshold)

            img = strong_batch['img']
            _, _, img_h, img_w = img.shape

            all_cls = []
            all_bboxes = []
            all_batch_idx = []

            for i, det in enumerate(nms_results):
                if det is not None and len(det) > 0:
                    # det: [N, 6] = [x1, y1, x2, y2, conf, cls]
                    boxes_xyxy = det[:, :4]
                    cls = det[:, 5]

                    # Convert xyxy to xywh normalized (YOLO loss format)
                    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / img_w
                    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / img_h
                    w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
                    h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
                    bboxes_xywh = torch.stack([cx, cy, w, h], dim=1)

                    all_cls.append(cls)
                    all_bboxes.append(bboxes_xywh)
                    all_batch_idx.append(torch.full((len(det),), i, dtype=torch.float32, device=det.device))

            device = img.device
            if all_cls:
                return {
                    'img': strong_batch['img'],
                    'cls': torch.cat(all_cls).unsqueeze(-1),
                    'bboxes': torch.cat(all_bboxes),
                    'batch_idx': torch.cat(all_batch_idx),
                }
            else:
                return {
                    'img': strong_batch['img'],
                    'cls': torch.zeros((0, 1), device=device),
                    'bboxes': torch.zeros((0, 4), device=device),
                    'batch_idx': torch.zeros(0, device=device),
                }

        return strong_batch
