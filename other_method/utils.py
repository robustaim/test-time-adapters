import os
import io
import math
import copy
import logging
from datetime import datetime
from collections import defaultdict
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_convert
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from supervision.metrics.mean_average_precision import MeanAveragePrecision
from supervision.detection.core import Detections

from ttadapters.datasets import (
    SHIFTClearDatasetForObjectDetection,
    SHIFTDiscreteSubsetForObjectDetection,
    BaseDataset,
)

class SHIFTCorruptedTaskDatasetForObjectDetection(SHIFTDiscreteSubsetForObjectDetection):
    def __init__(
            self, root: str, force_download: bool = False,
            train: bool = True, valid: bool = False,
            transform: Optional[Callable] = None, task: str = "clear", target_transform: Optional[Callable] = None
    ):
        super().__init__(
            root=root, force_download=force_download,
            train=train, valid=valid, subset_type=task_to_subset_types(task),
            transform=transform, target_transform=target_transform
        )

class LabelDataset(BaseDataset):
    def __init__(self, original_dataset, camera='front'):
        self.dataset = original_dataset
        self.camera = camera

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx][self.camera]
        return item['boxes2d'], item['boxes2d_classes']

def naive_collate_fn(batch):
    return batch
    
class DatasetAdapterForTransformers(BaseDataset):
    def __init__(self, original_dataset, camera='front'):
        self.dataset = original_dataset
        self.camera = camera

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx][self.camera]
        image = item['images'].squeeze(0)

        # Convert to COCO_Detection Format
        annotations = []
        target = dict(image_id=idx, annotations=annotations)
        for box, cls in zip(item['boxes2d'], item['boxes2d_classes']):
            x1, y1, x2, y2 = box.tolist()  # from Pascal VOC format (x1, y1, x2, y2)
            width, height = x2 - x1, y2 - y1
            annotations.append(dict(
                bbox=[x1, y1, width, height],  # to COCO format: [x, y, width, height]
                category_id=cls.item(),
                area=width * height,
                iscrowd=0
            ))

        # Following prepare_coco_detection_annotation's expected format
        # RT-DETR ImageProcessor converts the COCO bbox to center format (cx, cy, w, h) during preprocessing
        # But, eventually re-converts the bbox to Pascal VOC (x1, y1, x2, y2) format after post-processing
        return dict(image=image, target=target)
    
def collate_fn(batch, preprocessor=None):
    images = [item['image'] for item in batch]
    if preprocessor is not None:
        target = [item['target'] for item in batch]
        return preprocessor(images=images, annotations=target, return_tensors="pt")
    else:
        # If no preprocessor is provided, just assume images are already in tensor format
        return dict(
            pixel_values=dict(pixel_values=torch.stack(images)),
            labels=[dict(
                class_labels=item['boxes2d_classes'].long(),
                boxes=item["boxes2d"].float()
            ) for item in batch]
        )

def task_to_subset_types(task: str):
    T = SHIFTDiscreteSubsetForObjectDetection.SubsetType

    # weather
    if task == "cloudy":
        return T.CLOUDY_DAYTIME
    if task == "overcast":
        return T.OVERCAST_DAYTIME
    if task == "rainy":
        return T.RAINY_DAYTIME
    if task == "foggy":
        return T.FOGGY_DAYTIME

    # time
    if task == "night":
        return T.CLEAR_NIGHT
    if task in {"dawn", "dawn/dusk"}:
        return T.CLEAR_DAWN
    if task == "clear":
        return T.CLEAR_DAYTIME
    
    # simple
    if task == "normal":
        return T.NORMAL
    if task == "corrupted":
        return T.CORRUPTED

    raise ValueError(f"Unknown task: {task}")

def test(model, device, task, tta_raw_data, tta_valid_dataloader, reference_preprocessor, classes_list):
    targets = []
    predictions = []

    for idx, lables, inputs in zip(tqdm(range(len(tta_raw_data))), tta_raw_data, tta_valid_dataloader):
        sizes = [label['orig_size'].cpu().tolist() for label in inputs['labels']]

        with torch.no_grad():
            outputs = model(pixel_values=inputs['pixel_values'].to(device))

        results = reference_preprocessor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=0.0
        )

        detections = [Detections.from_transformers(results[i]) for i in range(len(results))]
        annotations = [Detections(
            xyxy=lables[i][0].cpu().numpy(),
            class_id=lables[i][1].cpu().numpy(),
        ) for i in range(len(lables))]

        targets.extend(annotations)
        predictions.extend(detections)

    mean_average_precision = MeanAveragePrecision().update(
    predictions=predictions,
    targets=targets,
    ).compute()
    per_class_map = {
        f"{classes_list[idx]}_mAP@0.95": mean_average_precision.ap_per_class[idx].mean()
        for idx in mean_average_precision.matched_classes
    }
    
    print(f"mAP@0.95_{task}: {mean_average_precision.map50_95:.3f}")
    print(f"mAP50_{task}: {mean_average_precision.map50:.3f}")
    print(f"mAP75_{task}: {mean_average_precision.map75:.3f}")
    for key, value in per_class_map.items():
        print(f"{key}_{task}: {value:.3f}")
    
    return {"mAP@0.95" : mean_average_precision.map50_95,
            "mAP50" : mean_average_precision.map50,
            "mAP75" : mean_average_precision.map75,
            "per_class_mAP@0.95" : per_class_map
            }

def agg_per_class(dicts):
    """dicts: per_class_map(dict)의 리스트. 예: [{"car_mAP@0.95":0.41, ...}, {...}]"""
    sums = defaultdict(float)
    counts = defaultdict(int)
    for d in dicts:
        for cls, val in d.items():
            sums[cls]  += float(val)
            counts[cls] += 1
    means = {cls: (sums[cls] / counts[cls]) for cls in sums}
    return means


def aggregate_runs(results_list):
    overall_sum = {"mAP@0.95": 0.0, "mAP50": 0.0, "mAP75": 0.0}
    n = len(results_list)

    per_class_maps = []

    for r in results_list:
        overall_sum["mAP@0.95"] += float(r["mAP@0.95"])
        overall_sum["mAP50"]    += float(r["mAP50"])

        overall_sum["mAP75"] += float(r["mAP75"])

        class_mAP = r["per_class_mAP@0.95"]
        per_class_means = agg_per_class([class_mAP])

    overall_mean = {k: (overall_sum[k] / n if n > 0 else 0.0) for k in overall_sum}

    return {
        "overall_sum": overall_sum,            # {"mAP@0.95": ..., "mAP50": ..., "map75": ...}
        "overall_mean": overall_mean,          # 위의 평균          # {"car_mAP@0.95": 합, ...}
        "per_class_mean@0.95": per_class_means,        # {"car_mAP@0.95": 평균, ...}
    }

def print_results(result):
    om = result["overall_mean"]
    print(f"mAP@0.95: {float(om['mAP@0.95']):.3f}")
    print(f"mAP50: {float(om['mAP50']):.3f}")
    print(f"mAP75: {float(om['mAP75']):.3f}")

    for k, v in result["per_class_mean@0.95"].items():
        print(f"{k}: {v:.3f}")

def setup_logger(save_dir=None, name="direct_method", level=logging.INFO, mirror_to_stdout=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    log_path = None
    # 파일 핸들러: save_dir가 주어졌을 때만 생성
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # 화면 핸들러(미러링)
    if mirror_to_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.propagate = False
    return logger, log_path

  # tqdm도 로깅에 정리되도록

class LoggerWriter(io.TextIOBase):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buf = ""
    def write(self, msg):
        # 줄 단위로 로깅(개행/부분문자 처리)
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.level(line)
    def flush(self):
        if self._buf.strip():
            self.level(self._buf.strip())
            self._buf = ""

def improve_test(device, batch_i, eval_every, 
                 model, task, best_map50,
                 tta_raw_data, tta_valid_dataloader, 
                 reference_preprocessor, classes_list):
    
    current_result = test(model, device, task, tta_raw_data, tta_valid_dataloader, reference_preprocessor, classes_list)
    current_map50 = current_result.get("mAP50", current_result.get("mAP@0.50", -1.0))

    improve = current_map50 >= best_map50

    return current_map50, improve

        

# ActMAD-specific function
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        out = torch.vstack(self.outputs)
        out = torch.mean(out, dim=0)
        return out

    def get_out_var(self):
        out = torch.vstack(self.outputs)
        out = torch.var(out, dim=0)
        return out
    
def extract_activation_alignment(model, method, device, data_root, reference_preprocessor, batch_size=32):
    train_dataloader = DataLoader(
        DatasetAdapterForTransformers(SHIFTClearDatasetForObjectDetection(root=data_root, train=True)), 
        batch_size=batch_size, collate_fn=partial(collate_fn, preprocessor=reference_preprocessor))
    # model unfreeze
    for k, v in model.named_parameters():
        v.requires_grad = True

    chosen_bn_layers = []
    if method == "actmad": 
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                chosen_bn_layers.append(m)

    else :
        for m in model.modules():
            if isinstance(m, RTDetrFrozenBatchNorm2d):
                chosen_bn_layers.append(m)
    # chosen_bn_layers
    """
    Since high-level representations are more sensitive to domain shift,
    only the later BN layers are selected. 
    The cutoff point is determined empirically.
    """
    n_chosen_layers = len(chosen_bn_layers)

    save_outputs = [SaveOutput() for _ in range(n_chosen_layers)]

    clean_mean_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
    clean_var_act_list = [AverageMeter() for _ in range(n_chosen_layers)]

    clean_mean_list_final = []
    clean_var_list_final = []
    # extract the activation alignment in train dataset
    print("Start extracting BN statistics from the training dataset")
    
    with torch.no_grad():
        for batch_i, input in enumerate(tqdm(train_dataloader)):
            img = input['pixel_values'].to(device, non_blocking=True)
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            model.eval()
            hook_list = [chosen_bn_layers[i].register_forward_hook(save_outputs[i]) for i in range(n_chosen_layers)]
            _ = model(img)

            for i in range(n_chosen_layers):
                clean_mean_act_list[i].update(save_outputs[i].get_out_mean())  # compute mean from clean data
                clean_var_act_list[i].update(save_outputs[i].get_out_var())  # compute variane from clean data

                save_outputs[i].clear()
                hook_list[i].remove()

        for i in range(n_chosen_layers):
            clean_mean_list_final.append(clean_mean_act_list[i].avg)  # [C, H, W]
            clean_var_list_final.append(clean_var_act_list[i].avg)  # [C, H, W]

        return clean_mean_list_final, clean_var_list_final, chosen_bn_layers
    
# DUA-specific function
def tensor_rot_90(x):
    x = TF.rotate(x, 90)
    return x


def tensor_rot_180(x):
    x = TF.rotate(x, 180)
    return x


def tensor_rot_270(x):
    x = TF.rotate(x, 270)
    return x


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)

def rotate_batch(batch, label):
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif label == 'expand':
        labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                            torch.zeros(len(batch), dtype=torch.long) + 1,
                            torch.zeros(len(batch), dtype=torch.long) + 2,
                            torch.zeros(len(batch), dtype=torch.long) + 3])
        batch = batch.repeat((4, 1, 1, 1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels

def get_adaption_inputs_batch(imgs, tr_transform_adapt, device, n = 32):
    batch_inputs = []
    for b in range(imgs.size(0)):
        img = imgs[b]
        inputs = [tr_transform_adapt(img) for _ in range(n)]
        inputs = torch.stack(inputs)
        batch_inputs.append(inputs)
    
    batch_inputs = torch.stack(batch_inputs)

    B, N, C, H, W = batch_inputs.shape
    batch_inputs = batch_inputs.view(B*N, C, H, W)

    inputs_ssh, _ = rotate_batch(batch_inputs, 'rand')

    return inputs_ssh.to(device, non_blocking=True)

def get_adaption_inputs_default(img, tr_transform_adapt, device, n=8):
    inputs = [(tr_transform_adapt(img)) for _ in range(n)]
    inputs = torch.stack(inputs)
    inputs_ssh, _ = rotate_batch(inputs, 'rand')
    inputs_ssh = inputs_ssh.to(device, non_blocking=True)
    return inputs_ssh

# Mean Teacher-specific function
def create_model(model, ema=False):
    model_copy = copy.deepcopy(model)

    if ema:
        for param in model_copy.parameters():
            param.detach_()
    return model_copy



def make_scheduler(optimizer, 
                   *, 
                   warmup_steps: int,
                   initial_lr: float,
                   decay_total_steps: int | None,
                   total_steps: float,
                   base_lr: float = 1e-4,
                   ):
    """
    Create a LambdaLR scheduler that mimics adjust_learning_rate
    """

    if decay_total_steps is not None:
        assert decay_total_steps >= total_steps, \
            "Expected decay_total_steps >= total_steps"
        
    def linear_rampup(step: int, ramp_steps: int) -> float:
        """Linear rampup"""
        if ramp_steps <= 0:   # End warm-up
            return 1.0
        if step >= ramp_steps:
            return 1.0
        if step <= 0:
            return 0.0
        return step / float(ramp_steps)


    def cosine_rampdown(step: int, total_decay_steps: int) -> float:
        step = max(0, min(step, total_decay_steps))
        return 0.5 * (math.cos(math.pi * step / float(total_decay_steps)) + 1.0)

    def lr_lambda(global_step: int) -> float:
        # Linear warm-up
        ramp = linear_rampup(global_step, warmup_steps)
        lr_val = ramp * (base_lr - initial_lr) + initial_lr

        if decay_total_steps is not None:
            decay_scale = cosine_rampdown(global_step, decay_total_steps)
            lr_val *= decay_scale

        return lr_val / base_lr

    return LambdaLR(optimizer, lr_lambda)

def make_label(pseudo_label : list, size):
    height, width = size
    def normalize_boxes(boxes, height, width):
        # 1. xyxy → cxcywh 
        boxes_cxcywh_norm = box_convert(boxes, 'xyxy', 'cxcywh')

        # 2. normalize (convert to actual pixel coordinates)
        boxes_cxcywh_norm[:, [0, 2]] /= width
        boxes_cxcywh_norm[:, [1, 3]] /= height
        return boxes_cxcywh_norm
    labels=[]
    for i in pseudo_label:        
        label = {"class_labels" : i["labels"],
                  "boxes" : normalize_boxes(i["boxes"], height, width)}
        labels.append(label)
    return labels

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    
# whw

def conv(in_planes, out_planes, kernel=3, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding, bias=bias)

def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):
    def __init__(self, planes, out_planes=None, stride=1, mode="parallel", bias=False):
        super(conv1x1, self).__init__()
        self.mode = mode
        self.conv = conv1x1_fonc(planes, out_planes, stride, bias=bias)

    def forward(self, x):
        y = self.conv(x)
        return y

class conv_task(nn.Module):
    def __init__(self, in_planes, planes, adapter=None, kernel=3, stride=1, padding=0, nb_tasks=20, is_proj=1, norm=None, th=0.9, r=32):
        super(conv_task, self).__init__()
        self.is_proj = is_proj
        self.conv = conv1x1_fonc(in_planes, planes, stride) if kernel == 1 else conv(in_planes, planes, kernel=kernel, stride=stride, padding=padding)
        self.mode = adapter

        if self.mode == 'parallel' and is_proj:
            if kernel == 1:
                self.down_proj = conv1x1(in_planes, planes // r, stride, mode=self.mode, bias=True)
                self.non_linear_func = nn.ReLU()
                self.up_proj = conv1x1(planes // r, planes, stride, mode=self.mode, bias=True)
                self.adapter_norm = nn.BatchNorm2d(planes)
                nn.init.kaiming_uniform_(self.down_proj.conv.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.conv.weight)
                nn.init.zeros_(self.down_proj.conv.bias)
                nn.init.zeros_(self.up_proj.conv.bias)
            else:
                self.down_proj = conv(in_planes, planes // r, kernel=kernel, stride=stride, padding=padding, bias=True)
                self.non_linear_func = nn.ReLU()
                self.up_proj = conv(planes // r, planes, kernel=(1, 1), stride=(1, 1), padding=0, bias=True)
                self.adapter_norm = nn.BatchNorm2d(planes)
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

            self.norm = RTDetrFrozenBatchNorm2d(planes) if norm is None else norm
            self.mean_feats = torch.zeros((1, in_planes))        # out_planes or in_planes
            self.num_feats = torch.zeros(nb_tasks)
            self.threshold = th

        # FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def forward(self, x):
        y = self.norm(self.conv(x))
        if self.mode == 'parallel' and self.is_proj:
            #adapt_x = self.adapter_norm(self.up_proj(self.non_linear_func(self.down_proj(x))))
            adapt_x = self.up_proj(self.non_linear_func(self.down_proj(x)))
            if self.scalar == 'cosine_sim':
                _x = x.mean(dim=[0, 2, 3]).detach()
                cos_sim = F.cosine_similarity(self.mean_feats.to(_x.device), _x.unsqueeze(0), dim=1)
                y = y + adapt_x * cos_sim
                self.mean_feats = 0.9 * self.mean_feats.to(_x.device) + 0.1 * _x[None, :]
            else:
                y = y + adapt_x * self.scalar.to(y.device)

        #y = self.norm(y)

        return y

def add_adapters_to_backbone(rtdetr_model, r=32, scalar=None, target_stages=[0, 1, 2, 3]):
    """
    Add parallel adapters to RT-DETR backbone (in-place modification)
    
    Args:
        rtdetr_model: Pre-trained RT-DETR model
        r: int, reduction ratio for adapter bottleneck (default: 32)
        scalar: str or None, scaling method ('learnable_scalar' or None)
        target_stages: list, which stages to add adapters to (default: [1, 2, 3])
    
    Returns:
        rtdetr_model: Same model with adapters added to backbone
    """
    
    # Access backbone encoder stages
    backbone_encoder = rtdetr_model.model.backbone.model.encoder
    stages = backbone_encoder.stages
    
    adapter_count = 0
    
    for stage_idx in target_stages:
        if stage_idx < len(stages):
            stage = stages[stage_idx]
            
            # Iterate through blocks in the stage 
            for idx, block in enumerate(stage.layers):
                # Check stride condition like in the original code
                if idx >= 0 and hasattr(block, 'layer') and isinstance(block.layer, nn.Sequential):
                    # Find the first conv layer in the bottleneck block that meets stride condition
                    for conv_idx, conv_layer in enumerate(block.layer):
                        if (hasattr(conv_layer, 'convolution') and 
                            conv_layer.convolution.stride[0] == 1):
                            
                            # Create adapter-enabled conv layer (same pattern as original)
                            new_conv = conv_task(
                                in_planes=conv_layer.convolution.in_channels,
                                planes=conv_layer.convolution.out_channels,
                                adapter='parallel',
                                kernel=conv_layer.convolution.kernel_size[0],
                                stride=conv_layer.convolution.stride[0],
                                padding=conv_layer.convolution.padding[0],
                                norm=conv_layer.normalization,
                                r=r,
                            )
                            
                            # Load original weights (same pattern as original)
                            load_weight = {k: conv_layer.convolution.state_dict()[k] 
                                         for k in conv_layer.convolution.state_dict() 
                                         if k in new_conv.conv.state_dict()}
                            new_conv.conv.load_state_dict(load_weight)
                            
                            # Replace the layer in block.layer (same as original: block.conv1 = new_conv1)
                            block.layer[conv_idx] = new_conv
                            adapter_count += 1
                            print(f"Added adapter to stage {stage_idx}, block {idx}, layer {conv_idx}")
                            break  # Only modify first eligible conv layer per block
    
    print(f"Added {adapter_count} adapters to backbone")
    return rtdetr_model

def freeze_backbone_except_adapters(model):
    """Freeze all backbone parameters except adapter parameters"""
    for name, param in model.named_parameters():
        if any(adapter_key in name for adapter_key in ['down_proj', 'up_proj', 'adapter']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model

def get_adapter_parameters(model):
    """Get only adapter parameters for optimization"""
    return [param for name, param in model.named_parameters() 
            if param.requires_grad and any(adapter_key in name for adapter_key in ['down_proj', 'up_proj', 'adapter'])]

# Main function
def add_adapters(rtdetr_model, device, reduction_ratio=32, target_stages=[0, 1, 2, 3]):
    """
    Add adapters to RT-DETR model and prepare for training
    
    Args:
        rtdetr_model: Pre-trained RT-DETR model
        reduction_ratio: int, adapter bottleneck reduction (default: 32)
        learnable_scale: bool, whether to use learnable scaling (default: True)
        target_stages: list, which stages to add adapters (default: [1, 2, 3])
    
    Returns:
        rtdetr_model: Model with adapters added and backbone frozen
    """
    
    # Add adapters to backbone
    rtdetr_model = add_adapters_to_backbone(rtdetr_model, r=reduction_ratio, target_stages=target_stages)

    rtdetr_model.eval()
    rtdetr_model.requires_grad_(False)

    # rtdetr_model = freeze_backbone_except_adapters(rtdetr_model)

    param = get_adapter_parameters(rtdetr_model)

    optimizer = torch.optim.SGD(param, lr=1e-4, momentum=0.05)

    
    # Count parameters
    total_params = sum(p.numel() for p in rtdetr_model.parameters())
    trainable_params = sum(p.numel() for p in rtdetr_model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.4f}")
    
    return rtdetr_model.to(device), optimizer

def outputs():
    return results, adapt_loss, 

def rtdetr_fg_gate_and_feats(outputs, num_classes=6, tau=0.5):
    # 1) 필수 출력
    logits = outputs["logits"]              # [B, N, C or C+1]
    query_feats = outputs["last_hidden_state"]  # [B, N, D]

    # 2) 배경 채널 유무 판단
    has_bg = (logits.shape[-1] == num_classes + 1)

    if has_bg: # 배경 존재 시
        probs = logits.softmax(-1)                          # [B, N, C+1]
        fg_scores, fg_preds = probs[..., :num_classes].max(-1)  # [B, N], [B, N]
        bg_scores = probs[..., -1]                          # [B, N]
    else:
        probs = logits.sigmoid()                            # [B, N, C]
        fg_scores, fg_preds = probs.max(-1)                 # [B, N], [B, N]
        bg_scores = 1.0 - fg_scores                         # 의사 배경 점수

    # 3) 임계값 게이팅(기존 코드의 0.5와 동일)
    valid = fg_scores >= tau
    sentinel = torch.full_like(fg_preds, num_classes)       # 무효 표식 = C
    fg_preds = torch.where(valid, fg_preds, sentinel)       # [B, N]
    fg_scores = torch.where(valid, fg_scores, bg_scores)    # [B, N]

    # 4) 기존 루프와 동일하게 쓰기 위해 평탄화
    flat_preds = fg_preds.reshape(-1)                       # [B*N]
    flat_feats = query_feats.reshape(-1, query_feats.size(-1))  # [B*N, D]

    return flat_preds, flat_feats, fg_scores.reshape(-1)

def make_t_stats(s_stats):
    t_stats = {}
    for k in s_stats["gl"]:
        mean, cov = s_stats["gl"][k]
        # self.template_cov["gl"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
        t_stats["gl"][k] = (mean, cov)
    return t_stats


def compute_fg_align_loss_with_rtdetr(
    num_classes,
    s_stats, t_stats,      # 소스/타깃 통계 dict: s_stats["fg"][k] = (mu_s, Sigma_s)
    template_cov,          # template_cov["fg"][k] (수치 안정화용)
    ema_n, ema_gamma,      # 클래스별 누적 카운터, EMA 계수
    device,
    tau=0.5,
    freq_weight=False,
    score_weight=None,     # 선택: 클래스별 점수 가중 (ex. flat_scores)
    clip_th=1e5
):
    logits = outputs["logits"]              # [B, N, C or C+1]
    query_feats = outputs["last_hidden_state"]  # [B, N, D]

    # 2) 배경 채널 유무 판단
    has_bg = (logits.shape[-1] == num_classes + 1)

    if has_bg: # 배경 존재 시
        probs = logits.softmax(-1)                          # [B, N, C+1]
        fg_scores, fg_preds = probs[..., :num_classes].max(-1)  # [B, N], [B, N]
        bg_scores = probs[..., -1]                          # [B, N]
    else:
        probs = logits.sigmoid()                            # [B, N, C]
        fg_scores, fg_preds = probs.max(-1)                 # [B, N], [B, N]
        bg_scores = 1.0 - fg_scores                         # 의사 배경 점수

    # 3) 임계값 게이팅(기존 코드의 0.5와 동일)
    valid = fg_scores >= tau
    sentinel = torch.full_like(fg_preds, num_classes)       # 무효 표식 = C
    fg_preds = torch.where(valid, fg_preds, sentinel)       # [B, N]
    fg_scores = torch.where(valid, fg_scores, bg_scores)    # [B, N]

    # 4) 기존 루프와 동일하게 쓰기 위해 평탄화
    flat_preds = fg_preds.reshape(-1)                       # [B*N]
    flat_feats = query_feats.reshape(-1, query_feats.size(-1))  # [B*N, D]

    loss_fg_align = torch.tensor(0.0, device=device)
    loss_n = 0

    valid_classes = flat_preds[flat_preds != num_classes].unique()
    for _k in valid_classes:
        k = int(_k.item())
        idx = (flat_preds == k)
        cur_feats = flat_feats[idx].to(device)               # [r_k, D]
        if cur_feats.numel() == 0:
            continue

        # ---- 타깃 평균 EMA 업데이트 (분산은 소스 것으로 고정) ----
        ema_n[k] += cur_feats.shape[0]
        mu_t_prev = t_stats["fg"][k][0].to(device)           # 이전 타깃 평균
        diff = cur_feats - mu_t_prev[None, :]                # x - mu_t(prev)
        delta = (1.0 / ema_gamma) * diff.sum(dim=0)
        mu_t = mu_t_prev + delta                             # 새 타깃 평균

        # ---- 대칭 KL 정렬 (분산: 소스 + 템플릿) ----
        mu_s, Sigma_s = s_stats["fg"][k][0].to(device), s_stats["fg"][k][1].to(device)
        Sigma = Sigma_s + template_cov["fg"][k].to(device)   # PD 보정/안정화
        s_dist = MVN(mu_s, Sigma)
        t_dist = MVN(mu_t, Sigma)
        cur_loss = 0.5 * (KL(s_dist, t_dist) + KL(t_dist, s_dist))

        # ---- (옵션) 클래스 불균형/점수 가중 ----
        if freq_weight:
            w = torch.log(max(ema_n.values()) / ema_n[k])    # dict면 적절히 수정
            cur_loss = cur_loss * min(w + 0.01, 10.0) ** 2
        if score_weight is not None:
            # 예: 해당 클래스의 평균 점수로 가중
            w_score = score_weight[idx].mean().to(device)
            cur_loss = cur_loss * w_score

        # ---- 안정화/누적 & 통계 업데이트 ----
        if torch.isfinite(cur_loss) and cur_loss < clip_th:
            loss_fg_align = loss_fg_align + cur_loss
            t_stats["fg"][k] = (mu_t.detach(), None)         # 평균만 유지(분산 고정)
            loss_n += 1

    return loss_fg_align, loss_n