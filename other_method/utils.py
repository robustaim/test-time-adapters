import os
import io
import logging
from datetime import datetime
from collections import defaultdict
from functools import partial
from typing import Optional, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

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
    if batch_i % eval_every == 0:
        current_result = test(model, device, task, tta_raw_data, tta_valid_dataloader, reference_preprocessor, classes_list)
        current_map50 = current_result.get("mAP50", current_result.get("mAP@0.50", -1.0))

        improve = current_map50 >= best_map50

    return current_map50, best_map50, improve

        

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
    
def extract_activation_alignment(model, device, data_root, reference_preprocessor, batch_size=32):
    train_dataloader = DataLoader(
        DatasetAdapterForTransformers(SHIFTClearDatasetForObjectDetection(root=data_root, train=True)), 
        batch_size=batch_size, collate_fn=partial(collate_fn, preprocessor=reference_preprocessor))
    # model unfreeze
    for k, v in model.named_parameters():
        v.requires_grad = True

    chosen_bn_layers = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
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