import copy
import time
import gc
import asyncio
import nest_asyncio
from contextlib import nullcontext

from tqdm.auto import tqdm

import torch
from torch import OutOfMemoryError
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

from supervision.detection.core import Detections
from supervision.metrics.mean_average_precision import MeanAveragePrecision

from ..models.base import BaseModel, ModelProvider
from ..datasets import DataPreparation


class DetectionEvaluator:
    def __init__(
        self, model: BaseModel | list[BaseModel], classes: list[str], data_preparation: DataPreparation, required_reset: bool = False,
        dtype=torch.float32, device=torch.device("cuda"), synchronize: bool = True, no_grad: bool = True
    ):
        self.do_parallel = isinstance(model, list)
        self.model = [m.to(device).to(dtype) for m in model] if self.do_parallel else model.to(device).to(dtype)
        self.data_preparation = data_preparation
        self.classes = classes
        self.required_reset = required_reset
        self.dtype = dtype
        self.device = device
        self.synchronize = synchronize
        self.no_grad = no_grad

    @staticmethod
    def evaluate_with_reset(
        model: BaseModel, desc: str, loader: DataLoader, loader_length: int, classes: list[str], data_preparation: DataPreparation,
        reset: bool = True, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda"),
        synchronize: bool = True, no_grad: bool = True, clear_tqdm_when_oom: bool = False
    ):
        torch.cuda.empty_cache(); torch.cuda.empty_cache(); torch.cuda.empty_cache()
        gc.collect(); gc.collect(); gc.collect()

        if reset:
            try:
                model.reset_adaptation()
            except NotImplementedError:
                print("WARNING: reset_adaptation() is not implemented for this model. Assuming the evaluation is running with deep-copy mode.")
                model = copy.deepcopy(model)

        map_metric = MeanAveragePrecision()
        predictions_list = []
        targets_list = []
        total_images = 0
        collapse_time = 0

        if no_grad:  # use no_grad for inference
            disable_grad = torch.no_grad
        else:  # let model decide gradient requirement
            disable_grad = nullcontext

        tqdm_loader = tqdm(loader, total=loader_length, desc=f"Evaluation for {desc}")
        try:
            with disable_grad():
                for batch in tqdm_loader:
                    if model.model_provider == ModelProvider.HuggingFace:
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        batch['labels'] = [{
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in label.items()
                        } for label in batch['labels']]
                        total_images += len(batch['labels'])
                    else:
                        total_images += len(batch)

                    with torch.autocast(device_type=device.type, dtype=dtype):
                        start = time.time()
                        match model.model_provider:
                            case ModelProvider.Detectron2:
                                outputs = model(batch)
                            case ModelProvider.Ultralytics:
                                outputs = model(*batch)
                            case ModelProvider.HuggingFace:
                                outputs = model(**batch)
                            case _:
                                raise ValueError(f"Unsupported model provider: {model.model_provider}")

                        if device.type == "cuda" and synchronize:
                            torch.cuda.synchronize()

                        collapse_time += time.time() - start

                    with torch.no_grad():
                        match model.model_provider:
                            case ModelProvider.Detectron2:
                                from detectron2.modeling.postprocessing import detector_postprocess
                                from detectron2.structures import Instances

                                for output, input_data in zip(outputs, batch):
                                    output = data_preparation.post_process(output)
                                    predictions_list.append(Detections.from_detectron2(output))

                                    gt_instances = Instances(image_size=input_data['instances'].image_size)
                                    gt_instances.pred_boxes = input_data['instances'].gt_boxes
                                    gt_instances.gt_classes = input_data['instances'].gt_classes
                                    gt_instances = detector_postprocess(  # recover original image size
                                        gt_instances, input_data['height'], input_data['width']
                                    )
                                    gt_instances.gt_boxes = gt_instances.pred_boxes

                                    targets_list.append(Detections(
                                        xyxy=gt_instances.gt_boxes.tensor.detach().cpu().numpy(),
                                        class_id=gt_instances.gt_classes.detach().cpu().numpy()
                                    ))
                            case ModelProvider.Ultralytics:
                                output = data_preparation.post_process(output)
                                pred_detection = Detections.from_ultralytics(output)
                                target_detection = Detections(
                                    xyxy=input_data['bboxes'].detach().cpu().numpy(),
                                    class_id=input_data['cls'].detach().cpu().numpy()
                                )
                                raise NotImplementedError("Ultralytics post_process is not implemented yet.")
                            case ModelProvider.HuggingFace:
                                sizes = [label['orig_size'].cpu().tolist() for label in batch['labels']]
                                outputs = data_preparation.post_process(outputs, target_sizes=sizes)
                                predictions_list.extend([  # dtype converting
                                    Detections.from_transformers({k: v.float() if isinstance(v, torch.Tensor) else v for k, v in output.items()})
                                    for output in outputs
                                ])
                                targets_list.extend([Detections(
                                    xyxy=(box_convert(label['boxes'], "cxcywh", "xyxy") * label['orig_size'].flip(0).repeat(2)).cpu().float().numpy(),
                                    class_id=label['class_labels'].cpu().float().numpy(),
                                ) for label in batch['labels']])
                            case _:
                                raise ValueError(f"Unsupported model provider: {model.model_provider}")
        except OutOfMemoryError as e:  # catch OOM error to close tqdm properly
            tqdm_loader.close()
            if clear_tqdm_when_oom:
                tqdm_loader.container.close()
            raise e

        map_metric.update(predictions=predictions_list, targets=targets_list)
        m_ap = map_metric.compute()

        per_class_map = {
            f"{classes[idx]}_mAP@0.50:0.95": m_ap.ap_per_class[idx].mean().item()
            for idx in m_ap.matched_classes
        }
        performances = {
            "fps": total_images / collapse_time,
            "collapse_time": collapse_time
        }

        result = {
            "mAP@0.50:0.95": m_ap.map50_95.item(),
            "mAP@0.50": m_ap.map50.item(),
            "mAP@0.75": m_ap.map75.item(),
            **performances,
            **per_class_map
        }
        return result

    @staticmethod
    def evaluate(
        model: BaseModel, desc: str, loader: DataLoader, loader_length: int, classes: list[str], data_preparation: DataPreparation,
        dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda"),
        synchronize: bool = True, no_grad: bool = True, clear_tqdm_when_oom: bool = False
    ):
        return DetectionEvaluator.evaluate_with_reset(
            model, desc, loader, loader_length, classes=classes, data_preparation=data_preparation,
            reset=False, dtype=dtype, device=device,
            synchronize=synchronize, no_grad=no_grad, clear_tqdm_when_oom=clear_tqdm_when_oom
        )

    async def evaluate_recursively(self, module: BaseModel | list[BaseModel], *args, **kwargs):
        if isinstance(module, list):
            try:  # run all
                return await asyncio.gather(*[self.evaluate_recursively(m, *args, **kwargs) for m in module])
            except OutOfMemoryError:  # on OOM, try to run half
                if self.device.type == "cuda":
                    torch.cuda.synchronize()  # ensure all coroutine are finished
                results = []
                sub_modules = [module[:len(module)//2], module[len(module)//2:]]
                sub_modules[0] = sub_modules[0] if len(sub_modules[0]) else sub_modules[0][0]
                sub_modules[1] = sub_modules[1] if len(sub_modules[1]) else sub_modules[1][0]

                for sub_module in sub_modules:
                    result = await self.evaluate_recursively(sub_module, *args, **kwargs)
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)
            except KeyboardInterrupt:  # handle keyboard interrupt
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                raise
            return results
        else:
            return await asyncio.to_thread(
                self.evaluate_with_reset,
                module, *args, **kwargs, reset=self.required_reset, classes=self.classes, data_preparation=self.data_preparation,
                dtype=self.dtype, device=self.device, synchronize=self.synchronize, no_grad=self.no_grad, clear_tqdm_when_oom=True
            )

    def __call__(self, *args, **kwargs):
        if self.do_parallel:
            nest_asyncio.apply()
            try:
                return asyncio.run(self.evaluate_recursively(self.model, *args, **kwargs))
            except KeyboardInterrupt:
                print("\nEvaluation interrupted by user")
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                raise
        return self.evaluate_with_reset(
            self.model, *args, **kwargs, reset=self.required_reset, classes=self.classes, data_preparation=self.data_preparation,
            dtype=self.dtype, device=self.device, synchronize=self.synchronize, no_grad=self.no_grad
        )
