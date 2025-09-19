import os
import time
import copy
import contextlib
from pathlib import Path
from functools import partial

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms import RandAugment

from supervision.detection.core import Detections

from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessorFast,
    RTDetrConfig,
)

from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d

from transformers.image_utils import AnnotationFormat

from safetensors.torch import load_file
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


from . import utils

class Baseline:
    def __init__(self, 
                 device,
                 data_root: Path="./data",
                 save_dir: Path="./checkpoints",
                 method: str="direct_method", 
                 model_arch: str="rt_detr",
                 reference_model_id: str="PekingU/rtdetr_r50vd",
                 image_size: int=640, 
                 class_num: int=6, 
                 valid_batch: int=16,
                 clean_bn_extract_batch: int=64,
                 model_states: Path="/home/elicer/ptta/RT-DETR_R50vd_SHIFT_CLEAR.safetensors",
                 lr: float=5e-5,
                 momentum: float=0.1,
                 mom_pre: float=0.01,
                 decay_factor: float=0.94,
                 min_momentum_constant: float=0.0001,
                 use_scheduler: bool=False,
                 warmup_steps: int=50,
                 decay_total_steps: int=20,
                 patience: int=10,
                 eval_every: int=10,
                 enable_log_file: bool=False):
        """
        Initialize the Baseline class.

        Args:
            device (torch.device): Device to run the model on (CPU or specific GPU).
            data_root (Path, optional): Root directory for datasets.
            save_dir (Path, optional): Directory to save checkpoints.
            method (str, optional): Another TTA method name. | ex) actmad, norm, dua, mean_teacher, wwh
            model_arch (str, optional): Model architecture type. | ex) rt_detr, yolo11, faster_rcnn
            reference_model_id (str, optional): Pre-trained model reference ID.
            image_size (int, optional): Input image size.
            class_num (int, optional): Number of classes for detection.
            valid_batch (int, optional): Number of valid dataset batch
            clean_bn_extract_batch (int, optional): Number of BN extract batch for ActMAD
            model_states (Path, optional): Path to the model weights.
            momentum (float, optional): Momentum value for updates.
            mom_pre (float, optional): Initial momentum before adaptation.
            decay_factor (float, optional): Decay factor for momentum update.
            min_momentum_constant (float, optional): Minimum momentum constant.
            patience (int, optional): Number of epochs to wait before early stopping.
            eval_every (int, optional): Frequency of evaluation during training (in epochs).
            enable_log_file (bool, optional): If True, save logs to a file as well as console. 
                                              If False, log only to console. Default=False.
        """
        self.device = device

        self.data_root=data_root
        self.save_dir=save_dir

        self.method = method

        self.model_arch = model_arch
        self.reference_model_id = reference_model_id

        self.image_size = image_size
        self.class_num = class_num
        self.valid_batch = valid_batch
        self.clean_bn_extract_batch = clean_bn_extract_batch

        self.model_states = model_states

        self.lr = lr
        self.momentum = momentum
        self.mom_pre = mom_pre
        self.decay_factor= decay_factor
        self.min_momentum_constant=min_momentum_constant
        self.use_scheduler=use_scheduler
        self.warmup_steps=warmup_steps # for mean-teacher method
        self.decay_total_steps=decay_total_steps # for mean-teacher method

        self.patience = patience
        self.eval_every = eval_every

        self.enable_log_file = enable_log_file

        # self.s_stats = torch.load("/workspace/ptta/other_method/WHW/rtdetr_feature_stats.pt")
        self.t_stats = {}

        self.reference_preprocessor = self.image_processor()
    

    def pretrained_model(self):
        """
        Load a pretrained model based on the model architecture.

        This method:
            - Selects the model according to `self.model_arch`.
            - Initializes the model.
            - Loads pretrained weights from `self.model_states`.
            - Freezes all model parameters (no gradient updates).

        Returns:
            torch.nn.Module: The pretrained model (moved to `self.device`).
        """

        # Loads the model according to model_arch
        if self.model_arch=="rt_detr":
            reference_model_id = self.reference_model_id
            reference_config = RTDetrConfig.from_pretrained(reference_model_id, torch_dtype=torch.float32, return_dict=True)
            reference_config.num_labels = self.class_num

            reference_config.image_size = self.image_size
            model = RTDetrForObjectDetection(config=reference_config)
            model_states = load_file(self.model_states)
            model.load_state_dict(model_states, strict=False)

            for param in model.parameters():
                param.requires_grad = False
            
            model.to(self.device)

            return model
        
        elif self.model_arch=="faster_rcnn":
            model = FastRCNN(model_arch="/workspace/ptta/other_method/fast_rcnn/configs/Base/SHIFT_faster_rcnn_R50_FPN_1x.yaml",
                            num_classes=self.class_num,
                            device=self.device,
                            weight_path="/workspace/ptta/faster_rcnn_r50_shift.pth"
                            )
            model.eval()
            
            return model

    def image_processor(self, resize: int=800, do_resize: bool=False):
        """
        Create and configure an image processor for the model architecture.

        This method:
            - Loads a pretrained image processor according to `self.model_arch`.
            - Sets the annotation format to COCO detection (bounding box format).
            - Adjusts the target image size (height and width).
            - Controls whether resizing is applied.

        Args:
            resize (int, optional): Target size (height and width) for image resizing.
                                    Default is 800.
            do_resize (bool, optional): Whether to apply resizing. Default is False.

        Returns:
            RTDetrImageProcessorFast: Configured image processor for preprocessing images.
        """
        if self.model_arch=="rt_detr":
            reference_preprocessor = RTDetrImageProcessorFast.from_pretrained(self.reference_model_id)
            reference_preprocessor.format = AnnotationFormat.COCO_DETECTION  # COCO Format / Detection BBOX Format
            reference_preprocessor.size = {"height": resize, "width": resize}
            reference_preprocessor.do_resize = do_resize

            return reference_preprocessor

        elif self.model_arch=="faster_rcnn":
            reference_preprocessor = RTDetrImageProcessorFast()
            reference_preprocessor.do_rescale = False
            reference_preprocessor.do_normalize = False
            reference_preprocessor.format = AnnotationFormat.COCO_DETECTION  # COCO Format / Detection BBOX Format
            reference_preprocessor.size = {"height": resize, "width": resize}
            reference_preprocessor.do_resize = do_resize

            return reference_preprocessor
    
    def get_method(self):
        methods = {
            "direct_method": self.direct_method,
            "actmad": self.actmad,
            "norm": self.norm,
            "dua": self.dua,
            "mean_teacher": self.mean_teacher,
            # "whw": self.whw
        }
        return methods[self.method]
    
    def train(self):
        # Load pretrained model and image processor
        model = self.pretrained_model()
        device = self.device
        
        # Setup logger (file + console or console-only)
        log_save_dir = self.save_dir if self.enable_log_file else None
        logger, log_path = utils.setup_logger(log_save_dir, name=self.method, mirror_to_stdout=True)
        if self.enable_log_file and log_path is not None:
            logger.info(f"Log file: {log_path}")
        else:
            logger.info("Log file disabled (console-only logging).")

        os.makedirs(self.save_dir, exist_ok=True)

        # Redirect stdout/stderr to logger and tqdm
        with contextlib.redirect_stdout(utils.LoggerWriter(logger, logger.info)), \
            contextlib.redirect_stderr(utils.LoggerWriter(logger, logger.error)), \
            logging_redirect_tqdm():

            all_results = []
            extras = {}

            # Extra configs for specific methods
            if self.method == "actmad":
                # ActMAD statistics 파일 경로를 other_method 폴더에 설정
                stats_save_path = Path("/workspace/ptta/other_method") / f"actmad_clean_statistics_{self.model_arch}.pt"

                # chosen_bn_layers는 항상 새로 생성 (decoder 제외, encoder/fpn만 사용)
                all_norm_layers = []
                decoder_layers = []

                for name, m in model.named_modules():
                    if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                        # decoder 부분의 LayerNorm 제외
                        if 'decoder' in name.lower():
                            decoder_layers.append((name, m))
                        else:
                            all_norm_layers.append(m)

                print(f"ActMAD: Found {len(all_norm_layers)} encoder/fpn normalization layers")
                print(f"ActMAD: Excluded {len(decoder_layers)} decoder normalization layers")

                # 후반부 50% 레이어만 사용 (고수준 특성에 집중)
                cutoff = len(all_norm_layers) // 2
                chosen_bn_layers = all_norm_layers[cutoff:]
                print(f"ActMAD: Using {len(chosen_bn_layers)}/{len(all_norm_layers)} normalization layers")
                extras["chosen_bn_layers"] = chosen_bn_layers

                # 저장된 statistics가 있는지 확인
                if stats_save_path.exists():
                    print(f"Loading saved ActMAD statistics from {stats_save_path}")
                    saved_stats = torch.load(stats_save_path)
                    extras["clean_mean_list_final"] = saved_stats["clean_mean_list_final"]
                    extras["clean_var_list_final"] = saved_stats["clean_var_list_final"]
                else:
                    print("Extracting ActMAD statistics from clean data...")
                    (
                        extras["clean_mean_list_final"],
                        extras["clean_var_list_final"],
                        _  # chosen_bn_layers는 이미 위에서 생성했으므로 무시
                    ) = utils.extract_activation_alignment(
                        model, self.method, device, self.data_root, self.reference_preprocessor, batch_size=self.clean_bn_extract_batch
                        )

                    # Statistics만 저장 (chosen_bn_layers는 저장하지 않음)
                    print(f"Saving ActMAD statistics to {stats_save_path}")
                    torch.save({
                        "clean_mean_list_final": extras["clean_mean_list_final"],
                        "clean_var_list_final": extras["clean_var_list_final"]
                    }, stats_save_path)

                extras["optimizer_actmad"]  = optim.SGD(
                    model.parameters(),
                    lr=self.lr,
                    momentum=0.95,
                    weight_decay=5e-4, 
                    nesterov=True  
                )
            
            elif self.method == "dua":
                extras["tr_transform_adapt"] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((224, 640)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
                
            elif self.method == "mean_teacher":
                extras["student_model"] = utils.create_model(model, ema=False)
                extras["teacher_model"] = utils.create_model(model, ema=True)

                for p in extras["student_model"].parameters():
                    p.requires_grad_(True)
                for p in extras["teacher_model"].parameters():
                    p.requires_grad_(False)

                extras["optimizer_mt"] = optim.SGD(
                extras["student_model"].parameters(),
                lr=self.lr, momentum=self.momentum
                )
                    
                extras["weak_augmentation"] = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                ])
                extras["strong_augmentation"] = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                ])

            elif self.method == "whw":
                extras["model"], extras["optimizer_whw"] = utils.add_adapters(model, device=self.device, reduction_ratio=32, target_stages=[0, 1, 2, 3])
                (
                    extras["clean_mean_list_final"],
                    extras["clean_var_list_final"],
                    extras["chosen_bn_layers"]
                ) = utils.extract_activation_alignment(
                    model, self.method, device, self.data_root, self.reference_preprocessor, batch_size=self.clean_bn_extract_batch
                    )
                

            (tta_cloudy_raw_data, tta_cloudy_valid_dataloader, 
            tta_overcast_raw_data, tta_overcast_valid_dataloader,
            tta_foggy_raw_data, tta_foggy_valid_dataloader,
            tta_rainy_raw_data, tta_rainy_valid_dataloader,
            tta_dawn_raw_data, tta_dawn_valid_dataloader,
            tta_night_raw_data, tta_night_valid_dataloader,
            tta_clear_raw_data, tta_clear_valid_dataloader,
            classes_list) = self.make_dataloader()

            # Loop over each corruption task
            counters = {"for": 0, "back": 0}
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            for task in ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]:
                logger.info(f"start {task}")

                # Build dataloaders
                if task == "cloudy" :
                    tta_raw_data = tta_cloudy_raw_data
                    tta_valid_dataloader = tta_cloudy_valid_dataloader

                elif task == "overcast":
                    tta_raw_data = tta_overcast_raw_data
                    tta_valid_dataloader = tta_overcast_valid_dataloader

                elif task == "foggy":
                    tta_raw_data = tta_foggy_raw_data
                    tta_valid_dataloader = tta_foggy_valid_dataloader

                elif task == "rainy":
                    tta_raw_data = tta_rainy_raw_data
                    tta_valid_dataloader = tta_rainy_valid_dataloader

                elif task == "dawn":
                    tta_raw_data = tta_dawn_raw_data
                    tta_valid_dataloader = tta_dawn_valid_dataloader

                elif task == "night":
                    tta_raw_data = tta_night_raw_data
                    tta_valid_dataloader = tta_night_valid_dataloader

                elif task == "clear":
                    tta_raw_data = tta_clear_raw_data
                    tta_valid_dataloader = tta_clear_valid_dataloader

                # Run selected TTA method
                method_fn = self.get_method()
                result = method_fn(
                    model=model,
                    task=task,
                    tta_raw_data=tta_raw_data,
                    tta_valid_dataloader=tta_valid_dataloader,
                    reference_preprocessor=self.reference_preprocessor,
                    classes_list=classes_list,
                    counters=counters,
                    **extras,
                )

                all_results.append(result)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # Aggregate and report results across tasks
            each_task_map_list = utils.aggregate_runs(all_results)
            utils.print_results(each_task_map_list)

            # evaluation
            fwd = int(counters.get("for", 0))
            bwd = int(counters.get("back", 0))
            fps = counters["for"] / elapsed
            print("")
            print(f"{self.method}-----------------------------------------------")
            print(f" Forward (For.) : {fwd:,}")
            print(f" Backward (Back.) : {bwd:,}")
            print(f" FPS : {fps:.2f} img/s")


    def make_dataloader(self):
        #tta cloudy
        tta_cloudy_valid_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=True, task="cloudy"
        )

        tta_cloudy_raw_data = DataLoader(
            utils.LabelDataset(tta_cloudy_valid_dataset), 
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=utils.naive_collate_fn
        )
        
        tta_cloudy_valid_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_cloudy_valid_dataset),
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=partial(utils.collate_fn, preprocessor=self.reference_preprocessor)
        )

        # overcast
        tta_overcast_valid_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=True, task="overcast"
        )

        tta_overcast_raw_data = DataLoader(
            utils.LabelDataset(tta_overcast_valid_dataset), 
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=utils.naive_collate_fn
        )
        
        tta_overcast_valid_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_overcast_valid_dataset),
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=partial(utils.collate_fn, preprocessor=self.reference_preprocessor)
        )

        # foggy
        tta_foggy_valid_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=True, task="foggy"
        )

        tta_foggy_raw_data = DataLoader(
            utils.LabelDataset(tta_foggy_valid_dataset), 
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=utils.naive_collate_fn
        )
        
        tta_foggy_valid_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_foggy_valid_dataset),
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=partial(utils.collate_fn, preprocessor=self.reference_preprocessor)
        )

        # rainy
        tta_rainy_valid_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=True, task="rainy"
        )

        tta_rainy_raw_data = DataLoader(
            utils.LabelDataset(tta_rainy_valid_dataset), 
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=utils.naive_collate_fn
        )
        
        tta_rainy_valid_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_rainy_valid_dataset),
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=partial(utils.collate_fn, preprocessor=self.reference_preprocessor)
        )

        # dawn
        tta_dawn_valid_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=True, task="dawn"
        )

        tta_dawn_raw_data = DataLoader(
            utils.LabelDataset(tta_dawn_valid_dataset), 
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=utils.naive_collate_fn
        )
        
        tta_dawn_valid_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_dawn_valid_dataset),
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=partial(utils.collate_fn, preprocessor=self.reference_preprocessor)
        )

        # night
        tta_night_valid_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=True, task="night"
        )

        tta_night_raw_data = DataLoader(
            utils.LabelDataset(tta_night_valid_dataset), 
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=utils.naive_collate_fn
        )
        
        tta_night_valid_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_night_valid_dataset),
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=partial(utils.collate_fn, preprocessor=self.reference_preprocessor)
        )

        # clear
        tta_clear_valid_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=True, task="clear"
        )

        tta_clear_raw_data = DataLoader(
            utils.LabelDataset(tta_clear_valid_dataset), 
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=utils.naive_collate_fn
        )
        
        tta_clear_valid_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_clear_valid_dataset),
            batch_size=self.valid_batch, 
            shuffle=False,
            collate_fn=partial(utils.collate_fn, preprocessor=self.reference_preprocessor)
        )

        return (tta_cloudy_raw_data, tta_cloudy_valid_dataloader,
                tta_overcast_raw_data, tta_overcast_valid_dataloader,
                tta_foggy_raw_data, tta_foggy_valid_dataloader,
                tta_rainy_raw_data, tta_rainy_valid_dataloader,
                tta_dawn_raw_data, tta_dawn_valid_dataloader,
                tta_night_raw_data, tta_night_valid_dataloader,
                tta_clear_raw_data, tta_clear_valid_dataloader,
                 tta_cloudy_valid_dataset.classes)

    def direct_method(self, *, model, task, 
                      tta_raw_data, tta_valid_dataloader, reference_preprocessor,
                      classes_list, counters, **_):
        model.eval()
        if self.model_arch == "rt_detr":
            evaluator = utils.RTDETR_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)
        elif self.model_arch == "faster_rcnn":
            evaluator = utils.FastRCNN_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)

        for batch_i, labels, input in zip(tqdm(range(len(tta_raw_data))), tta_raw_data, tta_valid_dataloader):
            img = input['pixel_values'].to(self.device, non_blocking=True)
            with torch.no_grad():   
                if self.model_arch=="rt_detr":
                    outputs = model(img)
                elif self.model_arch=="faster_rcnn":
                    outputs = model(img, labels)

            evaluator.add(outputs, labels)
        
        result = evaluator.compute()
            
        return result

    def actmad(self, *, model, task, tta_raw_data, tta_valid_dataloader,
               reference_preprocessor, classes_list,
               clean_mean_list_final, clean_var_list_final, chosen_bn_layers, optimizer_actmad,
               counters, **_):

        # Unfreeze model parameters for ActMAD
        for param in model.parameters():
            param.requires_grad = True

        n_chosen_layers = len(chosen_bn_layers)

        l1_loss = nn.L1Loss(reduction='mean')

        if self.model_arch == "rt_detr":
            evaluator = utils.RTDETR_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)
        elif self.model_arch == "faster_rcnn":
            evaluator = utils.FastRCNN_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)
        
        for batch_i, labels, input in zip(tqdm(range(len(tta_raw_data))), tta_raw_data, tta_valid_dataloader):
            model.train()
            for m in model.modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                    m.eval()

            optimizer_actmad.zero_grad()
            save_outputs_tta = [utils.SaveOutput() for _ in range(n_chosen_layers)]

            hook_list_tta = [chosen_bn_layers[x].register_forward_hook(save_outputs_tta[x])
                            for x in range(n_chosen_layers)]
            
            img = input['pixel_values'].to(self.device, non_blocking=True)
            outputs = model(img)
            counters["for"]+= 1

            batch_mean_tta = [save_outputs_tta[x].get_out_mean() for x in range(n_chosen_layers)]
            batch_var_tta = [save_outputs_tta[x].get_out_var() for x in range(n_chosen_layers)]

            loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(self.device)
            loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(self.device)

            for i in range(n_chosen_layers):
                loss_mean += l1_loss(batch_mean_tta[i].to(self.device), clean_mean_list_final[i].to(self.device))
                loss_var += l1_loss(batch_var_tta[i].to(self.device), clean_var_list_final[i].to(self.device))
                
            loss =  loss_mean +  loss_var

            loss.backward()

            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer_actmad.step()
            counters["back"]+= 1

            for z in range(n_chosen_layers):
                save_outputs_tta[z].clear()
                hook_list_tta[z].remove()
            
            evaluator.add(outputs, labels)
        
        result = evaluator.compute()

        return result

    def norm(self, model, task,
             tta_raw_data, tta_valid_dataloader, 
             reference_preprocessor, classes_list, 
             counters, **_):
        
        if self.model_arch == "rt_detr":
            evaluator = utils.RTDETR_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)
        elif self.model_arch == "faster_rcnn":
            evaluator = utils.FastRCNN_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)

        for batch_i, labels, input in zip(tqdm(range(len(tta_raw_data))), tta_raw_data, tta_valid_dataloader):
            model.eval()
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = self.momentum
                    module.train()
            img = input['pixel_values'].to(self.device, non_blocking=True)

            outputs = model(img)
            counters["for"]+= 1

            model.eval()
            evaluator.add(outputs, labels)
        
        result = evaluator.compute()

        return result

    def dua(self, *, model, task, tta_raw_data, tta_valid_dataloader,
             reference_preprocessor, classes_list, tr_transform_adapt,
             counters, **_):

        # Initialize DUA parameters
        mom_pre = self.mom_pre

        # Collect only nn.BatchNorm2d layers for DUA (exclude RTDetrFrozenBatchNorm2d)
        bn_layers = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and not isinstance(m, RTDetrFrozenBatchNorm2d):
                bn_layers.append(m)

        if self.model_arch == "rt_detr":
            evaluator = utils.RTDETR_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)
        elif self.model_arch == "faster_rcnn":
            evaluator = utils.FastRCNN_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)

        for batch_i, labels, input in zip(tqdm(range(len(tta_raw_data))), tta_raw_data, tta_valid_dataloader):
            model.eval()

            # Update momentum for this batch (ContinualTTA style)
            momentum = mom_pre + self.min_momentum_constant

            # Register forward hooks to capture activations and update running stats for BatchNorm2d only
            hooks = []

            def get_dua_hook(momentum_val):
                def hook(module, input, output):
                    # Calculate batch statistics
                    x = input[0]
                    batch_mean = x.mean(dim=[0,2,3])
                    batch_var = x.var(dim=[0,2,3], unbiased=True)

                    # Update running statistics (ContinualTTA DUA style)
                    module.running_mean = (1 - momentum_val) * module.running_mean + momentum_val * batch_mean
                    module.running_var = (1 - momentum_val) * module.running_var + momentum_val * batch_var
                return hook

            # Register hooks only for nn.BatchNorm2d layers
            for layer in bn_layers:
                hook = layer.register_forward_hook(get_dua_hook(momentum))
                hooks.append(hook)

            img = input['pixel_values'].to(self.device, non_blocking=True)

            # Process each image in the batch separately for DUA augmentation
            augmented_imgs = []
            for i in range(img.shape[0]):
                single_img = img[i]  # Shape: [C, H, W]
                aug_img = utils.get_adaption_inputs_default(single_img, tr_transform_adapt, self.device)
                augmented_imgs.append(aug_img)

            # Stack all augmented images back to batch
            augmented_img = torch.cat(augmented_imgs, dim=0)

            # DUA adaptation with augmented images
            _ = model(augmented_img)  # Just for BatchNorm adaptation
            counters["for"]+= 1

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Update mom_pre for next batch (ContinualTTA style)
            mom_pre *= self.decay_factor

            # Evaluation with original images (no augmentation)
            model.eval()
            original_img = input['pixel_values'].to(self.device, non_blocking=True)
            outputs = model(original_img)
            evaluator.add(outputs, labels)

        result = evaluator.compute()

        return result
    
    def mean_teacher(self, task, tta_raw_data, tta_valid_dataloader, 
                    reference_preprocessor, classes_list, student_model, teacher_model,
                    optimizer_mt, weak_augmentation, strong_augmentation, 
                    counters, **_):
        
        student_model = student_model
        teacher_model = teacher_model

        teacher_model.eval()

        optimizer = optimizer_mt
        
        cudnn.benchmark = True
        global_step = 0

        if self.use_scheduler:
            scheduler= utils.make_scheduler(
                optimizer=optimizer_mt,
                warmup_steps=self.warmup_steps,
                initial_lr=(self.lr/100),
                decay_total_steps=self.decay_total_steps,
                total_steps=len(tta_valid_dataloader),
                base_lr=self.lr
            )
        
        if self.model_arch == "rt_detr":
            evaluator = utils.RTDETR_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)
        elif self.model_arch == "faster_rcnn":
            evaluator = utils.FastRCNN_Evaluator(class_list = classes_list, task = task, reference_preprocessor= reference_preprocessor)
            
        for batch_i, labels, input in zip(tqdm(range(len(tta_raw_data))), tta_raw_data, tta_valid_dataloader):
            # 추후 image에 augment넣는 작업 추가 - fixmatch style 이용
            imgs = input['pixel_values'].to(self.device, non_blocking=True)

            student_input = torch.stack([strong_augmentation(img) for img in imgs], dim=0) # strong aug
            teacher_input = torch.stack([weak_augmentation(img)  for img in imgs], dim=0) # weak aug

            student_model.train()
            teacher_model.eval()

            # make pseudo label
            with torch.no_grad():
                teacher_outputs = teacher_model(teacher_input)
                counters["for"]+= 1

            # Create target_sizes matching the batch size
            batch_size = teacher_input.shape[0]
            sizes = torch.tensor([[800, 1280]] * batch_size, device=self.device)

            teacher_results = reference_preprocessor.post_process_object_detection(
                teacher_outputs, target_sizes=sizes, threshold=0.5
            )

            pseudo_label = utils.make_label(teacher_results, (800, 1280))
            
            # student model train
            optimizer.zero_grad(set_to_none=True)
            student_outputs = student_model(pixel_values=student_input, labels=pseudo_label)
            counters["for"]+= 1

            # student_outputs에서 loss 꺼내기.
            loss = student_outputs.loss

            loss.backward()
            optimizer.step()
            counters["back"]+= 1

            if self.use_scheduler:
                scheduler.step()

            utils.update_ema_variables(student_model, teacher_model, alpha=0.99, global_step=global_step)
            global_step += 1
            evaluator.add(teacher_outputs, labels)
        
        result = evaluator.compute()

        return result

    # def whw(self, save_dir, task, best_state, best_map50, no_imp_streak,
    #         tta_train_dataloader, tta_raw_data, tta_valid_dataloader, 
    #         reference_preprocessor, classes_list, optimizer_whw, model, clean_mean_list_final, clean_var_list_final, **_):
    #     for batch_i, input in enumerate(tqdm(tta_train_dataloader)):
    #         imgs = input['pixel_values'].to(self.device, non_blocking=True)
    #         model.eval()
    #         utils.freeze_backbone_except_adapters(model) # adapter를 제외한 다른 parameter들 freeze
    #         cur_used = True
    #         # div_thr = 2 * sum(model.s_div.values()) * 

    #         # for weight regularization
    #         init_weights = []
    #         for p_idx, _p in enumerate(optimizer_whw.param_groups):
    #             p = _p['params'][0]
    #             init_weights.append(p.clone().detach())
            
    #         outputs, losses = model(imgs)
    #         total_loss = sum([losses[k] for k in losses])
    #         if total_loss > 0 and cur_used:
    #             total_loss.backward()
    #             optimizer_whw.step()
    #         else:
    #             pass
    #         optimizer_whw.zero_grad()

    #         with torch.no_grad():
    #             # 여기서 val_dataset으로 가끔씩 평가하는 부분 나옴.
                
            


        

        # TODO
        # ContinualTTA_ObjectDetection/detectron2/modeling/configure_adaptation_model.py에서
        # model에 adapter 붙이는 부분 구현
        # 나머지 학습 코드 구현
        # 다른 method WHW에서 실험한것처럼 코드 고치기.