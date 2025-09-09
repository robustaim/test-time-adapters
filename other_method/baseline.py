import os
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


import utils

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
                 clean_bn_extract_batch: int=32,
                 model_states: Path="/home/elicer/ptta/RT-DETR_R50vd_SHIFT_CLEAR.safetensors",
                 lr: float=1e-4,
                 momentum: float=0.1,
                 mom_pre: float=0.07,
                 decay_factor: float=0.97,
                 min_momentum_constant: float=0.01,
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

        self.s_stats = torch.load("/workspace/ptta/other_method/WHW/rtdetr_feature_stats.pt")
        self.t_stats = {}
    

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
        else :
            raise NotImplementedError(f"Unsupported model_arch: {self.model_arch}")
            
        return model.to(self.device)

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
        else:
            raise NotImplementedError(f"Unsupported model_arch: {self.model_arch}")

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
        reference_preprocessor = self.image_processor()

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
                (
                    extras["clean_mean_list_final"],
                    extras["clean_var_list_final"],
                    extras["chosen_bn_layers"]
                ) = utils.extract_activation_alignment(
                    model, self.method, device, self.data_root, reference_preprocessor, batch_size=self.clean_bn_extract_batch
                    )      

                extras["optimizer_actmad"]  = optim.SGD(model.parameters(), lr=1e-5, momentum=0.3, weight_decay=1e-4, nesterov=True)
            
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
                    model, self.method, device, self.data_root, reference_preprocessor, batch_size=self.clean_bn_extract_batch
                    )
                
            # Save clean state of the model
            carry_state = copy.deepcopy(model.state_dict())
            
            # Loop over each corruption task
            for task in ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]:
                logger.info(f"start {task}")

                # restore model from carry_state (previous adapted state)
                model.load_state_dict(carry_state)
                best_state = copy.deepcopy(model.state_dict())
                best_map50 = -1.0
                no_imp_streak = 0

                # Build dataloaders
                tta_train_dataloader, tta_raw_data, tta_valid_dataloader, classes_list = self.make_dataloader(task, reference_preprocessor)

                # Run selected TTA method
                method_fn = self.get_method()
                carry_state, final_result = method_fn(
                    model=model,
                    save_dir=self.save_dir,
                    best_map50=best_map50,
                    no_imp_streak=no_imp_streak,
                    task=task,
                    best_state=best_state,
                    tta_train_dataloader=tta_train_dataloader,
                    tta_raw_data=tta_raw_data,
                    tta_valid_dataloader=tta_valid_dataloader,
                    reference_preprocessor=reference_preprocessor,
                    classes_list=classes_list,
                    **extras,
                )
                all_results.append(final_result)

            # Aggregate and report results across tasks
            each_task_map_list = utils.aggregate_runs(all_results)
            utils.print_results(each_task_map_list)

    def make_dataloader(self, task, reference_preprocessor):
        # tta train
        tta_train_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=False, task=task
        )

        tta_train_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_train_dataset),
            batch_size=1, 
            collate_fn=partial(utils.collate_fn, preprocessor=reference_preprocessor)
        )
        
        #tta valid
        tta_valid_dataset = utils.SHIFTCorruptedTaskDatasetForObjectDetection(
            root=self.data_root, valid=True, task=task
        )

        tta_raw_data = DataLoader(
            utils.LabelDataset(tta_valid_dataset), 
            batch_size=self.valid_batch, 
            collate_fn=utils.naive_collate_fn
        )
        
        tta_valid_dataloader = DataLoader(
            utils.DatasetAdapterForTransformers(tta_valid_dataset),
            batch_size=self.valid_batch, 
            collate_fn=partial(utils.collate_fn, preprocessor=reference_preprocessor)
        )

        return tta_train_dataloader, tta_raw_data, tta_valid_dataloader, tta_train_dataset.classes

    def direct_method(self, *, model, task, 
                      tta_raw_data, tta_valid_dataloader,
                      reference_preprocessor, classes_list, **_):
        final_result = utils.test(model, self.device, task, tta_raw_data, tta_valid_dataloader, reference_preprocessor, classes_list)
        carry_state = copy.deepcopy(model.state_dict())
        
        return carry_state, final_result

    def actmad(self, *, model, save_dir, task, best_state, best_map50, no_imp_streak,
               tta_train_dataloader, tta_raw_data, tta_valid_dataloader,
               reference_preprocessor, classes_list,
               clean_mean_list_final, clean_var_list_final, chosen_bn_layers, optimizer_actmad, **_):
        
        n_chosen_layers = len(chosen_bn_layers)
        
        l1_loss = nn.L1Loss(reduction='mean')

        for batch_i, input in enumerate(tqdm(tta_train_dataloader)):
            model.train()
            for m in model.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()

            optimizer_actmad.zero_grad()
            save_outputs_tta = [utils.SaveOutput() for _ in range(n_chosen_layers)]

            hook_list_tta = [chosen_bn_layers[x].register_forward_hook(save_outputs_tta[x])
                            for x in range(n_chosen_layers)]
            
            img = input['pixel_values'].to(self.device, non_blocking=True)
            _ = model(img)

            batch_mean_tta = [save_outputs_tta[x].get_out_mean() for x in range(n_chosen_layers)]
            batch_var_tta = [save_outputs_tta[x].get_out_var() for x in range(n_chosen_layers)]

            loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(self.device)
            loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(self.device)

            for i in range(n_chosen_layers):
                loss_mean += l1_loss(batch_mean_tta[i].to(self.device), clean_mean_list_final[i].to(self.device))
                loss_var += l1_loss(batch_var_tta[i].to(self.device), clean_var_list_final[i].to(self.device))
                
            loss = loss_mean + loss_var

            loss.backward()
            optimizer_actmad.step()

            for z in range(n_chosen_layers):
                save_outputs_tta[z].clear()
                hook_list_tta[z].remove()
            
            model.eval()
            current_map50, improve = utils.improve_test(self.device, batch_i, self.patience, self.eval_every, 
                                         model, task, tta_raw_data, tta_valid_dataloader, 
                                         reference_preprocessor, classes_list)
            if improve:
                print(f"[{task}] batch {batch_i}: mAP50 {best_map50:.4f} -> {current_map50:.4f} ✔")
                best_map50 = current_map50
                best_state = copy.deepcopy(model.state_dict())
                no_imp_streak = 0

            else:
                no_imp_streak += 1
                print(f"[{task}] batch {batch_i}: mAP50 {current_map50:.4f} (no-imp {no_imp_streak}/{self.patience})")
                if no_imp_streak >= self.patience:
                    print(f"[{task}] Early stop at batch {batch_i} (no improvement {self.patience} times).")
                    break

        model.load_state_dict(best_state)
        with torch.inference_mode():
            final_result = utils.test(model, self.device, task, tta_raw_data, tta_valid_dataloader, reference_preprocessor, classes_list)
        
        save_path = os.path.join(save_dir, f"ActMAD_model_{task}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved adapted ActMAD model after task [{task}] → {save_path}")
        
        carry_state = copy.deepcopy(model.state_dict())

        return carry_state, final_result

    def norm(self, model, save_dir, task, best_state, best_map50, no_imp_streak, 
             tta_train_dataloader, tta_raw_data, tta_valid_dataloader, 
             reference_preprocessor, classes_list, **_):
        for batch_i, input in enumerate(tqdm(tta_train_dataloader)):
            model.eval()
            for module in model.modules():
                if isinstance(module, RTDetrFrozenBatchNorm2d):
                    module.momentum = self.momentum
                    module.train()
            img = input['pixel_values'].to(self.device, non_blocking=True)

            _ = model(img)
            model.eval()
            if batch_i % self.eval_every == 0:
                current_map50, best_map50, improve = utils.improve_test(self.device, batch_i, self.eval_every,
                                            model, task, best_map50, tta_raw_data, tta_valid_dataloader, 
                                            reference_preprocessor, classes_list)
                if improve:
                    print(f"[{task}] batch {batch_i}: mAP50 {best_map50:.4f} -> {current_map50:.4f} ✔")
                    best_map50 = current_map50
                    best_state = copy.deepcopy(model.state_dict())
                    no_imp_streak = 0

                else:
                    no_imp_streak += 1
                    print(f"[{task}] batch {batch_i}: mAP50 {current_map50:.4f} (no-imp {no_imp_streak}/{self.patience})")
                    if no_imp_streak >= self.patience:
                        print(f"[{task}] Early stop at batch {batch_i} (no improvement {self.patience} times).")
                        break

        model.load_state_dict(best_state)
        with torch.inference_mode():
            final_result = utils.test(model, self.device, task, tta_raw_data, tta_valid_dataloader, reference_preprocessor, classes_list)
        
        save_path = os.path.join(save_dir, f"NORM_model_{task}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved adapted NORM model after task [{task}] → {save_path}")
        
        carry_state = copy.deepcopy(model.state_dict())

        return carry_state, final_result

    def dua(self, model, save_dir, task, best_state, best_map50, no_imp_streak, 
             tta_train_dataloader, tta_raw_data, tta_valid_dataloader, 
             reference_preprocessor, classes_list, tr_transform_adapt, **_):
        mom_pre = self.mom_pre
        for batch_i, input in enumerate(tqdm(tta_train_dataloader)):
            model.eval()
            mom_new = (mom_pre * self.decay_factor)
            for m in model.modules():
                if isinstance(m, RTDetrFrozenBatchNorm2d):
                    m.train()
                    m.momentum = mom_new + self.min_momentum_constant
            mom_pre = mom_new

            img = input['pixel_values'].squeeze(0).to(self.device, non_blocking=True)
            img = utils.get_adaption_inputs_default(img, tr_transform_adapt, self.device)

            _ = model(img)
            model.eval()
            if batch_i % self.eval_every == 0:
                current_map50, improve = utils.improve_test(self.device, batch_i, self.patience, self.eval_every, 
                                            model, task, tta_raw_data, tta_valid_dataloader, 
                                            reference_preprocessor, classes_list)
                if improve:
                    print(f"[{task}] batch {batch_i}: mAP50 {best_map50:.4f} -> {current_map50:.4f} ✔")
                    best_map50 = current_map50
                    best_state = copy.deepcopy(model.state_dict())
                    no_imp_streak = 0

                else:
                    no_imp_streak += 1
                    print(f"[{task}] batch {batch_i}: mAP50 {current_map50:.4f} (no-imp {no_imp_streak}/{self.patience})")
                    if no_imp_streak >= self.patience:
                        print(f"[{task}] Early stop at batch {batch_i} (no improvement {self.patience} times).")
                        break

        model.load_state_dict(best_state)
        with torch.inference_mode():
            final_result = utils.test(model, self.device, task, tta_raw_data, tta_valid_dataloader, reference_preprocessor, classes_list)
        
        save_path = os.path.join(save_dir, f"DUA_model_{task}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved adapted DUA model after task [{task}] → {save_path}")
        
        carry_state = copy.deepcopy(model.state_dict())

        return carry_state, final_result
    def mean_teacher(self, model, save_dir, task, best_state, best_map50, no_imp_streak,
            tta_train_dataloader, tta_raw_data, tta_valid_dataloader, 
            reference_preprocessor, classes_list, student_model, teacher_model,
            optimizer_mt, weak_augmentation, strong_augmentation, **_):
        
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
                total_steps=len(tta_train_dataloader),
                base_lr=self.lr
            )
        
        for batch_i, input in enumerate(tqdm(tta_train_dataloader)):
            # 추후 image에 augment넣는 작업 추가 - fixmatch style 이용
            imgs = input['pixel_values'].to(self.device, non_blocking=True)

            student_input = torch.stack([strong_augmentation(img) for img in imgs], dim=0) # strong aug
            teacher_input = torch.stack([weak_augmentation(img)  for img in imgs], dim=0) # weak aug

            student_model.train()
            teacher_model.eval()

            # make pseudo label
            with torch.no_grad():
                teacher_outputs = teacher_model(teacher_input)

            sizes = torch.tensor([[800, 1280]], device=self.device)

            teacher_results = reference_preprocessor.post_process_object_detection(
                teacher_outputs, target_sizes=sizes, threshold=0.3
            )

            pseudo_label = utils.make_label(teacher_results, (800, 1280))
            
            # student model train
            optimizer.zero_grad(set_to_none=True)
            student_outputs = student_model(pixel_values=student_input, labels=pseudo_label)
            # student_outputs에서 loss 꺼내기.
            loss = student_outputs.loss

            loss.backward()
            optimizer.step()

            if self.use_scheduler:
                scheduler.step()

            utils.update_ema_variables(student_model, teacher_model, alpha=0.999, global_step=global_step)
            global_step += 1

            current_map50, improve = utils.improve_test(self.device, batch_i, self.eval_every,
                                                teacher_model, task, best_map50, tta_raw_data, tta_valid_dataloader, 
                                                reference_preprocessor, classes_list)
            if improve:
                print(f"[{task}] batch {batch_i}: mAP50 {best_map50:.4f} -> {current_map50:.4f} ✔")
                best_map50 = current_map50
                best_state = copy.deepcopy(teacher_model.state_dict())
                no_imp_streak = 0

            else:
                no_imp_streak += 1
                print(f"[{task}] batch {batch_i}: mAP50 {current_map50:.4f} (no-imp {no_imp_streak}/{self.patience})")
                if no_imp_streak >= self.patience:
                    print(f"[{task}] Early stop at batch {batch_i} (no improvement {self.patience} times).")
                    break

        teacher_model.load_state_dict(best_state)
        with torch.inference_mode():
            final_result = utils.test(teacher_model, self.device, task, tta_raw_data, tta_valid_dataloader, reference_preprocessor, classes_list)
        
        save_path = os.path.join(save_dir, f"mean_teacher_model_{task}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved adapted Mean-Teacher model after task [{task}] → {save_path}")
        
        carry_state = copy.deepcopy(model.state_dict())

        return carry_state, final_result

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
