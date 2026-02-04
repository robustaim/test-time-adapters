from typing import List, Optional, Literal
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from ..base import AdaptationEngine, AdaptationConfig
from ...models.base import BaseModel


@dataclass
class CascadedNormConfig(AdaptationConfig):
    """Configuration for CascadedNorm."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-3

    temperature: float = 0.01
    saturation_limit: float = 100.0
    param_regularization: float = 0.0001
    
    cascade_mode: Literal["all", "single", "single_last", "selected"] = "single_last"  # Focus on high-level features
    cascade_indices: Optional[List[int]] = None
    anchor_granularity: Literal["block", "stage", "fpn"] = "fpn"  # "block": per-block, "stage": per-layer, "fpn": FPN outputs (WHW-identical)


class DifferentiableHistogramStretcher(nn.Module):
    """Differentiable histogram stretching."""

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def soft_percentile(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Differentiable percentile approximation."""
        x_flat = x.flatten()
        n = x_flat.shape[0]
        p = p.to(x.device)

        idx = (p / 100.0) * (n - 1)
        indices = torch.arange(n, device=x.device, dtype=x.dtype)
        weights = F.softmax(-(indices - idx).abs() / (self.temperature * n), dim=0)

        sorted_x, _ = torch.sort(x_flat)
        return (weights * sorted_x).sum()

    def stretch_channel(self, channel, clip_low, clip_high, gamma):
        """Apply stretching to single channel with gamma correction."""
        low_val = self.soft_percentile(channel, clip_low)
        high_val = self.soft_percentile(channel, clip_high)

        scale = 50.0
        clipped = low_val + F.softplus((channel - low_val) * scale) / scale
        clipped = high_val - F.softplus((high_val - clipped) * scale) / scale

        range_val = high_val - low_val + 1e-6
        normalized = (clipped - low_val) / range_val

        gamma_corrected = torch.pow(normalized + 1e-6, gamma)

        return torch.clamp(gamma_corrected * 255.0, 0, 255)

    def forward(self, image, clip_low, clip_high, gamma):
        """Apply stretching to image with gamma correction."""
        C = image.shape[0]
        stretched = torch.zeros_like(image)

        for c in range(C):
            stretched[c] = self.stretch_channel(image[c], clip_low, clip_high, gamma)

        return stretched


class GammaTransform(nn.Module):
    """Learnable parameters for noise reduction with gamma correction."""
    noise_floor_init = 2.0
    gamma_init = 1.0

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.saturation_limit = torch.tensor(config.saturation_limit, requires_grad=False)
        self.noise_floor = nn.Parameter(torch.tensor(self.noise_floor_init))
        self.gamma = nn.Parameter(torch.tensor(self.gamma_init))

        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self, img):
        """Get constrained parameters and apply transform."""
        noise_floor = self.noise_floor.clamp(min=0.0, max=20.0)  # pass range 20~100
        gamma = self.gamma.clamp(min=0.5, max=2.0)  # gamma range 0.5~2.0

        transformed = self.stretcher(img, noise_floor, self.saturation_limit, gamma)

        return transformed, (noise_floor, self.saturation_limit, gamma)

    def get_regularization_loss(self):
        """Compute regularization loss relative to initialization anchors."""
        return (self.noise_floor - self.noise_floor_init).pow(2) + (self.gamma - self.gamma_init).pow(2)


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Manages transformation and norm layer statistics.

    Integrates GammaTransform controller and tracks normalization layers.
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        # Transform controller with integrated stretcher
        self.transform_controller = GammaTransform(config)

        # Anchors
        self.anchors: List[nn.Module] = []

    def forward(self, img):
        transformed, params = self.transform_controller(img)

        # Residual Form
        output = 0.5 * transformed + 0.5 * img

        return output, params

    def compute_alignment_loss(self) -> torch.Tensor:
        """Compute alignment loss from anchors."""
        if not self.anchors:
            return torch.tensor(0.0, device=self.transform_controller.noise_floor.device)
            
        # Sum KL losses from all anchors
        total_loss = sum([a.loss for a in self.anchors])

        if self.config.param_regularization > 0:
            reg_loss = self.transform_controller.get_regularization_loss()
            total_loss = total_loss + reg_loss * self.config.param_regularization

        return total_loss

    def online_parameters(self):
        return self.transform_controller.parameters()


class CascadedAnchor(nn.Module):
    """
    Inserted into the network to measure feature statistics.
    Modes:
        - training=True: Updates running_mean/var (Calibration).
        - training=False: Computes KL Divergence Loss (Adaptation).
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('source_mean', torch.tensor(0.0))
        self.register_buffer('source_var', torch.tensor(1.0))
        self.register_buffer('count', torch.tensor(0.0))
        self.loss = torch.tensor(0.0)
        
    def _update_stats(self, x: torch.Tensor):
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        if self.count == 0:
            self.source_mean = batch_mean
            self.source_var = batch_var
            self.count = torch.tensor(1.0)
        else:
            n = self.count
            self.count += 1
            self.source_mean = self.source_mean + (batch_mean - self.source_mean) / self.count
            self.source_var = self.source_var + (batch_var - self.source_var) / self.count

    def _compute_kl_loss(self, x: torch.Tensor):
        q_mean = x.mean()
        q_var = x.var(unbiased=False) + 1e-6 
        p_mean = self.source_mean.to(x.device)
        p_var = self.source_var.to(x.device) + 1e-6
        sigma_p = torch.sqrt(p_var)
        sigma_q = torch.sqrt(q_var)
        term1 = torch.log(sigma_q / sigma_p)
        term2 = (p_var + (p_mean - q_mean)**2) / (2 * q_var)
        self.loss = term1 + term2 - 0.5

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self._update_stats(x)
                self.loss = torch.tensor(0.0, device=x.device)
        else:
            self._compute_kl_loss(x)
        return x


class CascadedNormEngine(AdaptationEngine):
    """
    CascadedNorm: Input transformation for BN/LN alignment.

    Transforms input to match norm layer source statistics.
    Works with both BatchNorm and LayerNorm.
    """
    model_name: str = "CascadedNormEngine"

    def __init__(self, base_model: BaseModel, config: CascadedNormConfig):
        self.cascaded_norm: CascadedNorm  # will be initialized in _pre_init()
        self.cascaded_norm_state: dict
        self.config = config

        super().__init__(base_model, config)

    def _pre_init(self):
        # Transformation modules
        self.cascaded_norm = CascadedNorm(self.config)

    def _post_init(self):
        self.cascaded_norm.to(self.device)
        self.cascaded_norm_state = {key: value.cpu() for key, value in self.cascaded_norm.state_dict().items()}

        # Insert Anchors and register to CascadedNorm
        self._insert_anchors()
        
        # Collect anchors based on granularity
        if self.config.anchor_granularity == "fpn":
            # FPN anchors are stored in backbone attributes
            if hasattr(self.base_model, 'backbone'):
                if hasattr(self.base_model.backbone, '_fpn_anchors'):
                    # Detectron2: ModuleDict
                    self.cascaded_norm.anchors = list(self.base_model.backbone._fpn_anchors.values())
                elif hasattr(self.base_model.backbone, '_backbone_anchors'):
                    # RT-DETR: ModuleList
                    self.cascaded_norm.anchors = list(self.base_model.backbone._backbone_anchors)
                else:
                    # Fallback: search in modules
                    self.cascaded_norm.anchors = [m for m in self.base_model.modules() if isinstance(m, CascadedAnchor)]
            else:
                # YOLO or fallback
                self.cascaded_norm.anchors = [m for m in self.base_model.modules() if isinstance(m, CascadedAnchor)]
        else:
            # Block/Stage: search in all modules
            self.cascaded_norm.anchors = [m for m in self.base_model.modules() if isinstance(m, CascadedAnchor)]

        # Stats
        self._stats = {'alignment_losses': [], 'transform_params': []}


    def _insert_anchors(self):
        """
        Insert anchors after Residual Blocks, Stages, or FPN outputs.
        Supports filtering via cascade_mode and granularity control.
        """
        if self.config.anchor_granularity == "fpn":
            self._insert_anchors_fpn()
        elif self.config.anchor_granularity == "stage":
            self._insert_anchors_stage()
        else:
            self._insert_anchors_block()
    
    def _insert_anchors_fpn(self):
        """
        Insert anchors to intercept FPN outputs (WHW-identical).
        Supports: Detectron2 (FPN), RT-DETR (Hybrid Encoder), YOLO (PAN Neck)
        """
        print("[CascadedNormEngine] Injecting anchors into model (FPN-based, WHW-identical)...")
        
        # Detect model provider
        if self.model_provider == ModelProvider.Detectron2:
            self._insert_anchors_fpn_detectron2()
        elif self.model_provider == ModelProvider.HuggingFace:
            self._insert_anchors_fpn_rtdetr()
        elif self.model_provider == ModelProvider.Ultralytics:
            self._insert_anchors_fpn_yolo()
        else:
            print(f"[WARNING] FPN insertion not supported for {self.model_provider}. Falling back to stage-based.")
            self._insert_anchors_stage()
    
    def _insert_anchors_fpn_detectron2(self):
        """Insert anchors for Detectron2 FPN outputs (ResNet50/SwinT Faster RCNN)."""
        # Detectron2 FPN structure: backbone returns dict {'p2', 'p3', 'p4', 'p5', 'p6'}
        # We need to wrap backbone.forward to insert anchors after FPN outputs
        
        original_backbone_forward = self.base_model.backbone.forward
        fpn_anchors = nn.ModuleDict()
        
        # Determine which FPN levels to use
        fpn_levels = ['p2', 'p3', 'p4', 'p5']  # Standard FPN levels
        
        if self.config.cascade_mode == "single_last":
            fpn_levels = ['p5']  # Only highest level
        elif self.config.cascade_mode == "selected" and self.config.cascade_indices:
            all_levels = ['p2', 'p3', 'p4', 'p5']
            fpn_levels = [all_levels[i] for i in self.config.cascade_indices if i < len(all_levels)]
        
        print(f"Target FPN Levels: {fpn_levels}")
        
        # Create anchors for each level
        for level in fpn_levels:
            fpn_anchors[level] = CascadedAnchor()
            print(f"  -> Inserting Anchor after FPN level '{level}'")
        
        # Wrap backbone forward
        def fpn_forward_with_anchors(x):
            features = original_backbone_forward(x)
            
            # Insert anchors after each FPN level
            for level in fpn_levels:
                if level in features:
                    features[level] = fpn_anchors[level](features[level])
            
            return features
        
        self.base_model.backbone.forward = fpn_forward_with_anchors
        self.base_model.backbone._fpn_anchors = fpn_anchors  # Store reference
        
        print(f"[CascadedNormEngine] Inserted {len(fpn_anchors)} FPN anchors (Detectron2)")
    
    def _insert_anchors_fpn_rtdetr(self):
        """Insert anchors for RT-DETR multi-scale features."""
        # RT-DETR: backbone outputs multi-scale features
        # We wrap backbone.forward to insert anchors
        
        original_backbone_forward = self.base_model.backbone.forward
        backbone_anchors = nn.ModuleList()
        
        # RT-DETR typically has 3-4 feature levels
        num_levels = 4 if self.config.cascade_mode == "all" else 1
        
        for i in range(num_levels):
            backbone_anchors.append(CascadedAnchor())
            print(f"  -> Inserting Anchor after backbone level {i}")
        
        def backbone_forward_with_anchors(x):
            features = original_backbone_forward(x)
            
            # features is typically a list or dict
            if isinstance(features, dict):
                for i, (k, v) in enumerate(features.items()):
                    if i < len(backbone_anchors):
                        features[k] = backbone_anchors[i](v)
            elif isinstance(features, (list, tuple)):
                features = [backbone_anchors[i](f) if i < len(backbone_anchors) else f 
                           for i, f in enumerate(features)]
            
            return features
        
        self.base_model.backbone.forward = backbone_forward_with_anchors
        self.base_model.backbone._backbone_anchors = backbone_anchors
        
        print(f"[CascadedNormEngine] Inserted {len(backbone_anchors)} backbone anchors (RT-DETR)")
    
    def _insert_anchors_fpn_yolo(self):
        """Insert anchors for YOLO PAN neck outputs."""
        # YOLO structure: model.model[0-9] = backbone, [10-15] = neck (PAN), [16-22] = head
        # We insert anchors after neck outputs
        
        neck_indices = [12, 15, 18]  # Typical YOLO neck output indices
        
        if self.config.cascade_mode == "single_last":
            neck_indices = [18]  # Only last neck output
        elif self.config.cascade_mode == "selected" and self.config.cascade_indices:
            all_indices = [12, 15, 18]
            neck_indices = [all_indices[i] for i in self.config.cascade_indices if i < len(all_indices)]
        
        print(f"Target Neck Indices: {neck_indices}")
        
        # Wrap neck layers with anchors
        for idx in neck_indices:
            if idx < len(self.base_model.model):
                original_layer = self.base_model.model[idx]
                anchor = CascadedAnchor()
                self.base_model.model[idx] = nn.Sequential(original_layer, anchor)
                print(f"  -> Inserting Anchor after neck layer {idx}")
        
        print(f"[CascadedNormEngine] Inserted {len(neck_indices)} neck anchors (YOLO)")
    

    def _insert_anchors_stage(self):
        """
        Insert anchors after each stage.
        WHW-style coarse-grained approach.
        Supports: ResNet (layer1-4), SwinT (layers.0-3), YOLO (backbone stages)
        """
        print("[CascadedNormEngine] Injecting anchors into model (Stage-based, WHW-style)...")
        
        stages = []
        
        # Strategy 1: ResNet-style (layer1, layer2, layer3, layer4)
        for name, module in self.base_model.named_children():
            if 'layer' in name.lower() and any(char.isdigit() for char in name):
                stages.append((name, module))
        
        # Strategy 2: Swin Transformer (layers.0, layers.1, layers.2, layers.3)
        if not stages:
            if hasattr(self.base_model, 'layers'):
                layers_module = self.base_model.layers
                if isinstance(layers_module, nn.ModuleList) or isinstance(layers_module, nn.Sequential):
                    for idx, layer in enumerate(layers_module):
                        stages.append((f'layers.{idx}', layer))
        
        # Strategy 3: YOLO backbone (model.0 ~ model.N, typically model.9 is last backbone)
        if not stages:
            if hasattr(self.base_model, 'model'):
                # YOLO typically has backbone ending around index 9-10
                # We'll detect by looking for downsampling layers
                model_seq = self.base_model.model
                if isinstance(model_seq, nn.Sequential) or isinstance(model_seq, nn.ModuleList):
                    # Heuristic: Collect every 3rd layer as "stage" (YOLO has 3 layers per stage typically)
                    for idx in [2, 4, 6, 8]:  # Common YOLO stage boundaries
                        if idx < len(model_seq):
                            stages.append((f'model.{idx}', model_seq[idx]))
        
        # Fallback: Use block-based if no stages found
        if not stages:
            print("[WARNING] No stages detected. Falling back to block-based insertion.")
            self._insert_anchors_block()
            return
        
        total_stages = len(stages)
        print(f"Total Stages Found: {total_stages} ({[name for name, _ in stages]})")
        
        # Determine target indices
        target_indices = set()
        if self.config.cascade_mode == "all":
            target_indices = set(range(total_stages))
        elif self.config.cascade_mode == "single_last":
            target_indices = {total_stages - 1}
        elif self.config.cascade_mode == "selected" and self.config.cascade_indices:
            target_indices = {i if i >= 0 else total_stages + i for i in self.config.cascade_indices}
        
        print(f"Target Stage Indices: {sorted(list(target_indices))}")
        
        # Inject anchors
        for idx, (stage_name, stage_module) in enumerate(stages):
            if idx in target_indices:
                print(f"  -> Inserting Anchor after Stage '{stage_name}' (index {idx})")
                anchor = CascadedAnchor()
                
                # Handle different parent structures
                if '.' in stage_name:
                    # Nested (e.g., 'layers.0', 'model.2')
                    parent_name, child_idx = stage_name.rsplit('.', 1)
                    parent = getattr(self.base_model, parent_name)
                    parent[int(child_idx)] = nn.Sequential(stage_module, anchor)
                else:
                    # Direct child (e.g., 'layer1')
                    setattr(self.base_model, stage_name, nn.Sequential(stage_module, anchor))

    
    def _insert_anchors_block(self):
        """
        Insert anchors after individual Residual Blocks.
        Fine-grained approach.
        """
        print("[CascadedNormEngine] Injecting anchors into model (Block-based)...")
        
        self.block_count = 0
        total_blocks_found = 0
        
        # 1. First Pass: Count total blocks
        def count_blocks(module):
            count = 0
            for name, child in module.named_children():
                class_name = child.__class__.__name__
                is_block = "Block" in class_name or "Bottleneck" in class_name
                if is_block:
                    count += 1
                else:
                    count += count_blocks(child)
            return count
            
        total_blocks_found = count_blocks(self.base_model)
        print(f"Total Blocks Found: {total_blocks_found}")

        # 2. Determine target indices
        target_indices = set()
        if self.config.cascade_mode == "all":
            target_indices = set(range(total_blocks_found))
        elif self.config.cascade_mode == "single_last":
            target_indices = {total_blocks_found - 1}
        elif self.config.cascade_mode == "selected" and self.config.cascade_indices:
            target_indices = {i if i >= 0 else total_blocks_found + i for i in self.config.cascade_indices}

        print(f"Target Block Indices: {sorted(list(target_indices))}")

        # 3. Injection Pass
        self.current_idx = 0
        
        def inject_recursive(module):
            for name, child in module.named_children():
                class_name = child.__class__.__name__
                is_block = "Block" in class_name or "Bottleneck" in class_name
                
                if is_block:
                    if self.current_idx in target_indices:
                        print(f"  -> Inserting Anchor after Block #{self.current_idx} ({class_name})")
                        anchor = CascadedAnchor()
                        setattr(module, name, nn.Sequential(child, anchor))
                    self.current_idx += 1
                else:
                    inject_recursive(child)
        
        inject_recursive(self.base_model)

    def fit(self, source_loader, max_samples=2000):
        print(f"[CascadedNormEngine] Starting Calibration ({max_samples} samples)...")
        self.base_model.eval()
        for anchor in self.cascaded_norm.anchors:
            anchor.train()
            anchor.count.fill_(0)
            
        count = 0
        pbar = tqdm(source_loader, desc="Calibrating", unit="batch")
        with torch.no_grad():
            for batch in pbar:
                img = None
                
                # Handle dict input
                if isinstance(batch, dict):
                    for key in ['pixel_values', 'img', 'image']:
                        if key in batch:
                            img = batch[key]
                            break
                    if img is None:
                        continue
                # Handle list/tuple input
                elif isinstance(batch, (list, tuple)):
                    img = batch[0]
                    # Check if first element is a dict (e.g., detectron2 format)
                    if isinstance(img, dict):
                        for key in ['pixel_values', 'img', 'image']:
                            if key in img:
                                img = img[key]
                                break
                # Handle tensor input
                else:
                    img = batch
                
                # Final check: img should be a tensor
                if not isinstance(img, torch.Tensor):
                    continue
                
                img = img.to(self.device)
                if img.max() <= 1.0: img = img * 255.0
                
                self.base_model(img)
                count += img.size(0)
                pbar.set_postfix({'samples': count, 'target': max_samples})
                if count >= max_samples:
                    break
        
        pbar.close()
        for anchor in self.cascaded_norm.anchors:
            anchor.eval()
        print(f"[CascadedNormEngine] Calibration Complete. {len(self.cascaded_norm.anchors)} anchors.")

    def online_parameters(self):
        """Only transformation parameters."""
        return self.cascaded_norm.online_parameters()

    def _transform_batch(self, imgs):
        """Transform batch."""
        transformed_list = []
        params_list = []

        for i in range(imgs.shape[0]):
            transformed, params = self.cascaded_norm(imgs[i])
            transformed_list.append(transformed)
            params_list.append(params)

        return torch.stack(transformed_list, dim=0), params_list

    def forward(self, batched_inputs=None, **kwargs):
        """Forward with transformation and alignment."""
        # Handle case where inputs are passed as kwargs (e.g. model(**batch))
        is_kwargs = False
        if batched_inputs is None and kwargs:
            batched_inputs = kwargs
            is_kwargs = True

        if not self.adapting:
            if is_kwargs:
                return self.base_model(**batched_inputs)
            return self.base_model(batched_inputs)

        if isinstance(batched_inputs, torch.Tensor):
            return self._forward_tensor(batched_inputs)
        elif isinstance(batched_inputs, dict):
            return self._forward_dict(batched_inputs, unpack_args=is_kwargs)
        return self._forward_dict_list(batched_inputs)

    def _forward_tensor(self, imgs):
        """Handle tensor input."""
        imgs = imgs.to(self._device)

        original_scale = imgs.max() <= 1.0
        if original_scale:
            imgs = imgs * 255.0

        imgs_transformed, params_list = self._transform_batch(imgs)

        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))

        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        outputs = self.base_model(model_input)

        # Compute Loss
        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        total_loss = alignment_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())

        return outputs

    def _forward_dict(self, input_dict, unpack_args=False):
        """Handle dictionary input (RT-DETR, YOLO)."""
        if 'pixel_values' in input_dict:
            img_key = 'pixel_values'
        elif 'img' in input_dict:
            img_key = 'img'
        else:
            if unpack_args:
                return self.base_model(**input_dict)
            return self.base_model(input_dict)

        imgs = input_dict[img_key].to(self._device)
        
        original_scale = imgs.max() <= 1.0
        if original_scale:
            imgs = imgs * 255.0

        imgs_transformed, params_list = self._transform_batch(imgs)

        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))

        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        
        new_input = input_dict.copy()
        new_input[img_key] = model_input
        
        if unpack_args:
            outputs = self.base_model(**new_input)
        else:
            outputs = self.base_model(new_input)

        # Compute Loss
        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        total_loss = alignment_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())

        return outputs

    def _forward_dict_list(self, batched_inputs):
        """Handle list of dicts."""
        transformed_inputs = []

        for input_dict in batched_inputs:
            if 'image' not in input_dict:
                transformed_inputs.append(input_dict)
                continue

            img = input_dict['image'].to(self._device)
            original_scale = img.max() <= 1.0
            if original_scale:
                img = img * 255.0

            img_transformed, params = self.cascaded_norm(img)
            self._stats['transform_params'].append(tuple(p.item() for p in params))

            new_input = input_dict.copy()
            new_input['image'] = img_transformed / 255.0 if original_scale else img_transformed
            transformed_inputs.append(new_input)

        outputs = self.base_model(transformed_inputs)

        # Compute Loss
        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        total_loss = alignment_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())

        return outputs

    def reset(self, reset_stats=False):
        """Reset model to initial state."""
        self.cascaded_norm.load_state_dict(self.cascaded_norm_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try:
            self.optimizer.zero_grad()
        except:
            pass
        if reset_stats:
            self._stats = {'alignment_losses': [], 'transform_params': []}

    @property
    def stats(self):
        """Get statistics."""
        if not self._stats['alignment_losses']:
            return None

        params_array = np.array(self._stats['transform_params'])

        return {
            'num_steps': len(self._stats['alignment_losses']),
            'mean_loss': np.mean(self._stats['alignment_losses']),
            'final_loss': self._stats['alignment_losses'][-1],
            'mean_noise_floor': np.mean(params_array[:, 0]),
            'mean_saturation_limit': np.mean(params_array[:, 1]),
            'mean_gamma': np.mean(params_array[:, 2]),
        }

    def to(self, *args, **kwargs):
        """Move to device."""
        result = super().to(*args, **kwargs)
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result
