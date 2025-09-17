import sys
from pathlib import Path

# 현재 폴더: ptta/other_method/ActMAD/
# ptta 바로 위의 디렉토리를 sys.path에 추가
PROJECT_PARENT = Path.cwd().parents[0]  # -> ptta/ 의 부모 디렉토리
print(PROJECT_PARENT)
sys.path.insert(0, str(PROJECT_PARENT))

import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessorFast,
    RTDetrConfig,
)
from os import path
from ttadapters.datasets import SHIFTClearDatasetForObjectDetection
import utils
from transformers.image_utils import AnnotationFormat
from functools import partial
from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d
from safetensors.torch import load_file

def collect_rtdetr_feature_statistics( 
    dataloader,
    device='cuda',
    num_classes=6,
    max_samples=2000,
    save_path='models/rtdetr_feature_stats.pt'
):
    reference_model_id = "PekingU/rtdetr_r50vd"
    reference_config = RTDetrConfig.from_pretrained(reference_model_id, torch_dtype=torch.float32, return_dict=True)
    reference_config.num_labels = 6

    reference_config.image_size = 800
    model = RTDetrForObjectDetection(config=reference_config)
    model_states = load_file("/workspace/ptta/RT-DETR_R50vd_SHIFT_CLEAR.safetensors")
    model.load_state_dict(model_states, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    """
    RT-DETR 모델의 feature 통계를 수집하는 통합 함수
    
    Args:
        model: RT-DETR 모델
        dataloader: 학습 데이터 로더
        device: 디바이스
        num_classes: 클래스 수
        max_samples: 최대 샘플 수
        save_path: 저장 경로
    
    Returns:
        feat_stats: 수집된 통계 딕셔너리
    """
    
    model = model.to(device)
    model.eval()
    
    # ResNet backbone 접근
    resnet_backbone = model.model.backbone.model
    
    # Feature 저장용 컨테이너
    gl_features = {}
    fg_features = {k: [] for k in range(num_classes)}
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting features")):
            if sample_count >= max_samples:
                break
                
            # 입력 준비
            img = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch.get('labels', None)
            
            # === 1. Backbone features 수집 ===
            features = resnet_backbone(img)
            
            # features 처리 - tuple인 경우 처리
            processed_features = {}
            if isinstance(features, dict):
                for k, v in features.items():
                    if isinstance(v, tuple):
                        processed_features[k] = v[0]
                    else:
                        processed_features[k] = v
            elif isinstance(features, (list, tuple)):
                for i, feat in enumerate(features):
                    if isinstance(feat, tuple):
                        processed_features[f"p{i}"] = feat[0]
                    else:
                        processed_features[f"p{i}"] = feat
            else:
                if isinstance(features, tuple):
                    processed_features["p0"] = features[0]
                else:
                    processed_features["p0"] = features
            
            # Global average pooling 후 저장
            for k, feat in processed_features.items():
                if torch.is_tensor(feat) and len(feat.shape) == 4:  # [B, C, H, W]
                    pooled = feat.mean(dim=[2, 3]).detach()
                    if k not in gl_features:
                        gl_features[k] = pooled.cpu()
                    else:
                        gl_features[k] = torch.cat([gl_features[k], pooled.cpu()], dim=0)
            
            # === 2. Query features (클래스별) 수집 ===
            try:
                outputs = model(img)
                if "logits" in outputs and "last_hidden_state" in outputs:
                    logits = outputs["logits"]
                    query_feats = outputs["last_hidden_state"]
                    
                    # 클래스 예측 및 필터링
                    has_bg = (logits.shape[-1] == num_classes + 1)
                    if has_bg:
                        probs = logits.softmax(-1)
                        fg_scores, fg_preds = probs[..., :num_classes].max(-1)
                    else:
                        probs = logits.sigmoid()
                        fg_scores, fg_preds = probs.max(-1)
                    
                    valid = fg_scores >= 0.5
                    
                    # 클래스별 features 저장
                    B, N, D = query_feats.shape
                    for b in range(B):
                        for n in range(N):
                            if valid[b, n]:
                                cls = fg_preds[b, n].item()
                                if cls < num_classes:
                                    fg_features[cls].append(query_feats[b, n].detach().cpu())
            except Exception as e:
                print(f"Error collecting query features: {e}")
            
            sample_count += img.shape[0]
    
    # === 3. 통계 계산 ===
    feat_stats = {}
    
    # Global level 통계
    feat_stats["gl"] = {}
    for k in gl_features:
        if gl_features[k].shape[0] > 1:
            mean = gl_features[k].mean(dim=0)
            try:
                cov = torch.cov(gl_features[k].T)
                if torch.isnan(cov).any():
                    cov = torch.diag(gl_features[k].var(dim=0) + 1e-6)
                else:
                    cov = cov + torch.eye(cov.shape[0]) * 1e-6
            except:
                cov = torch.diag(gl_features[k].var(dim=0) + 1e-6)
            
            feat_stats["gl"][k] = (mean, cov)
            print(f"Collected global stats for {k}: shape {mean.shape}")
    
    # Foreground 클래스별 통계
    feat_stats["fg"] = {}
    for cls, feats_list in fg_features.items():
        if len(feats_list) > 10:
            feats = torch.stack(feats_list, dim=0)
            mean = feats.mean(dim=0)
            try:
                cov = torch.cov(feats.T)
                if torch.isnan(cov).any():
                    cov = torch.diag(feats.var(dim=0) + 1e-6)
                else:
                    cov = cov + torch.eye(cov.shape[0]) * 1e-6
            except:
                cov = torch.diag(feats.var(dim=0) + 1e-6)
            
            feat_stats["fg"][cls] = (mean, cov)
            print(f"Collected fg stats for class {cls}: {len(feats_list)} samples")
    
    # BatchNorm/LayerNorm 통계 - RT-DETR용 수정
    feat_stats["bn_stats"] = []
    for name, module in resnet_backbone.named_modules():
        try:
            # RTDetrFrozenBatchNorm2d 처리
            if 'FrozenBatchNorm' in module.__class__.__name__:
                if hasattr(module, 'weight') and hasattr(module, 'bias'):
                    # eps 속성이 없으면 기본값 사용
                    eps = getattr(module, 'eps', 1e-5)
                    
                    if hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
                        mean = module.bias - module.weight * module.running_mean / torch.sqrt(module.running_var + eps)
                        var = (module.weight / torch.sqrt(module.running_var + eps)) ** 2
                    else:
                        # running stats가 없으면 weight/bias만 사용
                        mean = module.bias
                        var = module.weight ** 2
                    
                    feat_stats["bn_stats"].append((mean.detach(), var.detach()))
                    
            elif isinstance(module, torch.nn.BatchNorm2d):
                feat_stats["bn_stats"].append(
                    (module.running_mean.detach(), module.running_var.detach())
                )
                
            elif isinstance(module, (torch.nn.LayerNorm, LayerNorm)):
                if hasattr(module, 'weight') and hasattr(module, 'bias'):
                    feat_stats["bn_stats"].append(
                        (module.bias.detach(), module.weight.detach()**2)
                    )
        except Exception as e:
            print(f"Warning: Failed to process module {name}: {e}")
            continue
    
    # KL divergence 기준치
    feat_stats["kl_div"] = {}
    for k in gl_features:
        if gl_features[k].shape[0] > 20:
            try:
                full_mean = gl_features[k].mean(dim=0)
                full_cov = torch.cov(gl_features[k].T)
                if torch.isnan(full_cov).any() or torch.isinf(full_cov).any():
                    full_cov = torch.diag(gl_features[k].var(dim=0) + 1e-6)
                else:
                    full_cov = full_cov + torch.eye(full_cov.shape[0]) * 1e-6
                
                part_mean = gl_features[k][:20].mean(dim=0)
                part_cov = torch.cov(gl_features[k][:20].T)
                if torch.isnan(part_cov).any() or torch.isinf(part_cov).any():
                    part_cov = torch.diag(gl_features[k][:20].var(dim=0) + 1e-6)
                else:
                    part_cov = part_cov + torch.eye(part_cov.shape[0]) * 1e-6
                
                dist1 = torch.distributions.MultivariateNormal(full_mean, full_cov)
                dist2 = torch.distributions.MultivariateNormal(part_mean, part_cov)
                
                kl_div = (torch.distributions.kl.kl_divergence(dist1, dist2) + 
                         torch.distributions.kl.kl_divergence(dist2, dist1)) / 2
                
                feat_stats["kl_div"][k] = kl_div.item()
                print(f"In-domain KL divergence for {k}: {kl_div.item():.3f}")
            except Exception as e:
                print(f"KL divergence calculation failed for {k}: {e}")
                feat_stats["kl_div"][k] = 1.0
    
    # === 4. 저장 ===
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(feat_stats, save_path)
        print(f"\nFeature statistics saved to {save_path}")
    
    # 수집 결과 출력
    print("\n=== Collected Statistics ===")
    print(f"Global levels: {list(feat_stats['gl'].keys())}")
    print(f"Foreground classes with stats: {list(feat_stats['fg'].keys())}")
    print(f"BN/LN layers: {len(feat_stats['bn_stats'])}")
    print(f"KL baselines: {feat_stats['kl_div']}")
    
    return feat_stats

# 사용 예시
if __name__ == "__main__":
    from functools import partial
    reference_preprocessor = RTDetrImageProcessorFast.from_pretrained("PekingU/rtdetr_r50vd")
    reference_preprocessor.format = AnnotationFormat.COCO_DETECTION  # COCO Format / Detection BBOX Format
    reference_preprocessor.size = {"height": 800, "width": 800}
    reference_preprocessor.do_resize = False

    data_root = path.normpath(path.join(Path.cwd(), "..", "data"))
    
    # 데이터로더 준비
    train_dataset = SHIFTClearDatasetForObjectDetection(root=data_root, train=True)

    train_dataloader = DataLoader(
        utils.DatasetAdapterForTransformers(train_dataset),
        batch_size=8, 
        collate_fn=partial(utils.collate_fn, preprocessor=reference_preprocessor)
    )
    
    # 통계 수집
    feat_stats = collect_rtdetr_feature_statistics(
        dataloader=train_dataloader,
        device='cuda:0',
        num_classes=6,
        max_samples=2000,
        save_path='/workspace/ptta/other_method/WHW/rtdetr_feature_stats.pt'
    )