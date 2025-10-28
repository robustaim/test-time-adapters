"""
YOLO v8_transforms 추가 예시

현재 구현과 YOLO v8_transforms를 사용하는 구현의 차이를 보여줍니다.
"""

from typing import Callable
import warnings
import random
import torch
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes
from torchvision.transforms.v2.functional import convert_bounding_box_format
import numpy as np


# ============================================================================
# 현재 구현 (기본 transforms)
# ============================================================================

class CurrentYOLO11DataPreparation:
    """현재 구현 - 기본적인 augmentation만 사용"""

    def __init__(self, dataset, img_size=640, evaluation_mode=False):
        self.dataset = dataset
        self.img_size = img_size
        self.evaluation_mode = evaluation_mode

        # 단순한 transform
        self.train_transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=(img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),  # 이것만!
        ])

        self.valid_transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=(img_size, img_size)),
        ])


# ============================================================================
# YOLO v8_transforms 추가 버전
# ============================================================================

class YOLOv8StyleDataPreparation:
    """YOLO v8_transforms를 사용하는 버전 - 고급 augmentation 포함"""

    def __init__(
        self,
        dataset,
        img_size=640,
        evaluation_mode=False,
        # YOLO v8 augmentation 파라미터
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=0.0,  # Rotation degrees
        translate=0.1,  # Translation
        scale=0.5,    # Scaling
        shear=0.0,    # Shear
        perspective=0.0,  # Perspective
        flipud=0.0,   # Vertical flip probability
        fliplr=0.5,   # Horizontal flip probability
        mosaic=1.0,   # Mosaic augmentation probability
        mixup=0.0,    # MixUp augmentation probability
    ):
        self.dataset = dataset
        self.img_size = img_size
        self.evaluation_mode = evaluation_mode

        # YOLO augmentation 파라미터 저장
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic = mosaic
        self.mixup = mixup

        if not evaluation_mode:
            # Training transforms - YOLO v8 스타일
            self.train_transforms = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                # 1. Color augmentation (HSV)
                T.ColorJitter(
                    brightness=hsv_v,
                    contrast=hsv_v,
                    saturation=hsv_s,
                    hue=hsv_h
                ),
                # 2. Geometric augmentation
                T.RandomAffine(
                    degrees=degrees,
                    translate=(translate, translate),
                    scale=(1.0 - scale, 1.0 + scale),
                    shear=shear
                ),
                # 3. RandomPerspective
                T.RandomPerspective(
                    distortion_scale=perspective,
                    p=0.5 if perspective > 0 else 0.0
                ),
                # 4. Flips
                T.RandomHorizontalFlip(p=fliplr),
                T.RandomVerticalFlip(p=flipud),
                # 5. Final resize
                T.Resize(size=(img_size, img_size)),
            ])
        else:
            self.train_transforms = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size=(img_size, img_size)),
            ])

        self.valid_transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=(img_size, img_size)),
        ])

    def __getitem__(self, idx):
        """단일 이미지 로드"""
        image, metadata = self.dataset[idx]

        # Mosaic augmentation (확률적 적용)
        if not self.evaluation_mode and random.random() < self.mosaic:
            return self._mosaic_augmentation(idx)

        # 일반 transform
        return self.transforms(image, metadata)

    def transforms(self, image, metadata):
        """기본 transform"""
        bboxes = metadata['boxes2d']
        bbox_classes = metadata['boxes2d_classes']
        original_height, original_width = metadata['original_hw']

        if not isinstance(bboxes, BoundingBoxes):
            bboxes = BoundingBoxes(
                bboxes,
                format=BoundingBoxFormat.XYXY,
                canvas_size=(original_height, original_width)
            )

        # Apply transforms
        if not self.evaluation_mode:
            image, bboxes = self.train_transforms(image, bboxes)
        else:
            image, bboxes = self.valid_transforms(image, bboxes)

        # Convert to normalized YOLO format
        h, w = image.shape[-2:]
        xyxy = bboxes if isinstance(bboxes, torch.Tensor) else torch.tensor(bboxes)

        x1, y1, x2, y2 = xyxy.unbind(-1)
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        ww = (x2 - x1) / w
        hh = (y2 - y1) / h
        normalized_boxes = torch.stack([cx, cy, ww, hh], dim=-1)

        return image, {
            'pixel_values': image,
            'class_labels': bbox_classes,
            'boxes': normalized_boxes,
            'orig_size': torch.tensor([original_height, original_width]),
            'size': torch.tensor([h, w])
        }

    def _mosaic_augmentation(self, idx):
        """
        Mosaic augmentation - 4개 이미지를 조합

        구조:
        +-------+-------+
        | img1  | img2  |
        +-------+-------+
        | img3  | img4  |
        +-------+-------+
        """
        # 4개 이미지 랜덤 선택
        indices = [idx] + random.choices(range(len(self.dataset)), k=3)

        # Mosaic 캔버스 생성 (2배 크기)
        mosaic_size = self.img_size * 2
        mosaic_img = torch.zeros((3, mosaic_size, mosaic_size), dtype=torch.float32)

        all_boxes = []
        all_classes = []

        # 중심점 (랜덤)
        center_x = int(random.uniform(self.img_size * 0.5, self.img_size * 1.5))
        center_y = int(random.uniform(self.img_size * 0.5, self.img_size * 1.5))

        for i, index in enumerate(indices):
            image, metadata = self.dataset[index]

            # 간단한 transform만 적용
            image = T.ToImage()(image)
            image = T.ToDtype(torch.float32, scale=True)(image)

            # 각 이미지를 mosaic의 4분면에 배치
            if i == 0:  # top-left
                x1, y1 = 0, 0
                x2, y2 = center_x, center_y
            elif i == 1:  # top-right
                x1, y1 = center_x, 0
                x2, y2 = mosaic_size, center_y
            elif i == 2:  # bottom-left
                x1, y1 = 0, center_y
                x2, y2 = center_x, mosaic_size
            else:  # bottom-right
                x1, y1 = center_x, center_y
                x2, y2 = mosaic_size, mosaic_size

            # 이미지 리사이즈 및 배치
            h, w = y2 - y1, x2 - x1
            resized = T.Resize((h, w))(image)
            mosaic_img[:, y1:y2, x1:x2] = resized

            # Bounding box 변환
            bboxes = metadata['boxes2d']
            bbox_classes = metadata['boxes2d_classes']

            if len(bboxes) > 0:
                # 원본 이미지 크기
                orig_h, orig_w = metadata['original_hw']

                # bbox 좌표를 mosaic 좌표계로 변환
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.clone()
                else:
                    bboxes = torch.tensor(bboxes, dtype=torch.float32)

                # XYXY 형식으로 변환 후 스케일 조정
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / orig_w * w + x1
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / orig_h * h + y1

                all_boxes.append(bboxes)
                all_classes.extend(bbox_classes.tolist() if isinstance(bbox_classes, torch.Tensor) else bbox_classes)

        # 모든 박스 합치기
        if len(all_boxes) > 0:
            all_boxes = torch.cat(all_boxes, dim=0)
        else:
            all_boxes = torch.zeros((0, 4))
        all_classes = torch.tensor(all_classes)

        # Mosaic 이미지를 원래 크기로 리사이즈
        mosaic_img = T.Resize((self.img_size, self.img_size))(mosaic_img)

        # 박스도 리사이즈에 맞게 조정
        if len(all_boxes) > 0:
            all_boxes[:, [0, 2]] *= self.img_size / mosaic_size
            all_boxes[:, [1, 3]] *= self.img_size / mosaic_size

            # 클리핑
            all_boxes = torch.clamp(all_boxes, 0, self.img_size)

            # 너무 작은 박스 제거
            widths = all_boxes[:, 2] - all_boxes[:, 0]
            heights = all_boxes[:, 3] - all_boxes[:, 1]
            valid = (widths > 2) & (heights > 2)
            all_boxes = all_boxes[valid]
            all_classes = all_classes[valid]

        # YOLO format으로 변환
        if len(all_boxes) > 0:
            x1, y1, x2, y2 = all_boxes.unbind(-1)
            cx = (x1 + x2) / 2 / self.img_size
            cy = (y1 + y2) / 2 / self.img_size
            w = (x2 - x1) / self.img_size
            h = (y2 - y1) / self.img_size
            normalized_boxes = torch.stack([cx, cy, w, h], dim=-1)
        else:
            normalized_boxes = torch.zeros((0, 4))

        return mosaic_img, {
            'pixel_values': mosaic_img,
            'class_labels': all_classes,
            'boxes': normalized_boxes,
            'orig_size': torch.tensor([self.img_size, self.img_size]),
            'size': torch.tensor([self.img_size, self.img_size])
        }

    def collate_fn(self, batch):
        """Collate function - MixUp도 여기서 적용 가능"""
        images = torch.stack([item[1]['pixel_values'] for item in batch])
        labels = [item[1] for item in batch]

        # MixUp augmentation (확률적 적용)
        if not self.evaluation_mode and random.random() < self.mixup:
            images, labels = self._mixup_augmentation(images, labels)

        return {
            'pixel_values': images,
            'labels': labels
        }

    def _mixup_augmentation(self, images, labels):
        """
        MixUp augmentation - 두 이미지를 블렌딩

        mixed_image = alpha * image1 + (1 - alpha) * image2
        """
        batch_size = len(images)

        # Beta 분포에서 alpha 샘플링
        alpha = np.random.beta(32.0, 32.0)

        # 배치 내 이미지를 섞음
        indices = torch.randperm(batch_size)

        # 이미지 믹싱
        mixed_images = alpha * images + (1 - alpha) * images[indices]

        # 라벨도 두 개를 유지 (detection에서는 단순히 concat)
        mixed_labels = []
        for i in range(batch_size):
            label1 = labels[i]
            label2 = labels[indices[i]]

            # 두 이미지의 박스를 모두 포함
            boxes1 = label1['boxes']
            boxes2 = label2['boxes']
            classes1 = label1['class_labels']
            classes2 = label2['class_labels']

            mixed_labels.append({
                'pixel_values': mixed_images[i],
                'boxes': torch.cat([boxes1, boxes2], dim=0),
                'class_labels': torch.cat([classes1, classes2], dim=0) if isinstance(classes1, torch.Tensor)
                              else torch.tensor(list(classes1) + list(classes2)),
                'orig_size': label1['orig_size'],
                'size': label1['size']
            })

        return mixed_images, mixed_labels


# ============================================================================
# 비교 요약
# ============================================================================

comparison = """
┌──────────────────────────────────────────────────────────────────────────┐
│                      현재 구현 vs YOLO v8_transforms                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  현재 구현 (Basic):                                                       │
│  ✓ ToImage, ToDtype                                                      │
│  ✓ Resize                                                                │
│  ✓ RandomHorizontalFlip                                                  │
│                                                                          │
│  YOLO v8_transforms (Advanced):                                          │
│  ✓ ColorJitter (HSV augmentation)                                        │
│  ✓ RandomAffine (rotation, translation, scale, shear)                   │
│  ✓ RandomPerspective (원근 변환)                                          │
│  ✓ RandomHorizontalFlip + RandomVerticalFlip                            │
│  ✓ Mosaic (4개 이미지 조합) ⭐                                            │
│  ✓ MixUp (2개 이미지 블렌딩) ⭐                                           │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  주요 차이점:                                                             │
│                                                                          │
│  1. Mosaic Augmentation                                                  │
│     - 4개 이미지를 하나로 조합                                            │
│     - Small object detection 성능 향상                                    │
│     - 다양한 스케일과 맥락 학습                                            │
│                                                                          │
│  2. MixUp Augmentation                                                   │
│     - 2개 이미지를 블렌딩                                                 │
│     - Regularization 효과                                                │
│     - 모델의 일반화 성능 향상                                             │
│                                                                          │
│  3. Color & Geometric Augmentation                                      │
│     - HSV 조정으로 조명 변화에 강건                                       │
│     - Affine/Perspective로 다양한 각도 학습                              │
│                                                                          │
│  4. 구현 복잡도                                                           │
│     - Mosaic: collate_fn에서 여러 이미지 접근 필요                       │
│     - MixUp: batch 단위 처리 필요                                        │
│     - 더 많은 메모리와 계산량 필요                                        │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  성능 영향 (일반적):                                                      │
│  - Mosaic + MixUp: mAP +2~5% 향상 가능                                   │
│  - 학습 시간: 20~30% 증가                                                │
│  - Small object detection: 큰 폭 개선                                    │
│                                                                          │
│  참고:                                                                   │
│  - YOLOv8: mosaic=1.0, mixup=0.15                                        │
│  - YOLOv11: mosaic=0.0, mixup=0.0 (공식 weights는 단순하게 학습!)        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
"""

print(comparison)
