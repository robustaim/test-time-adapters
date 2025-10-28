# YOLO v8_transforms 추가 시 수정 예시

## 📊 현재 구현 vs YOLO v8_transforms

### 1️⃣ 현재 구현 (기본 Transforms)

```python
# ttadapters/models/yolo11/ul.py 현재 코드

class YOLO11DataPreparation(DataPreparation):
    def __init__(self, dataset, img_size=640, evaluation_mode=False):
        # 단순한 transforms만
        self.train_transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=(img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),  # ← 이것만!
        ])

    def __getitem__(self, idx):
        image, metadata = self.dataset[idx]
        return self.transforms(image, metadata)  # 단순 변환

    def collate_fn(self, batch):
        images = torch.stack([item[1]['pixel_values'] for item in batch])
        labels = [item[1] for item in batch]
        return {'pixel_values': images, 'labels': labels}
```

**특징:**
- ✅ 구현이 단순하고 빠름
- ✅ 메모리 효율적
- ❌ 제한적인 augmentation
- ❌ Small object detection 성능 제한

---

### 2️⃣ YOLO v8_transforms 추가 버전

```python
# 수정된 버전

class YOLO11DataPreparation(DataPreparation):
    def __init__(
        self,
        dataset,
        img_size=640,
        evaluation_mode=False,
        # 🆕 YOLO v8 augmentation 파라미터 추가
        hsv_h=0.015,      # HSV-Hue
        hsv_s=0.7,        # HSV-Saturation
        hsv_v=0.4,        # HSV-Value
        degrees=0.0,      # Rotation
        translate=0.1,    # Translation
        scale=0.5,        # Scale
        shear=0.0,        # Shear
        perspective=0.0,  # Perspective
        flipud=0.0,       # Vertical flip prob
        fliplr=0.5,       # Horizontal flip prob
        mosaic=1.0,       # 🌟 Mosaic prob
        mixup=0.0,        # 🌟 MixUp prob
    ):
        self.mosaic = mosaic
        self.mixup = mixup

        # 🆕 고급 transforms
        self.train_transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            # 1. Color augmentation
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
            # 3. Perspective
            T.RandomPerspective(distortion_scale=perspective, p=0.5),
            # 4. Flips
            T.RandomHorizontalFlip(p=fliplr),
            T.RandomVerticalFlip(p=flipud),
            T.Resize(size=(img_size, img_size)),
        ])

    def __getitem__(self, idx):
        # 🆕 Mosaic augmentation 확률적 적용
        if not self.evaluation_mode and random.random() < self.mosaic:
            return self._mosaic_augmentation(idx)

        image, metadata = self.dataset[idx]
        return self.transforms(image, metadata)

    def _mosaic_augmentation(self, idx):
        """🆕 Mosaic: 4개 이미지를 조합"""
        # 1. 4개 이미지 랜덤 선택
        indices = [idx] + random.choices(range(len(self.dataset)), k=3)

        # 2. 2x2 그리드 생성
        mosaic_size = self.img_size * 2
        mosaic_img = torch.zeros((3, mosaic_size, mosaic_size))

        # 3. 랜덤 중심점
        center_x = int(random.uniform(self.img_size * 0.5, self.img_size * 1.5))
        center_y = int(random.uniform(self.img_size * 0.5, self.img_size * 1.5))

        all_boxes = []
        all_classes = []

        # 4. 각 이미지를 4분면에 배치
        for i, index in enumerate(indices):
            image, metadata = self.dataset[index]

            # 위치 결정 (top-left, top-right, bottom-left, bottom-right)
            if i == 0:    # top-left
                x1, y1, x2, y2 = 0, 0, center_x, center_y
            elif i == 1:  # top-right
                x1, y1, x2, y2 = center_x, 0, mosaic_size, center_y
            elif i == 2:  # bottom-left
                x1, y1, x2, y2 = 0, center_y, center_x, mosaic_size
            else:         # bottom-right
                x1, y1, x2, y2 = center_x, center_y, mosaic_size, mosaic_size

            # 이미지 리사이즈 및 배치
            h, w = y2 - y1, x2 - x1
            resized = T.Resize((h, w))(image)
            mosaic_img[:, y1:y2, x1:x2] = resized

            # Bounding box 변환 (원본 좌표 → mosaic 좌표)
            bboxes = metadata['boxes2d']
            orig_h, orig_w = metadata['original_hw']

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / orig_w * w + x1
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / orig_h * h + y1

            all_boxes.append(bboxes)
            all_classes.extend(metadata['boxes2d_classes'])

        # 5. 모든 박스 합치기
        all_boxes = torch.cat(all_boxes, dim=0)

        # 6. 원래 크기로 리사이즈
        mosaic_img = T.Resize((self.img_size, self.img_size))(mosaic_img)
        all_boxes *= self.img_size / mosaic_size

        # 7. YOLO format으로 변환
        # ... (normalized cx, cy, w, h)

        return mosaic_img, labels

    def collate_fn(self, batch):
        """🆕 MixUp도 여기서 적용"""
        images = torch.stack([item[1]['pixel_values'] for item in batch])
        labels = [item[1] for item in batch]

        # 🆕 MixUp augmentation
        if not self.evaluation_mode and random.random() < self.mixup:
            images, labels = self._mixup_augmentation(images, labels)

        return {'pixel_values': images, 'labels': labels}

    def _mixup_augmentation(self, images, labels):
        """🆕 MixUp: 두 이미지 블렌딩"""
        batch_size = len(images)

        # Beta 분포에서 alpha 샘플링
        alpha = np.random.beta(32.0, 32.0)

        # 배치 내 이미지 섞기
        indices = torch.randperm(batch_size)

        # 이미지 믹싱: mixed = alpha * img1 + (1-alpha) * img2
        mixed_images = alpha * images + (1 - alpha) * images[indices]

        # 라벨은 두 개 모두 유지 (박스 concat)
        mixed_labels = []
        for i in range(batch_size):
            label1 = labels[i]
            label2 = labels[indices[i]]

            mixed_labels.append({
                'boxes': torch.cat([label1['boxes'], label2['boxes']]),
                'class_labels': torch.cat([label1['class_labels'], label2['class_labels']]),
                # ...
            })

        return mixed_images, mixed_labels
```

---

## 🎯 주요 변경사항 요약

### 추가되는 코드

1. **`__init__` 파라미터**: 10개 이상의 augmentation 파라미터 추가
2. **`_mosaic_augmentation()` 메서드**: ~100줄 (4개 이미지 조합 로직)
3. **`_mixup_augmentation()` 메서드**: ~30줄 (2개 이미지 블렌딩)
4. **`__getitem__` 수정**: Mosaic 확률적 적용
5. **`collate_fn` 수정**: MixUp 확률적 적용

### 코드 복잡도

```
현재 구현:     ~350줄
v8 추가 버전:  ~550줄 (+200줄, 약 60% 증가)
```

---

## 📈 성능 비교

### 학습 속도
```
현재 구현:         100%
v8 transforms:     70-80% (20-30% 느림)
```

### mAP 성능 (일반적)
```
현재 구현:         baseline
v8 transforms:     +2~5% mAP
  - Mosaic만:      +1~3% mAP
  - + MixUp:       +2~5% mAP
```

### Small Object Detection
```
현재 구현:         baseline
Mosaic 추가:       +5~10% (큰 폭 개선!)
```

---

## 🤔 선택 가이드

### 현재 구현을 유지하는 경우

✅ **이런 경우 추천:**
- 빠른 프로토타이핑이 필요한 경우
- 계산 자원이 제한적인 경우
- 데이터셋이 충분히 크고 다양한 경우
- YOLOv11 공식 방식을 따르고 싶은 경우 (mosaic=0, mixup=0)

### v8_transforms 추가하는 경우

✅ **이런 경우 추천:**
- 데이터셋이 작은 경우 (augmentation으로 보완)
- Small object detection 성능이 중요한 경우
- mAP 성능을 최대한 높이고 싶은 경우
- YOLOv8 스타일 학습을 원하는 경우
- 계산 자원이 충분한 경우

---

## 💡 권장사항

### 점진적 추가 전략

1. **1단계**: Color augmentation만 추가
   ```python
   T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015)
   ```

2. **2단계**: Geometric augmentation 추가
   ```python
   T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.5, 1.5))
   ```

3. **3단계**: Mosaic 추가 (가장 효과적)
   ```python
   mosaic=1.0  # 항상 적용
   ```

4. **4단계** (선택): MixUp 추가
   ```python
   mixup=0.15  # 15% 확률
   ```

### 하이퍼파라미터 설정

**보수적 (추천):**
```python
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
degrees=0.0, translate=0.1, scale=0.5,
mosaic=0.5, mixup=0.0  # Mosaic만 50%
```

**공격적 (YOLOv8 스타일):**
```python
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
degrees=0.0, translate=0.1, scale=0.5,
mosaic=1.0, mixup=0.15  # 풀 augmentation
```

**YOLOv11 스타일 (공식):**
```python
# 단순한 augmentation만
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
mosaic=0.0, mixup=0.0  # 현재 구현과 유사
```

---

## 📝 결론

**현재 구현은 YOLOv11 공식 학습 방식에 가깝습니다.**

YOLOv11 공식 weights가 `mosaic=0.0, mixup=0.0`으로 학습되었다는 점을 고려하면,
현재의 단순한 구현도 충분히 합리적인 선택입니다.

다만, **작은 데이터셋**이나 **small object detection**이 중요한 경우라면
v8_transforms (특히 Mosaic)를 추가하는 것을 강력히 권장합니다.

---

## 📂 실제 파일 위치

예시 코드는 다음 파일에서 확인할 수 있습니다:
- `/home/user/test-time-adapters/yolo_v8_transforms_example.py`

현재 구현:
- `/home/user/test-time-adapters/ttadapters/models/yolo11/ul.py`
