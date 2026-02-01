# Mean Teacher Implementation

## 개요
`MeanTeacherEngine`은 원본 모델(Student)과 그 파라미터의 Exponential Moving Average(EMA)를 유지하는 Teacher 모델 간의 Consistency를 이용해 적응하는 방법입니다.

## 구현 상세 (`mean_teacher.py`)

### 1. 구조 (Teacher-Student)
*   **Teacher Model**: 초기화 시 Student(Base Model)를 복제하여 생성하며, Gradient Update를 하지 않습니다(`requires_grad=False`).
*   **Student Model**: Base Model 그 자체이며, Test 데이터를 통해 학습(Adaptation)됩니다.

### 2. Augmentation & Pseudo-Labeling
*   **RandAugmentMC**: `augment_strength_n`, `augment_strength_m` 파라미터를 받아 `AutoContrast`, `Sharpness`, `Shear`, `Solarize` 등 다양한 강한 증강 기법을 적용합니다.
*   **Flow**:
    1.  **Teacher**: 원본(Weak) 이미지에 대해 추론을 수행하여 Pseudo-Label(Confidence > Threshold)을 생성합니다.
    2.  **Student**: 강하게 증강된(Strong) 이미지와 Teacher가 만든 Pseudo-Label을 사용하여 학습(Loss 계산 및 Backprop)합니다.

### 3. EMA Update
*   매 Step(Forward)마다 Student의 파라미터를 Teacher에 반영합니다.
    $$ \theta_{teacher} = \alpha \cdot \theta_{teacher} + (1 - \alpha) \cdot \theta_{student} $$
*   최종 추론 결과는 안정적인 **Teacher Model**의 출력을 반환합니다.

## 특징
*   **Consistency Regularization**: 같은 데이터의 다른 뷰(Weak/Strong)에 대해 모델이 일관된 예측을 하도록 유도합니다.
*   **Model Agnostic**: Detectron2 및 RT-DETR의 입력/출력 구조(Dict vs Keyword Args)를 분기 처리하여 모두 지원합니다.

## 사용법
```python
config = MeanTeacherConfig(ema_alpha=0.99, conf_threshold=0.3)
engine = MeanTeacherEngine(model, config)

engine.online(True)
output = engine(input) # 내부적으로 Teacher의 출력을 반환
```
