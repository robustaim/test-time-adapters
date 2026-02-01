# Test-Time Adaptation Methods

이 디렉토리는 다양한 Test-Time Adaptation (TTA) 방법론들의 구현을 포함하고 있습니다. 각 방법론은 `AdaptationEngine` (`base.py`)을 상속받아 구현되었으며, Detectron2 및 RT-DETR 등 다양한 모델 구조를 지원하도록 추상화되었습니다.

## 구현된 방법론 목록

1.  **[ActMAD](ActMAD_README.md)** (`actmad.py`)
    *   **설명**: Clean Data에서 추출한 Activation Statistics(Mean, Variance)와 Test Batch의 Statistics 간의 차이를 최소화하는 Loss를 통해 모델을 적응시킵니다.
    *   **특징**: `clean_mean_list`, `clean_var_list`를 사전에 추출하여 저장/로드합니다.

2.  **[DUA](DUA_README.md)** (`dua.py`)
    *   **설명**: Dynamic Update Adaptation. 배치마다 Normalization Layer의 Running Statistics를 동적으로 업데이트(Momentum Decay)하여 적응합니다.
    *   **특징**: 파라미터 업데이트(Backprop) 없이 Forward Pass 내에서 통계치를 수정합니다.

3.  **[NormAdaptation](NormAdaptation_README.md)** (`norm_adaptation.py`)
    *   **설명**: Source-Weighted Norm / Prediction Re-Norm. Source Domain의 통계치와 Current Batch의 통계치를 혼합(Blending)하여 정규화를 수행합니다.
    *   **특징**: `source_sum` 하이퍼파라미터를 통해 혼합 비율을 조절합니다.

4.  **[MeanTeacher](MeanTeacher_README.md)** (`mean_teacher.py`)
    *   **설명**: Teacher-Student 구조를 이용한 적응. Teacher 모델(Student의 EMA)의 예측을 Pseudo-Label로 사용하여 Student(Original) 모델을 학습시킵니다.
    *   **특징**: `RandAugmentMC`를 이용한 강한 Augmentation을 적용하여 Consistency를 학습합니다.

5.  **[WHW](WHW_README.md)** (`whw.py`)
    *   **설명**: Weighted Histogram / Continual TTA. 별도의 Adapter 모듈(`ParallelAdapter`, `ConvTaskWrapper`)을 부착하고, Global/Foreground Feature의 분포(KL Divergence)를 맞추도록 학습합니다.
    *   **특징**: Feature Alignment Loss, Loss/Divergence 기반의 Skipping Logic, Block 단위 Adapter Injection 등 복잡한 로직을 포함합니다.

---

## 공통 구조 (`AdaptationEngine`)

모든 엔진은 `AdaptationEngine`을 상속받으며 다음 공통 기능을 제공합니다:
- **`online(mode=True)`**: Adaptation 모드 진입/해제.
- **`reset()`**: 모델 상태 초기화.
- **`fit()`**: (필요 시) Clean Data 통계 추출.
- **Model Agnostic**: Detectron2 및 HuggingFace RT-DETR 등 다양한 Provider 지원.
