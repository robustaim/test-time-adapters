# NormAdaptation (Source-Weighted / Prediction Re-Norm) Implementation

## 개요
`NormAdaptationEngine`은 Test Batch의 통계치와 Source Domain(Training)의 통계치를 적절한 비율로 혼합(Blending)하여 정규화(Normalization)에 사용하는 방법입니다. TENT와 유사하지만 파라미터 업데이트 대신 통계치 혼합에 중점을 둡니다.

## 구현 상세 (`norm_adaptation.py`)

### 1. 레이어 래핑 및 설정
*   **Target**: BatchNorm 계열 Layer (`BatchNorm2d`, `FrozenBatchNorm2d`, `RTDetrFrozenBatchNorm2d`).
*   **Parameter**: `source_sum` (혼합 가중치를 결정하는 상수).

### 2. 통계 혼합 로직 (`norm_forward`)
*   **Alpha Calculation**: 현재 배치의 크기($N$)와 `source_sum`($S$)을 이용해 혼합 비율 $\alpha$를 계산합니다.
    $$ \alpha = \frac{N}{S + N} $$
*   **Stats Blending**:
    $$ \mu_{blended} = (1 - \alpha) \cdot \mu_{source} + \alpha \cdot \mu_{batch} $$
    $$ \sigma^2_{blended} = (1 - \alpha) \cdot \sigma^2_{source} + \alpha \cdot \sigma^2_{batch} $$
*   **Manual Normalization**: `running_mean/var`를 영구적으로 수정하지 않고, **해당 Forward Pass에서만** 계산된 $\mu_{blended}, \sigma^2_{blended}$를 사용하여 정규화를 수행합니다.
    $$ y = \frac{x - \mu_{blended}}{\sqrt{\sigma^2_{blended} + \epsilon}} \cdot \gamma + \beta $$

## 특징
*   **Batch Size Robustness**: Batch Size가 작을 때는 Source 통계에 의존하고, 클 때는 Batch 통계를 더 많이 반영하여 안정성을 확보합니다.
*   **Instance-wise**: DUA와 달리 Running Stats를 영구적으로 오염시키지 않는 방식(Local Blending)으로 구현되었습니다. (단, 엔진 설정에 따라 영구 업데이트 방식 변형 가능)

## 사용법
```python
config = NormAdaptationConfig(source_sum=128)
engine = NormAdaptationEngine(model, config)

engine.online(True)
output = engine(input)
```
