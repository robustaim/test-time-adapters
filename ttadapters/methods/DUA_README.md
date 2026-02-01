# DUA (Dynamic Update Adaptation) Implementation

## 개요
`DUAEngine`은 별도의 Loss Backpropagation 없이, Batch Normalization Layer의 Running Statistics(Moving Average)를 Test Batch의 통계로 동적으로 업데이트하여 적응하는 방법입니다.

## 구현 상세 (`dua.py`)

### 1. 레이어 래핑 (`_wrap_layer`)
*   **Target Identification**: Adaptation 대상이 되는 Normalization Layer를 식별합니다. (Backbone, Encoder 등)
*   **Stats Backup**: 원본 `running_mean`, `running_var`를 백업(`original_running_mean`)하여 `reset()` 시 복구할 수 있도록 합니다.
*   **Instance Patching**: 대상 Layer의 `forward` 메서드를 `dua_forward`로 교체합니다.

### 2. Dynamic Update 로직 (`dua_forward`)
*   **Forward Interception**: Forward Pass가 호출될 때마다 다음을 수행합니다:
    1.  입력 Feature Map(`x`)의 현재 Batch Mean/Var를 계산합니다.
    2.  Decaying Momentum(`mom_pre`)을 계산합니다. ($Momentum = Mom_{pre} + MinConstant$)
    3.  **Running Stats Update**: 
        $$ \mu_{running} = (1 - m) \cdot \mu_{running} + m \cdot \mu_{batch} $$
        $$ \sigma^2_{running} = (1 - m) \cdot \sigma^2_{running} + m \cdot \sigma^2_{batch} $$
    4.  Momentum Decay: $Mom_{pre} \leftarrow Mom_{pre} \times DecayFactor$
*   **Normalization**: (선택적) 업데이트된 Running Stats를 사용하여 Manual Normalization을 수행하거나, 업데이트된 상태로 Original Forward를 호출합니다.

## 특징
*   **No Backprop**: Gradient 계산이 필요 없어 매우 빠릅니다.
*   **Continuous**: 데이터가 들어올수록 Momentum이 줄어들며 점진적으로 통계가 안정화됩니다.

## 사용법
```python
config = DUAConfig(decay_factor=0.94)
engine = DUAEngine(model, config)

engine.online(True) # DUA 모드 활성화 (Running Stats가 변하기 시작함)
output = engine(input)
```
