# ActMAD (Activation Mean Alignment and Discrepancy) Implementation

## 개요
`ActMADEngine`은 Clean Source Data에서 추출한 Activation Statistics(Mean, Variance)를 Target Test Data의 Statistics와 정렬(Alignment)시킴으로써 도메인 적응을 수행하는 방법입니다.

## 구현 상세 (`actmad.py`)

### 1. 통계 추출 및 저장 (`fit`)
*   **Layer 식별**: `_identify_layers` 메서드를 통해 모델 내의 Normalization Layer(`BatchNorm2d`, `LayerNorm`, `FrozenBatchNorm2d` 등)를 자동으로 탐색합니다.
*   **Filtering**: Config(`adaptation_layers`)에 따라 Backbone, Encoder, Decoder 중 적응 대상을 선별합니다. (기본: Backbone + Encoder의 후반부 레이어)
*   **Hooking**: `fit` 메서드 실행 시, 선별된 레이어에 Forward Hook을 등록하여 Clean Dataset에 대한 Activation Mean/Var를 축적(`accum_means`, `accum_vars`)하고 평균을 내어 저장합니다.

### 2. 적응 프로세스 (`forward`)
*   **Online Mode**: `online(True)` 호출 시, 저장된 Clean Statistics(`clean_mean_list`, `clean_var_list`)를 로드합니다.
*   **Loss Calculation**:
    1.  현재 Test Batch에 대한 Forward Pass를 수행하며 각 레이어의 Activation Statistics(`current_batch_means/vars`)를 수집합니다.
    2.  저장된 Clean Statistics와의 차이(L1 Loss)를 계산합니다.
    3.  `Alignment Loss = L1(Current_Mean, Clean_Mean) + L1(Current_Var, Clean_Var)`
*   **Optimization**: 계산된 Loss를 통해 모델 파라미터를 업데이트합니다.

## 사용법
```python
config = ActMADConfig(
    statistic_save_path="./stats.pt",
    adaptation_layers="backbone+encoder"
)
engine = ActMADEngine(model, config)

# 1. 초기 통계 추출 (최초 1회)
engine.fit(clean_dataset)

# 2. 테스트 타임 적응
engine.online(True)
output = engine(input_image)
```
