# WHW (Weighted Histogram / Continual TTA) Implementation

## 개요
`WHWEngine`은 Continual Test-Time Adaptation을 위해 제안된 복합적인 방법론입니다. 모델의 Block에 별도의 Adapter 모듈을 삽입하고, Source Domain의 Feature 분포(Global & Foreground)와 일치하도록 Adapter를 학습시킵니다.

## 구현 상세 (`whw.py`)

### 1. Adapter Injection
*   **ParallelAdapter**: ResNet Block의 Conv2 출력에 병렬로 부착되는 Adapter입니다. (`ParallelAdapterWithProjection`)
*   **ConvTaskWrapper**: Conv1 Layer를 감싸서 Adapter와 병렬로 동작하게 하는 Wrapper입니다.
*   **Structure**: Detectron2의 ResNet Backbone 구조(`bottom_up.resN`)를 순회하며 Adapter를 동적으로 삽입합니다.

### 2. Feature Alignment Strategy
*   **Statistics Collection**: Clean Dataset에서 Class별 Foreground Feature와 Image별 Global Feature의 Mean/Covariance를 추출하여 저장합니다.
*   **Alignment Loss (KL Divergence)**:
    *   **Global Alignment**: Backbone Global Pooling Feature의 분포를 Source 분포와 매칭합니다.
    *   **Foreground Alignment**: ROI Heads를 통과한 Box Feature 중 High Confidence Prediction에 해당하는 Feature들의 분포를 Source Class 분포와 매칭합니다.
    *   두 분포 간의 Symmetric KL Divergence를 Loss로 사용합니다.

### 3. Skipping Logic
*   **Efficiency**: 매 배치를 학습하면 느리기 때문에, Loss 변화량(EMA)이나 통계적 임계값(Statistical Threshold)을 기준으로 중복/불필요한 적응 Step을 건너뛰는(Skip) 로직이 포함되어 있습니다.

## 특징
*   **Complex Modification**: 단순 파라미터 튜닝이 아닌, 모델 구조 자체를 변형(Adapter 추가)합니다.
*   **Feature-Level Adaptation**: 최종 출력이 아닌 중간 Feature Map의 분포를 교정합니다.

## 사용법
```python
config = WHWConfig(source_feat_stats="parsed_stats.pt")
engine = WHWEngine(model, config)

# 1. Source 통계 로드 (필수)
# engine.fit(clean_dataset) # 혹은 사전 저장된 파일 로드

engine.online(True)
output = engine(input)
```
