# Test-Time Adaptation via Input Normalization: Theoretical Evolution and Experimental Findings

## Executive Summary

본 보고서는 CascadedNorm 방법론의 이론적 진화 과정을 기술합니다. 초기 "소스 통계 정보 정렬" 이론에서 시작하여 "Bypass" 이론으로의 리프레이밍 시도, 그리고 통합 이론 실패까지의 과정을 실험 결과와 수식을 통해 설명합니다.

---

## 1. 초기 이론: 소스 통계 정보 정렬 (V3)

### 1.1 기본 가설

**핵심 아이디어**: Input transformation을 통해 normalization layer의 입력을 source domain statistics에 align

### 1.2 BatchNorm 처리

**BN의 Inference 동작:**

$$y = \gamma \cdot \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}} + \beta$$

여기서:
- $\mu_{\text{running}}, \sigma^2_{\text{running}}$: Training에서 수집한 running statistics
- $\gamma, \beta$: Learned affine parameters

**Alignment 전략:**

Target domain input $x_{\text{target}}$을 transform하여 batch statistics가 source running statistics와 일치하도록:

$$\mathcal{L}_{\text{BN}} = \|\text{mean}(x_{\text{transformed}}) - \mu_{\text{running}}\|^2 + \|\text{var}(x_{\text{transformed}}) - \sigma^2_{\text{running}}\|^2$$

**구현:**

채널별 통계를 scalar로 평균화:

```python
source_mean = module.running_mean.mean()  # (C,) → scalar
source_var = module.running_var.mean()

batch_mean = x.mean(dim=(0,2,3)).mean()  # → scalar
loss = MSE(batch_mean, source_mean)
```

### 1.3 LayerNorm 처리

**LN의 동작:**

$$y = \gamma \cdot \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}} + \beta$$

여기서:
- $\mu_{\text{batch}}, \sigma^2_{\text{batch}}$: Batch에서 즉시 계산 (running stats 없음)

**초기 구현:**

$$\mathcal{L}_{\text{LN}} = \|\text{mean}(x_{\text{transformed}}) - 0\|^2 + \|\text{var}(x_{\text{transformed}}) - 1\|^2$$

Target을 $(0, 1)$로 설정.

### 1.4 이론적 불일치 발견

**교수님 질문**: "LN은 source 통계 정보가 없는데 어떻게 loss를 만드는가?"

**문제 분석:**

1. BN: $\mu_{\text{running}}, \sigma^2_{\text{running}}$ (source 정보) 존재
2. LN: Running stats 없음 → Source 정보 부재
3. $(0, 1)$ target의 이론적 근거 불명확

**근본적 의문:**

> LN input이 $\mathcal{N}(0, 1)$이 되어야 할 이유가 없다.

---

## 2. Bypass 이론으로의 리프레이밍

### 2.1 새로운 가설

**핵심 아이디어**: Input을 $(0, 1)$로 normalize → Normalization layer가 identity에 가까워짐 (Bypass)

### 2.2 LayerNorm Bypass 수식

**LN forward:**

$$y = \gamma \cdot \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} + \beta$$

**Input이 $x \sim \mathcal{N}(0, 1)$이면:**

$$\mathbb{E}[x] \approx 0, \quad \text{Var}[x] \approx 1$$

따라서:

$$y \approx \gamma \cdot \frac{x - 0}{\sqrt{1 + \epsilon}} + \beta \approx \gamma \cdot x + \beta$$

**결론**: Normalization step이 identity → Affine만 적용 (Bypass!)

### 2.3 LN Bypass의 이론적 타당성

**Domain-agnostic Output:**

Clear domain:
$$x_{\text{clear}} \sim \mathcal{N}(0, 1) \Rightarrow y_{\text{clear}} = \gamma x_{\text{clear}} + \beta$$

Foggy domain (transformed):
$$x_{\text{foggy}} \sim \mathcal{N}(0, 1) \Rightarrow y_{\text{foggy}} = \gamma x_{\text{foggy}} + \beta$$

**Distribution 일치:**

$$p(y_{\text{clear}}) = p(y_{\text{foggy}})$$

→ Drift-free adaptation!

### 2.4 BN에 Bypass 적용 시도

**기존 BN (Running stats 사용):**

$$y = \gamma \cdot \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}} + \beta$$

**Bypass BN (Batch stats 사용):**

$$y = \gamma \cdot \frac{x - \mathbb{E}_{\text{batch}}[x]}{\sqrt{\text{Var}_{\text{batch}}[x] + \epsilon}} + \beta$$

**Input $x \sim \mathcal{N}(0, 1)$이면:**

$$y \approx \gamma \cdot x + \beta$$

→ LN과 동일한 bypass 메커니즘!

**통합 이론 기대:**

> BN + LN 모두 bypass → 통합된 이론적 프레임워크

---

## 3. BN Per-channel 수정의 예상치 못한 성공

### 3.1 수정 내용

**변경 사항:**

1. **Per-channel alignment**: Scalar averaging 제거
2. **First BN skip**: Stem BN 보존
3. **BN input $(0, 1)$ target**: Channel-wise

### 3.2 Per-channel Alignment 수식

**기존 (V3):**

$$\mathcal{L}_{\text{BN}} = \left\|\frac{1}{C}\sum_{c=1}^C \mu_c - \bar{\mu}_{\text{source}}\right\|^2$$

**수정 (V4):**

$$\mathcal{L}_{\text{BN}} = \sum_{c=1}^C \left(\|\mu_c - 0\|^2 + \|\sigma^2_c - 1\|^2\right)$$

각 채널 독립적으로 $(0, 1)$ align.

### 3.3 실험 결과

**Parameter Stability:**

| Domain | clip_low (R1) | clip_low (R2) | Drift |
|--------|---------------|---------------|-------|
| Cloudy | 2.01 | 2.03 | ✓ Stable |
| Foggy | 2.21 | 2.21 | ✓ Stable |
| Clear | 2.00 | 2.00 | ✓ Stable |

**Performance:**

- Cloudy mAP: 0.48 (excellent)
- Average mAP: 0.42 (state-of-the-art)

**결론**: **이론적 근거 없이 성공!**

### 3.4 이론적 혼란

**문제:**

1. BN은 여전히 **running stats 사용** (bypass 아님)
2. Input $(0, 1)$이어도 BN이 정상 작동
3. 왜 성공하는지 불명확

**가능한 설명 (추측):**

```
Pixel per-channel (0,1) → Conv → Features
                                   ↓
                               BN (running stats)
```

- Pixel-level normalization이 feature distribution에 영향?
- Conv weights가 $(0,1)$ input에 최적화되어 있음?
- Per-channel prior가 regularization 역할?

**명확한 이론 없음.**

---

## 4. BN Bypass 실험의 실패

### 4.1 실험 설정

**가설**: Bypass 이론이 맞다면 BN도 batch stats 사용 시 성공해야 함

**구현:**

```python
# BN forward override
batch_mean = x.mean(dim=(0,2,3))  # Per-channel
batch_var = x.var(dim=(0,2,3))

normalized = (x - batch_mean) / sqrt(batch_var)
output = normalized * gamma + beta  # Bypass!
```

**Input alignment**: Per-channel $(0, 1)$ (동일)

### 4.2 실험 결과

**Performance:**

- Cloudy mAP: **0.308** (vs 0.48 without bypass)
- Significant degradation!

**비교:**

| | Original BN | Bypass BN |
|---|---|---|
| **Forward** | Running stats | Batch stats |
| **Cloudy mAP** | 0.48 | 0.308 |
| **Result** | ✓ Success | ✗ Failure |

### 4.3 Optimizer 변수 제거

**추가 실험**: SGD → Adam (per-channel optimization 개선)

**결과**: 여전히 실패 (mAP 증가 없음)

**결론**: Optimizer가 아닌 근본적 문제

### 4.4 이론적 모순

**BN Bypass 실패 이유 (가설):**

**Learned $\gamma, \beta$는 running stats 기준:**

Training 시:
$$y_{\text{train}} = \gamma \cdot \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}}}} + \beta$$

Bypass 시:
$$y_{\text{bypass}} = \gamma \cdot \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}}}} + \beta$$

**Mismatch:**

$$\mu_{\text{running}} \neq \mu_{\text{batch}}, \quad \sigma^2_{\text{running}} \neq \sigma^2_{\text{batch}}$$

→ $\gamma, \beta$가 다른 normalized distribution에 적용됨 → 성능 저하

---

## 5. LN Feature-wise Alignment 시도

### 5.1 새로운 접근

**가설**: LN도 per-feature alignment하면 BN처럼 성공?

**"Alignment Structure Matching Principle":**

> Normalization layer의 구조와 alignment 구조가 일치해야 stability 달성

### 5.2 구현

**Stats 측정:**

```python
# Input: (B, N, D), normalized_shape=(D,)
batch_dims = (0, 1)  # Batch + sequence
current_mean = x.mean(dim=batch_dims)  # (D,)
current_var = x.var(dim=batch_dims)    # (D,)
```

**Loss:**

$$\mathcal{L}_{\text{LN}} = \sum_{d=1}^D \left(\|\mu_d - 0\|^2 + \|\sigma^2_d - 1\|^2\right)$$

### 5.3 실험 결과

**Parameter Drift 여전히 존재:**

| Round | clip_low (Cloudy) | gamma (Cloudy) |
|-------|-------------------|----------------|
| R1 | 2.53 | 1.39 |
| R2 | 4.87 | 0.50 |
| R3 | 6.31 | 0.50 |

**문제**: 계속 증가 (BN의 안정성과 대조)

**가능한 원인:**

1. **Patch Embedding 간극**:
   ```
   Pixel transform → [Patch Embed] → LN
                        ↑ Indirect connection
   ```

2. **Architecture 차이**:
   - CNN: Pixel → Conv → BN (direct)
   - ViT: Pixel → Patch Embed → LN (indirect)

---

## 6. 결론 및 미해결 과제

### 6.1 실험적 발견 요약

| | BN (Original) | BN (Bypass) | LN (Feature-wise) |
|---|---|---|---|
| **Alignment** | Per-channel (0,1) | Per-channel (0,1) | Per-feature (0,1) |
| **Normalization** | Running stats | Batch stats | Batch stats |
| **mAP** | 0.48 ✓ | 0.308 ✗ | ~0.39 (drift) |
| **Stability** | ✓ | ✗ | ✗ |

### 6.2 이론적 불일치

**Bypass 이론의 예측:**

1. LN bypass: $(0,1)$ input → LN ≈ identity ✓ (수식 성립)
2. BN bypass: $(0,1)$ input → BN ≈ identity ✓ (수식 성립)

**실험 결과:**

1. LN: Bypass이지만 **drift 발생** ✗
2. BN: Bypass 아닌데 **성공** ✗

**모순:**

> 이론과 실험이 일치하지 않음

### 6.3 미해결 질문

**Q1. BN이 running stats로 왜 성공하는가?**

- Input $(0,1)$ per-channel
- BN은 running stats 사용 (mismatch)
- 하지만 성능 최고 + stability

**Q2. LN feature-wise가 왜 실패하는가?**

- BN과 구조적 대칭 (per-channel ↔ per-feature)
- 하지만 drift 발생

**Q3. Pixel-level transform과 Embedding-level norm의 관계?**

- ViT: Pixel → Patch Embed → LN
- Indirect connection이 문제?

**Q4. 통합 이론이 가능한가?**

- BN: Empirical success, theoretical unclear
- LN: Theoretical clear, empirical failure
- 어떻게 통합?

### 6.4 향후 연구 방향

**Option 1: Embedding-level Transform**

```
Pixel → Patch Embed → [Transform] → LN
```

Transform을 embedding space로 이동

**Option 2: Architecture-specific Theory**

BN과 LN을 별개 이론으로 설명:
- BN:  Per-channel regularization prior
- LN: Different mechanism (아직 불명확)

**Option 3: Source Stats Alignment 재고**

Bypass 포기, V3 original로 회귀:
- BN: Running stats target
- LN: ? (여전히 불명확)

---

## 7. Mathematical Appendix

### 7.1 Batch Normalization

**Forward (Inference):**

$$\begin{align}
\hat{x} &= \frac{x - \mathbb{E}[\mathcal{X}_{\text{train}}]}{\sqrt{\text{Var}[\mathcal{X}_{\text{train}}] + \epsilon}} \\
y &= \gamma \hat{x} + \beta
\end{align}$$

**Channel-wise (4D tensor):**

For input $x \in \mathbb{R}^{B \times C \times H \times W}$:

$$\mu_c = \frac{1}{BHW}\sum_{b,h,w} x_{b,c,h,w}, \quad c \in \{1, \ldots, C\}$$

### 7.2 Layer Normalization

**Forward:**

$$\begin{align}
\mu &= \mathbb{E}_{\text{features}}[x] \\
\sigma^2 &= \text{Var}_{\text{features}}[x] \\
y &= \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\end{align}$$

**For ViT (3D tensor):**

Input $x \in \mathbb{R}^{B \times N \times D}$, normalized_shape=$(D,)$:

$$\mu_b = \frac{1}{D}\sum_{d=1}^D x_{b,n,d}, \quad \forall b, n$$

### 7.3 Alignment Loss (General Form)

**Per-element:**

$$\mathcal{L}_{\text{align}} = \sum_{i} \left(\|\mu_i - \mu^{\text{target}}_i\|^2 + \|\sigma^2_i - (\sigma^2)^{\text{target}}_i\|^2\right)$$

where $i$ indexes over channels (BN) or features (LN).

---

## References

**Key Insights:**

1. Ioffe & Szegedy (2015): Batch Normalization
2. Ba et al. (2016): Layer Normalization
3. V3 Implementation: Source statistics alignment
4. V4 Implementation: Per-channel/feature $(0,1)$ alignment

**Experimental Setup:**

- Model: Swin-T, YOLO11
- Dataset: ACDC (7 domains)
- Metrics: mAP, parameter drift

---

## Acknowledgments

본 연구는 Test-Time Adaptation 문제에 대한 새로운 접근을 시도하였으나 통합 이론에는 도달하지 못했습니다. BN에서의 예상치 못한 성공과 LN에서의 지속적인 어려움은 향후 연구의 중요한 방향을 제시합니다.
