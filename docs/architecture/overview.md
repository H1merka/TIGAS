# Обзор архитектуры TIGAS

## Высокоуровневая схема

```
┌─────────────────────────────────────────────────────────────────┐
│                         TIGASModel                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Input Image [B, 3, H, W]                 ││
│  │                           │                                  ││
│  │                    _normalize_input()                        ││
│  │                    [0,1] → [-1,1]                           ││
│  │                           │                                  ││
│  │          ┌────────────────┼────────────────┐                 ││
│  │          ▼                ▼                ▼                 ││
│  │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        ││
│  │   │  Perceptual  │ │   Spectral   │ │ Statistical  │        ││
│  │   │  Extractor   │ │   Analyzer   │ │  Estimator   │        ││
│  │   │  (CNN)       │ │  (FFT-based) │ │  (Moments)   │        ││
│  │   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘        ││
│  │          │                │                │                 ││
│  │          │      ┌─────────┴─────────┐      │                 ││
│  │          │      ▼                   ▼      │                 ││
│  │          │  Cross-Modal        Cross-Modal │                 ││
│  │          │  Attention          Attention   │                 ││
│  │          │      │                   │      │                 ││
│  │          └──────┼───────────────────┼──────┘                 ││
│  │                 ▼                   ▼                        ││
│  │          ┌─────────────────────────────────┐                 ││
│  │          │    Adaptive Feature Fusion      │                 ││
│  │          │    (Learned Weights)            │                 ││
│  │          └────────────────┬────────────────┘                 ││
│  │                           │                                  ││
│  │                    Self-Attention                           ││
│  │                           │                                  ││
│  │          ┌────────────────┴────────────────┐                 ││
│  │          ▼                                 ▼                 ││
│  │   ┌──────────────┐                  ┌──────────────┐        ││
│  │   │  Regression  │                  │   Binary     │        ││
│  │   │    Head      │                  │ Classifier   │        ││
│  │   │  [B, 1]      │                  │  [B, 2]      │        ││
│  │   └──────────────┘                  └──────────────┘        ││
│  │          │                                 │                 ││
│  │          ▼                                 ▼                 ││
│  │       score                            logits                ││
│  │    [0.0 - 1.0]                     [real, fake]             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Два режима работы

### Fast Mode (default)

Оптимизированная архитектура для обучения:

```
Input → Perceptual Extractor → Aggregator
                                    ↓
                            ┌───────┴───────┐
                            ▼               ▼
                     Aux Branch        Perceptual
                     (lightweight)      Features
                            │               │
                            └───────┬───────┘
                                    ▼
                              Fast Fusion
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                   Regression              Classifier
                   Head (score)            Head (logits)
```

**Характеристики:**
- Только perceptual ветвь + lightweight aux branch
- ~2.5M параметров
- Быстрее в ~2-3x
- Подходит для обучения

### Full Mode

Полная архитектура со всеми ветвями:

```
Input ──┬──→ Perceptual Extractor ──→ Perceptual Features
        │
        ├──→ Spectral Analyzer ──→ Spectral Features
        │                              │
        │                              ▼
        │                     Cross-Modal Attention
        │                     (Spectral → Perceptual)
        │
        └──→ Statistical Estimator ──→ Statistical Features
                                           │
                                           ▼
                                  Cross-Modal Attention
                                  (Statistical → Perceptual)
                                           │
                                           ▼
                                  Adaptive Feature Fusion
                                  (3 streams, learned weights)
                                           │
                                           ▼
                                     Self-Attention
                                           │
                               ┌───────────┴───────────┐
                               ▼                       ▼
                          Regression              Classifier
```

**Характеристики:**
- Все три ветви + cross-modal attention
- ~5M параметров
- Более точный
- Подходит для inference

---

## Поток данных

### 1. Входная нормализация

```python
def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
    # [0, 1] → [-1, 1]
    if x.min() >= 0 and x.max() <= 1:
        x = x * 2 - 1
    # Clamp для гарантии
    x = torch.clamp(x, -1.0, 1.0)
    return x
```

**Важно:** Модель ожидает вход в диапазоне `[-1, 1]`. Нормализация выполняется автоматически.

### 2. Извлечение признаков

#### Perceptual Branch
```python
perceptual_features = self.perceptual_extractor(x)
# Returns: List[Tensor] с 4 масштабами
# [scale1/2, scale1/4, scale1/8, scale1/16]

# Агрегация
perceptual_concat = torch.cat([
    F.adaptive_avg_pool2d(feat, 1).flatten(1)
    for feat in perceptual_features
], dim=1)
perceptual_feat = self.perceptual_aggregator(perceptual_concat)
# Shape: [B, feature_dim]
```

#### Spectral Branch (Full Mode)
```python
spectral_feat, spectral_aux = self.spectral_analyzer(x)
# spectral_feat: [B, feature_dim]
# spectral_aux: {'freq_features': Tensor}
```

#### Statistical Branch (Full Mode)
```python
statistical_feat, stat_aux = self.statistical_estimator(x, update_prototypes=True)
# statistical_feat: [B, feature_dim]
# stat_aux: {'statistics': Tensor, 'prototypes': Tensor}
```

### 3. Cross-Modal Attention (Full Mode)

```python
# Spectral attending to perceptual context
spectral_attended = self.spectral_to_perceptual_attn(
    query=spectral_feat.unsqueeze(1),
    key_value=perceptual_feat.unsqueeze(1)
).squeeze(1)

# Statistical attending to perceptual context
stat_attended = self.stat_to_perceptual_attn(
    query=statistical_feat.unsqueeze(1),
    key_value=perceptual_feat.unsqueeze(1)
).squeeze(1)
```

### 4. Feature Fusion

```python
# Full Mode: Adaptive fusion с learned weights
fused = self.feature_fusion([
    perceptual_feat,
    spectral_attended,
    stat_attended
])
fused = self.self_attention(fused.unsqueeze(1)).squeeze(1)

# Fast Mode: Simple concatenation
fused = self.fast_fusion(torch.cat([perceptual_feat, aux_feat], dim=1))
```

### 5. Output Generation

```python
# Regression head: score [0, 1]
score = self.regression_head(fused)
score = torch.clamp(score, 0.0, 1.0)

# Classification head: logits [real, fake]
logits = self.binary_classifier(fused)
logits = torch.clamp(logits, -10.0, 10.0)

return {'score': score, 'logits': logits}
```

---

## Размерности тензоров

| Точка | Shape | Описание |
|-------|-------|----------|
| Input | `[B, 3, 256, 256]` | RGB изображение |
| После stem | `[B, 32, 256, 256]` | Base channels |
| Stage 1 | `[B, 64, 128, 128]` | 1/2 scale |
| Stage 2 | `[B, 128, 64, 64]` | 1/4 scale |
| Stage 3 | `[B, 256, 32, 32]` | 1/8 scale |
| Stage 4 | `[B, 512, 16, 16]` | 1/16 scale |
| Perceptual aggregated | `[B, 256]` | Feature dim |
| Spectral features | `[B, 256]` | Feature dim |
| Statistical features | `[B, 256]` | Feature dim |
| Fused features | `[B, 256]` | Feature dim |
| Score | `[B, 1]` | Output score |
| Logits | `[B, 2]` | Class logits |

---

## Численная стабильность

### Защита от NaN/Inf

```python
# В Attention
attn = torch.clamp(attn, min=-1e4, max=1e4)
if torch.isnan(attn).any():
    attn = torch.ones_like(attn) / attn.shape[-1]  # Uniform fallback

# В Output
score = torch.clamp(score, 0.0, 1.0)
logits = torch.clamp(logits, -10.0, 10.0)

# В Loss
if torch.isnan(loss) or torch.isinf(loss):
    raise RuntimeError("NaN/Inf detected!")  # Fail-fast
```

### AMP (Mixed Precision)

```python
# Spectral Analyzer: FFT в float32
with torch.amp.autocast('cuda', enabled=False):
    x_float = x.float()
    freq = torch.fft.rfft2(x_float)
    freq_mag = torch.abs(freq)
```

---

## Визуализация архитектуры

### Perceptual Extractor

```
┌────────────────────────────────────────────────┐
│            MultiScaleFeatureExtractor           │
│  ┌──────────────────────────────────────────┐  │
│  │ Stem: Conv2d(3→32) + BN + ReLU           │  │
│  └─────────────────┬────────────────────────┘  │
│                    ▼                            │
│  ┌──────────────────────────────────────────┐  │
│  │ Stage1: [Conv + BN + ReLU] × 2           │  │
│  │         + GatedResidualBlock × 1         │  │
│  │         + CBAM                           │  │
│  │         Output: [B, 64, H/2, W/2]        │──┼──→ Scale 1/2
│  └─────────────────┬────────────────────────┘  │
│                    ▼                            │
│  ┌──────────────────────────────────────────┐  │
│  │ Stage2: Similar, Output: [B,128,H/4,W/4] │──┼──→ Scale 1/4
│  └─────────────────┬────────────────────────┘  │
│                    ▼                            │
│  ┌──────────────────────────────────────────┐  │
│  │ Stage3: Output: [B, 256, H/8, W/8]       │──┼──→ Scale 1/8
│  └─────────────────┬────────────────────────┘  │
│                    ▼                            │
│  ┌──────────────────────────────────────────┐  │
│  │ Stage4: Output: [B, 512, H/16, W/16]     │──┼──→ Scale 1/16
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

### Regression Head

```
┌────────────────────────────────────────┐
│           Regression Head               │
│  ┌──────────────────────────────────┐  │
│  │ Linear(256 → 128) + LN + ReLU    │  │
│  │ Dropout(0.1)                     │  │
│  └────────────────┬─────────────────┘  │
│                   ▼                     │
│  ┌──────────────────────────────────┐  │
│  │ Linear(128 → 64) + LN + ReLU     │  │
│  │ Dropout(0.1)                     │  │
│  └────────────────┬─────────────────┘  │
│                   ▼                     │
│  ┌──────────────────────────────────┐  │
│  │ Linear(64 → 1) + Sigmoid         │  │
│  └──────────────────────────────────┘  │
│                   │                     │
│                   ▼                     │
│            score ∈ [0, 1]               │
└────────────────────────────────────────┘
```
