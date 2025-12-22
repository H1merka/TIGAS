# Компоненты архитектуры

## Feature Extractors

### MultiScaleFeatureExtractor

Основной экстрактор перцептивных признаков.

**Расположение:** [tigas/models/feature_extractors.py](../tigas/models/feature_extractors.py)

```python
class MultiScaleFeatureExtractor(nn.Module):
    """
    Многомасштабный экстрактор признаков с:
    - Gated residual blocks
    - CBAM attention на каждом масштабе
    - 4 выходных масштаба
    """
```

#### Архитектура

```
Input [B, 3, H, W]
       │
       ▼
┌─────────────────┐
│  Stem Block     │  Conv2d(3→32, k=3) + BN + ReLU
│  [B, 32, H, W]  │
└────────┬────────┘
         │
   ┌─────┴─────┐
   ▼           │
┌──────────────┴──────────────────────────────────────┐
│  Stage 1 (channels=64)                              │
│  ├─ ConvBlock: Conv+BN+ReLU → Conv+BN+ReLU         │
│  │  Downsampling: stride=2 → [B, 64, H/2, W/2]     │
│  ├─ GatedResidualBlock × 1                         │
│  └─ CBAM Attention                                 │
└──────────────┬──────────────────────────────────────┘
               │ → features[0]: [B, 64, H/2, W/2]
   ┌───────────┘
   ▼
┌───────────────────────────────────────────────────┐
│  Stage 2 (channels=128)                            │
│  Similar structure → [B, 128, H/4, W/4]            │
└──────────────┬────────────────────────────────────┘
               │ → features[1]: [B, 128, H/4, W/4]
   ┌───────────┘
   ▼
┌───────────────────────────────────────────────────┐
│  Stage 3 (channels=256)                            │
│  Similar structure → [B, 256, H/8, W/8]            │
└──────────────┬────────────────────────────────────┘
               │ → features[2]: [B, 256, H/8, W/8]
   ┌───────────┘
   ▼
┌───────────────────────────────────────────────────┐
│  Stage 4 (channels=512)                            │
│  Similar structure → [B, 512, H/16, W/16]          │
└──────────────┬────────────────────────────────────┘
               │ → features[3]: [B, 512, H/16, W/16]
               ▼
         Output: List[Tensor]
```

#### Параметры

| Параметр | Default | Описание |
|----------|---------|----------|
| `in_channels` | 3 | RGB input |
| `base_channels` | 32 | Начальные каналы |
| `num_stages` | 4 | Количество стадий |

#### Использование

```python
from tigas.models.feature_extractors import MultiScaleFeatureExtractor

extractor = MultiScaleFeatureExtractor(
    in_channels=3,
    base_channels=32
)

x = torch.randn(4, 3, 256, 256)
features = extractor(x)

# features[0]: [4, 64, 128, 128]   - 1/2 scale
# features[1]: [4, 128, 64, 64]    - 1/4 scale
# features[2]: [4, 256, 32, 32]    - 1/8 scale
# features[3]: [4, 512, 16, 16]    - 1/16 scale
```

---

### SpectralAnalyzer

Анализатор частотных характеристик изображения.

**Расположение:** [tigas/models/feature_extractors.py](../tigas/models/feature_extractors.py)

```python
class SpectralAnalyzer(nn.Module):
    """
    Анализатор спектральных признаков:
    - FFT анализ магнитуды
    - Частотные band extraction
    - Энергетическое распределение
    """
```

#### Архитектура

```
Input [B, 3, H, W]
       │
       ▼ (float32 для FFT)
┌─────────────────────────────────────────┐
│  torch.fft.rfft2(x)                     │
│  → Complex spectrum [B, 3, H, H//2+1]   │
│  → Magnitude: torch.abs(freq)           │
└────────────────┬────────────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Low-freq     │    │ High-freq    │
│ Band         │    │ Band         │
│ (DC - 0.2)   │    │ (0.2 - 0.5)  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 ▼
┌────────────────────────────────────────┐
│ Energy Distribution Features           │
│ + Band Statistics                       │
│ + Spectral Centroid                    │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│ MLP: Linear layers + normalization     │
│ → [B, feature_dim]                     │
└────────────────────────────────────────┘
```

#### Параметры

| Параметр | Default | Описание |
|----------|---------|----------|
| `feature_dim` | 256 | Выходная размерность |

#### Использование

```python
from tigas.models.feature_extractors import SpectralAnalyzer

analyzer = SpectralAnalyzer(feature_dim=256)

x = torch.randn(4, 3, 256, 256)
spectral_features, aux_data = analyzer(x)

# spectral_features: [4, 256]
# aux_data: {'freq_features': Tensor}
```

---

### StatisticalEstimator

Статистический анализатор со сравнением с прототипами.

**Расположение:** [tigas/models/feature_extractors.py](../tigas/models/feature_extractors.py)

```python
class StatisticalEstimator(nn.Module):
    """
    Статистический оценщик:
    - Моменты распределения (mean, var, skewness, kurtosis)
    - Learnable prototypes для real/fake классов
    - Сравнение с прототипами
    """
```

#### Архитектура

```
Input [B, 3, H, W]
       │
       ▼
┌─────────────────────────────────────────┐
│  Spatial Statistics Extraction          │
│  ├─ Mean per channel                    │
│  ├─ Variance                           │
│  ├─ Skewness (3rd moment)              │
│  └─ Kurtosis (4th moment)              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Prototype Comparison                   │
│  ├─ Real prototype (learnable)          │
│  ├─ Fake prototype (learnable)          │
│  └─ Distance computation                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Feature MLP                            │
│  → [B, feature_dim]                     │
└─────────────────────────────────────────┘
```

#### Параметры

| Параметр | Default | Описание |
|----------|---------|----------|
| `feature_dim` | 256 | Выходная размерность |
| `num_prototypes` | 2 | real + fake |

---

## Attention Mechanisms

### CrossModalAttention

Внимание между модальностями (spectral → perceptual, statistical → perceptual).

**Расположение:** [tigas/models/attention.py](../tigas/models/attention.py)

```python
class CrossModalAttention(nn.Module):
    """
    Cross-modal attention:
    - Query из одной модальности
    - Key/Value из другой модальности
    - Multi-head self-attention механизм
    """
```

#### Архитектура

```
Query: [B, 1, D]     Key/Value: [B, 1, D]
      │                    │
      ▼                    ▼
┌───────────────┐    ┌───────────────┐
│ W_q Linear    │    │ W_kv Linear   │
└───────┬───────┘    └───────┬───────┘
        │                    │
        │            ┌───────┴───────┐
        │            ▼               ▼
        │       ┌────────┐     ┌────────┐
        │       │   K    │     │   V    │
        │       └────┬───┘     └────┬───┘
        │            │              │
        ▼            ▼              │
    ┌────────────────────┐         │
    │  Q @ K^T / √d_k    │         │
    │  + Clamp [-1e4,1e4]│         │
    │  + Softmax         │         │
    └─────────┬──────────┘         │
              │                    │
              ▼                    ▼
         ┌──────────────────────────┐
         │   Attention @ V          │
         │   + Output Linear        │
         └──────────────────────────┘
                    │
                    ▼
              [B, 1, D]
```

#### Численная стабильность

```python
# Защита от NaN в attention
attn = torch.clamp(attn, min=-1e4, max=1e4)

# Fallback при NaN
if torch.isnan(attn).any():
    attn = torch.ones_like(attn) / attn.shape[-1]
```

#### Использование

```python
from tigas.models.attention import CrossModalAttention

attention = CrossModalAttention(
    query_dim=256,
    key_dim=256,
    num_heads=4,
    dropout=0.1
)

query = torch.randn(4, 1, 256)   # Spectral features
key_value = torch.randn(4, 1, 256)  # Perceptual features

output = attention(query=query, key_value=key_value)
# output: [4, 1, 256]
```

---

### SelfAttention

Self-attention для финальных объединённых признаков.

**Расположение:** [tigas/models/attention.py](../tigas/models/attention.py)

```python
class SelfAttention(nn.Module):
    """
    Standard multi-head self-attention.
    """
```

#### Использование

```python
from tigas.models.attention import SelfAttention

self_attn = SelfAttention(
    dim=256,
    num_heads=4,
    dropout=0.1
)

x = torch.randn(4, 1, 256)
output = self_attn(x)  # [4, 1, 256]
```

---

## Fusion Modules

### AdaptiveFeatureFusion

Адаптивное объединение признаков из нескольких источников.

**Расположение:** [tigas/models/layers.py](../tigas/models/layers.py)

```python
class AdaptiveFeatureFusion(nn.Module):
    """
    Объединение признаков с learnable весами:
    - Каждый stream имеет свой вес
    - Softmax нормализация весов
    - Взвешенная сумма
    """
```

#### Архитектура

```
Features: [feat_1, feat_2, feat_3]
    │         │         │
    │         │         │
    ▼         ▼         ▼
┌────────────────────────────┐
│   Learned Weights          │
│   α₁, α₂, α₃ (trainable)   │
│                            │
│   w_i = softmax(α_i)       │
└────────────────────────────┘
              │
              ▼
┌────────────────────────────┐
│   Weighted Sum             │
│   out = Σ w_i * feat_i     │
└────────────────────────────┘
              │
              ▼
        [B, feature_dim]
```

#### Использование

```python
from tigas.models.layers import AdaptiveFeatureFusion

fusion = AdaptiveFeatureFusion(
    feature_dim=256,
    num_features=3
)

perceptual = torch.randn(4, 256)
spectral = torch.randn(4, 256)
statistical = torch.randn(4, 256)

fused = fusion([perceptual, spectral, statistical])
# fused: [4, 256]

# Посмотреть веса
weights = torch.softmax(fusion.weights, dim=0)
print(weights)  # tensor([0.33, 0.33, 0.33])
```

---

## Building Blocks

### GatedResidualBlock

Residual block с gating механизмом.

**Расположение:** [tigas/models/layers.py](../tigas/models/layers.py)

```python
class GatedResidualBlock(nn.Module):
    """
    Residual block с gate для контроля информационного потока.
    """
```

#### Архитектура

```
Input x
   │
   ├───────────────────────────┐
   │                           │
   ▼                           │
┌─────────────────┐            │
│ Conv + BN + ReLU│            │
└────────┬────────┘            │
         │                     │
         ▼                     │
┌─────────────────┐            │
│ Conv + BN       │            │
└────────┬────────┘            │
         │                     │
         ▼                     │
┌─────────────────┐            │
│ Gate: Sigmoid   │            │
│ g = σ(W*x)      │            │
└────────┬────────┘            │
         │                     │
         ▼                     ▼
┌─────────────────────────────────┐
│ output = g * residual + (1-g)*x │
└─────────────────────────────────┘
```

---

### CBAM (Convolutional Block Attention Module)

Channel + Spatial attention.

**Расположение:** [tigas/models/layers.py](../tigas/models/layers.py)

```python
class CBAM(nn.Module):
    """
    CBAM: Channel Attention + Spatial Attention
    """
```

#### Архитектура

```
Input x [B, C, H, W]
       │
       ▼
┌──────────────────────────────┐
│    Channel Attention         │
│    ┌───────────────────────┐ │
│    │ AvgPool + MaxPool     │ │
│    │ → MLP → σ → weights   │ │
│    └───────────────────────┘ │
│    x = x * weights           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Spatial Attention         │
│    ┌───────────────────────┐ │
│    │ AvgPool + MaxPool     │ │
│    │ (channel-wise)        │ │
│    │ → Conv7×7 → σ         │ │
│    └───────────────────────┘ │
│    x = x * spatial_weights   │
└──────────────────────────────┘
               │
               ▼
        Output [B, C, H, W]
```

---

## Output Heads

### RegressionHead

MLP для предсказания score ∈ [0, 1].

```python
class RegressionHead(nn.Module):
    def __init__(self, in_features, hidden_features=128, dropout=0.1):
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.LayerNorm(hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, 1),
            nn.Sigmoid()
        )
```

### BinaryClassifier

MLP для бинарной классификации [real, fake].

```python
class BinaryClassifier(nn.Module):
    def __init__(self, in_features, hidden_features=128, dropout=0.1):
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 2)  # No softmax - используется CrossEntropyLoss
        )
```

---

## Модульная интеграция

### Полная сборка компонентов

```python
# В TIGASModel.__init__

# Feature Extractors
self.perceptual_extractor = MultiScaleFeatureExtractor(in_channels=3)
self.spectral_analyzer = SpectralAnalyzer(feature_dim=256)  # Full mode
self.statistical_estimator = StatisticalEstimator(feature_dim=256)  # Full mode

# Aggregators
self.perceptual_aggregator = nn.Sequential(
    nn.Linear(64 + 128 + 256 + 512, 256),
    nn.LayerNorm(256),
    nn.ReLU()
)

# Attention (Full mode)
self.spectral_to_perceptual_attn = CrossModalAttention(256, 256, num_heads=4)
self.stat_to_perceptual_attn = CrossModalAttention(256, 256, num_heads=4)
self.self_attention = SelfAttention(256, num_heads=4)

# Fusion
self.feature_fusion = AdaptiveFeatureFusion(feature_dim=256, num_features=3)

# Output heads
self.regression_head = RegressionHead(256)
self.binary_classifier = BinaryClassifier(256)
```

---

## Пример полного forward pass

```python
def forward(self, x, return_features=False):
    # 1. Нормализация входа
    x = self._normalize_input(x)
    
    # 2. Perceptual features
    perceptual_features = self.perceptual_extractor(x)
    perceptual_concat = torch.cat([
        F.adaptive_avg_pool2d(f, 1).flatten(1) 
        for f in perceptual_features
    ], dim=1)
    perceptual_feat = self.perceptual_aggregator(perceptual_concat)
    
    if self.fast_mode:
        # Fast mode: simplified path
        aux_feat = self.aux_branch(perceptual_features[-1])
        fused = self.fast_fusion(torch.cat([perceptual_feat, aux_feat], dim=1))
    else:
        # Full mode: all branches
        spectral_feat, _ = self.spectral_analyzer(x)
        statistical_feat, _ = self.statistical_estimator(x)
        
        # Cross-modal attention
        spectral_attended = self.spectral_to_perceptual_attn(
            spectral_feat.unsqueeze(1), 
            perceptual_feat.unsqueeze(1)
        ).squeeze(1)
        
        stat_attended = self.stat_to_perceptual_attn(
            statistical_feat.unsqueeze(1),
            perceptual_feat.unsqueeze(1)
        ).squeeze(1)
        
        # Fusion
        fused = self.feature_fusion([perceptual_feat, spectral_attended, stat_attended])
        fused = self.self_attention(fused.unsqueeze(1)).squeeze(1)
    
    # 3. Output
    score = self.regression_head(fused)
    logits = self.binary_classifier(fused)
    
    return {'score': score, 'logits': logits}
```
