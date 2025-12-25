# Функции потерь (Losses)

## Обзор системы потерь

TIGAS использует комбинированную функцию потерь, объединяющую три компонента:

```
Total Loss = w₁ × L_regression + w₂ × L_classification + w₃ × L_ranking
```

**Расположение:** [tigas/training/losses.py](../tigas/training/losses.py)

---

## CombinedLoss

Основной класс комбинированной потери.

```python
from tigas.training.losses import CombinedLoss

loss_fn = CombinedLoss(
    regression_weight=1.0,
    classification_weight=0.3,
    ranking_weight=0.2,
    regression_type='mse'  # или 'smoothl1'
)
```

### Параметры

| Параметр | Default | Описание |
|----------|---------|----------|
| `regression_weight` | 1.0 | Вес регрессионной потери |
| `classification_weight` | 0.3 | Вес классификационной потери |
| `ranking_weight` | 0.2 | Вес ranking потери |
| `regression_type` | 'mse' | Тип регрессии: `'mse'` или `'smoothl1'` |

---

## Компоненты потерь

### 1. Regression Loss

Минимизирует разницу между предсказанным score и ground truth.

```
L_regression = MSE(predicted_score, target_label)
```

#### MSE Loss (default)

```python
L = (1/N) × Σ (score_i - label_i)²
```

**Характеристики:**
- Чувствителен к выбросам
- Более строгий к большим ошибкам
- Подходит для задач с точными метками

#### SmoothL1 Loss

```python
L = (1/N) × Σ smooth_l1(score_i - label_i)

где smooth_l1(x) = 0.5 × x²           если |x| < 1
                 = |x| - 0.5          иначе
```

**Характеристики:**
- Робастен к выбросам
- Линейный рост для больших ошибок
- Рекомендуется при шумных данных

#### Использование

```python
# MSE (default)
loss_fn = CombinedLoss(regression_type='mse')

# SmoothL1
loss_fn = CombinedLoss(regression_type='smoothl1')
```

---

### 2. Classification Loss

Cross-Entropy потеря для бинарной классификации (real vs fake).

```python
L_classification = CrossEntropy(logits, binary_labels)
```

#### Формула

```
L = -(1/N) × Σ [y_i × log(p_real) + (1-y_i) × log(p_fake)]

где p = softmax(logits)
    y_i = 1 для real, 0 для fake
```

#### Логика преобразования меток

```python
# labels: float [0.0, 1.0] → binary indices [0, 1]
binary_labels = (labels > 0.5).long()
# label > 0.5 → class 1 (real)
# label ≤ 0.5 → class 0 (fake)
```

---

### 3. Ranking Loss

Margin Ranking Loss обеспечивает правильное упорядочивание score'ов.

```python
L_ranking = MarginRankingLoss(score_real, score_fake, y=+1, margin=0.5)
```

#### Формула

```
L = max(0, margin - (score_real - score_fake))

Цель: score_real > score_fake + margin
```

#### Логика формирования пар

```python
# Разделение батча на real и fake
real_mask = labels > 0.5
fake_mask = labels <= 0.5

real_scores = scores[real_mask]
fake_scores = scores[fake_mask]

# Создание пар (все комбинации real × fake)
for score_r in real_scores:
    for score_f in fake_scores:
        loss += max(0, margin - (score_r - score_f))
```

**Важно:** Ranking loss вычисляется только если в батче есть оба класса.

---

## Веса компонентов

### Рекомендуемые значения

| Сценарий | Regression | Classification | Ranking |
|----------|------------|----------------|---------|
| Default | 1.0 | 0.3 | 0.2 |
| Акцент на score | 1.5 | 0.2 | 0.1 |
| Акцент на классификацию | 0.5 | 1.0 | 0.2 |
| Максимальное разделение | 1.0 | 0.3 | 0.5 |

### Обоснование default весов

1. **Regression (1.0)** - основная задача, наивысший приоритет
2. **Classification (0.3)** - вспомогательная задача, помогает изучить дискриминативные признаки
3. **Ranking (0.2)** - регуляризатор, обеспечивает правильный порядок

---

## Численная стабильность

### Защита от NaN/Inf

```python
def forward(self, scores, logits, labels):
    # Проверка входов
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        raise RuntimeError("NaN/Inf detected in scores!")
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("NaN/Inf detected in logits!")
    
    # ... вычисление потерь ...
    
    # Проверка выхода
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        raise RuntimeError(f"NaN/Inf detected in total loss!")
    
    return total_loss, loss_dict
```

### Fail-fast политика

При обнаружении NaN/Inf обучение немедленно останавливается с RuntimeError. Это предотвращает:
- Бесполезное продолжение обучения
- Порчу чекпоинтов
- Трату вычислительных ресурсов

---

## Использование в обучении

### Стандартный forward

```python
outputs = model(images)
# outputs['score']: [B, 1] - предсказанный score
# outputs['logits']: [B, 2] - logits для классификации

total_loss, loss_dict = loss_fn(
    scores=outputs['score'].squeeze(-1),  # [B]
    logits=outputs['logits'],              # [B, 2]
    labels=labels                          # [B]
)

# loss_dict содержит:
# - 'regression': значение регрессионной потери
# - 'classification': значение классификационной потери
# - 'ranking': значение ranking потери (или 0 если невозможно)
# - 'total': итоговая потеря
```

### С TIGASTrainer

```python
from tigas.training import TIGASTrainer

trainer = TIGASTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    regression_weight=1.0,
    classification_weight=0.3,
    ranking_weight=0.2
)

trainer.train(num_epochs=50)
```

---

## Кастомные функции потерь

### Наследование от CombinedLoss

```python
from tigas.training.losses import CombinedLoss
import torch.nn as nn

class CustomLoss(CombinedLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Дополнительные компоненты
        self.consistency_loss = nn.MSELoss()
    
    def forward(self, scores, logits, labels, **extra):
        # Базовые потери
        total, loss_dict = super().forward(scores, logits, labels)
        
        # Кастомный компонент
        if 'features' in extra:
            consistency = self.consistency_loss(
                extra['features']['scale1'],
                extra['features']['scale2']
            )
            total = total + 0.1 * consistency
            loss_dict['consistency'] = consistency.item()
        
        return total, loss_dict
```

### Добавление Focal Loss

```python
class FocalCombinedLoss(CombinedLoss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
    
    def _classification_loss(self, logits, labels):
        binary_labels = (labels > 0.5).long()
        ce = F.cross_entropy(logits, binary_labels, reduction='none')
        
        # Focal weighting
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        
        return focal.mean()
```

---

## Мониторинг потерь

### TensorBoard

```python
# Автоматически логируется TIGASTrainer:
# - Loss/train_total
# - Loss/train_regression
# - Loss/train_classification
# - Loss/train_ranking
# - Loss/val_total
# - Loss/val_regression
# - Loss/val_classification
# - Loss/val_ranking
```

### Анализ баланса потерь

```python
# В training loop
_, loss_dict = loss_fn(scores, logits, labels)

print(f"Regression: {loss_dict['regression']:.4f}")
print(f"Classification: {loss_dict['classification']:.4f}")
print(f"Ranking: {loss_dict['ranking']:.4f}")

# Соотношение (должно быть сбалансировано)
total_unweighted = (
    loss_dict['regression'] +
    loss_dict['classification'] +
    loss_dict['ranking']
)
print(f"Regression %: {loss_dict['regression']/total_unweighted*100:.1f}%")
```

### Диагностика проблем

| Симптом | Причина | Решение |
|---------|---------|---------|
| Regression >> Classification | Несбалансированные данные | Увеличить `classification_weight` |
| Ranking = 0 | Нет обоих классов в батче | Увеличить batch_size |
| NaN в regression | Слишком большой LR | Уменьшить learning rate |
| Classification не падает | Плохие признаки | Проверить perceptual extractor |

---

## Gradients и дифференцируемость

Все компоненты полностью дифференцируемы:

```python
# Граф вычислений
total_loss.backward()

# Градиенты распространяются через:
# - Regression: score → regression_head → fused_features → ...
# - Classification: logits → classifier → fused_features → ...
# - Ranking: score → regression_head → fused_features → ...
```

### Использование как loss для генеративных моделей

```python
from tigas import TIGAS

tigas = TIGAS(auto_download=True, device='cuda')

# В training loop генератора
generated_images = generator(noise)

# TIGAS как дифференцируемая метрика качества
tigas_score = tigas(generated_images)
generator_loss = 1.0 - tigas_score.mean()  # Максимизируем "реальность"

generator_loss.backward()  # Градиенты идут через TIGAS в генератор
```
