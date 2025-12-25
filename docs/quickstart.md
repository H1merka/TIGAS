# Быстрый старт

## Инференс (оценка изображений)

### Способ 1: Автозагрузка модели

```python
from tigas import TIGAS

# Модель автоматически загружается из HuggingFace Hub
tigas = TIGAS(auto_download=True)

# Оценка одного изображения
score = tigas('image.jpg')
print(f"Authenticity score: {score.item():.4f}")
# 1.0 = реальное изображение
# 0.0 = сгенерированное изображение
```

### Способ 2: Локальный чекпоинт

```python
from tigas import TIGAS

tigas = TIGAS(checkpoint_path='checkpoints/best_model.pt')
score = tigas('image.jpg')
```

### Способ 3: Высокоуровневая функция

```python
from tigas import compute_tigas_score

# Одна строка для оценки
score = compute_tigas_score('image.jpg', auto_download=True)
print(f"Score: {score:.4f}")
```

---

## Batch-обработка

### Обработка директории

```python
from tigas import TIGAS

tigas = TIGAS(auto_download=True)

# Все изображения в директории
results = tigas.compute_directory(
    'path/to/images/',
    return_paths=True,
    batch_size=32
)

for path, score in results.items():
    status = "REAL" if score > 0.5 else "FAKE"
    print(f"{path}: {score:.4f} ({status})")
```

### Обработка тензоров

```python
import torch
from tigas import TIGAS

tigas = TIGAS(auto_download=True, device='cuda')

# Batch изображений [B, C, H, W]
images = torch.randn(8, 3, 256, 256)
scores = tigas(images)

print(f"Scores shape: {scores.shape}")  # [8, 1]
print(f"Mean score: {scores.mean():.4f}")
```

---

## Использование как Loss-функция

TIGAS полностью дифференцируема и может использоваться для обучения генеративных моделей:

```python
import torch
from tigas import TIGAS

# Инициализация
tigas = TIGAS(checkpoint_path='model.pt', device='cuda')

# В цикле обучения генератора
for noise in dataloader:
    # Генерация изображений
    generated = generator(noise)
    
    # TIGAS как loss: максимизируем "реалистичность"
    authenticity = tigas(generated)
    loss = 1.0 - authenticity.mean()
    
    # Backpropagation
    loss.backward()
    optimizer.step()
```

---

## Командная строка (CLI)

### Оценка одного изображения

```bash
python scripts/evaluate.py --image test.jpg --auto_download
```

Вывод:
```
TIGAS Score: 0.8234
Assessment:  Likely REAL/Natural
Confidence:  High
```

### Оценка директории

```bash
python scripts/evaluate.py \
    --image_dir images/ \
    --checkpoint checkpoints/best_model.pt \
    --batch_size 32 \
    --output results.json \
    --plot
```

Вывод:
```
Processing 1000 images...

Statistics:
  Mean:   0.6234
  Std:    0.2145
  Min:    0.0123
  Max:    0.9876
  
  Real (>0.5):  623 (62.3%)
  Fake (<=0.5): 377 (37.7%)

Results saved to: results.json
Plot saved to: score_distribution.png
```

---

## Получение промежуточных признаков

```python
from tigas import TIGAS

tigas = TIGAS(auto_download=True)

# Получение features
outputs = tigas('image.jpg', return_features=True)

score = outputs['score']
features = outputs['features']

print(f"Score: {score.item():.4f}")
print(f"Available features: {list(features.keys())}")
# ['perceptual', 'fused', 'multi_scale']

print(f"Fused features shape: {features['fused'].shape}")
# torch.Size([1, 256])
```

---

## Информация о модели

```python
from tigas import TIGAS

tigas = TIGAS(auto_download=True)

info = tigas.get_model_info()
print(f"Total parameters: {info['total_parameters']:,}")
print(f"Trainable: {info['trainable_parameters']:,}")
print(f"Size: {info['model_size_mb']:.2f} MB")
print(f"Device: {info['device']}")
print(f"Image size: {info['img_size']}")
```

---

## Кэширование модели

### Информация о кэше

```python
from tigas import cache_info

info = cache_info()
# Выводит информацию о закэшированных моделях
```

### Очистка кэша

```python
from tigas import clear_cache

clear_cache()  # Удаляет все закэшированные модели
```

---

## Интерпретация результатов

| Score | Интерпретация | Уверенность |
|-------|---------------|-------------|
| 0.8 - 1.0 | Скорее всего реальное | Высокая |
| 0.5 - 0.8 | Вероятно реальное | Средняя |
| 0.3 - 0.5 | Вероятно сгенерированное | Средняя |
| 0.0 - 0.3 | Скорее всего сгенерированное | Высокая |

> **Примечание**: Пороговое значение 0.5 является стандартным, но может быть настроено в зависимости от требований к precision/recall.

---

## Следующие шаги

- [API Reference](api/reference.md) — полное описание классов и методов
- [Обучение модели](training/configuration.md) — как обучить свою модель
- [Архитектура](architecture/overview.md) — как работает TIGAS внутри
