# Запуск и мониторинг обучения

## Быстрый старт

### Минимальная команда

```bash
python scripts/train_script.py --data_root ./dataset
```

### Рекомендуемая команда (GPU + AMP)

```bash
python scripts/train_script.py \
    --data_root ./dataset \
    --epochs 50 \
    --batch_size 16 \
    --use_amp \
    --output_dir ./checkpoints
```

---

## Вывод при запуске

```
============================================================
ЗАПУСК ОБУЧЕНИЯ TIGAS
============================================================

[ДАТАСЕТ] /path/to/dataset
   [РЕЖИМ] Структура real/fake
   |- Real изображений: 10000
   |- Fake изображений: 10000
   `- Всего: 20000 изображений

[УСТРОЙСТВО] CUDA
   [OK] GPU обнаружен:
   |- Название: NVIDIA GeForce RTX 3080
   |- Память: 10.0 GB
   |- CUDA версия: 11.8
   `- Обучение будет выполняться на GPU (быстро)

[ПАРАМЕТРЫ ОБУЧЕНИЯ]
   |- Эпох: 50
   |- Размер батча: 16
   |- Скорость обучения: 1.25e-05
   |- Размер изображений: 256x256
   `- Воркеров: 0

[СОЗДАНИЕ МОДЕЛИ]...
   [РЕЖИМ] FAST (оптимизированный)
   [OK] Модель создана:
      |- Параметров: 2,456,789
      |- Обучаемых: 2,456,789
      |- Размер: 9.37 MB
      `- Модель на GPU: cuda:0

[СОЗДАНИЕ ДАТАЛОАДЕРОВ]...
   [РЕЖИМ] Использование структуры real/fake
   [OK] Даталоадеры созданы

[КОНФИГУРАЦИЯ LOSS FUNCTION]...
   Regression weight:     1.0
   Classification weight: 0.3
   Ranking weight:        0.2

[НАЧАЛО ОБУЧЕНИЯ]...
============================================================
```

---

## Мониторинг в реальном времени

### Progress bar

```
Epoch 0: 100%|██████████| 1250/1250 [02:34<00:00, 8.10it/s, loss=0.4523]

Epoch 0 - Train Loss: 0.4523
Val Loss: 0.4012, Accuracy: 0.7234
Learning Rate: 0.000012
[checkpoint] saved: checkpoints/checkpoint_epoch_0.pt
[checkpoint] best saved: checkpoints/best_model.pt
```

### Метрики каждой эпохи

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| Train Loss | Общий loss на train | ↓ Уменьшается |
| Val Loss | Общий loss на val | ↓ Уменьшается |
| Accuracy | Доля верных предсказаний | ↑ Увеличивается |
| Learning Rate | Текущий LR | По scheduler |

---

## TensorBoard

### Запуск

```bash
# В отдельном терминале
tensorboard --logdir checkpoints/logs --port 6006
```

Открыть в браузере: `http://localhost:6006`

### Доступные графики

**Scalars:**
- `train/batch_loss` — loss каждого batch
- `train/total` — epoch training loss
- `train/regression`, `train/classification`, `train/ranking` — компоненты loss
- `val/total` — validation loss
- `val/accuracy` — validation accuracy
- `train/learning_rate` — LR по эпохам

### Сравнение экспериментов

```bash
# Разные директории для разных экспериментов
python scripts/train_script.py --output_dir ./exp_lr_001
python scripts/train_script.py --output_dir ./exp_lr_0001

# Сравнение в TensorBoard
tensorboard --logdir_spec exp1:./exp_lr_001/logs,exp2:./exp_lr_0001/logs
```

---

## Checkpoints

### Автоматическое сохранение

| Файл | Когда сохраняется | Содержимое |
|------|-------------------|------------|
| `checkpoint_epoch_N.pt` | Каждые `save_interval` эпох | Полное состояние |
| `best_model.pt` | При улучшении val_loss | Полное состояние |
| `latest_model.pt` | После каждой эпохи | Полное состояние |
| `training_history.json` | В конце обучения | История метрик |

### Интервал сохранения

```python
save_interval=5  # Сохранять каждые 5 эпох
```

### Проверка содержимого чекпоинта

```python
import torch

checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')

print(f"Epoch: {checkpoint['epoch']}")
print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
print(f"Global step: {checkpoint['global_step']}")

# История обучения
train_hist = checkpoint['train_history']
val_hist = checkpoint['val_history']

print(f"Training epochs recorded: {len(train_hist)}")
```

---

## Прерывание и возобновление

### Graceful shutdown (Ctrl+C)

```
^C
[ПРЕРВАНО] Обучение прервано пользователем
[СОХРАНЕНО] Текущее состояние сохранено в: checkpoints/latest_model.pt
```

### Возобновление

```bash
python scripts/train_script.py \
    --data_root ./dataset \
    --resume checkpoints/latest_model.pt \
    --epochs 20  # Ещё 20 эпох
```

### Логирование resume

```
[RESUME] Будет загружен чекпоинт: checkpoints/latest_model.pt
   |- --epochs 20 означает: ещё 20 эпох
   `- Начинаем с эпохи 35

Resuming from epoch 34, will train 20 more epochs
Epochs: 35 -> 54
```

---

## Обработка ошибок

### NaN в loss

```
RuntimeError: [TIGAS LOSS] NaN/Inf in Regression Loss detected!
Scores stats - min: nan, max: nan, mean: nan, std: nan
Labels stats - min: 0.000000, max: 1.000000
This indicates a problematic batch (possibly corrupted images).
Run: python scripts/validate_dataset.py --data_root <dataset>
```

**Решение:**
```bash
python scripts/validate_dataset.py \
    --dataset_dir ./dataset \
    --remove_corrupted \
    --update_csv
```

### Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB
```

**Решения:**
1. Уменьшить `--batch_size`
2. Использовать `--fast_mode` (по умолчанию)
3. Уменьшить `--img_size`
4. Отключить AMP: `--no_use_amp`

### Медленное обучение на Windows

**Симптом:** Training очень медленный, GPU utilization низкий

**Решение:**
```bash
# Убедиться что num_workers=0
python scripts/train_script.py --data_root ./dataset --num_workers 0
```

---

## Анализ результатов

### training_history.json

```json
{
  "train": [
    {"total": 0.5234, "regression": 0.3456, "classification": 0.1234, "ranking": 0.0544},
    {"total": 0.4567, ...},
    ...
  ],
  "val": [
    {"total": 0.4890, "regression": 0.3123, "classification": 0.1200, "accuracy": 0.7234},
    ...
  ]
}
```

### Визуализация истории

```python
import json
import matplotlib.pyplot as plt

with open('checkpoints/training_history.json') as f:
    history = json.load(f)

train_loss = [e['total'] for e in history['train']]
val_loss = [e['total'] for e in history['val']]
val_acc = [e.get('accuracy', 0) for e in history['val']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_loss, label='Train')
ax1.plot(val_loss, label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Loss curves')

ax2.plot(val_acc)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## Оценка обученной модели

### После обучения

```bash
# Тест на test split
python scripts/evaluate.py \
    --image_dir ./dataset/test/ \
    --checkpoint checkpoints/best_model.pt \
    --output test_results.json \
    --plot
```

### Сравнение моделей

```python
from tigas import TIGAS

# Загрузка разных чекпоинтов
model_v1 = TIGAS(checkpoint_path='checkpoints/checkpoint_epoch_10.pt')
model_v2 = TIGAS(checkpoint_path='checkpoints/best_model.pt')

# Сравнение на тестовом изображении
score_v1 = model_v1('test.jpg')
score_v2 = model_v2('test.jpg')

print(f"Model v1: {score_v1.item():.4f}")
print(f"Model v2: {score_v2.item():.4f}")
```

---

## Типичный workflow

```bash
# 1. Валидация датасета
python scripts/validate_dataset.py --dataset_dir ./dataset --remove_corrupted

# 2. Первичное обучение
python scripts/train_script.py \
    --data_root ./dataset \
    --epochs 50 \
    --batch_size 16 \
    --use_amp

# 3. Оценка результатов
python scripts/evaluate.py \
    --image_dir ./dataset/test/ \
    --checkpoint checkpoints/best_model.pt \
    --plot

# 4. Fine-tuning (если нужно)
python scripts/train_script.py \
    --data_root ./dataset \
    --resume checkpoints/best_model.pt \
    --epochs 20 \
    --lr 0.00001 \
    --reset_lr

# 5. Финальная оценка
python scripts/evaluate.py \
    --image_dir ./test_images/ \
    --checkpoint checkpoints/best_model.pt \
    --output final_results.json
```
