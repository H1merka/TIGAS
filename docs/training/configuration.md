# Конфигурация обучения

## Обзор параметров

### Основные параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--data_root` | PATH | **Обязательный.** Путь к датасету |
| `--epochs` | 50 | Количество эпох |
| `--batch_size` | 16 | Размер batch |
| `--lr` | 0.0000125 | Learning rate |
| `--output_dir` | ./checkpoints | Директория для сохранения |

---

## Режимы модели

### Fast Mode (по умолчанию)

```bash
python scripts/train_script.py --data_root ./dataset --fast_mode
```

| Характеристика | Значение |
|----------------|----------|
| Ветви | Perceptual + Aux branch |
| Параметры | ~2.5M |
| Память GPU | ~4 GB (batch=16) |
| Скорость | Быстро |
| Применение | Обучение, прототипирование |

### Full Mode

```bash
python scripts/train_script.py --data_root ./dataset --full_mode
```

| Характеристика | Значение |
|----------------|----------|
| Ветви | Perceptual + Spectral + Statistical |
| Attention | Cross-modal + Self-attention |
| Параметры | ~5M |
| Память GPU | ~8 GB (batch=16) |
| Скорость | Медленнее |
| Применение | Максимальная точность |

---

## Learning Rate

### Выбор LR

| Сценарий | Рекомендуемый LR |
|----------|------------------|
| С AMP (mixed precision) | 0.0000125 - 0.0001 |
| Без AMP | 0.0001 - 0.0003 |
| Fine-tuning | 0.00001 - 0.00005 |
| С большим batch | Увеличить пропорционально |

### LR Scheduling

По умолчанию используется **Cosine Annealing** с warmup:

```
LR
 ^
 |    /‾‾‾‾‾‾‾‾‾‾‾‾\
 |   /              ‾‾‾‾‾‾‾‾‾\
 |  /                          ‾‾‾→
 | /
 +--+----------------------------->
   warmup        training        min_lr
```

**Параметры scheduler:**
```python
scheduler_config = {
    'scheduler_type': 'cosine',
    'num_epochs': 50,
    'warmup_epochs': 5,
    'min_lr': 0.000001
}
```

---

## Loss функция

### Компоненты loss

```
Total Loss = w₁ × Regression + w₂ × Classification + w₃ × Ranking
```

| Компонент | Вес (default) | Назначение |
|-----------|---------------|------------|
| Regression | 1.0 | MSE между score и label |
| Classification | 0.3 | CrossEntropy для real/fake |
| Ranking | 0.2 | MarginRanking: real > fake |

### Настройка весов

```bash
python scripts/train_script.py \
    --data_root ./dataset \
    --regression_weight 1.0 \
    --classification_weight 0.3 \
    --ranking_weight 0.2
```

### Рекомендации по настройке

| Ситуация | Рекомендация |
|----------|--------------|
| Нестабильное обучение | Уменьшить classification_weight |
| Плохое разделение классов | Увеличить ranking_weight |
| Неточные scores | Увеличить regression_weight |

---

## Mixed Precision (AMP)

### Включение AMP

```bash
python scripts/train_script.py --data_root ./dataset --use_amp
```

### Преимущества
- Ускорение обучения в 1.5-2x
- Уменьшение использования памяти
- Возможность увеличить batch_size

### Настройки GradScaler (консервативные)

```python
# Настроены для стабильности
GradScaler(
    init_scale=128.0,      # Низкий начальный scale
    growth_factor=1.5,     # Медленный рост
    backoff_factor=0.75,   # Быстрый откат при overflow
    growth_interval=100    # Редкое увеличение scale
)
```

### Когда НЕ использовать AMP
- На CPU (нет поддержки)
- При частых NaN (попробуйте без AMP)
- На старых GPU (compute capability < 7.0)

---

## Batch Size

### Рекомендации

| GPU VRAM | Fast Mode | Full Mode |
|----------|-----------|-----------|
| 6 GB | 16 | 8 |
| 8 GB | 32 | 16 |
| 12 GB | 64 | 32 |
| 24 GB | 128 | 64 |

### Gradient Accumulation

Для эффективного увеличения batch size без увеличения памяти:

```python
# В trainer настроено:
gradient_accumulation_steps=1  # Default

# Effective batch = batch_size × gradient_accumulation_steps
```

---

## DataLoader Workers

### Windows

```bash
--num_workers 0  # Default, рекомендуется
```

На Windows многопроцессность DataLoader имеет высокий overhead из-за spawn (вместо fork).

### Linux/Mac

```bash
--num_workers 4  # или количество CPU cores
```

### Параметры DataLoader

```python
DataLoader(
    ...,
    pin_memory=True,          # Быстрее GPU transfer
    persistent_workers=True,   # Не пересоздавать workers
    prefetch_factor=2,        # Prefetch batches
    drop_last=True            # Для стабильного batch_size
)
```

---

## Checkpoint и Resume

### Структура чекпоинта

```python
checkpoint = {
    'epoch': int,
    'global_step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,
    'scaler_state_dict': OrderedDict,  # Для AMP
    'best_val_loss': float,
    'train_history': List[Dict],
    'val_history': List[Dict]
}
```

### Сохраняемые файлы

| Файл | Описание |
|------|----------|
| `checkpoint_epoch_N.pt` | Чекпоинт каждой N эпохи |
| `best_model.pt` | Лучшая модель по val_loss |
| `latest_model.pt` | Последний чекпоинт |
| `training_history.json` | История метрик |
| `logs/` | TensorBoard логи |

### Resume обучения

```bash
# Продолжить с той же конфигурацией
python scripts/train_script.py \
    --data_root ./dataset \
    --resume checkpoints/latest_model.pt \
    --epochs 20  # Ещё 20 эпох

# Продолжить с новым LR
python scripts/train_script.py \
    --data_root ./dataset \
    --resume checkpoints/latest_model.pt \
    --epochs 20 \
    --lr 0.0001 \
    --reset_lr

# Полный сброс LR и scheduler (новый warmup)
python scripts/train_script.py \
    --data_root ./dataset \
    --resume checkpoints/best_model.pt \
    --epochs 30 \
    --lr 0.0003 \
    --reset_lr \
    --reset_scheduler
```

---

## Early Stopping

### Параметры

```python
early_stopping_patience=15  # Эпох без улучшения до остановки
```

### Логика
1. После каждой валидации сравниваем `val_loss` с лучшим
2. Если улучшение — сбрасываем счётчик, сохраняем best_model
3. Если нет — увеличиваем счётчик
4. При достижении patience — останавливаем обучение

---

## Validation

### Интервал валидации

```python
validate_interval=1  # Валидация каждую эпоху
```

### Метрики валидации

| Метрика | Описание |
|---------|----------|
| `val_loss` | Общий validation loss |
| `val_regression` | Regression компонент |
| `val_classification` | Classification компонент |
| `val_accuracy` | Accuracy (threshold=0.5) |

---

## TensorBoard

### Включение

```python
use_tensorboard=True  # По умолчанию в trainer
```

### Просмотр логов

```bash
tensorboard --logdir checkpoints/logs
```

### Логируемые метрики

- `train/batch_loss` — loss каждого batch
- `train/total`, `train/regression`, etc. — epoch losses
- `val/total`, `val/accuracy` — validation метрики
- `train/learning_rate` — текущий LR

---

## Пример полной конфигурации

```bash
python scripts/train_script.py \
    --data_root ./dataset \
    --use_csv \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --img_size 256 \
    --use_amp \
    --fast_mode \
    --regression_weight 1.0 \
    --classification_weight 0.3 \
    --ranking_weight 0.2 \
    --output_dir ./checkpoints \
    --num_workers 0 \
    --device cuda
```
