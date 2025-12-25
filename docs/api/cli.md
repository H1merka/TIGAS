# Командная строка (CLI)

## Скрипты

TIGAS предоставляет несколько скриптов для работы с моделью:

| Скрипт | Назначение |
|--------|------------|
| `scripts/evaluate.py` | Инференс/оценка изображений |
| `scripts/train_script.py` | Обучение модели |
| `scripts/validate_dataset.py` | Валидация датасета |
| `scripts/example_usage.py` | Примеры использования |

---

## evaluate.py — Оценка изображений

### Основное использование

```bash
# Одно изображение с автозагрузкой модели
python scripts/evaluate.py --image test.jpg --auto_download

# Одно изображение с локальным чекпоинтом
python scripts/evaluate.py --image test.jpg --checkpoint model.pt

# Директория изображений
python scripts/evaluate.py --image_dir images/ --auto_download
```

### Полный список параметров

#### Входные данные (взаимоисключающие)

| Параметр | Описание |
|----------|----------|
| `--image PATH` | Путь к одному изображению |
| `--image_dir PATH` | Путь к директории с изображениями |

#### Конфигурация модели

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--checkpoint PATH` | `None` | Путь к чекпоинту модели |
| `--auto_download` | `False` | Автозагрузка из HuggingFace Hub |
| `--img_size INT` | `256` | Размер входного изображения |

#### Опции вывода

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--output PATH` | `None` | Сохранить результаты в JSON |
| `--plot` | `False` | Построить график распределения |
| `--plot_path PATH` | `None` | Путь для сохранения графика |

#### Опции обработки

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--batch_size INT` | `32` | Размер batch для директории |
| `--device STR` | auto | Устройство (cuda/cpu) |

#### Опции отображения

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--verbose` | `False` | Подробный вывод |
| `--show_top INT` | `None` | Показать топ N реальных/fake |

### Примеры

```bash
# Базовая оценка
python scripts/evaluate.py --image photo.jpg --auto_download

# Оценка директории с сохранением результатов
python scripts/evaluate.py \
    --image_dir dataset/test/ \
    --checkpoint checkpoints/best_model.pt \
    --output results.json \
    --plot \
    --batch_size 64

# Показать топ-10 реальных и fake
python scripts/evaluate.py \
    --image_dir images/ \
    --auto_download \
    --show_top 10 \
    --verbose

# CPU-only
python scripts/evaluate.py \
    --image test.jpg \
    --auto_download \
    --device cpu
```

### Формат вывода

#### Одиночное изображение

```
============================================================
Evaluating: test.jpg
============================================================

TIGAS Score: 0.8234
Assessment:  Likely REAL/Natural
Confidence:  High
```

#### Директория

```
Processing 1000 images...

============================================================
Results Summary
============================================================
Statistics:
  Mean:   0.6234
  Std:    0.2145
  Min:    0.0123
  Max:    0.9876
  
Classification (threshold=0.5):
  Real (>0.5):  623 (62.3%)
  Fake (<=0.5): 377 (37.7%)

Top 5 Most Real:
  1. real_photo_001.jpg: 0.9876
  2. real_photo_042.jpg: 0.9823
  ...

Top 5 Most Fake:
  1. generated_001.jpg: 0.0123
  2. ai_art_015.jpg: 0.0234
  ...

Results saved to: results.json
```

### Формат JSON результатов

```json
{
  "summary": {
    "total_images": 1000,
    "mean_score": 0.6234,
    "std_score": 0.2145,
    "min_score": 0.0123,
    "max_score": 0.9876,
    "real_count": 623,
    "fake_count": 377
  },
  "results": {
    "images/photo1.jpg": 0.8234,
    "images/photo2.jpg": 0.7123,
    ...
  }
}
```

---

## train_script.py — Обучение модели

### Основное использование

```bash
# Базовое обучение (структура real/fake)
python scripts/train_script.py --data_root ./dataset --epochs 50

# Обучение с CSV аннотациями
python scripts/train_script.py --data_root ./dataset --use_csv --epochs 50

# Продолжение обучения
python scripts/train_script.py \
    --data_root ./dataset \
    --resume checkpoints/latest_model.pt \
    --epochs 10
```

### Полный список параметров

#### Обязательные

| Параметр | Описание |
|----------|----------|
| `--data_root PATH` | Путь к датасету |

#### Модель

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--img_size INT` | `256` | Размер изображений |
| `--fast_mode` | `True` | Быстрая архитектура (default) |
| `--full_mode` | `False` | Полная архитектура (все ветви) |

#### Обучение

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--epochs INT` | `50` | Количество эпох |
| `--batch_size INT` | `16` | Размер batch |
| `--lr FLOAT` | `0.0000125` | Learning rate |
| `--num_workers INT` | `0` | Воркеры DataLoader (0 для Windows) |

#### Loss функция

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--regression_weight FLOAT` | `1.0` | Вес regression loss |
| `--classification_weight FLOAT` | `0.3` | Вес classification loss |
| `--ranking_weight FLOAT` | `0.2` | Вес ranking loss |

#### Mixed Precision

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--use_amp` | `False` | Включить AMP |
| `--no_use_amp` | — | Отключить AMP |

#### Режим данных

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--use_csv` | `False` | Использовать CSV аннотации |

#### Resume

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--resume PATH` | `None` | Путь к чекпоинту для продолжения |
| `--reset_lr` | `False` | Сбросить LR при resume |
| `--reset_scheduler` | `False` | Сбросить scheduler при resume |

#### Вывод

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--output_dir PATH` | `./checkpoints` | Директория для чекпоинтов |
| `--device STR` | auto | Устройство (cuda/cpu) |

### Примеры

```bash
# Быстрое обучение на GPU с AMP
python scripts/train_script.py \
    --data_root ./dataset \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0003 \
    --use_amp \
    --fast_mode

# Полное обучение (все ветви)
python scripts/train_script.py \
    --data_root ./dataset \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0001 \
    --full_mode \
    --use_amp

# Обучение с CSV
python scripts/train_script.py \
    --data_root ./TIGAS_dataset \
    --use_csv \
    --epochs 50 \
    --batch_size 16

# Продолжение с новым LR
python scripts/train_script.py \
    --data_root ./dataset \
    --resume checkpoints/best_model.pt \
    --epochs 20 \
    --lr 0.0001 \
    --reset_lr \
    --reset_scheduler
```

---

## validate_dataset.py — Валидация датасета

### Использование

```bash
# Проверка датасета
python scripts/validate_dataset.py \
    --dataset_dir /path/to/dataset \
    --csv_file train.csv

# Автоудаление повреждённых изображений
python scripts/validate_dataset.py \
    --dataset_dir /path/to/dataset \
    --csv_file train.csv \
    --remove_corrupted \
    --update_csv
```

### Параметры

| Параметр | Описание |
|----------|----------|
| `--dataset_dir PATH` | Путь к датасету |
| `--csv_file PATH` | CSV файл аннотаций |
| `--remove_corrupted` | Удалить повреждённые файлы |
| `--update_csv` | Обновить CSV после удаления |

> **Важно:** Рекомендуется запускать валидацию перед обучением для предотвращения NaN ошибок.

---

## Коды возврата

| Код | Описание |
|-----|----------|
| `0` | Успешное выполнение |
| `1` | Ошибка (файл не найден, ошибка модели, и т.д.) |

---

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `CUDA_VISIBLE_DEVICES` | Выбор GPU |
| `TIGAS_CACHE_DIR` | Директория кэша моделей |

```bash
# Использовать только GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --image test.jpg --auto_download

# CPU-only
CUDA_VISIBLE_DEVICES="" python scripts/evaluate.py --image test.jpg --auto_download
```
