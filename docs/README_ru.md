# TIGAS - Trained Image Generation Authenticity Score

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)

**TIGAS** — нейросетевая метрика для оценки подлинности и реалистичности изображений, разработанная для различения реальных/натуральных изображений от сгенерированных ИИ/поддельных.

## Описание

TIGAS предоставляет непрерывную оценку в диапазоне [0, 1]:
- **1.0** — натуральное/реальное изображение
- **0.0** — сгенерированное/поддельное изображение

### Ключевые особенности

- **Мультимодальный анализ**: комбинирует взаимодополняющие подходы к анализу
  - Перцептивные признаки (многомасштабная CNN)
  - Спектральный анализ (частотная область)
  - Статистическая согласованность (анализ распределений)
  - Локально-глобальная когерентность

- **Полностью дифференцируема**: может использоваться как
  - Метрика оценки качества изображений
  - Функция потерь для обучения генеративных моделей
  - Метрика оценки для задач генерации изображений

- **Гибкое развертывание**:
  - Вычисление на основе модели (обученная нейросеть)
  - Вычисление на основе компонентов (без обученной модели)

## Установка

### Базовая установка

```bash
git clone https://github.com/H1merka/TIGAS.git
cd TIGAS
pip install -r requirements.txt
pip install -e .
```

### С поддержкой CUDA

```bash
pip install -r requirements_cuda.txt
```

### Зависимости

**Основные зависимости:**
- PyTorch >= 2.2.0
- torchvision >= 0.17.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- scikit-learn >= 1.3.0
- Pillow >= 10.0.0
- OpenCV >= 4.8.0
- pandas >= 2.0.0

## Быстрый старт

### Python API

```python
from tigas import TIGAS, compute_tigas_score
import torch

# Метод 1: Высокоуровневая функция
score = compute_tigas_score('image.jpg', checkpoint_path='model.pt')
print(f"TIGAS Score: {score:.4f}")

# Метод 2: Объектно-ориентированный API
tigas = TIGAS(checkpoint_path='model.pt', img_size=256, device='cuda')
score = tigas('image.jpg')  # Одно изображение
scores = tigas(torch.randn(4, 3, 256, 256))  # Батч
scores = tigas.compute_directory('path/to/images/')  # Директория

# Метод 3: Автозагрузка модели из HuggingFace Hub
tigas = TIGAS(auto_download=True)  # Автоматически загружает модель из Hub
score = tigas('image.jpg')

# Метод 4: Как функция потерь
generated_images = torch.randn(4, 3, 256, 256, requires_grad=True)
scores = tigas(generated_images)
loss = 1.0 - scores.mean()  # Максимизация подлинности
loss.backward()
```

### Командная строка

```bash
# Оценка одного изображения
python scripts/evaluate.py --image path/to/image.jpg --checkpoint model.pt

# Оценка директории
python scripts/evaluate.py --image_dir path/to/images/ --checkpoint model.pt --batch_size 32

# С автозагрузкой модели из HuggingFace Hub
python scripts/evaluate.py --image_dir images/ --auto_download

# С сохранением результатов и визуализацией
python scripts/evaluate.py --image_dir images/ --output results.json --plot
```

## Обучение

### Структура данных

**Режим директорий:**
```
dataset/
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── fake/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

**Режим CSV:**
```
dataset/
├── train/
│   ├── images/
│   └── annotations01.csv
├── val/
│   └── ...
└── test/
    └── ...
```
Валидация датасета (обязательный шаг)

**ВАЖНО**: Перед обучением необходимо проверить целостность датасета:

```bash
python scripts/validate_dataset.py \
  --dataset_dir /path/to/data \
  --csv_file train.csv \
  --remove_corrupted \
  --update_csv
```

Это удалит поврежденные изображения и обновит CSV, предотвращая ошибки во время обучения.

### Запуск обучения

```bash
# Быстрое обучение (Fast Mode - по умолчанию, оптимизировано для скорости)
python scripts/train_script.py \
  --data_root /path/to/data \
  --epochs 50 \
  --batch_size 16 \
  --img_size 128 \
  --lr 0.0003 \
  --use_amp \
  --output_dir ./checkpoints

# Полное обучение (Full Mode - все ветви модели, выше точность)
python scripts/train_script.py \
  --data_root /path/to/data \
  --epochs 50 \
  --batch_size 8 \
  --img_size 256 \
  --lr 0.0001 \
  --use_amp \
  --full_mode \
  --output_dir ./checkpoints

# Обучение с CSV аннотациями (рекомендуется)
python scripts/train_script.py \
  --data_root /path/to/data \
  --use_csv \
  --epochs 50 \
  --use_amp

# Продолжение обучения с чекпоинта (ещё N эпох)
python scripts/train_script.py \
  --data_root data/ \
  --resume checkpoints/best_model.pt \
  --epochs 10 \
  --lr 0.0001 \
  --reset_lr

# Продолжение с полным сбросом LR и scheduler
python scripts/train_script.py \
  --data_root data/ \
  --resume checkpoints/best_model.pt \
  --epochs 10 \
  --lr 0.0003 \
  --reset_lr \
  --reset_scheduler
```

### Параметры обучения

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--data_root` | Путь к данным | Обязательный |
| `--epochs` | Количество эпох (при resume — ещё N эпох) | 50 |
| `--batch_size` | Размер батча | 16 |
| `--lr` | Скорость обучения | 0.0000125 |
| `--use_csv` | Использовать CSV аннотации | False |
| `--img_size` | Размер изображения | 256 |
| `--output_dir` | Директория чекпоинтов | ./checkpoints |
| `--device` | Устройство (cuda/cpu) | auto |
| `--num_workers` | Воркеры DataLoader (0 для Windows) | 0 |
| `--use_amp` | Mixed Precision Training | False |
| `--fast_mode` | Быстрая архитектура (оптимизирована) | True |
| `--full_mode` | Полная архитектура (все ветви) | False |
| `--resume` | Путь к чекпоинту для продолжения | None |
| `--reset_lr` | Сбросить LR при resume | False |
| `--reset_scheduler` | Сбросить scheduler при resume | False |

## Архитектура

### Модель TIGASModel

Многоветвевая нейронная сеть, включающая:

1. **Многомасштабный экстрактор признаков**
   - 4-этапный CNN backbone (разрешения 1/2, 1/4, 1/8, 1/16)
   - Сохраняет высокочастотные детали для обнаружения артефактов
   - Дизайн, вдохновленный EfficientNet

2. **Спектральный анализатор**
   - Анализ частотной области на основе FFT
   - Обнаружение артефактов GAN (шахматные паттерны, неестественные спектры)
   - Извлечение радиального профиля из спектра мощности

3. **Статистический оценщик моментов**
   - Анализ согласованности распределений
   - Обучаемая статистика натуральных изображений
   - Сопоставление моментов с априорными данными

4. **Механизмы внимания**
   - Self-Attention для захвата дальних зависимостей
   - Cross-Modal Attention для слияния признаков разных модальностей

5. **Адаптивное слияние признаков**
   - Обучаемое взвешивание 3 потоков признаков
   - Комбинирование перцептивных, спектральных и статистических признаков

## Структура проекта

```
TIGAS/
├── tigas/                          # Основной пакет
│   ├── __init__.py                # Инициализация и экспорты
│   ├── api.py                     # Высокоуровневый API (класс TIGAS)
│   │
│   ├── models/                    # Архитектуры нейросетей
│   │   ├── tigas_model.py        # Основная модель TIGASModel
│   │   ├── feature_extractors.py # Экстракторы признаков
│   │   ├── attention.py          # Механизмы внимания
│   │   ├── layers.py             # Пользовательские слои
│   │   └── constants.py          # Константы конфигурации
│   │
│   ├── metrics/                   # Модули вычисления метрик
│   │   ├── tigas_metric.py       # Основной калькулятор метрики
│   │   └── components.py         # Компоненты метрик
│   │
│   ├── data/                      # Загрузка и предобработка данных
│   │   ├── dataset.py            # Классы датасетов
│   │   ├── loaders.py            # Создание DataLoader
│   │   └── transforms.py         # Аугментации и трансформации
│   │
│   ├── training/                  # Инфраструктура обучения
│   │   ├── trainer.py            # Основной класс тренера
│   │   ├── losses.py             # Функции потерь
│   │   └── optimizers.py         # Оптимизаторы и планировщики
│   │
│   └── utils/                     # Утилиты
│       ├── config.py             # Управление конфигурацией
│       ├── input_processor.py    # Обработка входных данных
│       └── visualization.py      # Визуализация
│
├── scripts/                       # Исполняемые скрипты
│   ├── evaluate.py              # Скрипт оценки/инференса
│   ├── example_usage.py          # Примеры использования
│   └── train_script.py           # Скрипт обучения
│
├── setup.py                     # Конфигурация пакета
├── requirements.txt             # Зависимости
├── requirements_cuda.txt        # CUDA-зависимости
└── LICENSE                      # Лицензия MIT
```

## Примеры использования

### 1. Базовое использование

```python
from tigas import TIGAS

tigas = TIGAS(checkpoint_path='model.pt')
score = tigas('test_image.jpg')
print(f"Оценка подлинности: {score:.4f}")
```

### 2. Пакетная обработка

```python
from tigas import TIGAS
import torch

tigas = TIGAS(checkpoint_path='model.pt', device='cuda')
images = torch.randn(8, 3, 256, 256)
scores = tigas(images)
print(f"Средняя оценка: {scores.mean():.4f}")
```

### 3. Извлечение признаков
import torch

tigas = TIGAS(checkpoint_path='model.pt')
image = torch.randn(1, 3, 256, 256)
outputs = tigas(image, return_features=True)

score = outputs['score']
features = outputs['features']
print(f"Оценка: {score.item():.4f}")
print(f"Доступные признаки: {list(features.keys())}")
print(f"Размерность слитых признаков: {features['fused']
tigas = TIGAS(checkpoint_path='model.pt', return_features=True)
score, features = tigas('image.jpg')
print(f"Размерность признаков: {features.shape}")
```

### 4. Использование как функции потерь

```python
from tigas import TIGAS
Обработка директории с изображениями

```python
from tigas import TIGAS

tigas = TIGAS(checkpoint_path='model.pt')

# Получить оценки для всех изображений
results = tigas.compute_directory(
    'path/to/images/',
    return_paths=True,
    batch_size=32
)

for img_path, score in results.items():
    print(f"{img_path}: {score:.4f}"n()
loss.backward()
```

### 5. Метрика на основе компонентов

```python
from tigas.metrics import TIGASMetric

metric = TIGASMetric(use_model=False)
score = metric.compute(image_tensor)
```

## Требования к изображениям

- **Форматы**: JPG, JPEG, PNG, BMP
- **Разрешение**: по умолчанию 256x256 (настраивается)
- Изображения автоматически масштабируются при необходимости
- Нормализация в диапазон [-1, 1]

## Возможности обучения

- **Mixed Precision Training**: ускоренное обучение с AMP
- **Gradient Accumulation**: для больших эффективных размеров батча
- **Learning Rate Scheduling**: косинусное затухание, warmup
- **Early Stopping**: автоматическая остановка при переобучении
- **TensorBoard Logging**: визуализация процесса обучения
- **Checkpoint Management**: сохранение и загрузка моделей

## Лицензия

Проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).

## Авторы

- Дмитрий Моргенштерн

## Ссылки

- [Репозиторий GitHub](https://github.com/H1merka/TIGAS)