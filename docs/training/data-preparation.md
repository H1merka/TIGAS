# Подготовка данных для обучения

## Поддерживаемые форматы данных

TIGAS поддерживает два формата организации датасета:

### 1. Структура директорий (real/fake)

Простой формат — изображения разделены по папкам:

```
dataset/
├── real/           # label = 1.0 (реальные изображения)
│   ├── img001.jpg
│   ├── img002.png
│   └── ...
└── fake/           # label = 0.0 (сгенерированные изображения)
    ├── gen001.jpg
    ├── gen002.png
    └── ...
```

**Использование:**
```bash
python scripts/train_script.py --data_root ./dataset
```

### 2. CSV аннотации (рекомендуется для больших датасетов)

Формат с явными сплитами train/val/test:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── annotations01.csv
├── val/
│   ├── images/
│   └── annotations01.csv
└── test/
    ├── images/
    └── annotations01.csv
```

**Формат CSV:**
```csv
image_path,label
images/img001.jpg,1
images/img002.jpg,0
train/images/photo.png,1
```

| Колонка | Описание |
|---------|----------|
| `image_path` | Относительный путь к изображению (от корня или от папки split) |
| `label` | 1 = real, 0 = fake |

**Использование:**
```bash
python scripts/train_script.py --data_root ./dataset --use_csv
```

---

## Требования к изображениям

### Поддерживаемые форматы
- JPEG (.jpg, .jpeg, .JPEG)
- PNG (.png)
- BMP (.bmp)

### Рекомендуемые параметры
| Параметр | Рекомендация |
|----------|--------------|
| Минимальный размер | 256×256 px |
| Оптимальный размер | 256×256 или 512×512 px |
| Цветовое пространство | RGB (3 канала) |
| Битовая глубина | 8 bit |

> **Примечание:** Изображения автоматически конвертируются в RGB и ресайзятся до `img_size`.

---

## Валидация датасета

### Зачем нужна валидация

Повреждённые или некорректные изображения могут вызвать:
- NaN в loss во время обучения
- Crash тренировочного процесса
- Некорректную сходимость модели

### Запуск валидации

```bash
python scripts/validate_dataset.py \
    --dataset_dir /path/to/dataset \
    --csv_file train/annotations01.csv
```

### Автоматическое исправление

```bash
python scripts/validate_dataset.py \
    --dataset_dir /path/to/dataset \
    --csv_file train/annotations01.csv \
    --remove_corrupted \
    --update_csv
```

Это:
1. Найдёт все повреждённые изображения
2. Удалит их с диска
3. Обновит CSV, убрав соответствующие записи

### Что проверяется

| Проверка | Описание |
|----------|----------|
| Существование файла | Путь из CSV указывает на существующий файл |
| Открываемость | Файл может быть открыт как изображение |
| RGB конвертация | Изображение конвертируется в RGB без ошибок |
| Размер | Изображение имеет ненулевые размеры |
| Корректность данных | Нет NaN/Inf в пикселях |

---

## Баланс классов

### Проверка баланса

```python
from pathlib import Path

dataset = Path('./dataset')
real_count = len(list((dataset / 'real').glob('**/*.jpg')))
fake_count = len(list((dataset / 'fake').glob('**/*.jpg')))

print(f"Real: {real_count}")
print(f"Fake: {fake_count}")
print(f"Ratio: {real_count / fake_count:.2f}")
```

### Рекомендации

| Дисбаланс | Рекомендация |
|-----------|--------------|
| < 1:2 | Нормально, обучение стабильно |
| 1:2 - 1:5 | Рассмотреть weighted sampling или class weights |
| > 1:5 | Сильный дисбаланс, использовать oversampling меньшего класса |

### Балансировка при создании датасета

```python
from tigas.data.dataset import RealFakeDataset

# Автоматическая балансировка
dataset = RealFakeDataset(
    real_images=real_paths,
    fake_images=fake_paths,
    balance=True  # Выравнивает количество real/fake
)
```

---

## Аугментации

### Уровни аугментации

| Уровень | Описание | Применение |
|---------|----------|------------|
| `light` | Только resize + crop | Чувствительные данные |
| `medium` (default) | + flip, color jitter | Большинство случаев |
| `heavy` | + rotation | Маленькие датасеты |

### Применяемые трансформации

**Light:**
- Resize (1.1x target size)
- RandomCrop
- ToTensor
- Normalize [-1, 1]

**Medium (добавляет):**
- RandomHorizontalFlip (p=0.5)
- ColorJitter (brightness, contrast, saturation, hue)

**Heavy (добавляет):**
- RandomRotation (±10°)

### Настройка

```bash
# В train_script.py (внутренне)
# augment_level устанавливается через конфигурацию loaders

# Для кастомных аугментаций:
from tigas.data.transforms import get_train_transforms

transform = get_train_transforms(
    img_size=256,
    augment_level='heavy',
    normalize=True
)
```

---

## Источники данных

### Рекомендуемые датасеты для обучения

| Датасет | Описание | Размер |
|---------|----------|--------|
| [FFHQ](https://github.com/NVlabs/ffhq-dataset) | Лица высокого качества | 70K |
| [LSUN](https://www.yf.io/p/lsun) | Различные категории | 1M+ |
| [ImageNet](https://www.image-net.org/) | Разнообразные объекты | 1.2M |
| [COCO](https://cocodataset.org/) | Натуральные сцены | 330K |

### Датасеты сгенерированных изображений

| Источник | Генератор | Примечание |
|----------|-----------|------------|
| [This Person Does Not Exist](https://thispersondoesnotexist.com/) | StyleGAN | Лица |
| [Midjourney](https://midjourney.com/) | Diffusion | Художественные |
| [DALL-E](https://openai.com/dall-e-2) | Diffusion | Разнообразные |
| [Stable Diffusion](https://stability.ai/) | Diffusion | Open-source |

---

## Организация больших датасетов

### Структура для >100K изображений

```
TIGAS_dataset/
├── train/
│   ├── real/
│   │   ├── batch_001/
│   │   │   ├── img_00001.jpg
│   │   │   └── ...
│   │   ├── batch_002/
│   │   └── ...
│   ├── fake/
│   │   ├── stylegan/
│   │   ├── midjourney/
│   │   └── stable_diffusion/
│   └── annotations01.csv
├── val/
│   └── ...
└── test/
    └── ...
```

### Создание CSV для структуры

```python
import pandas as pd
from pathlib import Path

def create_csv(root_dir, output_csv):
    records = []
    
    for img_path in Path(root_dir).rglob('*.jpg'):
        # Определяем label по пути
        if 'real' in str(img_path):
            label = 1
        elif 'fake' in str(img_path):
            label = 0
        else:
            continue
        
        rel_path = img_path.relative_to(root_dir)
        records.append({'image_path': str(rel_path), 'label': label})
    
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} records")

# Использование
create_csv('dataset/train', 'dataset/train/annotations01.csv')
```

---

## Troubleshooting

### Проблема: "No images found"

```
[ОШИБКА] No images found in /path/to/dataset
```

**Решение:**
1. Проверьте структуру директорий (должны быть папки `real/` и `fake/`)
2. Проверьте расширения файлов (.jpg, .png, .bmp)
3. Убедитесь, что пути не содержат спецсимволов

### Проблема: "NaN loss during training"

**Причины:**
1. Повреждённые изображения
2. Слишком высокий learning rate
3. Некорректная нормализация

**Решение:**
```bash
# Валидация датасета
python scripts/validate_dataset.py --dataset_dir ./dataset --remove_corrupted
```

### Проблема: "Out of memory"

**Решения:**
1. Уменьшить `batch_size`
2. Уменьшить `img_size`
3. Использовать `--fast_mode` (по умолчанию)
4. Отключить `use_cache` в датасете
