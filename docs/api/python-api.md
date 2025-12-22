# Python API Reference

## Основные классы

### TIGAS

Главный класс для вычисления TIGAS score.

```python
from tigas import TIGAS
```

#### Конструктор

```python
TIGAS(
    checkpoint_path: Optional[str] = None,
    img_size: int = 256,
    device: Optional[str] = None,
    auto_download: bool = True
)
```

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `checkpoint_path` | `str \| None` | `None` | Путь к чекпоинту модели. Если `None`, используется автозагрузка. |
| `img_size` | `int` | `256` | Размер входного изображения |
| `device` | `str \| None` | `None` | Устройство ('cuda' или 'cpu'). Автоопределение если `None`. |
| `auto_download` | `bool` | `True` | Автоматически загрузить модель из HuggingFace Hub |

**Пример:**

```python
# Автозагрузка
tigas = TIGAS(auto_download=True)

# Локальный чекпоинт
tigas = TIGAS(checkpoint_path='model.pt', device='cuda')

# CPU-only
tigas = TIGAS(auto_download=True, device='cpu')
```

---

#### forward / __call__

```python
def forward(
    x: Union[torch.Tensor, str, Path, Image.Image, List],
    return_features: bool = False
) -> Union[torch.Tensor, Dict]
```

Вычисление TIGAS score.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `x` | `Tensor \| str \| Path \| PIL.Image \| List` | Входные данные |
| `return_features` | `bool` | Вернуть промежуточные признаки |

**Поддерживаемые входные форматы:**
- `torch.Tensor`: `[B, C, H, W]` или `[C, H, W]`
- `str` / `Path`: путь к файлу или директории
- `PIL.Image`: объект PIL Image
- `List`: список любых из вышеперечисленных

**Возвращает:**
- `torch.Tensor`: score `[B, 1]` если `return_features=False`
- `Dict`: `{'score': Tensor, 'features': Dict}` если `return_features=True`

**Примеры:**

```python
# Строковый путь
score = tigas('image.jpg')

# PIL Image
from PIL import Image
img = Image.open('image.jpg')
score = tigas(img)

# Тензор
tensor = torch.randn(4, 3, 256, 256)
scores = tigas(tensor)  # [4, 1]

# Директория
scores = tigas('images/')  # Обрабатывает все изображения

# С признаками
outputs = tigas('image.jpg', return_features=True)
score = outputs['score']
features = outputs['features']
```

---

#### compute_image

```python
def compute_image(image_path: str) -> float
```

Вычисление score для одного изображения.

**Возвращает:** `float` — TIGAS score

**Пример:**

```python
score = tigas.compute_image('test.jpg')
print(f"Score: {score:.4f}")  # 0.8234
```

---

#### compute_directory

```python
def compute_directory(
    directory: str,
    return_paths: bool = False,
    batch_size: int = 32,
    max_images: Optional[int] = None
) -> Union[np.ndarray, Dict[str, float]]
```

Вычисление scores для всех изображений в директории.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `directory` | `str` | — | Путь к директории |
| `return_paths` | `bool` | `False` | Вернуть словарь {путь: score} |
| `batch_size` | `int` | `32` | Размер batch |
| `max_images` | `int \| None` | `None` | Максимум изображений |

**Пример:**

```python
# Массив scores
scores = tigas.compute_directory('images/')
print(f"Mean: {scores.mean():.4f}")

# Словарь с путями
results = tigas.compute_directory('images/', return_paths=True)
for path, score in results.items():
    print(f"{path}: {score:.4f}")
```

---

#### compute_batch

```python
def compute_batch(
    images: torch.Tensor,
    batch_size: int = 32
) -> torch.Tensor
```

Batch-обработка с автоматическим разбиением.

**Пример:**

```python
large_batch = torch.randn(1000, 3, 256, 256)
scores = tigas.compute_batch(large_batch, batch_size=64)
```

---

#### get_model_info

```python
def get_model_info() -> Dict[str, Any]
```

Информация о модели.

**Возвращает:**

```python
{
    'total_parameters': 1234567,
    'trainable_parameters': 1234567,
    'model_size_mb': 4.71,
    'device': 'cuda',
    'img_size': 256
}
```

---

#### save_model

```python
def save_model(save_path: str)
```

Сохранение модели в файл.

---

## Функции

### compute_tigas_score

```python
from tigas import compute_tigas_score

score = compute_tigas_score(
    image: Union[str, Path, Image.Image, torch.Tensor],
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    auto_download: bool = True
) -> float
```

Быстрая функция для оценки одного изображения.

**Пример:**

```python
score = compute_tigas_score('image.jpg', auto_download=True)
```

> **Примечание:** Создаёт новый экземпляр TIGAS при каждом вызове. Для множественных вызовов используйте класс `TIGAS`.

---

### load_tigas

```python
from tigas import load_tigas

tigas = load_tigas(
    checkpoint_path: str,
    device: Optional[str] = None
) -> TIGAS
```

Загрузка предобученной модели.

**Пример:**

```python
tigas = load_tigas('checkpoints/best_model.pt')
```

---

## Model Hub функции

### get_default_model_path

```python
from tigas import get_default_model_path

path = get_default_model_path(
    auto_download: bool = True,
    show_progress: bool = True
) -> Optional[str]
```

Получение пути к модели по умолчанию.

---

### download_default_model

```python
from tigas import download_default_model

path = download_default_model(
    model_filename: str = "best_model.pt",
    force_download: bool = False,
    show_progress: bool = True
) -> str
```

Принудительная загрузка модели.

---

### clear_cache

```python
from tigas import clear_cache

clear_cache()
```

Очистка кэша моделей.

---

### cache_info

```python
from tigas import cache_info

info = cache_info() -> dict
```

Информация о кэше.

**Возвращает:**

```python
{
    'cache_dir': '~/.cache/tigas/models/',
    'cache_exists': True,
    'models': ['best_model.pt'],
    'total_size_mb': 4.71
}
```

---

## TIGASMetric

Низкоуровневый класс для вычисления метрики.

```python
from tigas.metrics import TIGASMetric

metric = TIGASMetric(
    model: Optional[TIGASModel] = None,
    device: str = 'cuda'
)
```

### Методы

#### forward

```python
def forward(
    images: torch.Tensor,
    return_features: bool = False
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]
```

#### compute_pairwise

```python
def compute_pairwise(
    images1: torch.Tensor,
    images2: torch.Tensor
) -> torch.Tensor
```

Сравнение двух наборов изображений.

#### compute_dataset_statistics

```python
def compute_dataset_statistics(
    dataloader: DataLoader,
    max_samples: Optional[int] = None
) -> Dict[str, float]
```

Статистика по датасету.

---

## TIGASModel

Основная нейросетевая архитектура.

```python
from tigas.models import TIGASModel, create_tigas_model
```

### Конструктор

```python
TIGASModel(
    img_size: int = 256,
    in_channels: int = 3,
    base_channels: int = 32,
    feature_dim: int = 256,
    num_attention_heads: int = 8,
    dropout: float = 0.1,
    pretrained_backbone: bool = False,
    fast_mode: bool = True
)
```

### Параметр fast_mode

| `fast_mode` | Описание |
|-------------|----------|
| `True` (default) | Только перцептивная ветвь + лёгкий aux branch. Быстрее, меньше памяти. |
| `False` | Все три ветви + cross-modal attention. Точнее, но медленнее. |

### forward

```python
def forward(
    x: torch.Tensor,
    return_features: bool = False,
    update_prototypes: bool = False
) -> Dict[str, torch.Tensor]
```

**Возвращает:**

```python
{
    'score': Tensor[B, 1],     # Authenticity score [0, 1]
    'logits': Tensor[B, 2],   # Classification logits
    'features': {...}          # (если return_features=True)
}
```

### create_tigas_model (factory)

```python
model = create_tigas_model(
    img_size: int = 256,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    fast_mode: bool = True,
    **kwargs
) -> TIGASModel
```

---

## InputProcessor

Обработка различных типов входных данных.

```python
from tigas.utils import InputProcessor

processor = InputProcessor(
    img_size: int = 256,
    device: Optional[str] = None,
    normalize: bool = True
)
```

### process

```python
def process(
    x: Union[torch.Tensor, str, Path, Image.Image, List]
) -> torch.Tensor
```

Преобразует любой поддерживаемый вход в тензор `[B, C, H, W]`.
