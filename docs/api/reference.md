# API Reference — Классы и функции

## Полная иерархия модулей

```
tigas/
├── __init__.py          # Публичные экспорты
├── api.py               # TIGAS, compute_tigas_score, load_tigas
├── model_hub.py         # get_default_model_path, download_default_model, clear_cache, cache_info
│
├── models/
│   ├── tigas_model.py   # TIGASModel, create_tigas_model
│   ├── feature_extractors.py  # MultiScaleFeatureExtractor, SpectralAnalyzer, StatisticalMomentEstimator
│   ├── attention.py     # SelfAttention, CrossModalAttention, CBAM
│   ├── layers.py        # FrequencyBlock, AdaptiveFeatureFusion, GatedResidualBlock
│   └── constants.py     # DEFAULT_FEATURE_DIM, etc.
│
├── metrics/
│   ├── tigas_metric.py  # TIGASMetric, compute_tigas_batch
│   └── components.py    # PerceptualDistance, SpectralDivergence, StatisticalConsistency
│
├── data/
│   ├── dataset.py       # TIGASDataset, RealFakeDataset, CSVDataset, PairedDataset
│   ├── loaders.py       # create_dataloaders, create_dataloaders_from_csv
│   └── transforms.py    # get_train_transforms, get_val_transforms, get_inference_transforms
│
├── training/
│   ├── trainer.py       # TIGASTrainer
│   ├── losses.py        # TIGASLoss, CombinedLoss, ContrastiveLoss, FocalLoss
│   └── optimizers.py    # create_optimizer, create_scheduler
│
└── utils/
    ├── config.py        # Конфигурация
    ├── input_processor.py  # InputProcessor
    └── visualization.py # Визуализация
```

---

## tigas (top-level)

### Экспорты из `__init__.py`

```python
from tigas import (
    # Основной API
    TIGAS,
    compute_tigas_score,
    load_tigas,
    
    # Метрика
    TIGASMetric,
    
    # Model Hub
    get_default_model_path,
    download_default_model,
    clear_cache,
    cache_info,
)
```

---

## tigas.models

### TIGASModel

```python
class TIGASModel(nn.Module):
    """
    Основная архитектура TIGAS.
    
    Многоветвевая нейросеть:
    - fast_mode=True: перцептивная ветвь + aux branch
    - fast_mode=False: все три ветви + cross-modal attention
    """
    
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        base_channels: int = 32,
        feature_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        pretrained_backbone: bool = False,
        fast_mode: bool = True
    ): ...
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        update_prototypes: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'score': Tensor[B, 1],   # [0, 1]
                'logits': Tensor[B, 2],  # для classification loss
                'features': {...}        # если return_features=True
            }
        """
    
    def compute_tigas(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience метод для получения только score."""
    
    def get_model_size(self) -> Dict[str, int]:
        """Информация о размере модели."""
```

### create_tigas_model

```python
def create_tigas_model(
    img_size: int = 256,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    fast_mode: bool = True,
    **kwargs
) -> TIGASModel:
    """Фабричная функция для создания модели."""
```

### MultiScaleFeatureExtractor

```python
class MultiScaleFeatureExtractor(nn.Module):
    """
    Извлечение признаков на 4 масштабах (1/2, 1/4, 1/8, 1/16).
    
    Args:
        in_channels: Входные каналы (3 для RGB)
        base_channels: Базовое количество каналов (32)
        stages: Количество блоков на каждом этапе [2, 3, 4, 3]
    """
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            List из 4 feature maps на разных масштабах
        """
```

### SpectralAnalyzer

```python
class SpectralAnalyzer(nn.Module):
    """
    Анализ частотной области через FFT.
    Детектирует артефакты GAN (checkerboard patterns, etc.)
    
    Args:
        in_channels: Входные каналы
        hidden_dim: Размерность скрытого слоя
    """
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            output: Tensor[B, hidden_dim]
            aux: {'freq_features': Tensor}
        """
```

### StatisticalMomentEstimator

```python
class StatisticalMomentEstimator(nn.Module):
    """
    Оценка статистической согласованности с естественными изображениями.
    Использует learnable prototypes.
    
    Args:
        in_channels: Входные каналы
        feature_dim: Размерность выходных признаков
    """
    
    def forward(
        self,
        x: torch.Tensor,
        update_prototypes: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            output: Tensor[B, feature_dim]
            aux: {'statistics': Tensor, 'prototypes': Tensor}
        """
```

### Attention классы

```python
class SelfAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.1
    ): ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C] -> [B, N, C]"""


class CrossModalAttention(nn.Module):
    """Cross-modal attention для fusion разных модальностей."""
    
    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ): ...
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor: ...


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
```

---

## tigas.training

### TIGASTrainer

```python
class TIGASTrainer:
    """
    Полнофункциональный trainer для TIGAS.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Checkpoint management
    - TensorBoard logging
    - Early stopping
    - LR scheduling
    """
    
    def __init__(
        self,
        model: TIGASModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        device: str = 'cuda',
        output_dir: str = './checkpoints',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 0.5,
        log_interval: int = 50,
        save_interval: int = 1,
        validate_interval: int = 5,
        early_stopping_patience: int = 10,
        use_tensorboard: bool = False,
    ): ...
    
    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None,
        reset_lr: bool = False,
        reset_scheduler: bool = False,
        new_lr: Optional[float] = None
    ):
        """
        Args:
            num_epochs: Количество ДОПОЛНИТЕЛЬНЫХ эпох при resume
            resume_from: Путь к чекпоинту
            reset_lr: Не восстанавливать LR из чекпоинта
            reset_scheduler: Сбросить scheduler
            new_lr: Установить конкретный LR
        """
    
    def train_epoch(self) -> Dict[str, float]: ...
    def validate(self) -> Dict[str, float]: ...
    def save_checkpoint(self, is_best: bool = False): ...
    def load_checkpoint(self, checkpoint_path: str, **kwargs): ...
```

### Loss функции

```python
class TIGASLoss(nn.Module):
    """
    Основной loss: regression + classification + ranking.
    
    Args:
        regression_weight: Вес MSE/SmoothL1 loss (default: 1.0)
        classification_weight: Вес CrossEntropy (default: 0.5)
        ranking_weight: Вес MarginRanking (default: 0.3)
        use_smooth_l1: Использовать SmoothL1 вместо MSE
        margin: Margin для ranking loss
    """
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'total': Tensor,
                'regression': Tensor,
                'classification': Tensor,
                'ranking': Tensor
            }
        """


class CombinedLoss(nn.Module):
    """
    Комбинированный loss с опциональными компонентами.
    
    Args:
        use_tigas_loss: Использовать TIGASLoss
        use_contrastive: Добавить ContrastiveLoss
        use_regularization: L2 регуляризация
        tigas_loss_config: Конфигурация TIGASLoss
        reg_weight: Вес L2 регуляризации
    """


class ContrastiveLoss(nn.Module):
    """Contrastive loss для feature embeddings."""


class FocalLoss(nn.Module):
    """Focal loss для работы с class imbalance."""
```

### Optimizers и Schedulers

```python
def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',  # 'adam', 'adamw', 'sgd'
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: tuple = (0.9, 0.999),
    **kwargs
) -> torch.optim.Optimizer: ...


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'cosine',  # 'cosine', 'step', 'plateau', 'onecycle', 'none'
    num_epochs: int = 100,
    steps_per_epoch: Optional[int] = None,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
    **kwargs
) -> Optional[LRScheduler]: ...
```

---

## tigas.data

### Dataset классы

```python
class TIGASDataset(Dataset):
    """
    Датасет со структурой root/{real,fake}/*.
    
    Args:
        root: Путь к корню датасета
        transform: Трансформации изображений
        split: 'train', 'val', 'test'
        use_cache: Кэшировать изображения в памяти
    """


class CSVDataset(Dataset):
    """
    Датасет с CSV аннотациями.
    
    CSV формат: image_path,label
    label: 1 = real, 0 = fake
    
    Args:
        csv_file: Путь к CSV (относительный или абсолютный)
        root_dir: Корневая директория для путей в CSV
        transform: Трансформации
        use_cache: Кэширование
        validate_paths: Проверять существование файлов
    """


class RealFakeDataset(Dataset):
    """Датасет с явными списками real/fake изображений."""


class PairedDataset(Dataset):
    """Датасет для парного обучения (real, fake) пар."""
```

### DataLoader функции

```python
def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 12,
    train_split: float = 0.8,
    val_split: float = 0.1,
    augment_level: str = 'medium',
    pin_memory: bool = True,
    shuffle: bool = True
) -> Dict[str, DataLoader]:
    """
    Создание train/val/test loaders из структуры real/fake.
    
    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """


def create_dataloaders_from_csv(
    data_root: str,
    train_csv: str = 'train/annotations01.csv',
    val_csv: str = 'val/annotations01.csv',
    test_csv: str = 'test/annotations01.csv',
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 12,
    augment_level: str = 'medium',
    pin_memory: bool = True,
    shuffle: bool = True,
    use_cache: bool = False,
    validate_paths: bool = True
) -> Dict[str, DataLoader]:
    """Создание loaders из CSV аннотаций."""
```

### Transforms

```python
def get_train_transforms(
    img_size: int = 256,
    normalize: bool = True,
    augment_level: str = 'medium'  # 'light', 'medium', 'heavy'
) -> Compose:
    """
    Тренировочные трансформации:
    - Resize + RandomCrop
    - RandomHorizontalFlip
    - ColorJitter
    - ToTensor
    - Normalize [-1, 1]
    """


def get_val_transforms(
    img_size: int = 256,
    normalize: bool = True
) -> Compose:
    """
    Валидационные трансформации:
    - Resize + CenterCrop
    - ToTensor
    - Normalize [-1, 1]
    """


def get_inference_transforms(
    img_size: int = 256,
    normalize: bool = True
) -> Compose:
    """
    Инференс трансформации:
    - Resize (force square)
    - ToTensor
    - Normalize [-1, 1]
    """


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Денормализация [-1, 1] -> [0, 1]"""
```

---

## tigas.utils

### InputProcessor

```python
class InputProcessor:
    """
    Унифицированная обработка входных данных.
    
    Поддерживает:
    - torch.Tensor
    - str / Path (путь к файлу)
    - PIL.Image
    - List (список любых из вышеперечисленных)
    """
    
    def __init__(
        self,
        img_size: int = 256,
        device: Optional[str] = None,
        normalize: bool = True
    ): ...
    
    def process(
        self,
        x: Union[torch.Tensor, str, Path, Image.Image, List]
    ) -> torch.Tensor:
        """
        Returns:
            Tensor [B, C, H, W] на указанном device
        """
```

---

## Константы (tigas.models.constants)

```python
# Архитектура
DEFAULT_FEATURE_DIM = 256
DEFAULT_BASE_CHANNELS = 32
DEFAULT_ATTENTION_HEADS = 8
DEFAULT_STAGES = [2, 3, 4, 3]

# Нормализация входа
INPUT_MIN = -1.0
INPUT_MAX = 1.0

# Regression head
REGRESSION_HIDDEN_DIM_RATIO = 2
REGRESSION_FINAL_DIM_RATIO = 4

# Инициализация весов
LINEAR_WEIGHT_STD = 0.02
```
