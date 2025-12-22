# Продвинутые техники обучения

## Fine-tuning предобученной модели

### Стратегия 1: Полный fine-tuning

```bash
python scripts/train_script.py \
    --data_root ./new_dataset \
    --resume checkpoints/pretrained_model.pt \
    --epochs 20 \
    --lr 0.00005 \
    --reset_lr \
    --reset_scheduler
```

### Стратегия 2: Замораживание backbone

```python
from tigas.models import create_tigas_model

model = create_tigas_model(checkpoint_path='pretrained.pt')

# Заморозить feature extractor
for param in model.perceptual_extractor.parameters():
    param.requires_grad = False

# Обучить только головы
# (regression_head и binary_classifier остаются trainable)
```

### Стратегия 3: Discriminative learning rates

```python
from torch.optim import AdamW

# Разные LR для разных частей модели
param_groups = [
    {'params': model.perceptual_extractor.parameters(), 'lr': 1e-5},
    {'params': model.regression_head.parameters(), 'lr': 1e-4},
    {'params': model.binary_classifier.parameters(), 'lr': 1e-4},
]

optimizer = AdamW(param_groups)
```

---

## Кастомные Loss функции

### Добавление нового компонента loss

```python
from tigas.training.losses import CombinedLoss, TIGASLoss
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tigas_loss = TIGASLoss(
            regression_weight=1.0,
            classification_weight=0.3,
            ranking_weight=0.2
        )
    
    def forward(self, outputs, labels, model=None):
        # Базовые losses
        losses = self.tigas_loss(outputs, labels)
        
        # Добавляем custom компонент (например, confidence loss)
        scores = outputs['score']
        
        # Штраф за неуверенные предсказания (близкие к 0.5)
        confidence_loss = -torch.abs(scores - 0.5).mean()
        losses['confidence'] = confidence_loss
        
        # Обновляем total
        losses['total'] = losses['total'] + 0.1 * confidence_loss
        
        return losses
```

### Focal Loss для несбалансированных данных

```python
from tigas.training.losses import FocalLoss

# Использование в CombinedLoss
loss_fn = CombinedLoss(
    use_tigas_loss=True,
    tigas_loss_config={
        'regression_weight': 1.0,
        'classification_weight': 0.0,  # Отключить стандартный CE
    }
)

# Добавить Focal Loss отдельно
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
```

---

## Кастомные аугментации

### Создание специализированных трансформаций

```python
import torchvision.transforms as T
from tigas.data.transforms import RandomJPEGCompression, RandomGaussianNoise

def get_custom_transforms(img_size=256):
    return T.Compose([
        T.Resize(int(img_size * 1.2)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        
        # Имитация JPEG артефактов (важно для детекции)
        RandomJPEGCompression(quality_range=(60, 95)),
        
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomRotation(15),
        
        T.ToTensor(),
        
        # Gaussian noise (имитация камеры)
        RandomGaussianNoise(std_range=(0.0, 0.05)),
        
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
```

### CutMix / MixUp аугментации

```python
from tigas.data.transforms import MixUp

mixup = MixUp(alpha=0.2)

# В training loop:
for images, labels in dataloader:
    if random.random() < 0.5:  # 50% chance
        # Получить другой batch
        images2, labels2 = next(iter(dataloader))
        images, lam = mixup(images, images2)
        labels = lam * labels + (1 - lam) * labels2
    
    outputs = model(images)
    loss = loss_fn(outputs, labels)
```

---

## Multi-GPU обучение

### DataParallel (простой способ)

```python
import torch
from tigas.models import create_tigas_model

model = create_tigas_model()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model = model.to('cuda')
```

### DistributedDataParallel (рекомендуется)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size):
    setup(rank, world_size)
    
    model = create_tigas_model().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # ... training loop
    
    dist.destroy_process_group()

# Запуск
# torchrun --nproc_per_node=2 train_ddp.py
```

---

## Оптимизация памяти

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class TIGASModelWithCheckpointing(TIGASModel):
    def _forward_fast(self, x, return_features=False):
        # Используем checkpointing для тяжёлых операций
        perceptual_features = checkpoint(
            self.perceptual_extractor,
            x,
            use_reentrant=False
        )
        # ... остальной код
```

### Накопление градиентов

```python
gradient_accumulation_steps = 4
effective_batch_size = batch_size * gradient_accumulation_steps

for i, (images, labels) in enumerate(dataloader):
    outputs = model(images)
    loss = loss_fn(outputs, labels) / gradient_accumulation_steps
    loss.backward()
    
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Оптимизация batch size

```python
# Автоматический подбор batch size
def find_optimal_batch_size(model, dataset, start_batch=128, min_batch=4):
    batch_size = start_batch
    
    while batch_size >= min_batch:
        try:
            loader = DataLoader(dataset, batch_size=batch_size)
            batch = next(iter(loader))
            
            # Пробный forward + backward
            model.zero_grad()
            outputs = model(batch[0].cuda())
            loss = outputs['score'].mean()
            loss.backward()
            
            print(f"Batch size {batch_size}: OK")
            return batch_size
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                batch_size //= 2
                print(f"OOM, trying batch size {batch_size}")
            else:
                raise
    
    return min_batch
```

---

## Эксперименты и отслеживание

### Структура экспериментов

```
experiments/
├── exp_001_baseline/
│   ├── config.yaml
│   ├── checkpoints/
│   └── logs/
├── exp_002_lr_sweep/
│   ├── config.yaml
│   └── ...
└── exp_003_full_mode/
    └── ...
```

### Конфигурационный файл

```yaml
# config.yaml
experiment:
  name: exp_001_baseline
  seed: 42

data:
  root: /path/to/dataset
  img_size: 256
  batch_size: 16
  num_workers: 0

model:
  fast_mode: true
  feature_dim: 256

training:
  epochs: 50
  lr: 0.0000125
  use_amp: true
  
loss:
  regression_weight: 1.0
  classification_weight: 0.3
  ranking_weight: 0.2
```

### Интеграция с Weights & Biases

```python
import wandb

wandb.init(
    project="tigas",
    config={
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": "fast_mode" if args.fast_mode else "full_mode"
    }
)

# В training loop
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_accuracy": accuracy,
    "learning_rate": current_lr
})

# В конце
wandb.finish()
```

---

## Domain Adaptation

### Обучение на новом домене

Если нужно адаптировать модель под новый тип изображений:

```python
# 1. Заморозить нижние слои
for name, param in model.named_parameters():
    if 'perceptual_extractor.stem' in name or 'perceptual_extractor.stage1' in name:
        param.requires_grad = False

# 2. Использовать маленький LR
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

# 3. Обучить на новом домене
```

### Протоколы статистического модуля

```python
# Сброс learned prototypes для нового домена
model.statistical_estimator.natural_prototypes.zero_()
model.statistical_estimator.prototypes_initialized.fill_(False)
model.statistical_estimator.prototype_update_count.zero_()
```

---

## Экспорт модели

### Для inference (только веса)

```python
# Сохранение только state_dict
torch.save(model.state_dict(), 'model_weights.pt')

# Загрузка
model = create_tigas_model()
model.load_state_dict(torch.load('model_weights.pt'))
```

### TorchScript

```python
model.eval()

# Scripting
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')

# Tracing (для фиксированного input shape)
example_input = torch.randn(1, 3, 256, 256)
traced = torch.jit.trace(model, example_input)
traced.save('model_traced.pt')
```

### ONNX

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy_input,
    "tigas.onnx",
    input_names=['image'],
    output_names=['score', 'logits'],
    dynamic_axes={
        'image': {0: 'batch_size'},
        'score': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    },
    opset_version=14
)
```

---

## Benchmarking

### Скорость inference

```python
import time
import torch

model = TIGAS(auto_download=True, device='cuda')
model.model.eval()

# Warmup
for _ in range(10):
    _ = model(torch.randn(1, 3, 256, 256).cuda())

# Benchmark
torch.cuda.synchronize()
start = time.time()

n_iterations = 100
for _ in range(n_iterations):
    _ = model(torch.randn(1, 3, 256, 256).cuda())

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Average inference time: {elapsed/n_iterations*1000:.2f} ms")
print(f"Throughput: {n_iterations/elapsed:.1f} images/sec")
```

### Profiling

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    _ = model(torch.randn(1, 3, 256, 256).cuda())

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```
