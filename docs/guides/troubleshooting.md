# Решение проблем (Troubleshooting)

## Содержание

- [Ошибки установки](#ошибки-установки)
- [Проблемы с CUDA и GPU](#проблемы-с-cuda-и-gpu)
- [Ошибки обучения](#ошибки-обучения)
- [Проблемы с данными](#проблемы-с-данными)
- [Ошибки inference](#ошибки-inference)
- [Проблемы с памятью](#проблемы-с-памятью)
- [Численная нестабильность](#численная-нестабильность)

---

## Ошибки установки

### `ModuleNotFoundError: No module named 'tigas'`

**Причина:** Пакет не установлен в текущем окружении.

**Решение:**
```bash
# Убедитесь, что активировано нужное окружение
conda activate myenv  # или source venv/bin/activate

# Установите пакет
pip install -e .
```

### `ERROR: Could not find a version that satisfies the requirement torch>=2.2`

**Причина:** PyTorch не найден или версия слишком старая.

**Решение:**
```bash
# CPU версия
pip install torch>=2.2.0 torchvision>=0.17.0

# CUDA 12.1
pip install torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cu118
```

### `ImportError: DLL load failed` (Windows)

**Причина:** Отсутствуют Visual C++ Redistributable.

**Решение:**
1. Скачайте и установите [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Перезагрузите систему
3. Переустановите PyTorch

---

## Проблемы с CUDA и GPU

### `RuntimeError: CUDA out of memory`

**Симптом:**
```
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB
```

**Решения:**

1. **Уменьшить batch size:**
   ```bash
   python scripts/train_script.py --batch_size 8  # вместо 16
   ```

2. **Использовать gradient accumulation:**
   ```bash
   python scripts/train_script.py --batch_size 4 --accumulation_steps 4
   ```

3. **Включить AMP (mixed precision):**
   ```bash
   python scripts/train_script.py --use_amp
   ```

4. **Очистить кэш CUDA:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

5. **Мониторинг памяти:**
   ```python
   print(torch.cuda.memory_summary())
   ```

### `RuntimeError: CUDA error: device-side assert triggered`

**Причина:** Обычно ошибка индексации или неверные метки.

**Решение:**
```bash
# Запустить с CUDA_LAUNCH_BLOCKING для детальной ошибки
CUDA_LAUNCH_BLOCKING=1 python scripts/train_script.py
```

Частые причины:
- Метки вне диапазона [0, 1]
- Неверный размер тензоров
- NaN/Inf в данных

### `torch.cuda.is_available() возвращает False`

**Проверка:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
```

**Решения:**

1. **Проверить драйвер NVIDIA:**
   ```bash
   nvidia-smi
   ```

2. **Проверить совместимость версий:**
   - CUDA Toolkit версия должна соответствовать драйверу
   - PyTorch CUDA версия должна соответствовать системной

3. **Переустановить PyTorch с правильной CUDA версией:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

---

## Ошибки обучения

### `RuntimeError: NaN/Inf detected in scores!`

**Причина:** Численная нестабильность в модели.

**Решения:**

1. **Уменьшить learning rate:**
   ```bash
   python scripts/train_script.py --lr 0.000005
   ```

2. **Отключить AMP:**
   ```bash
   python scripts/train_script.py  # без --use_amp
   ```

3. **Проверить данные:**
   ```python
   from tigas.data import TIGASDataset
   from torch.utils.data import DataLoader
   
   dataset = TIGASDataset('data/', transform=...)
   loader = DataLoader(dataset, batch_size=1)
   
   for images, labels in loader:
       if torch.isnan(images).any():
           print("NaN в изображениях!")
       if torch.isinf(images).any():
           print("Inf в изображениях!")
   ```

4. **Использовать gradient clipping:**
   ```python
   # В TIGASTrainer это включено по умолчанию
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

### `Loss не уменьшается`

**Диагностика:**
```python
# Проверить, что модель обновляется
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.6f}")
```

**Решения:**

1. **Увеличить learning rate:**
   ```bash
   python scripts/train_script.py --lr 0.0001
   ```

2. **Проверить баланс классов:**
   ```python
   import os
   real_count = len(os.listdir('data/real'))
   fake_count = len(os.listdir('data/fake'))
   print(f"Real: {real_count}, Fake: {fake_count}")
   # Должно быть примерно равно
   ```

3. **Увеличить количество эпох:**
   ```bash
   python scripts/train_script.py --epochs 100
   ```

4. **Проверить правильность меток:**
   ```python
   # real/ должен давать label=1.0
   # fake/ должен давать label=0.0
   ```

### `Validation loss растёт (overfitting)`

**Решения:**

1. **Увеличить dropout:**
   ```python
   model = TIGASModel(fast_mode=True)
   # Увеличить dropout в конфиге модели
   ```

2. **Уменьшить размер модели:**
   ```bash
   python scripts/train_script.py --fast_mode  # Меньше параметров
   ```

3. **Добавить аугментацию:**
   ```python
   from tigas.data.transforms import TIGASTransform
   transform = TIGASTransform(augmentation_level='heavy')
   ```

4. **Early stopping (включён по умолчанию):**
   ```bash
   python scripts/train_script.py --patience 10
   ```

---

## Проблемы с данными

### `No images found in directory`

**Проверка:**
```python
import os
from pathlib import Path

data_root = Path('data/')
real_dir = data_root / 'real'
fake_dir = data_root / 'fake'

for ext in ['*.jpg', '*.png', '*.bmp', '*.jpeg']:
    real = list(real_dir.glob(ext))
    fake = list(fake_dir.glob(ext))
    print(f"{ext}: real={len(real)}, fake={len(fake)}")
```

**Решения:**

1. **Проверить структуру директорий:**
   ```
   data/
   ├── real/     # НЕ data/real/images/
   │   └── *.jpg
   └── fake/
       └── *.jpg
   ```

2. **Проверить расширения файлов:**
   - Поддерживаются: `.jpg`, `.jpeg`, `.png`, `.bmp`
   - Case-sensitive на Linux

3. **Проверить права доступа:**
   ```bash
   ls -la data/real/
   ```

### `RuntimeError: stack expects each tensor to be equal size`

**Причина:** Изображения разных размеров.

**Решение:**
```python
from tigas.data.transforms import TIGASTransform

# Transform гарантирует единый размер
transform = TIGASTransform(img_size=256)
dataset = TIGASDataset('data/', transform=transform)
```

### `Некорректные метки`

**Проверка:**
```python
dataset = TIGASDataset('data/')
for i in range(10):
    img, label = dataset[i]
    path = dataset.image_paths[i]
    print(f"{path}: label={label}")
    # real/ → label=1.0
    # fake/ → label=0.0
```

---

## Ошибки inference

### `RuntimeError: Error(s) in loading state_dict`

**Причина:** Несовместимость архитектуры модели.

**Решения:**

1. **Проверить режим модели:**
   ```python
   # Если чекпоинт сохранён в fast_mode=True
   model = TIGASModel(fast_mode=True)
   model.load_state_dict(checkpoint['model_state_dict'])
   
   # Если в full_mode
   model = TIGASModel(fast_mode=False)
   ```

2. **Загрузить с strict=False (осторожно):**
   ```python
   model.load_state_dict(checkpoint['model_state_dict'], strict=False)
   ```

3. **Проверить формат чекпоинта:**
   ```python
   checkpoint = torch.load('model.pt')
   print(checkpoint.keys())
   # Должно быть: ['model_state_dict', 'optimizer_state_dict', ...]
   
   # Или просто state_dict:
   print(type(checkpoint))  # dict с весами
   ```

### `Модель предсказывает всегда одинаковый score`

**Диагностика:**
```python
tigas = TIGAS(auto_download=True)

# Тест на разных изображениях
for img_path in ['real1.jpg', 'real2.jpg', 'fake1.jpg', 'fake2.jpg']:
    score = tigas(img_path)
    print(f"{img_path}: {score.item():.4f}")
```

**Решения:**

1. **Проверить, что модель в eval режиме:**
   ```python
   tigas.model.eval()
   ```

2. **Проверить нормализацию входа:**
   ```python
   from torchvision import transforms
   
   transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(256),
       transforms.ToTensor(),  # → [0, 1]
   ])
   # Модель сама нормализует в [-1, 1]
   ```

3. **Перезагрузить модель:**
   ```python
   tigas = TIGAS(auto_download=True, force_download=True)
   ```

---

## Проблемы с памятью

### `MemoryError` или `Killed` (Linux OOM)

**Решения:**

1. **Уменьшить num_workers:**
   ```bash
   python scripts/train_script.py --num_workers 0
   ```

2. **Использовать pin_memory=False:**
   ```python
   loader = DataLoader(dataset, pin_memory=False)
   ```

3. **Уменьшить batch_size:**
   ```bash
   python scripts/train_script.py --batch_size 4
   ```

4. **Освободить RAM перед обучением:**
   ```python
   import gc
   gc.collect()
   ```

### `DataLoader worker killed`

**Причина:** Нехватка shared memory или слишком много workers.

**Решения:**

1. **Windows - использовать num_workers=0:**
   ```bash
   python scripts/train_script.py --num_workers 0
   ```

2. **Linux - увеличить shared memory:**
   ```bash
   # Docker
   docker run --shm-size=8g ...
   
   # Или использовать tmpfs
   ```

3. **Уменьшить prefetch_factor:**
   ```python
   loader = DataLoader(dataset, prefetch_factor=1)
   ```

---

## Численная нестабильность

### `NaN в attention`

**Симптом:** 
```
Warning: NaN detected in attention, using uniform fallback
```

**Решения:**

1. **Это warning, не error** - модель автоматически использует fallback

2. **Для устранения:**
   ```bash
   # Уменьшить LR
   python scripts/train_script.py --lr 0.000005
   
   # Отключить AMP
   python scripts/train_script.py  # без --use_amp
   ```

### `GradScaler: Overflow detected, skipping update`

**Это нормально при AMP** - scaler автоматически подстраивается.

**Если слишком частое:**
```python
# Уменьшить начальный scale
scaler = torch.amp.GradScaler(
    init_scale=64,  # вместо 128
    growth_factor=1.25  # вместо 1.5
)
```

### `Exploding gradients`

**Диагностика:**
```python
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm}")
# Если > 100, это проблема
```

**Решение:**
```python
# Gradient clipping (включён по умолчанию в TIGASTrainer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Общие рекомендации

### Сбор диагностической информации

```python
import torch
import platform
import sys

print("=== System Info ===")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Минимальный воспроизводимый пример

При создании issue приложите:

```python
# 1. Версии
import torch
print(torch.__version__)

# 2. Код для воспроизведения
from tigas import TIGAS
tigas = TIGAS(auto_download=True)
result = tigas('test_image.jpg')

# 3. Полный traceback ошибки
```

### Логирование для отладки

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Теперь TIGAS будет выводить подробные логи
from tigas import TIGAS
tigas = TIGAS(auto_download=True)
```
