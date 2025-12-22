# Установка и настройка

## Системные требования

### Минимальные требования
- **Python**: 3.8+
- **RAM**: 8 GB
- **Диск**: 1 GB (для модели и зависимостей)

### Рекомендуемые требования (для обучения)
- **Python**: 3.10+
- **RAM**: 16+ GB
- **GPU**: NVIDIA с 6+ GB VRAM (CUDA 11.8+)
- **Диск**: 10+ GB (для датасетов)

---

## Установка

### Базовая установка (CPU)

```bash
# Клонирование репозитория
git clone https://github.com/H1merka/TIGAS.git
cd TIGAS

# Установка зависимостей
pip install -r requirements.txt

# Установка пакета в режиме разработки
pip install -e .
```

### Установка с поддержкой CUDA (GPU)

```bash
# Клонирование репозитория
git clone https://github.com/H1merka/TIGAS.git
cd TIGAS

# Установка CUDA-зависимостей
pip install -r requirements_cuda.txt

# Установка пакета
pip install -e .
```

### Установка через pip (будущее)

```bash
# Пока не опубликован в PyPI
# pip install tigas-metric
```

---

## Зависимости

### Основные зависимости

| Пакет | Версия | Назначение |
|-------|--------|------------|
| `torch` | >= 2.2.0 | Основной фреймворк |
| `torchvision` | >= 0.17.0 | Трансформации изображений |
| `numpy` | >= 1.24.0 | Численные операции |
| `scipy` | >= 1.10.0 | Научные вычисления |
| `scikit-learn` | >= 1.3.0 | ML утилиты |
| `Pillow` | >= 10.0.0 | Обработка изображений |
| `opencv-python` | >= 4.8.0 | Компьютерное зрение |
| `pandas` | >= 2.0.0 | Работа с CSV |
| `tqdm` | >= 4.64.0 | Прогресс-бары |

### Опциональные зависимости

| Пакет | Назначение |
|-------|------------|
| `huggingface-hub` | Автозагрузка модели из HF Hub |
| `tensorboard` | Визуализация обучения |
| `matplotlib` | Построение графиков |
| `seaborn` | Визуализация данных |

---

## Проверка установки

### Быстрый тест

```bash
python test/test_package_quick.py
```

Ожидаемый вывод:
```
============================================================
TIGAS Package - Quick Smoke Test
============================================================

[1/6] Testing imports...
  ✓ Main API imports
  ✓ TIGASMetric import
  ✓ Model hub imports
  ✓ Version attribute

[2/6] Testing model creation...
  ✓ TIGASModel creation (fast_mode)
  ✓ TIGASModel creation (full_mode)
  ✓ create_tigas_model factory
...
```

### Полный тест

```bash
python test/test_package_full.py
```

### Проверка CUDA

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

## Настройка окружения

### Создание виртуального окружения

```bash
# С помощью venv
python -m venv tigas_env
source tigas_env/bin/activate  # Linux/Mac
tigas_env\Scripts\activate     # Windows

# С помощью conda
conda create -n tigas python=3.10
conda activate tigas
```

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `TIGAS_CACHE_DIR` | Директория кэша моделей | `~/.cache/tigas/models/` |
| `CUDA_VISIBLE_DEVICES` | GPU для использования | Все доступные |

---

## Windows-специфичные настройки

### DataLoader workers

На Windows рекомендуется `num_workers=0` из-за overhead при создании процессов:

```python
# Автоматически определено в train_script.py
--num_workers 0  # Default для Windows
```

### Кодировка консоли

При проблемах с русскими символами:
```bash
# PowerShell
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8

# cmd.exe
chcp 65001
```

---

## Обновление

```bash
cd TIGAS
git pull origin main
pip install -e . --upgrade
```

---

## Удаление

```bash
# Удаление пакета
pip uninstall tigas-metric

# Очистка кэша моделей
python -c "from tigas import clear_cache; clear_cache()"

# Удаление репозитория
rm -rf TIGAS/
```
