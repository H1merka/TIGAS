# TIGAS Documentation

## Trained Image Generation Authenticity Score

TIGAS — нейросетевая метрика для детектирования AI-сгенерированных изображений. Выдаёт непрерывную оценку [0, 1], где 1.0 = реальное изображение, 0.0 = сгенерированное.

## Содержание документации

### Начало работы
- [Установка и настройка](installation.md)
- [Быстрый старт](quickstart.md)

### API Reference
- [Python API](api/python-api.md)
- [Командная строка (CLI)](api/cli.md)
- [Классы и функции](api/reference.md)

### Обучение модели
- [Подготовка данных](training/data-preparation.md)
- [Конфигурация обучения](training/configuration.md)
- [Запуск и мониторинг](training/running.md)
- [Продвинутые техники](training/advanced.md)

### Архитектура
- [Обзор архитектуры](architecture/overview.md)
- [Компоненты модели](architecture/components.md)
- [Функции потерь](architecture/losses.md)

### Руководства
- [Интеграция как Loss-функция](guides/loss-function.md)
- [HuggingFace Hub](guides/huggingface.md)
- [Устранение неполадок](guides/troubleshooting.md)

---

## Ключевые особенности

| Функция | Описание |
|---------|----------|
| **Дифференцируемость** | Может использоваться как loss-функция для обучения генеративных моделей |
| **Мультимодальный анализ** | Объединяет перцептивные, спектральные и статистические признаки |
| **Два режима** | Fast Mode для обучения, Full Mode для максимальной точности |
| **Auto-download** | Автоматическая загрузка модели из HuggingFace Hub |
| **Windows-оптимизация** | Настройки по умолчанию оптимизированы для Windows |

## Быстрый пример

```python
from tigas import TIGAS

# Автозагрузка модели из HuggingFace Hub
tigas = TIGAS(auto_download=True)

# Оценка изображения
score = tigas('image.jpg')
print(f"Authenticity: {score:.4f}")  # 1.0 = real, 0.0 = fake

# Как loss-функция для генеративных моделей
loss = 1.0 - tigas(generated_images).mean()
loss.backward()
```

## Ссылки

- **GitHub**: [H1merka/TIGAS](https://github.com/H1merka/TIGAS)
- **HuggingFace**: [H1merka/TIGAS](https://huggingface.co/H1merka/TIGAS)
- **Лицензия**: MIT
