import argparse
import sys
import os
from pathlib import Path

# Настройка кодировки для Windows
if sys.platform == 'win32':
    try:
        # Пытаемся установить UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        # Если не получается, используем cp866 для консоли Windows
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='cp866')

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from tigas.models.tigas_model import create_tigas_model
from tigas.data.loaders import create_dataloaders, create_dataloaders_from_csv
from tigas.training.trainer import TIGASTrainer
from tigas.training.losses import CombinedLoss


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучение модели TIGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Обязательные аргументы
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Путь к датасету с папками 'real' и 'fake'"
    )
    
    # Опциональные аргументы - модель
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Размер входных изображений (по умолчанию: 256)"
    )
    
    # Опциональные аргументы - обучение
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Количество эпох обучения (по умолчанию: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Размер батча (по умолчанию: 16 для стабильности)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0000125,
        help="Скорость обучения (по умолчанию: 0.0000125, очень консервативная для AMP стабильности)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,  # Windows: 0 быстрее из-за spawn overhead
        help="Количество воркеров для загрузки данных (по умолчанию: 0 для Windows)"
    )
    
    # Опциональные аргументы - выход
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Папка для сохранения чекпоинтов (по умолчанию: ./checkpoints)"
    )
    
    # Опциональные аргументы - возобновление
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Путь к чекпоинту для продолжения обучения"
    )
    
    # Опциональные аргументы - устройство
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="Устройство для обучения (по умолчанию: автоматически)"
    )

    # Опциональные аргументы - режим CSV
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="Использовать CSV файлы аннотаций вместо структуры real/fake"
    )

    # Опциональные аргументы - смешанная точность
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Использовать Automatic Mixed Precision (по умолчанию: False для стабильности)"
    )
    parser.add_argument(
        "--no_use_amp",
        action="store_false",
        dest="use_amp",
        help="Отключить Automatic Mixed Precision"
    )

    # Опциональные аргументы - оптимизация памяти
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="Использовать gradient checkpointing для сохранения памяти (медленнее)"
    )
    
    # Опциональные аргументы - режим модели
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        default=True,
        help="Использовать быструю архитектуру модели (по умолчанию: True)"
    )
    parser.add_argument(
        "--full_mode",
        action="store_true",
        help="Использовать полную архитектуру модели (медленнее, но точнее)"
    )

    # Опциональные аргументы - Loss weights (критично важно!)
    parser.add_argument(
        "--regression_weight",
        type=float,
        default=1.0,
        help="Вес regression loss (по умолчанию: 1.0, основной loss)"
    )
    parser.add_argument(
        "--classification_weight",
        type=float,
        default=0.3,
        help="Вес classification loss (по умолчанию: 0.3, уменьшено с 0.5)"
    )
    parser.add_argument(
        "--ranking_weight",
        type=float,
        default=0.2,
        help="Вес ranking loss (по умолчанию: 0.2, уменьшено с 0.3)"
    )

    return parser.parse_args()


def main():
    """Основная функция обучения."""
    args = parse_args()
    
    # Проверка существования датасета
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ОШИБКА] Датасет не найден по пути {data_root}")
        return

    # Валидация структуры датасета в зависимости от режима
    if args.use_csv:
        # CSV режим - проверяем структуру train/val/test и CSV файлы
        train_dir = data_root / 'train'
        val_dir = data_root / 'val'
        test_dir = data_root / 'test'

        train_csv = train_dir / 'annotations01.csv'
        val_csv = val_dir / 'annotations01.csv'
        test_csv = test_dir / 'annotations01.csv'

        missing_dirs = []
        if not train_dir.exists():
            missing_dirs.append('train')
        if not val_dir.exists():
            missing_dirs.append('val')
        if not test_dir.exists():
            missing_dirs.append('test')

        if missing_dirs:
            print(f"[ОШИБКА] Не найдены директории: {', '.join(missing_dirs)}")
            print(f"   CSV режим требует структуру: train/, val/, test/")
            return

        missing_csvs = []
        if not train_csv.exists():
            missing_csvs.append('train/annotations01.csv')
        if not val_csv.exists():
            missing_csvs.append('val/annotations01.csv')
        if not test_csv.exists():
            missing_csvs.append('test/annotations01.csv')

        if missing_csvs:
            print(f"[ОШИБКА] Не найдены CSV файлы: {', '.join(missing_csvs)}")
            return

        # Подсчет количества записей в CSV
        import pandas as pd
        train_count = len(pd.read_csv(train_csv))
        val_count = len(pd.read_csv(val_csv))
        test_count = len(pd.read_csv(test_csv))
        total_count = train_count + val_count + test_count

        print("\n" + "="*60)
        print("ЗАПУСК ОБУЧЕНИЯ TIGAS")
        print("="*60)
        print(f"\n[ДАТАСЕТ] {data_root}")
        print(f"   [РЕЖИМ] CSV аннотации")
        print(f"   |- Train: {train_count:,} изображений")
        print(f"   |- Val: {val_count:,} изображений")
        print(f"   |- Test: {test_count:,} изображений")
        print(f"   `- Всего: {total_count:,} изображений")
    else:
        # Директории режим - проверяем папки real/fake
        real_dir = data_root / 'real'
        fake_dir = data_root / 'fake'

        if not real_dir.exists():
            print(f"[ОШИБКА] Папка 'real' не найдена в {data_root}")
            print(f"   Для CSV режима используйте флаг --use_csv")
            return

        if not fake_dir.exists():
            print(f"[ОШИБКА] Папка 'fake' не найдена в {data_root}")
            print(f"   Для CSV режима используйте флаг --use_csv")
            return

        # Подсчет количества изображений
        real_images = list(real_dir.glob('**/*.jpg')) + list(real_dir.glob('**/*.png'))
        fake_images = list(fake_dir.glob('**/*.jpg')) + list(fake_dir.glob('**/*.png'))

        print("\n" + "="*60)
        print("ЗАПУСК ОБУЧЕНИЯ TIGAS")
        print("="*60)
        print(f"\n[ДАТАСЕТ] {data_root}")
        print(f"   [РЕЖИМ] Структура real/fake")
        print(f"   |- Real изображений: {len(real_images)}")
        print(f"   |- Fake изображений: {len(fake_images)}")
        print(f"   `- Всего: {len(real_images) + len(fake_images)}")
    
    # Определение устройства
    cuda_available = torch.cuda.is_available()
    
    if args.device:
        device = args.device
        if device == 'cuda' and not cuda_available:
            print(f"\n[ПРЕДУПРЕЖДЕНИЕ] CUDA запрошена, но недоступна!")
            print(f"   Переключаемся на CPU")
            device = 'cpu'
    else:
        # Автоматически выбираем GPU, если доступен
        device = 'cuda' if cuda_available else 'cpu'
    
    print(f"\n[УСТРОЙСТВО] {device.upper()}")
    if device == 'cuda' and cuda_available:
        print(f"   [OK] GPU обнаружен:")
        print(f"   |- Название: {torch.cuda.get_device_name(0)}")
        print(f"   |- Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   |- CUDA версия: {torch.version.cuda}")
        print(f"   `- Обучение будет выполняться на GPU (быстро)")
    else:
        print(f"   [ВНИМАНИЕ] GPU недоступен, используется CPU")
        print(f"   |- Обучение будет медленнее")
        if not cuda_available:
            print(f"   |- CUDA не установлена или драйверы не найдены")
        print(f"   `- Рекомендуется использовать GPU для ускорения")
    
    # Параметры обучения
    print(f"\n[ПАРАМЕТРЫ ОБУЧЕНИЯ]")
    print(f"   |- Эпох: {args.epochs}")
    print(f"   |- Размер батча: {args.batch_size}")
    print(f"   |- Скорость обучения: {args.lr}")
    print(f"   |- Размер изображений: {args.img_size}x{args.img_size}")
    print(f"   `- Воркеров: {args.num_workers}")
    
    # Создание модели
    print(f"\n[СОЗДАНИЕ МОДЕЛИ]...")
    fast_mode = not args.full_mode  # fast_mode по умолчанию, если не указан --full_mode
    model = create_tigas_model(
        img_size=args.img_size,
        base_channels=32,
        feature_dim=256,
        num_attention_heads=8,
        dropout=0.1,
        fast_mode=fast_mode
    )
    print(f"   [РЕЖИМ] {'FAST (оптимизированный)' if fast_mode else 'FULL (все ветви)'}")
    
    # Переносим модель на устройство сразу
    model = model.to(device)
    
    model_info = model.get_model_size()
    print(f"   [OK] Модель создана:")
    print(f"      |- Параметров: {model_info['total_parameters']:,}")
    print(f"      |- Обучаемых: {model_info['trainable_parameters']:,}")
    print(f"      |- Размер: {model_info['model_size_mb']:.2f} MB")
    
    # Проверяем, что модель на правильном устройстве
    if device == 'cuda':
        next_param_device = next(model.parameters()).device
        if next_param_device.type == 'cuda':
            print(f"      `- Модель на GPU: {next_param_device}")
        else:
            print(f"      `- [ОШИБКА] Модель не на GPU!")
    else:
        print(f"      `- Модель на CPU")
    
    # Создание даталоадеров
    print(f"\n[СОЗДАНИЕ ДАТАЛОАДЕРОВ]...")
    try:
        if args.use_csv:
            # Режим CSV - загрузка из файлов аннотаций
            print(f"   [РЕЖИМ] Использование CSV аннотаций")
            dataloaders = create_dataloaders_from_csv(
                data_root=str(data_root),
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_workers=args.num_workers,
                augment_level='medium'
            )
        else:
            # Режим директорий - традиционная структура real/fake
            print(f"   [РЕЖИМ] Использование структуры real/fake")
            dataloaders = create_dataloaders(
                data_root=str(data_root),
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_workers=args.num_workers,
                train_split=0.8,
                val_split=0.1,
                augment_level='medium'
            )
        print(f"   [OK] Даталоадеры созданы")
    except Exception as e:
        print(f"   [ОШИБКА] Ошибка при создании даталоадеров: {e}")
        return
    
    # Создание функции потерь с настраиваемыми весами
    print(f"\n[КОНФИГУРАЦИЯ LOSS FUNCTION]...")
    print(f"   Regression weight:     {args.regression_weight}")
    print(f"   Classification weight: {args.classification_weight}")
    print(f"   Ranking weight:        {args.ranking_weight}")
    
    loss_fn = CombinedLoss(
        use_tigas_loss=True,
        use_contrastive=False,
        tigas_loss_config={
            'regression_weight': args.regression_weight,
            'classification_weight': args.classification_weight,
            'ranking_weight': args.ranking_weight,
            'use_smooth_l1': True,
            'margin': 0.5
        }
    )
    
    # Создание тренера
    print(f"\n[ИНИЦИАЛИЗАЦИЯ ТРЕНЕРА]...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = TIGASTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        loss_fn=loss_fn,
        optimizer_config={
            'optimizer_type': 'adamw',
            'learning_rate': args.lr,
            'weight_decay': 0.0001
        },
        scheduler_config={
            'scheduler_type': 'cosine',
            'num_epochs': args.epochs,
            'warmup_epochs': 5,
            'min_lr': 0.000001
        },
        device=device,
        output_dir=str(output_dir),
        use_amp=args.use_amp and (device == 'cuda'),  # AMP только на GPU и если включено
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_interval=50,  # Реже логировать для скорости
        save_interval=5,
        validate_interval=1,
        early_stopping_patience=15,
        use_tensorboard=True
    )
    
    print(f"   [OK] Тренер инициализирован")
    print(f"   [ПУТЬ] Чекпоинты будут сохраняться в: {output_dir.absolute()}")
    
    # Pre-training dataset check (optional but recommended)
    print(f"\n[РЕКОМЕНДАЦИЯ] Перед началом обучения рекомендуется проверить датасет:")
    print(f"   python scripts/validate_dataset.py --dataset_dir {args.data_root}")
    print(f"   Это займёт ~5-10 минут и помогает избежать NaN ошибок позже")
    
    # Финальная проверка устройства
    if device == 'cuda':
        print(f"\n   [GPU] Обучение будет выполняться на видеокарте (GPU)")
        if args.use_amp:
            print(f"   [GPU] Mixed Precision (AMP) включен для ускорения")
        else:
            print(f"   [GPU] Mixed Precision (AMP) отключен (будет медленнее)")
    else:
        print(f"\n   [CPU] Обучение будет выполняться на процессоре (CPU)")
        print(f"   [CPU] Это будет медленнее, чем на GPU")
    
    # Resume info (загрузка произойдёт внутри trainer.train())
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n[RESUME] Будет загружен чекпоинт: {args.resume}")
        else:
            print(f"\n[WARNING] Чекпоинт не найден: {args.resume}")
            print(f"   Начинаем обучение с нуля")
            args.resume = None  # Сбросить чтобы не передавать в train()
    
    # Обучение
    print(f"\n[НАЧАЛО ОБУЧЕНИЯ]...")
    print("="*60 + "\n")
    
    try:
        trainer.train(num_epochs=args.epochs, resume_from=args.resume)
        
        print("\n" + "="*60)
        print("[УСПЕХ] ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("="*60)
        print(f"\n[РЕЗУЛЬТАТЫ] Сохранены в: {output_dir.absolute()}")
        print(f"   |- best_model.pt - лучшая модель")
        print(f"   |- latest_model.pt - последняя модель")
        print(f"   `- logs/ - логи TensorBoard")
        
    except KeyboardInterrupt:
        print("\n\n[ПРЕРВАНО] Обучение прервано пользователем")
        print(f"[СОХРАНЕНО] Текущее состояние сохранено в: {output_dir / 'latest_model.pt'}")
    except Exception as e:
        print(f"\n\n[ОШИБКА] Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

