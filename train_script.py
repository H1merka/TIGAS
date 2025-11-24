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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from tigas.models.tigas_model import create_tigas_model
from tigas.data.loaders import create_dataloaders
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
        help="Размер батча (по умолчанию: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Скорость обучения (по умолчанию: 0.0001)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Количество воркеров для загрузки данных (по умолчанию: 4)"
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
    
    return parser.parse_args()


def main():
    """Основная функция обучения."""
    args = parse_args()
    
    # Проверка существования датасета
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ОШИБКА] Датасет не найден по пути {data_root}")
        print(f"   Убедитесь, что путь правильный и содержит папки 'real' и 'fake'")
        return
    
    real_dir = data_root / 'real'
    fake_dir = data_root / 'fake'
    
    if not real_dir.exists():
        print(f"[ОШИБКА] Папка 'real' не найдена в {data_root}")
        return
    
    if not fake_dir.exists():
        print(f"[ОШИБКА] Папка 'fake' не найдена в {data_root}")
        return
    
    # Подсчет количества изображений
    real_images = list(real_dir.glob('**/*.jpg')) + list(real_dir.glob('**/*.png'))
    fake_images = list(fake_dir.glob('**/*.jpg')) + list(fake_dir.glob('**/*.png'))
    
    print("\n" + "="*60)
    print("ЗАПУСК ОБУЧЕНИЯ TIGAS")
    print("="*60)
    print(f"\n[ДАТАСЕТ] {data_root}")
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
    model = create_tigas_model(
        img_size=args.img_size,
        base_channels=32,
        feature_dim=256,
        num_attention_heads=8,
        dropout=0.1
    )
    
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
    
    # Создание функции потерь
    loss_fn = CombinedLoss(
        use_tigas_loss=True,
        use_contrastive=False,
        tigas_loss_config={
            'regression_weight': 1.0,
            'classification_weight': 0.5,
            'ranking_weight': 0.3,
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
        use_amp=(device == 'cuda'),  # Mixed precision только для GPU
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_interval=10,
        save_interval=5,
        validate_interval=1,
        early_stopping_patience=15,
        use_tensorboard=True
    )
    
    print(f"   [OK] Тренер инициализирован")
    print(f"   [ПУТЬ] Чекпоинты будут сохраняться в: {output_dir.absolute()}")
    
    # Финальная проверка устройства
    if device == 'cuda':
        print(f"   [GPU] Обучение будет выполняться на видеокарте (GPU)")
        print(f"   [GPU] Mixed Precision (AMP) включен для ускорения")
    else:
        print(f"   [CPU] Обучение будет выполняться на процессоре (CPU)")
        print(f"   [CPU] Это будет медленнее, чем на GPU")
    
    # Загрузка чекпоинта, если указан
    if args.resume:
        print(f"\n[ЗАГРУЗКА ЧЕКПОИНТА] {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"   [OK] Чекпоинт загружен (эпоха {start_epoch})")
        except Exception as e:
            print(f"   [ОШИБКА] Ошибка при загрузке чекпоинта: {e}")
            print(f"   [ВНИМАНИЕ] Начинаем обучение с нуля")
            start_epoch = 0
    else:
        start_epoch = 0
    
    # Обучение
    print(f"\n[НАЧАЛО ОБУЧЕНИЯ]...")
    print("="*60 + "\n")
    
    try:
        trainer.train(
            num_epochs=args.epochs,
            resume_from=args.resume
        )
        
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

