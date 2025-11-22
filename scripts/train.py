"""
Training script for TIGAS model.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --data_root /path/to/data --num_epochs 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from tigas.models.tigas_model import create_tigas_model
from tigas.data.loaders import create_dataloaders
from tigas.training.trainer import TIGASTrainer
from tigas.training.losses import CombinedLoss
from tigas.utils.config import load_config, get_default_config, save_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train TIGAS model")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory with real/fake subdirectories")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")

    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (YAML or JSON)")

    # Model
    parser.add_argument("--img_size", type=int, default=256,
                        help="Input image size")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base number of channels")

    # Training
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = get_default_config()
        print("Using default config")

    # Override config with command-line args
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Data root: {args.data_root}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Image size: {config['model']['img_size']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")

    # Save config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    # Create model
    print("\nCreating model...")
    model = create_tigas_model(
        img_size=config['model']['img_size'],
        base_channels=config['model']['base_channels'],
        feature_dim=config['model']['feature_dim'],
        num_attention_heads=config['model']['num_attention_heads'],
        dropout=config['model']['dropout']
    )

    model_info = model.get_model_size()
    print(f"Model created:")
    print(f"  Total parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"  Model size: {model_info['model_size_mb']:.2f} MB")

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        data_root=args.data_root,
        batch_size=config['training']['batch_size'],
        img_size=config['model']['img_size'],
        num_workers=args.num_workers,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        augment_level=config['data']['augment_level']
    )

    # Create loss function
    loss_fn = CombinedLoss(
        use_tigas_loss=True,
        use_contrastive=False,
        tigas_loss_config=config['loss']
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = TIGASTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        loss_fn=loss_fn,
        optimizer_config={
            'optimizer_type': config['training']['optimizer'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay']
        },
        scheduler_config={
            'scheduler_type': config['training']['scheduler'],
            'num_epochs': config['training']['num_epochs'],
            'warmup_epochs': config['training']['warmup_epochs']
        },
        device=device,
        output_dir=args.output_dir,
        use_amp=config['training']['use_amp'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        log_interval=config['logging']['log_interval'],
        save_interval=config['logging']['save_interval'],
        validate_interval=config['logging']['validate_interval'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        use_tensorboard=config['logging']['use_tensorboard']
    )

    # Train
    print("\nStarting training...\n")
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        resume_from=args.resume
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
