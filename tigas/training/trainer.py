"""
TIGAS Trainer - Main training loop with all bells and whistles.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.amp
from pathlib import Path
from typing import Dict, Optional, Callable
import time
from tqdm import tqdm
import json

from ..models.tigas_model import TIGASModel
from .losses import CombinedLoss
from .optimizers import create_optimizer, create_scheduler


class TIGASTrainer:
    """
    Comprehensive trainer for TIGAS model.

    Features:
    - Mixed precision training
    - Gradient accumulation
    - Checkpoint saving/loading
    - TensorBoard logging
    - Early stopping
    - Learning rate scheduling
    - Validation monitoring
    """

    def __init__(
        self,
        model: TIGASModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './checkpoints',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 0.5,
        log_interval: int = 50,
        save_interval: int = 1,
        validate_interval: int = 5,
        early_stopping_patience: int = 10,
        use_tensorboard: bool = False,
    ):
        """
        Args:
            model: TIGAS model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer_config: Optimizer configuration
            scheduler_config: Scheduler configuration
            device: Device to train on
            output_dir: Directory for checkpoints and logs
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Logging interval (batches)
            save_interval: Checkpoint save interval (epochs)
            validate_interval: Validation interval (epochs)
            early_stopping_patience: Early stopping patience
            use_tensorboard: Use TensorBoard logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        if loss_fn is None:
            self.loss_fn = CombinedLoss()
        else:
            self.loss_fn = loss_fn

        # Optimizer
        optimizer_config = optimizer_config or {}
        self.optimizer = create_optimizer(self.model, **optimizer_config)

        # Scheduler
        scheduler_config = scheduler_config or {}
        scheduler_config['steps_per_epoch'] = len(train_loader)
        self.scheduler = create_scheduler(self.optimizer, **scheduler_config)

        # Training settings
        self.use_amp = use_amp and device == 'cuda'
        # Very conservative GradScaler for numerical stability
        # Lower init_scale + smaller growth_factor prevent overflow
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(
                'cuda',
                init_scale=128.0,      # ← Снижено с 256 для консервативности
                growth_factor=1.5,     # ← Медленный рост (было 2.0 по умолчанию)
                backoff_factor=0.75,   # ← Быстрый откат (было 0.5)
                growth_interval=100    # ← Реже повышать масштаб
            )
        else:
            self.scaler = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Logging and saving
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.validate_interval = validate_interval

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
            except ImportError:
                print("TensorBoard not available. Logging disabled.")
                self.use_tensorboard = False

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.train_history = []
        self.val_history = []

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'regression': 0.0,
            'classification': 0.0,
            'ranking': 0.0
        }
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with AMP using modern torch.amp API
            with torch.amp.autocast('cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                outputs = self.model(
                    images,
                    return_features=False,
                    update_prototypes=True
                )

                # Pre-loss diagnostic: check model outputs for NaN
                if torch.isnan(outputs['score']).any() or torch.isinf(outputs['score']).any():
                    raise RuntimeError(
                        f"[TRAINER] NaN/Inf detected in model score output at batch {batch_idx}\n"
                        f"Score stats: min={outputs['score'].min():.6f}, max={outputs['score'].max():.6f}, "
                        f"mean={outputs['score'].mean():.6f}, std={outputs['score'].std():.6f}\n"
                        f"This indicates an issue in the model forward pass or input data.\n"
                        f"Run: python scripts/check_dataset.py --data_root <dataset>"
                    )

                # Compute losses
                losses = self.loss_fn(outputs, labels, self.model)
                loss = losses['combined_total'] if 'combined_total' in losses else losses['total']

                # CRITICAL: Check for NaN loss and STOP immediately
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(
                        f"[TRAINER] NaN/Inf loss at batch {batch_idx}\n"
                        f"Loss breakdown:\n"
                        f"  - Regression: {losses.get('regression', 'N/A')}\n"
                        f"  - Classification: {losses.get('classification', 'N/A')}\n"
                        f"  - Ranking: {losses.get('ranking', 'N/A')}\n"
                        f"  - Total: {loss}\n"
                        f"Probable cause: Corrupted image(s) in this batch (batch_idx={batch_idx})\n"
                        f"Image indices in batch: {batch_idx * self.train_loader.batch_size} to "
                        f"{min((batch_idx + 1) * self.train_loader.batch_size, len(self.train_loader.dataset))}\n"
                        f"Action: Run check_dataset.py to validate and fix the dataset"
                    )

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key] += losses[key].item()

            num_batches += 1

            # Logging
            if batch_idx % self.log_interval == 0:
                current_loss = losses.get('combined_total', losses['total']).item()
                progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

                if self.use_tensorboard:
                    self.writer.add_scalar('train/batch_loss', current_loss, self.global_step)

        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = {
            'total': 0.0,
            'regression': 0.0,
            'classification': 0.0
        }
        num_batches = 0

        # Additional metrics
        correct_predictions = 0
        total_samples = 0

        for images, labels in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images, return_features=False)

            # Compute losses
            losses = self.loss_fn(outputs, labels)

            # Accumulate losses
            for key in val_losses.keys():
                if key in losses:
                    val_losses[key] += losses[key].item()

            # Compute accuracy
            predictions = (outputs['score'] > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            num_batches += 1

        # Average losses
        for key in val_losses.keys():
            val_losses[key] /= num_batches

        # Accuracy
        val_losses['accuracy'] = correct_predictions / total_samples

        return val_losses

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint with logging and error handling."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        checkpoint_path_str = str(checkpoint_path)

        try:
            # атомарная запись: сначала во временный файл, затем переименование
            temp_path = checkpoint_path.with_suffix('.pt.tmp')
            torch.save(checkpoint, str(temp_path))
            temp_path.replace(checkpoint_path)  # атомарно переименовать
            print(f"[checkpoint] saved: {checkpoint_path_str}")
        except Exception as e:
            print(f"[checkpoint] failed to save {checkpoint_path_str}: {e}", flush=True)

        if is_best:
            try:
                best_path = self.output_dir / 'best_model.pt'
                torch.save(checkpoint, str(best_path))
                print(f"[checkpoint] best saved: {best_path}")
            except Exception as e:
                print(f"[checkpoint] failed to save best model: {e}", flush=True)

        try:
            latest_path = self.output_dir / 'latest_model.pt'
            torch.save(checkpoint, str(latest_path))
        except Exception as e:
            print(f"[checkpoint] failed to save latest model: {e}", flush=True)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        reset_lr: bool = False,
        reset_scheduler: bool = False,
        new_lr: Optional[float] = None
    ):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            reset_lr: If True, don't restore LR from checkpoint
            reset_scheduler: If True, don't restore scheduler state
            new_lr: If provided, set this LR after loading
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False  # Явно указываем для подавления FutureWarning
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Восстановление scheduler (если не сброшен)
        if not reset_scheduler and checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif reset_scheduler:
            print("[resume] Scheduler state reset (starting fresh)")

        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        # Установка нового LR (если указан или reset_lr)
        if new_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"[resume] Learning rate set to: {new_lr}")
        elif reset_lr:
            # Сохраняем текущий LR из optimizer (был установлен при создании)
            print("[resume] Learning rate kept from config (not from checkpoint)")

        loaded_lr = self.optimizer.param_groups[0]['lr']
        print(f"Loaded checkpoint from epoch {self.current_epoch} (LR: {loaded_lr:.6f})")

    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None,
        reset_lr: bool = False,
        reset_scheduler: bool = False,
        new_lr: Optional[float] = None
    ):
        """
        Main training loop.

        Args:
            num_epochs: Number of ADDITIONAL epochs to train (not total!)
            resume_from: Path to checkpoint to resume from
            reset_lr: If True, don't restore LR from checkpoint
            reset_scheduler: If True, don't restore scheduler state  
            new_lr: If provided, set this LR after loading checkpoint
        """
        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(
                resume_from,
                reset_lr=reset_lr,
                reset_scheduler=reset_scheduler,
                new_lr=new_lr
            )

        # Вычисляем конечную эпоху
        start_epoch = self.current_epoch
        if resume_from:
            start_epoch += 1  # Начинаем со следующей эпохи
            end_epoch = start_epoch + num_epochs  # num_epochs = сколько ЕЩЁ эпох
            print(f"Resuming from epoch {self.current_epoch}, will train {num_epochs} more epochs")
            print(f"Epochs: {start_epoch} -> {end_epoch - 1}")
        else:
            end_epoch = num_epochs  # Без resume: num_epochs = всего эпох
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()
            self.train_history.append(train_losses)

            # Log training
            print(f"\nEpoch {epoch} - Train Loss: {train_losses['total']:.4f}")
            
            # Update current_epoch только после завершения эпохи
            # (для корректного resume если прервётся во время validation/save)
            self.current_epoch = epoch
            if self.use_tensorboard:
                for key, value in train_losses.items():
                    self.writer.add_scalar(f'train/{key}', value, epoch)

            # Validate
            if (epoch + 1) % self.validate_interval == 0:
                val_losses = self.validate()
                if val_losses:
                    self.val_history.append(val_losses)
                    print(f"Val Loss: {val_losses['total']:.4f}, Accuracy: {val_losses.get('accuracy', 0):.4f}")

                    if self.use_tensorboard:
                        for key, value in val_losses.items():
                            self.writer.add_scalar(f'val/{key}', value, epoch)

                    # Early stopping
                    if val_losses['total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.patience_counter = 0
                        self.save_checkpoint(is_best=True)
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(is_best=False)

            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_losses:
                        self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()

            # Log LR
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
            if self.use_tensorboard:
                self.writer.add_scalar('train/learning_rate', current_lr, epoch)

        # Save final checkpoint
        self.save_checkpoint(is_best=False)

        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        if self.use_tensorboard:
            self.writer.close()
