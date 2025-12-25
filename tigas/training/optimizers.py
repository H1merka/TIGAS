"""
Optimizer and scheduler configurations for TIGAS training.
"""

import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    OneCycleLR
)
from typing import Optional


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: tuple = (0.9, 0.999),
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model: Model to optimize
        optimizer_type: 'adam', 'adamw', or 'sgd'
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        momentum: Momentum for SGD
        betas: Betas for Adam/AdamW
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to biases and normalization layers
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    if optimizer_type.lower() == 'adam':
        optimizer = Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_epochs: int = 100,
    steps_per_epoch: Optional[int] = None,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
    **kwargs
) -> Optional[object]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_type: 'cosine', 'step', 'plateau', 'onecycle', or 'none'
        num_epochs: Total number of epochs
        steps_per_epoch: Steps per epoch (for OneCycleLR)
        min_lr: Minimum learning rate
        warmup_epochs: Warmup epochs
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance or None
    """
    if scheduler_type.lower() == 'none':
        return None

    if scheduler_type.lower() == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr,
            **kwargs
        )
    elif scheduler_type.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type.lower() == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=min_lr,
            **kwargs
        )
    elif scheduler_type.lower() == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")

        max_lr = optimizer.param_groups[0]['lr']
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Wrap with warmup if needed
    if warmup_epochs > 0 and scheduler_type != 'onecycle':
        scheduler = WarmupScheduler(
            optimizer,
            scheduler,
            warmup_epochs=warmup_epochs
        )

    return scheduler


class WarmupScheduler:
    """
    Learning rate warmup scheduler.
    Gradually increases LR from 0 to target LR over warmup_epochs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler: object,
        warmup_epochs: int = 5
    ):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Store initial LRs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase LR
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # Normal scheduling
            if hasattr(self.base_scheduler, 'step'):
                self.base_scheduler.step()

    def state_dict(self):
        """Return state dict."""
        return {
            'current_epoch': self.current_epoch,
            'base_scheduler': self.base_scheduler.state_dict() if hasattr(self.base_scheduler, 'state_dict') else None
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.current_epoch = state_dict['current_epoch']
        if state_dict['base_scheduler'] is not None and hasattr(self.base_scheduler, 'load_state_dict'):
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
