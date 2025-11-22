"""
Data loaders for TIGAS training.
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional, Dict
from pathlib import Path

from .dataset import TIGASDataset, RealFakeDataset, PairedDataset
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    augment_level: str = 'medium',
    pin_memory: bool = True,
    shuffle: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        data_root: Root directory with real/fake subdirectories
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of worker processes
        train_split: Fraction for training
        val_split: Fraction for validation
        augment_level: Augmentation level ('light', 'medium', 'heavy')
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle training data

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Create full dataset
    full_transform = get_train_transforms(img_size, augment_level=augment_level)
    full_dataset = TIGASDataset(root=data_root, transform=full_transform)

    # Calculate splits
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_inference_loader(
    image_dir: str,
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 4
) -> DataLoader:
    """Create dataloader for inference."""
    from .transforms import get_inference_transforms
    from torch.utils.data import Dataset
    from PIL import Image

    class InferenceDataset(Dataset):
        def __init__(self, image_dir, transform):
            self.image_paths = list(Path(image_dir).glob('**/*'))
            self.image_paths = [
                p for p in self.image_paths
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ]
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, str(self.image_paths[idx])

    transform = get_inference_transforms(img_size)
    dataset = InferenceDataset(image_dir, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return loader
