"""
Data loaders for TIGAS training.
"""

import torch
from torch.utils.data import DataLoader, random_split, Subset
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import numpy as np

from .dataset import TIGASDataset, RealFakeDataset, PairedDataset, CSVDataset
from .transforms import get_train_transforms, get_val_transforms


def stratified_split(
    dataset: TIGASDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/val/test split indices.
    
    Ensures each split has approximately the same ratio of real/fake images
    as the full dataset.
    
    Args:
        dataset: TIGASDataset with .samples attribute
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Use local RNG to avoid race conditions in multi-threaded environments
    rng = np.random.default_rng(seed)
    
    # Separate indices by class
    real_indices = []
    fake_indices = []
    
    for idx, (path, label) in enumerate(dataset.samples):
        if label == 1.0:
            real_indices.append(idx)
        else:
            fake_indices.append(idx)
    
    # Shuffle within each class using local RNG
    rng.shuffle(real_indices)
    rng.shuffle(fake_indices)
    
    def split_indices(indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """Split a list of indices according to ratios."""
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (
            indices[:train_end],
            indices[train_end:val_end],
            indices[val_end:]
        )
    
    # Split each class
    real_train, real_val, real_test = split_indices(real_indices)
    fake_train, fake_val, fake_test = split_indices(fake_indices)
    
    # Combine and shuffle using local RNG
    train_indices = real_train + fake_train
    val_indices = real_val + fake_val
    test_indices = real_test + fake_test
    
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    
    return train_indices, val_indices, test_indices


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 12,
    train_split: float = 0.8,
    val_split: float = 0.1,
    augment_level: str = 'medium',
    pin_memory: bool = True,
    shuffle: bool = True,
    stratified: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders with stratified splitting.

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
        stratified: Whether to use stratified splitting (preserves class ratios)

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Create transforms
    train_transform = get_train_transforms(img_size, augment_level=augment_level)
    val_transform = get_val_transforms(img_size)
    
    # Create full dataset with training transforms (we'll override for val/test)
    full_dataset = TIGASDataset(root=data_root, transform=train_transform)

    if stratified:
        # Stratified split - preserves class balance
        train_indices, val_indices, test_indices = stratified_split(
            full_dataset,
            train_ratio=train_split,
            val_ratio=val_split,
            seed=42
        )
        
        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        
        # For val/test, create new dataset with val transforms
        val_dataset_base = TIGASDataset(root=data_root, transform=val_transform)
        val_dataset = Subset(val_dataset_base, val_indices)
        test_dataset = Subset(val_dataset_base, test_indices)
        
        # Log class distribution
        train_real = sum(1 for i in train_indices if full_dataset.samples[i][1] == 1.0)
        val_real = sum(1 for i in val_indices if full_dataset.samples[i][1] == 1.0)
        test_real = sum(1 for i in test_indices if full_dataset.samples[i][1] == 1.0)
        
        print(f"Stratified split statistics:")
        print(f"  Train: {len(train_indices)} samples ({train_real} real, {len(train_indices)-train_real} fake)")
        print(f"  Val: {len(val_indices)} samples ({val_real} real, {len(val_indices)-val_real} fake)")
        print(f"  Test: {len(test_indices)} samples ({test_real} real, {len(test_indices)-test_real} fake)")
    else:
        # Random split (original behavior)
        total_size = len(full_dataset)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Random split (non-stratified):")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

    # NOTE: persistent_workers can cause memory issues with caching datasets
    # Only enable if num_workers > 0 and dataset doesn't use cache
    use_persistent = num_workers > 0 and not getattr(full_dataset, 'use_cache', False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None
    )

    print(f"Created dataloaders with {len(train_loader)} train batches")

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


def create_dataloaders_from_csv(
    data_root: str,
    train_csv: str = 'train/annotations01.csv',
    val_csv: str = 'val/annotations01.csv',
    test_csv: str = 'test/annotations01.csv',
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 12,
    augment_level: str = 'medium',
    pin_memory: bool = True,
    shuffle: bool = True,
    use_cache: bool = False,
    validate_paths: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders from CSV annotation files.

    Designed for datasets with pre-defined splits in CSV format.
    This is ideal for TIGAS dataset structure where train/val/test
    are already separated with CSV annotations.

    Args:
        data_root: Root directory containing CSV files and images
                  E.g., 'C:/Dev/dataset/dataset/TIGAS_dataset/TIGAS'
        train_csv: Path to training CSV (relative to data_root or absolute)
        val_csv: Path to validation CSV (relative to data_root or absolute)
        test_csv: Path to test CSV (relative to data_root or absolute)
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of worker processes
        augment_level: Augmentation level for training ('light', 'medium', 'heavy')
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle training data
        use_cache: Whether to cache loaded images in memory (use with caution for large datasets)
        validate_paths: Whether to validate all image paths exist (slower but safer)

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders

    Example:
        >>> dataloaders = create_dataloaders_from_csv(
        ...     data_root='C:/Dev/dataset/dataset/TIGAS_dataset/TIGAS',
        ...     batch_size=16,
        ...     img_size=256
        ... )
        >>> train_loader = dataloaders['train']
    """
    data_root = Path(data_root)

    # Get transforms
    train_transform = get_train_transforms(img_size, augment_level=augment_level)
    val_transform = get_val_transforms(img_size)

    # Create datasets
    print("\n" + "="*60)
    print("Creating dataloaders from CSV files")
    print("="*60)

    print("\n[1/3] Loading training dataset...")
    train_dataset = CSVDataset(
        csv_file=train_csv,
        root_dir=str(data_root),
        transform=train_transform,
        use_cache=use_cache,
        validate_paths=validate_paths
    )

    print("\n[2/3] Loading validation dataset...")
    val_dataset = CSVDataset(
        csv_file=val_csv,
        root_dir=str(data_root),
        transform=val_transform,
        use_cache=use_cache,
        validate_paths=validate_paths
    )

    print("\n[3/3] Loading test dataset...")
    test_dataset = CSVDataset(
        csv_file=test_csv,
        root_dir=str(data_root),
        transform=val_transform,
        use_cache=use_cache,
        validate_paths=validate_paths
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory if num_workers > 0 else False,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Print summary
    print("\n" + "="*60)
    print("Dataloaders created successfully!")
    print("="*60)
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Val:   {len(val_dataset):,} samples, {len(val_loader):,} batches")
    print(f"  Test:  {len(test_dataset):,} samples, {len(test_loader):,} batches")
    print("="*60 + "\n")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
