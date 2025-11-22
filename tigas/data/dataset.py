"""
Dataset classes for TIGAS training.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Optional, Callable, Tuple, List
import json
import random


class TIGASDataset(Dataset):
    """
    Base dataset for TIGAS training.
    Loads images with real/fake labels.

    Expected directory structure:
    root/
        real/
            img1.jpg
            img2.jpg
            ...
        fake/
            img1.jpg
            img2.jpg
            ...
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        split: str = 'train',
        use_cache: bool = False
    ):
        """
        Args:
            root: Root directory containing 'real' and 'fake' subdirectories
            transform: Image transformations
            split: 'train', 'val', or 'test'
            use_cache: Whether to cache images in memory
        """
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.use_cache = use_cache

        # Find all images
        self.samples = []
        self.cache = {} if use_cache else None

        # Real images (label = 1.0)
        real_dir = self.root / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), 1.0))

        # Fake images (label = 0.0)
        fake_dir = self.root / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), 0.0))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root}")

        print(f"Loaded {len(self.samples)} images for {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Load from cache if available
        if self.use_cache and img_path in self.cache:
            img = self.cache[img_path]
        else:
            img = Image.open(img_path).convert('RGB')
            if self.use_cache:
                self.cache[img_path] = img

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor([label], dtype=torch.float32)

        return img, label


class RealFakeDataset(Dataset):
    """
    Dataset with explicit real/fake lists.
    More flexible than TIGASDataset.
    """

    def __init__(
        self,
        real_images: List[str],
        fake_images: List[str],
        transform: Optional[Callable] = None,
        balance: bool = True
    ):
        """
        Args:
            real_images: List of paths to real images
            fake_images: List of paths to fake images
            transform: Image transformations
            balance: Whether to balance real/fake samples
        """
        self.transform = transform

        # Create samples list
        real_samples = [(path, 1.0) for path in real_images]
        fake_samples = [(path, 0.0) for path in fake_images]

        # Balance if requested
        if balance:
            min_len = min(len(real_samples), len(fake_samples))
            real_samples = random.sample(real_samples, min_len)
            fake_samples = random.sample(fake_samples, min_len)

        self.samples = real_samples + fake_samples
        random.shuffle(self.samples)

        print(f"Dataset: {len(real_samples)} real, {len(fake_samples)} fake images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor([label], dtype=torch.float32)

        return img, label


class PairedDataset(Dataset):
    """
    Paired dataset for comparing real and fake images.
    Useful for contrastive learning and comparison-based training.
    """

    def __init__(
        self,
        real_dir: str,
        fake_dir: str,
        transform: Optional[Callable] = None,
        pairs_per_epoch: Optional[int] = None
    ):
        """
        Args:
            real_dir: Directory with real images
            fake_dir: Directory with fake images
            transform: Image transformations
            pairs_per_epoch: Number of random pairs per epoch (default: min of real/fake count)
        """
        self.transform = transform

        # Load all image paths
        real_dir = Path(real_dir)
        fake_dir = Path(fake_dir)

        self.real_images = [
            str(p) for p in real_dir.glob('**/*')
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ]

        self.fake_images = [
            str(p) for p in fake_dir.glob('**/*')
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ]

        if not self.real_images or not self.fake_images:
            raise ValueError("Both real_dir and fake_dir must contain images")

        # Determine number of pairs
        if pairs_per_epoch is None:
            self.pairs_per_epoch = min(len(self.real_images), len(self.fake_images))
        else:
            self.pairs_per_epoch = pairs_per_epoch

        print(f"Paired dataset: {len(self.real_images)} real, {len(self.fake_images)} fake")

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Randomly sample one real and one fake image
        real_path = random.choice(self.real_images)
        fake_path = random.choice(self.fake_images)

        real_img = Image.open(real_path).convert('RGB')
        fake_img = Image.open(fake_path).convert('RGB')

        if self.transform is not None:
            real_img = self.transform(real_img)
            fake_img = self.transform(fake_img)

        real_label = torch.tensor([1.0], dtype=torch.float32)
        fake_label = torch.tensor([0.0], dtype=torch.float32)

        return real_img, real_label, fake_img, fake_label


class MultiSourceDataset(Dataset):
    """
    Dataset combining multiple sources of fake images.
    Useful for training on diverse generative models.
    """

    def __init__(
        self,
        real_dir: str,
        fake_sources: dict,
        transform: Optional[Callable] = None,
        source_weights: Optional[dict] = None
    ):
        """
        Args:
            real_dir: Directory with real images
            fake_sources: Dict of {source_name: directory_path}
            transform: Image transformations
            source_weights: Optional dict of {source_name: weight} for sampling
        """
        self.transform = transform
        self.samples = []

        # Load real images
        real_dir = Path(real_dir)
        for img_path in real_dir.glob('**/*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.samples.append((str(img_path), 1.0, 'real'))

        # Load fake images from each source
        for source_name, source_dir in fake_sources.items():
            source_dir = Path(source_dir)
            for img_path in source_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), 0.0, source_name))

        # Apply source weights if provided
        if source_weights is not None:
            weighted_samples = []
            for img_path, label, source in self.samples:
                weight = source_weights.get(source, 1.0)
                # Duplicate samples based on weight
                count = int(weight)
                weighted_samples.extend([(img_path, label, source)] * count)
            self.samples = weighted_samples

        random.shuffle(self.samples)

        # Print statistics
        real_count = sum(1 for _, label, _ in self.samples if label == 1.0)
        fake_count = len(self.samples) - real_count
        print(f"MultiSourceDataset: {real_count} real, {fake_count} fake from {len(fake_sources)} sources")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, label, source = self.samples[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor([label], dtype=torch.float32)

        return img, label, source


class MetadataDataset(Dataset):
    """
    Dataset with metadata (e.g., generator type, quality, etc.).
    Useful for conditional training or analysis.
    """

    def __init__(
        self,
        metadata_file: str,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            metadata_file: JSON file with format:
                [
                    {
                        "image_path": "path/to/image.jpg",
                        "label": 1.0,  # 1.0 for real, 0.0 for fake
                        "generator": "StyleGAN2",  # optional
                        "quality": "high"  # optional
                    },
                    ...
                ]
            transform: Image transformations
        """
        self.transform = transform

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded {len(self.metadata)} samples from {metadata_file}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self.metadata[idx]

        img_path = sample['image_path']
        label = sample['label']

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor([label], dtype=torch.float32)

        # Return metadata
        metadata = {k: v for k, v in sample.items() if k not in ['image_path', 'label']}

        return img, label, metadata
