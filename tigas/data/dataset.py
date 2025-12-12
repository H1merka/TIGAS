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
                    label = 1.0
                    self.samples.append((str(img_path), label))

        # Fake images (label = 0.0)
        fake_dir = self.root / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label = 0.0
                    self.samples.append((str(img_path), label))

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


class CSVDataset(Dataset):
    """
    Dataset loading from CSV annotation files.

    Designed for datasets with pre-defined splits and CSV annotations.
    Handles various path formats and automatically resolves file locations.

    CSV Format:
        image_path,label
        images\\ADM\\0_real\\img.jpg,1
        images\\ADM\\1_fake\\img.png,0

    Features:
    - Reads from CSV with image_path,label columns
    - Handles Windows/Unix path separators automatically
    - Supports relative and absolute paths
    - Extracts generator information from paths
    - Optional image caching for faster training

    IMPORTANT:
    For best performance and data integrity, use a cleaned CSV generated by:
        python scripts/validate_dataset.py \
            --dataset_dir <your_data> \
            --csv_file <csv_file> \
            --remove_corrupted \
            --update_csv
    
    This ensures all corrupted/problematic images are removed before training,
    avoiding runtime errors and maintaining consistent batch sizes.
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform: Optional[Callable] = None,
        use_cache: bool = False,
        validate_paths: bool = True,
        skip_corrupted: bool = False,
        corrupted_log_file: Optional[str] = None
    ):
        """
        Args:
            csv_file: Path to CSV file with columns: image_path, label
            root_dir: Root directory for resolving relative paths
            transform: Image transformations
            use_cache: Whether to cache loaded images in memory
            validate_paths: Whether to validate all image paths exist
            skip_corrupted: DEPRECATED - kept for backward compatibility.
                Use validate_dataset.py to clean data instead.
            corrupted_log_file: DEPRECATED - not used when skip_corrupted=False
        
        Note:
            For best performance, use a cleaned CSV generated by
            scripts/validate_dataset.py with --remove_corrupted --update_csv
        """
        import pandas as pd

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.skip_corrupted = skip_corrupted
        self.corrupted_log_file = corrupted_log_file
        self.corrupted_files = []

        # Load CSV
        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            csv_path = self.root_dir / csv_path

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Store CSV directory for resolving relative image paths
        csv_dir = csv_path.parent

        self.df = pd.read_csv(csv_path)

        # Validate columns
        if 'image_path' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError(f"CSV must have 'image_path' and 'label' columns. Found: {self.df.columns.tolist()}")

        # Convert paths to Path objects and make absolute
        self.samples = []
        invalid_count = 0

        for idx, row in self.df.iterrows():
            img_path_str = str(row['image_path'])
            label = float(row['label'])

            # Convert backslashes to forward slashes for cross-platform compatibility
            img_path_str = img_path_str.replace('\\', '/')

            # Build full path
            img_path = Path(img_path_str)
            if not img_path.is_absolute():
                # Relative path - resolve relative to CSV directory
                img_path = csv_dir / img_path

            # Validate path exists if requested
            if validate_paths:
                if not img_path.exists():
                    if self.skip_corrupted:
                        self.corrupted_files.append((str(img_path), "File not found"))
                        invalid_count += 1
                        continue
                    else:
                        raise FileNotFoundError(f"Image file not found: {img_path}")

            # Extract generator name from path (optional metadata)
            generator = self._extract_generator(img_path_str)

            self.samples.append({
                'path': str(img_path),
                'label': label,
                'generator': generator
            })

        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in CSV: {csv_path}")

        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid paths skipped from CSV")
        
        # Log corrupted files if requested
        if self.corrupted_log_file and self.corrupted_files:
            log_path = Path(self.corrupted_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"Corrupted files log from {csv_path.name}\n")
                f.write("=" * 80 + "\n\n")
                for file_path, reason in self.corrupted_files:
                    f.write(f"{file_path}\n  Reason: {reason}\n")
            print(f"Corrupted files logged to: {self.corrupted_log_file}")

        # Print statistics
        real_count = sum(1 for s in self.samples if s['label'] == 1.0)
        fake_count = len(self.samples) - real_count
        print(f"CSVDataset loaded from {csv_path.name}:")
        print(f"  Total: {len(self.samples)} images")
        print(f"  Real: {real_count} ({real_count/len(self.samples)*100:.1f}%)")
        print(f"  Fake: {fake_count} ({fake_count/len(self.samples)*100:.1f}%)")

    def _extract_generator(self, path_str: str) -> str:
        """
        Extract generator name from path.
        E.g., 'images/ADM/0_real/img.jpg' -> 'ADM'
        """
        parts = path_str.split('/')
        # Look for common generator names in path
        generators = ['ADM', 'biggan', 'DALLE2', 'face', 'gaugan', 'Glide',
                     'Midjourney', 'sd_xl', 'sd14', 'sd15', 'stargan', 'VQDM', 'wuk']

        for part in parts:
            if part in generators:
                return part

        return 'unknown'

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = sample['path']
        label = sample['label']

        # Load from cache if available
        if self.use_cache and img_path in self.cache:
            img = self.cache[img_path]
        else:
            # Note: skip_corrupted parameter kept for backward compatibility
            # but graceful error handling is disabled for performance.
            # Use cleaned CSV from validate_dataset.py to ensure data integrity.
            img = Image.open(img_path).convert('RGB')
            if self.use_cache:
                self.cache[img_path] = img

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor([label], dtype=torch.float32)

        return img, label

    def get_generator_stats(self) -> dict:
        """Get statistics per generator."""
        from collections import defaultdict
        stats = defaultdict(lambda: {'real': 0, 'fake': 0})

        for sample in self.samples:
            gen = sample['generator']
            if sample['label'] == 1.0:
                stats[gen]['real'] += 1
            else:
                stats[gen]['fake'] += 1

        return dict(stats)
