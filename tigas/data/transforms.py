"""
Data transformations and augmentations for TIGAS training.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import random
from typing import List, Callable
import numpy as np


class RandomJPEGCompression:
    """
    Random JPEG compression augmentation.
    Simulates real-world image degradation.
    """

    def __init__(self, quality_range: tuple = (70, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        # This requires PIL Image
        if hasattr(img, 'save'):
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            from PIL import Image
            return Image.open(buffer)
        return img


class RandomGaussianNoise:
    """Add random Gaussian noise."""

    def __init__(self, std_range: tuple = (0.0, 0.05)):
        self.std_range = std_range

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(img) * std
            return torch.clamp(img + noise, 0, 1)
        return img


class RandomGaussianBlur:
    """Random Gaussian blur with varying kernel size."""

    def __init__(self, kernel_size_range: tuple = (3, 7), sigma_range: tuple = (0.1, 2.0)):
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range

    def __call__(self, img):
        if random.random() > 0.5:
            kernel_size = random.choice(range(*self.kernel_size_range, 2))  # Odd only
            sigma = random.uniform(*self.sigma_range)
            return TF.gaussian_blur(img, kernel_size, sigma)
        return img


class MixUp:
    """
    MixUp augmentation for training.
    Mixes two images with a random weight.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, img1, img2):
        lam = np.random.beta(self.alpha, self.alpha)
        mixed = lam * img1 + (1 - lam) * img2
        return mixed, lam


class ColorJitter:
    """Enhanced color jittering."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, img):
        return self.color_jitter(img)


def get_train_transforms(
    img_size: int = 256,
    normalize: bool = True,
    augment_level: str = 'medium'
) -> T.Compose:
    """
    Get training transforms with optimized augmentations for speed.

    Args:
        img_size: Target image size
        normalize: Whether to normalize to [-1, 1]
        augment_level: 'light', 'medium', or 'heavy'

    Returns:
        transforms: Composed transforms (optimized for GPU speed)
    """
    transforms_list = []

    # Resize and crop (fast operations)
    transforms_list.extend([
        T.Resize(int(img_size * 1.1)),
        T.RandomCrop(img_size),
    ])

    # Basic augmentations (lightweight only)
    if augment_level in ['medium', 'heavy']:
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05
            ),
        ])

    # Minimal rotation for medium (removed heavy augmentations)
    if augment_level == 'heavy':
        transforms_list.append(T.RandomRotation(10))

    # Convert to tensor
    transforms_list.append(T.ToTensor())

    # Skip heavy tensor augmentations for speed
    # (RandomGaussianNoise and RandomGaussianBlur removed)
    # These can be added back if GPU memory allows

    # Normalization
    if normalize:
        # Normalize to [-1, 1]
        transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return T.Compose(transforms_list)


def get_val_transforms(
    img_size: int = 256,
    normalize: bool = True
) -> T.Compose:
    """
    Get validation/test transforms.

    Args:
        img_size: Target image size
        normalize: Whether to normalize to [-1, 1]

    Returns:
        transforms: Composed transforms
    """
    transforms_list = [
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ]

    if normalize:
        transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return T.Compose(transforms_list)


def get_inference_transforms(
    img_size: int = 256,
    normalize: bool = True
) -> T.Compose:
    """
    Get inference transforms (minimal processing).

    Args:
        img_size: Target image size
        normalize: Whether to normalize

    Returns:
        transforms: Composed transforms
    """
    transforms_list = [
        T.Resize((img_size, img_size)),  # Force square resize
        T.ToTensor(),
    ]

    if normalize:
        transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return T.Compose(transforms_list)


class DenormalizeTransform:
    """Denormalize images from [-1, 1] to [0, 1]."""

    def __call__(self, tensor):
        return tensor * 0.5 + 0.5


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1].

    Args:
        tensor: Input tensor

    Returns:
        Denormalized tensor
    """
    return tensor * 0.5 + 0.5
