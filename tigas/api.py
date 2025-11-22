"""
TIGAS Public API - Simple interface for using TIGAS metric.

Example usage:
    >>> from tigas import TIGAS, compute_tigas_score
    >>>
    >>> # Initialize TIGAS
    >>> tigas = TIGAS(checkpoint_path='path/to/checkpoint.pt')
    >>>
    >>> # Compute score for single image
    >>> score = tigas('path/to/image.jpg')
    >>> print(f"TIGAS score: {score:.3f}")
    >>>
    >>> # Compute scores for directory
    >>> scores = tigas.compute_directory('path/to/images/')
    >>>
    >>> # Use as PyTorch module
    >>> import torch
    >>> images = torch.randn(4, 3, 256, 256)
    >>> scores = tigas(images)
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from typing import Union, List, Dict, Optional
import numpy as np

from .models.tigas_model import TIGASModel, create_tigas_model
from .metrics.tigas_metric import TIGASMetric
from .data.transforms import get_inference_transforms


class TIGAS(nn.Module):
    """
    TIGAS - Neural Authenticity and Realism Index

    High-level API for computing TIGAS scores.

    Attributes:
        model: Underlying TIGASModel
        metric: TIGASMetric wrapper
        device: Computation device
        transform: Image preprocessing transforms
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        img_size: int = 256,
        device: Optional[str] = None,
        use_model: bool = True
    ):
        """
        Initialize TIGAS.

        Args:
            checkpoint_path: Path to pretrained checkpoint (optional)
            img_size: Input image size
            device: Device ('cuda' or 'cpu'). Auto-detected if None.
            use_model: Whether to use model-based computation
        """
        super().__init__()

        # Device setup
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Create or load model
        if use_model:
            if checkpoint_path and Path(checkpoint_path).exists():
                self.model = create_tigas_model(
                    img_size=img_size,
                    pretrained=True,
                    checkpoint_path=checkpoint_path
                )
                print(f"Loaded TIGAS model from {checkpoint_path}")
            else:
                self.model = create_tigas_model(img_size=img_size)
                if checkpoint_path:
                    print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained model.")
                else:
                    print("Using untrained TIGAS model.")

            self.model = self.model.to(device)
            self.model.eval()
        else:
            self.model = None

        # Create metric wrapper
        self.metric = TIGASMetric(
            model=self.model,
            use_model=use_model,
            device=device
        )

        # Image preprocessing
        self.transform = get_inference_transforms(img_size=img_size, normalize=True)
        self.img_size = img_size

    def forward(
        self,
        x: Union[torch.Tensor, str, Path, Image.Image, List],
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict]:
        """
        Compute TIGAS score(s).

        Args:
            x: Input(s). Can be:
               - torch.Tensor: [B, C, H, W] or [C, H, W]
               - str/Path: Path to image file or directory
               - PIL.Image: PIL Image object
               - List: List of any of the above
            return_features: Whether to return intermediate features

        Returns:
            scores: TIGAS score(s) [B, 1] or dict with features
        """
        # Handle different input types
        if isinstance(x, (str, Path)):
            x = Path(x)
            if x.is_dir():
                return self.compute_directory(str(x))
            else:
                x = Image.open(x).convert('RGB')

        if isinstance(x, Image.Image):
            x = self.transform(x).unsqueeze(0)  # [1, C, H, W]

        elif isinstance(x, list):
            # Process list of inputs
            processed = []
            for item in x:
                if isinstance(item, (str, Path)):
                    img = Image.open(item).convert('RGB')
                    img = self.transform(img)
                elif isinstance(item, Image.Image):
                    img = self.transform(item)
                elif isinstance(item, torch.Tensor):
                    img = item
                else:
                    raise TypeError(f"Unsupported type in list: {type(item)}")
                processed.append(img)
            x = torch.stack(processed)

        elif isinstance(x, torch.Tensor):
            # Handle single image [C, H, W]
            if x.ndim == 3:
                x = x.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

        # Ensure on correct device
        x = x.to(self.device)

        # Compute TIGAS
        with torch.no_grad():
            if return_features:
                return self.metric(x, return_components=True)
            else:
                return self.metric(x)

    def compute_image(self, image_path: str) -> float:
        """
        Compute TIGAS score for a single image.

        Args:
            image_path: Path to image file

        Returns:
            TIGAS score (float)
        """
        score = self.forward(image_path)
        return score.item()

    def compute_directory(
        self,
        directory: str,
        return_paths: bool = False,
        batch_size: int = 32
    ) -> Union[np.ndarray, Dict[str, float]]:
        """
        Compute TIGAS scores for all images in a directory.

        Args:
            directory: Path to directory
            return_paths: Whether to return dict with paths as keys
            batch_size: Batch size for processing

        Returns:
            scores: Array of scores or dict {path: score}
        """
        directory = Path(directory)
        image_paths = []

        # Find all images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(directory.glob(f'**/{ext}'))

        if not image_paths:
            print(f"No images found in {directory}")
            return np.array([])

        print(f"Processing {len(image_paths)} images...")

        # Process in batches
        all_scores = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [Image.open(p).convert('RGB') for p in batch_paths]
            batch_tensors = torch.stack([self.transform(img) for img in batch_images])

            scores = self.forward(batch_tensors)
            all_scores.append(scores.cpu().numpy())

        all_scores = np.concatenate(all_scores, axis=0).squeeze()

        if return_paths:
            return {str(path): score for path, score in zip(image_paths, all_scores)}
        else:
            return all_scores

    def compute_batch(
        self,
        images: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute TIGAS scores for a batch with automatic batching.

        Args:
            images: Images [N, C, H, W]
            batch_size: Batch size

        Returns:
            scores: TIGAS scores [N, 1]
        """
        N = images.size(0)
        all_scores = []

        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size]
            scores = self.forward(batch)
            all_scores.append(scores.cpu())

        return torch.cat(all_scores, dim=0)

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the model."""
        if self.model is None:
            return {"mode": "component-based", "model": None}

        info = self.model.get_model_size()
        info['device'] = str(self.device)
        info['img_size'] = self.img_size

        return info

    def save_model(self, save_path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("No model to save (component-based mode)")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'img_size': self.img_size
        }

        torch.save(checkpoint, save_path)
        print(f"Saved model to {save_path}")


def compute_tigas_score(
    image: Union[str, Path, Image.Image, torch.Tensor],
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> float:
    """
    Convenience function to compute TIGAS score for a single image.

    Args:
        image: Input image (path, PIL Image, or tensor)
        checkpoint_path: Path to pretrained checkpoint
        device: Device to use

    Returns:
        TIGAS score (float)

    Example:
        >>> score = compute_tigas_score('image.jpg', checkpoint_path='model.pt')
        >>> print(f"Score: {score:.3f}")
    """
    tigas = TIGAS(checkpoint_path=checkpoint_path, device=device)
    return tigas.compute_image(image) if isinstance(image, (str, Path)) else tigas(image).item()


def load_tigas(checkpoint_path: str, device: Optional[str] = None) -> TIGAS:
    """
    Load pretrained TIGAS model.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use

    Returns:
        TIGAS instance

    Example:
        >>> tigas = load_tigas('checkpoints/best_model.pt')
        >>> score = tigas('test_image.jpg')
    """
    return TIGAS(checkpoint_path=checkpoint_path, device=device)
