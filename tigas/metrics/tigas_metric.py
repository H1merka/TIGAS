"""
TIGAS Metric - Main metric computation class.
Provides model-based metric computation using trained neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, List
import warnings

from ..models.tigas_model import TIGASModel


class TIGASMetric(nn.Module):
    """
    TIGAS Metric Calculator.

    Uses trained TIGASModel for prediction.
    
    The metric is fully differentiable and can be used as:
    - Image quality assessment
    - Loss function for training generative models
    - Evaluation metric for image generation tasks
    """

    def __init__(
        self,
        model: Optional[TIGASModel] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Pretrained TIGASModel (required for inference)
            device: Device to run on
        """
        super().__init__()

        self.device = device

        if model is None:
            warnings.warn(
                "No model provided for TIGAS. "
                "Creating default model (untrained)."
            )
            model = TIGASModel()

        self.model = model.to(device)
        self.model.eval()

    def compute(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute TIGAS using trained model.

        Args:
            images: Input images [B, C, H, W]
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with 'score' and optionally 'features'
        """
        with torch.no_grad():
            outputs = self.model(
                images,
                return_features=return_features,
                update_prototypes=False
            )

        return {
            'score': outputs['score'],
            'features': outputs.get('features', None)
        }

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute TIGAS score.

        Args:
            images: Input images [B, C, H, W], range [-1, 1] or [0, 1]
            return_features: Whether to return intermediate features

        Returns:
            score: TIGAS score [B, 1] if return_features=False
            dict: Dictionary with scores and features if return_features=True
        """
        # Ensure images are on correct device
        images = images.to(self.device)

        # Note: Нормализация [0,1] -> [-1,1] теперь выполняется в TIGASModel._normalize_input()
        # Здесь оставляем pass-through для совместимости

        # Compute using trained model
        results = self.compute(images, return_features=return_features)

        if return_features:
            return results
        else:
            return results['score']

    def compute_pairwise(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise TIGAS scores between two sets of images.

        Useful for:
        - Comparing real vs generated images
        - Image-to-image translation evaluation

        Args:
            images1: First set [B, C, H, W]
            images2: Second set [B, C, H, W]

        Returns:
            scores: Pairwise scores [B, 1]
        """
        # Compute scores for both sets
        score1 = self.forward(images1)
        score2 = self.forward(images2)

        # Return absolute difference
        # Lower difference = more similar realism levels
        return torch.abs(score1 - score2)

    @torch.no_grad()
    def compute_dataset_statistics(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute TIGAS statistics over a dataset.

        Args:
            dataloader: DataLoader for the dataset
            max_samples: Maximum number of samples to process

        Returns:
            statistics: Dictionary with mean, std, etc.
        """
        scores = []
        num_samples = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            batch_scores = self.forward(images)
            scores.append(batch_scores.cpu())

            num_samples += images.size(0)
            if max_samples is not None and num_samples >= max_samples:
                break

        scores = torch.cat(scores, dim=0)

        return {
            'mean': scores.mean().item(),
            'std': scores.std().item(),
            'min': scores.min().item(),
            'max': scores.max().item(),
            'median': scores.median().item(),
            'num_samples': len(scores)
        }


def compute_tigas_batch(
    images: torch.Tensor,
    model: Optional[TIGASModel] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32
) -> torch.Tensor:
    """
    Compute TIGAS scores for a batch of images with automatic batching.

    Args:
        images: Input images [N, C, H, W]
        model: TIGASModel instance
        device: Device to use
        batch_size: Batch size for processing

    Returns:
        scores: TIGAS scores [N, 1]
    """
    metric = TIGASMetric(model=model, use_model=(model is not None), device=device)

    N = images.size(0)
    all_scores = []

    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size]
        scores = metric(batch)
        all_scores.append(scores.cpu())

    return torch.cat(all_scores, dim=0)
