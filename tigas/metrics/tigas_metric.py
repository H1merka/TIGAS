"""
TIGAS Metric - Main metric computation class.
Provides both model-based and model-free metric computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, List
import warnings

from ..models.tigas_model import TIGASModel
from .components import (
    PerceptualDistance,
    SpectralDivergence,
    StatisticalConsistency
)


class TIGASMetric(nn.Module):
    """
    TIGAS Metric Calculator.

    Can operate in two modes:
    1. Model-based: Uses trained TIGASModel for prediction
    2. Component-based: Combines individual metric components

    The metric is fully differentiable and can be used as:
    - Image quality assessment
    - Loss function for training generative models
    - Evaluation metric for image generation tasks
    """

    def __init__(
        self,
        model: Optional[TIGASModel] = None,
        use_model: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        component_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            model: Pretrained TIGASModel (if use_model=True)
            use_model: Whether to use model-based computation
            device: Device to run on
            component_weights: Weights for component metrics (if use_model=False)
        """
        super().__init__()

        self.use_model = use_model
        self.device = device

        if use_model:
            if model is None:
                warnings.warn(
                    "No model provided for model-based TIGAS. "
                    "Creating default model (untrained)."
                )
                model = TIGASModel()

            self.model = model.to(device)
            self.model.eval()
        else:
            # Component-based mode
            self.spectral_div = SpectralDivergence()
            self.stat_consistency = StatisticalConsistency()

            # Default weights for components
            if component_weights is None:
                component_weights = {
                    'spectral': 0.4,
                    'statistical': 0.4,
                    'spatial': 0.2
                }
            self.component_weights = component_weights

    def compute_model_based(
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

    def compute_component_based(
        self,
        images: torch.Tensor,
        reference_images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute TIGAS using individual components.

        Args:
            images: Input images [B, C, H, W]
            reference_images: Optional reference real images [B, C, H, W]

        Returns:
            Dictionary with scores and component values
        """
        B = images.size(0)

        # Spectral divergence
        spectral_div, spectral_info = self.spectral_div(images, reference_images)
        spectral_score = torch.exp(-spectral_div)  # Convert divergence to similarity

        # Statistical consistency
        stat_consistency, stat_moments = self.stat_consistency(images)
        stat_score = torch.exp(-stat_consistency)  # Convert to similarity

        # Spatial variance (simple measure of detail)
        # Real images typically have more spatial variance than generated
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=images.dtype,
            device=images.device
        ).view(1, 1, 3, 3)

        spatial_var = []
        for c in range(images.size(1)):
            channel = images[:, c:c+1, :, :]
            laplacian = F.conv2d(channel, laplacian_kernel, padding=1)
            var = laplacian.var(dim=[2, 3])
            spatial_var.append(var)

        spatial_var = torch.stack(spatial_var, dim=1).mean(dim=1)  # [B]

        # Normalize spatial variance to [0, 1] range
        # Higher variance = more realistic
        spatial_score = torch.sigmoid(spatial_var * 10 - 5)

        # Weighted combination
        final_score = (
            self.component_weights['spectral'] * spectral_score +
            self.component_weights['statistical'] * stat_score +
            self.component_weights['spatial'] * spatial_score
        )

        return {
            'score': final_score.unsqueeze(1),  # [B, 1]
            'spectral_score': spectral_score,
            'statistical_score': stat_score,
            'spatial_score': spatial_score,
            'spectral_info': spectral_info,
            'statistical_moments': stat_moments
        }

    def forward(
        self,
        images: torch.Tensor,
        reference_images: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute TIGAS score.

        Args:
            images: Input images [B, C, H, W], range [-1, 1] or [0, 1]
            reference_images: Optional reference images (for component-based)
            return_components: Whether to return component scores

        Returns:
            score: TIGAS score [B, 1] if return_components=False
            dict: Dictionary with scores and components if return_components=True
        """
        # Ensure images are on correct device
        images = images.to(self.device)
        if reference_images is not None:
            reference_images = reference_images.to(self.device)

        # Normalize to [-1, 1] if needed
        if images.min() >= 0 and images.max() <= 1:
            images = images * 2 - 1
        if reference_images is not None:
            if reference_images.min() >= 0 and reference_images.max() <= 1:
                reference_images = reference_images * 2 - 1

        # Compute based on mode
        if self.use_model:
            results = self.compute_model_based(images, return_features=return_components)
        else:
            results = self.compute_component_based(images, reference_images)

        if return_components:
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
