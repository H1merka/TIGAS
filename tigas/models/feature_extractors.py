"""
Feature extraction modules for TIGAS metric.
Implements multi-scale, spectral, and statistical feature extractors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np

from .layers import (
    FrequencyBlock, GatedResidualBlock, StatisticalPooling,
    MultiScaleConv
)
from .attention import CBAM, SpatialAttention


class MultiScaleFeatureExtractor(nn.Module):
    """
    Efficient multi-scale feature extractor.

    Inspired by EfficientNet and MobileNetV3 but customized for
    authenticity assessment. Extracts features at 4 scales:
    1/2, 1/4, 1/8, 1/16 of input resolution.

    Unlike standard classification networks, emphasizes:
    - High-frequency detail preservation (for artifact detection)
    - Multi-scale feature fusion
    - Spatial attention for artifact localization
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        stages: List[int] = [2, 3, 4, 3]
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # Initial convolution - preserve high-frequency info
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Multi-scale stages
        channels = base_channels
        self.stage1 = self._make_stage(channels, channels * 2, stages[0], stride=2)
        channels *= 2

        self.stage2 = self._make_stage(channels, channels * 2, stages[1], stride=2)
        channels *= 2

        self.stage3 = self._make_stage(channels, channels * 2, stages[2], stride=2)
        channels *= 2

        self.stage4 = self._make_stage(channels, channels * 2, stages[3], stride=2)
        channels *= 2

        # Channel dimensions for each stage output
        self.out_channels = [
            base_channels * 2,   # stage1: 1/2
            base_channels * 4,   # stage2: 1/4
            base_channels * 8,   # stage3: 1/8
            base_channels * 16   # stage4: 1/16
        ]

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Module:
        """Create a stage with multiple residual blocks."""
        layers = []

        # First block handles stride and channel change
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        # Subsequent blocks
        for _ in range(num_blocks - 1):
            layers.append(GatedResidualBlock(out_channels))

        # Add CBAM attention at the end of stage
        layers.append(CBAM(out_channels))

        return nn.ModuleList(layers)

    def _process_stage(self, x: torch.Tensor, stage: nn.ModuleList) -> torch.Tensor:
        """
        Обработать один этап экстрактора признаков.

        Args:
            x: Входной тензор
            stage: Список слоёв этапа

        Returns:
            Обработанный тензор
        """
        for layer in stage:
            if isinstance(layer, CBAM):
                x, _ = layer(x)
            else:
                x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            features: List of 4 feature maps at different scales
                     [scale1/2, scale1/4, scale1/8, scale1/16]
        """
        features = []
        x = self.stem(x)

        # Обработка всех этапов
        stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        for stage in stages:
            x = self._process_stage(x, stage)
            features.append(x)

        return features


class SpectralAnalyzer(nn.Module):
    """
    Spectral analysis module for detecting GAN artifacts in frequency domain.
    OPTIMIZED: Removed redundant FFT calls, simplified pipeline.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Simplified frequency feature extraction (no internal FFT blocks)
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Global pooling + projection (faster than StatisticalPooling)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Analyze spectral characteristics.
        OPTIMIZED: Single FFT, no loops, no meshgrid recreation.
        """
        # Single FFT call with AMP-compatible processing
        with torch.amp.autocast('cuda', enabled=False):
            x_float = x.float()
            freq = torch.fft.rfft2(x_float, dim=(-2, -1))  # rfft2 is faster than fft2
            freq_mag = torch.abs(freq)
            # Log magnitude for better dynamic range
            freq_mag_log = torch.log1p(freq_mag)  # log1p is numerically stable
        
        # Pad to match input spatial dims for conv (rfft2 output is smaller)
        # Use simple interpolation instead
        freq_mag_log = F.interpolate(
            freq_mag_log, 
            size=x.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).to(x.dtype)  # Back to original dtype for AMP

        # Extract features through convolutions
        freq_feat = self.freq_encoder(freq_mag_log)

        # Project to final feature dimension
        output = self.projection(freq_feat)

        # Minimal auxiliary outputs (for compatibility)
        aux = {
            'freq_features': freq_feat,
        }

        return output, aux


class StatisticalMomentEstimator(nn.Module):
    """
    Estimates statistical moments and compares with natural image statistics.
    OPTIMIZED: Single-scale processing, simplified statistics.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Single-scale feature extractor (removed multi-scale loop)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Fast pooling: mean + std + max = 3 stats
        # Output: (feature_dim // 2) * 3
        stat_dim = (feature_dim // 2) * 3

        # Learnable prototypes for natural image statistics
        self.register_buffer('natural_prototypes', torch.zeros(stat_dim))
        self.register_buffer('prototypes_initialized', torch.tensor(False))
        self.register_buffer('prototype_update_count', torch.tensor(0))
        self.prototype_momentum = 0.99

        # Comparison network
        self.comparison_net = nn.Sequential(
            nn.Linear(stat_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor):
        """Update natural image statistics prototypes (EMA)."""
        if not self.training:
            return
        
        batch_mean = features.mean(dim=0)
        
        if not self.prototypes_initialized:
            self.natural_prototypes.copy_(batch_mean)
            self.prototypes_initialized.fill_(True)
            return
        
        self.natural_prototypes.mul_(self.prototype_momentum).add_(
            batch_mean, alpha=1 - self.prototype_momentum
        )
        self.prototype_update_count += 1

    def forward(
        self,
        x: torch.Tensor,
        update_prototypes: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Estimate statistical consistency with natural images.
        OPTIMIZED: No loops, single-scale processing.
        """
        # Extract features
        feat = self.feature_extractor(x)
        
        # Fast statistical pooling (vectorized)
        B, C = feat.shape[:2]
        feat_flat = feat.view(B, C, -1)
        
        mean = feat_flat.mean(dim=2)
        std = feat_flat.std(dim=2)
        max_val, _ = feat_flat.max(dim=2)
        
        all_stats = torch.cat([mean, std, max_val], dim=1)

        # Update prototypes if training
        if update_prototypes:
            self.update_prototypes(all_stats)

        # Compare with prototypes
        output = self.comparison_net(all_stats)

        aux = {
            'statistics': all_stats,
            'prototypes': self.natural_prototypes.unsqueeze(0).expand(x.size(0), -1)
        }

        return output, aux
