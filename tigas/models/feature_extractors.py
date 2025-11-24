"""
Feature extraction modules for NAIR metric.
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

    Key insights:
    - GANs often produce distinctive frequency patterns
    - Checkerboard artifacts appear as specific frequency peaks
    - Natural images have characteristic spectral falloff
    - Generated images may lack high-frequency detail or have unnatural patterns
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Frequency domain feature extractors
        self.freq_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.freq_conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Frequency blocks for DCT analysis
        self.freq_blocks = nn.ModuleList([
            FrequencyBlock(hidden_dim),
            FrequencyBlock(hidden_dim)
        ])

        # Spectral statistics
        self.spectral_pooling = StatisticalPooling('all')

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),  # 5 statistics
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def compute_azimuthal_average(self, freq_map: torch.Tensor) -> torch.Tensor:
        """
        Compute azimuthal average of frequency spectrum.
        This captures the radial frequency distribution.
        """
        B, C, H, W = freq_map.shape

        # Create radial frequency bins
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(
            torch.arange(H, device=freq_map.device) - center_h,
            torch.arange(W, device=freq_map.device) - center_w,
            indexing='ij'
        )
        radius = torch.sqrt(x.float() ** 2 + y.float() ** 2)

        # Bin radii
        num_bins = min(H, W) // 2
        radial_profile = []

        for r in range(num_bins):
            mask = (radius >= r) & (radius < r + 1)
            if mask.sum() > 0:
                avg = (freq_map * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3)) / mask.sum()
                radial_profile.append(avg)

        if radial_profile:
            radial_profile = torch.stack(radial_profile, dim=1)  # [B, num_bins, C]
        else:
            radial_profile = torch.zeros(B, 1, C, device=freq_map.device)

        return radial_profile

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Analyze spectral characteristics.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            features: Spectral features [B, hidden_dim]
            aux_outputs: Auxiliary outputs for visualization/analysis
        """
        # Convert to frequency domain (FFT)
        freq = torch.fft.fft2(x, dim=(-2, -1))
        freq_mag = torch.abs(freq)  # Magnitude spectrum
        freq_phase = torch.angle(freq)  # Phase spectrum

        # Shift zero frequency to center
        freq_mag = torch.fft.fftshift(freq_mag, dim=(-2, -1))

        # Log magnitude for better dynamic range
        freq_mag_log = torch.log(freq_mag + 1e-8)

        # Extract frequency features
        freq_feat = self.freq_conv1(freq_mag_log)
        freq_feat = self.freq_conv2(freq_feat)

        # Apply frequency blocks
        spatial_feat, freq_feat_enhanced = self.freq_blocks[0](freq_feat)
        spatial_feat, freq_feat_enhanced = self.freq_blocks[1](spatial_feat)

        # Statistical pooling of frequency features
        freq_stats = self.spectral_pooling(freq_feat_enhanced)

        # Project to final feature dimension
        output = self.projection(freq_stats)

        # Auxiliary outputs for analysis
        aux = {
            'freq_magnitude': freq_mag,
            'freq_features': freq_feat_enhanced,
            'spatial_features': spatial_feat
        }

        return output, aux


class StatisticalMomentEstimator(nn.Module):
    """
    Estimates statistical moments and compares with natural image statistics.

    Natural images follow certain statistical distributions (e.g., heavy-tailed).
    Generated images may deviate from these distributions.

    This module:
    1. Computes local and global statistical moments
    2. Compares with learned prototypes of natural statistics
    3. Outputs a consistency score
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Multi-scale statistical extraction
        self.scales = [1, 2, 4]  # Different patch sizes

        # Local feature extractors for each scale
        self.local_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, feature_dim // 4, 3, padding=1),
                nn.BatchNorm2d(feature_dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 4, feature_dim // 4, 3, padding=1),
                nn.BatchNorm2d(feature_dim // 4),
                nn.ReLU(inplace=True)
            ) for _ in self.scales
        ])

        # Statistical pooling
        self.stat_pooling = StatisticalPooling('all')

        # Calculate actual feature dimension after extraction
        # Each extractor outputs feature_dim // 4 channels
        # StatisticalPooling('all') returns 5 stats per channel
        # We have len(self.scales) scales
        actual_stat_dim = (feature_dim // 4) * 5 * len(self.scales)

        # Learnable prototypes for natural image statistics
        # These will be updated during training to capture natural distributions
        self.register_buffer(
            'natural_prototypes',
            torch.randn(actual_stat_dim) * 0.01
        )
        self.prototype_momentum = 0.99

        # Comparison network
        self.comparison_net = nn.Sequential(
            nn.Linear(actual_stat_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def compute_local_statistics(
        self,
        x: torch.Tensor,
        patch_size: int
    ) -> torch.Tensor:
        """Compute statistics on local patches."""
        B, C, H, W = x.shape

        # Divide into patches
        if patch_size > 1:
            x = F.avg_pool2d(x, patch_size, patch_size)

        return x

    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor):
        """Update natural image statistics prototypes (EMA)."""
        if self.training:
            batch_mean = features.mean(dim=0)
            self.natural_prototypes.mul_(self.prototype_momentum).add_(
                batch_mean, alpha=1 - self.prototype_momentum
            )

    def forward(
        self,
        x: torch.Tensor,
        update_prototypes: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Estimate statistical consistency with natural images.

        Args:
            x: Input image [B, 3, H, W]
            update_prototypes: Whether to update natural prototypes

        Returns:
            features: Statistical consistency features [B, feature_dim]
            aux_outputs: Auxiliary statistical information
        """
        multi_scale_stats = []

        for scale, extractor in zip(self.scales, self.local_extractors):
            # Apply local feature extraction
            local_x = self.compute_local_statistics(x, scale)
            local_feat = extractor(local_x)

            # Compute statistical moments
            stats = self.stat_pooling(local_feat)
            multi_scale_stats.append(stats)

        # Concatenate all scales
        all_stats = torch.cat(multi_scale_stats, dim=1)

        # Update prototypes if training with real images
        if update_prototypes:
            self.update_prototypes(all_stats)

        # Compare with prototypes
        output = self.comparison_net(all_stats)

        # Auxiliary outputs
        aux = {
            'statistics': all_stats,
            'prototypes': self.natural_prototypes.unsqueeze(0).expand(x.size(0), -1)
        }

        return output, aux
