"""
Custom neural network layers for NAIR model.
Implements specialized components for authenticity assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class FrequencyBlock(nn.Module):
    """
    Frequency domain analysis block using DCT.
    Analyzes spectral characteristics to detect GAN artifacts.

    GAN-generated images often have distinctive frequency patterns:
    - Checkerboard artifacts in high frequencies
    - Unnatural spectral distributions
    - Mode collapse indicators in frequency domain
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels

        # Channel-wise frequency attention
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # Spectral feature extractor
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D Discrete Cosine Transform.
        Converts spatial domain to frequency domain.
        """
        # Efficient DCT using FFT
        X = torch.fft.fft2(x, dim=(-2, -1))
        # Take real part and normalize
        return X.real

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            spatial_features: Enhanced spatial features
            freq_features: Frequency domain features
        """
        # Apply DCT
        freq = self.dct2d(x)

        # Extract spectral features
        freq_feat = self.spectral_conv(freq)

        # Concatenate spatial and frequency for attention
        combined = torch.cat([x, freq_feat], dim=1)
        attention = self.freq_attention(combined)

        # Apply attention to spatial features
        spatial_out = x * attention

        return spatial_out, freq_feat


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive fusion of multi-modal features.
    Learns optimal weighting of different feature streams.
    """

    def __init__(self, num_streams: int, feature_dim: int):
        super().__init__()
        self.num_streams = num_streams
        self.feature_dim = feature_dim

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(
            torch.ones(num_streams) / num_streams
        )

        # Feature transformation before fusion
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_streams)
        ])

        # Post-fusion processing
        self.post_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, features_list: list) -> torch.Tensor:
        """
        Args:
            features_list: List of feature tensors [B, D] from different streams

        Returns:
            fused_features: Adaptively fused features [B, D]
        """
        assert len(features_list) == self.num_streams

        # Transform each feature stream
        transformed = [
            transform(feat)
            for transform, feat in zip(self.transforms, features_list)
        ]

        # Stack and apply softmax weights
        stacked = torch.stack(transformed, dim=1)  # [B, num_streams, D]
        weights = F.softmax(self.fusion_weights, dim=0)  # [num_streams]

        # Weighted sum
        fused = torch.sum(
            stacked * weights.view(1, -1, 1),
            dim=1
        )  # [B, D]

        # Post-fusion processing
        output = self.post_fusion(fused)

        return output


class GatedResidualBlock(nn.Module):
    """
    Gated residual block with learnable skip connections.
    Helps with gradient flow and adaptive feature selection.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        # Apply gating
        gate = self.gate(out)
        out = out * gate

        # Residual connection
        out = out + identity
        out = F.relu(out, inplace=True)

        return out


class StatisticalPooling(nn.Module):
    """
    Statistical pooling layer that computes multiple statistical moments.
    Captures distributional properties of features.
    """

    def __init__(self, pool_type: str = 'all'):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            stats: Statistical features [B, C * num_stats]
        """
        # Flatten spatial dimensions
        B, C = x.shape[:2]
        x_flat = x.view(B, C, -1)  # [B, C, H*W]

        stats = []

        # Mean
        if self.pool_type in ['all', 'mean']:
            mean = torch.mean(x_flat, dim=2)
            stats.append(mean)

        # Standard deviation
        if self.pool_type in ['all', 'std']:
            std = torch.std(x_flat, dim=2)
            stats.append(std)

        # Max pooling
        if self.pool_type in ['all', 'max']:
            max_val, _ = torch.max(x_flat, dim=2)
            stats.append(max_val)

        # Skewness (3rd moment)
        if self.pool_type == 'all':
            mean = torch.mean(x_flat, dim=2, keepdim=True)
            std = torch.std(x_flat, dim=2, keepdim=True) + 1e-8
            skewness = torch.mean(((x_flat - mean) / std) ** 3, dim=2)
            stats.append(skewness)

        # Kurtosis (4th moment)
        if self.pool_type == 'all':
            kurtosis = torch.mean(((x_flat - mean) / std) ** 4, dim=2)
            stats.append(kurtosis)

        # Concatenate all statistics
        output = torch.cat(stats, dim=1)

        return output


class MultiScaleConv(nn.Module):
    """
    Multi-scale convolution with parallel branches.
    Captures features at different receptive fields.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Different kernel sizes for multi-scale
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.branch2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.bn(out)

        return F.relu(out, inplace=True)
