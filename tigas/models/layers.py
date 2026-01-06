"""
Custom neural network layers for TIGAS model.
Implements specialized components for authenticity assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class FrequencyBlock(nn.Module):
    """
    Frequency domain analysis block using FFT-based high-frequency extraction.
    Analyzes spectral characteristics to detect GAN artifacts.

    GAN-generated images often have distinctive frequency patterns:
    - Checkerboard artifacts in high frequencies
    - Unnatural spectral distributions
    - Mode collapse indicators in frequency domain
    
    This block extracts frequency-domain features and uses learned attention
    to focus on discriminative frequency bands.
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
        
        # High-frequency filter (learnable threshold)
        self.high_freq_threshold = nn.Parameter(torch.tensor(0.3))

    def _extract_high_frequency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract high-frequency components using FFT.
        
        High frequencies often contain GAN artifacts like checkerboard patterns
        from transposed convolutions.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            high_freq: High-frequency components [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Compute FFT with AMP-safe float32
        with torch.amp.autocast('cuda', enabled=False):
            x_float = x.float()
            
            # 2D FFT
            freq = torch.fft.fft2(x_float, dim=(-2, -1))
            freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
            
            # Create high-pass filter mask
            # Distance from center (DC component)
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=x.device, dtype=torch.float32) - cy
            x_coord = torch.arange(W, device=x.device, dtype=torch.float32) - cx
            yy, xx = torch.meshgrid(y, x_coord, indexing='ij')
            
            # Normalized distance
            max_dist = np.sqrt(cy**2 + cx**2)
            dist = torch.sqrt(yy**2 + xx**2) / max_dist
            
            # Smooth high-pass filter (sigmoid transition)
            threshold = torch.sigmoid(self.high_freq_threshold * 10)  # Scale for sharper transition
            high_pass = torch.sigmoid((dist - threshold) * 20)  # Smooth transition
            high_pass = high_pass.view(1, 1, H, W)
            
            # Apply filter
            freq_filtered = freq_shifted * high_pass
            
            # Inverse FFT
            freq_unshifted = torch.fft.ifftshift(freq_filtered, dim=(-2, -1))
            high_freq = torch.fft.ifft2(freq_unshifted, dim=(-2, -1))
            high_freq = torch.real(high_freq)
        
        return high_freq.to(x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract frequency-aware features.
        
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            spatial_features: Enhanced spatial features with frequency attention
            freq_features: High-frequency extracted features
        """
        # Extract high-frequency components via FFT
        high_freq = self._extract_high_frequency(x)
        
        # Process high-frequency through conv layers
        freq_feat = self.spectral_conv(high_freq)

        # Concatenate spatial and frequency for attention
        combined = torch.cat([x, freq_feat], dim=1)
        attention = self.freq_attention(combined)

        # Apply attention to spatial features
        # This emphasizes regions where frequency artifacts are detected
        spatial_out = x * attention + x  # Residual connection

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

        # Learnable fusion weights with small random initialization
        # Random init breaks symmetry and allows each stream to specialize
        # Scale 0.1 keeps weights near uniform while enabling learning
        self.fusion_weights = nn.Parameter(
            torch.randn(num_streams) * 0.1 + 1.0 / num_streams
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
    Statistical pooling layer.
    OPTIMIZED: Removed expensive skewness/kurtosis calculations.
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

        # Fast statistics only: mean, std, max (3 stats instead of 5)
        mean = torch.mean(x_flat, dim=2)
        std = torch.std(x_flat, dim=2)
        max_val, _ = torch.max(x_flat, dim=2)

        # Concatenate statistics
        output = torch.cat([mean, std, max_val], dim=1)

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
