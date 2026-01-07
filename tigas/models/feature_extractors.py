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
    
    Implements comprehensive frequency-domain analysis:
    1. FFT magnitude analysis - detects unnatural spectral distributions
    2. FFT phase analysis - detects phase coherence artifacts from GANs
    3. Radial power spectrum - analyzes 1/f decay characteristic of natural images
    4. Azimuthal statistics - detects directional artifacts (e.g., checkerboard)
    
    GAN-generated images typically show:
    - Abnormal high-frequency peaks (checkerboard from transposed convolutions)
    - Deviation from natural 1/f^2 power spectrum decay
    - Phase incoherence patterns
    - Azimuthal asymmetry in frequency domain
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Number of radial bins for power spectrum analysis
        self.num_radial_bins = 32
        # Number of azimuthal bins for directional analysis
        self.num_azimuthal_bins = 8

        # Magnitude feature encoder
        self.mag_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Phase feature encoder
        self.phase_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
        )
        
        # Radial spectrum analyzer (1D conv over radial bins)
        # Input: radial power spectrum [B, C, num_radial_bins]
        self.radial_analyzer = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        
        # Combined feature dimension:
        # mag_encoder: hidden_dim // 2 (after global pool)
        # phase_encoder: hidden_dim // 4 (after global pool)  
        # radial_analyzer: hidden_dim // 4
        # azimuthal_stats: num_azimuthal_bins * in_channels * 2 (mean + std)
        azimuthal_dim = self.num_azimuthal_bins * in_channels * 2
        combined_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 4 + azimuthal_dim
        
        # Final projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Pre-compute coordinate grids (will be lazily initialized)
        self._radial_bins = None
        self._azimuthal_bins = None
        self._grid_size = None

    def _compute_grids(self, H: int, W: int, device: torch.device):
        """Compute radial and azimuthal coordinate grids for frequency analysis."""
        if self._grid_size == (H, W) and self._radial_bins is not None:
            return  # Already computed for this size
            
        # Create coordinate grid centered at DC component
        # For rfft2 output: width is W//2+1, height is H
        W_freq = W // 2 + 1
        
        # Frequency coordinates (normalized to [-1, 1] for full spectrum)
        fy = torch.fft.fftfreq(H, device=device)
        fx = torch.fft.rfftfreq(W, device=device)
        
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
        
        # Radial distance from DC (center)
        radius = torch.sqrt(fx_grid ** 2 + fy_grid ** 2)
        max_radius = radius.max()
        radius_normalized = radius / (max_radius + 1e-8)
        
        # Bin indices for radial averaging
        self._radial_bins = (radius_normalized * (self.num_radial_bins - 1)).long()
        self._radial_bins = self._radial_bins.clamp(0, self.num_radial_bins - 1)
        
        # Azimuthal angle
        angle = torch.atan2(fy_grid, fx_grid + 1e-8)  # [-pi, pi]
        angle_normalized = (angle + np.pi) / (2 * np.pi)  # [0, 1]
        
        # Bin indices for azimuthal analysis
        self._azimuthal_bins = (angle_normalized * self.num_azimuthal_bins).long()
        self._azimuthal_bins = self._azimuthal_bins.clamp(0, self.num_azimuthal_bins - 1)
        
        self._grid_size = (H, W)

    def _compute_radial_spectrum(
        self, 
        freq_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute radial power spectrum (average magnitude at each radius).
        
        This captures the 1/f decay characteristic - natural images follow
        approximately 1/f^2 decay, while GAN images often deviate.
        
        OPTIMIZED: Vectorized scatter operation instead of O(B*C) loops.
        
        Args:
            freq_mag: FFT magnitude [B, C, H, W_freq]
            
        Returns:
            radial_spectrum: [B, C, num_radial_bins]
        """
        B, C, H, W_freq = freq_mag.shape
        
        # Flatten spatial dimensions
        freq_flat = freq_mag.view(B, C, -1)  # [B, C, H*W_freq]
        bins_flat = self._radial_bins.view(-1)  # [H*W_freq]
        
        # Count pixels in each bin (computed once, shared across batch/channels)
        counts = torch.zeros(
            self.num_radial_bins, 
            device=freq_mag.device, dtype=freq_mag.dtype
        )
        ones = torch.ones_like(bins_flat, dtype=freq_mag.dtype)
        counts.scatter_add_(0, bins_flat, ones)
        counts = counts.clamp(min=1)  # Avoid division by zero
        
        # VECTORIZED: Expand bins for batch scatter operation
        # Shape: [B, C, H*W_freq] - same as freq_flat
        bins_expanded = bins_flat.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        
        # Initialize output tensor
        radial_spectrum = torch.zeros(
            B, C, self.num_radial_bins, 
            device=freq_mag.device, dtype=freq_mag.dtype
        )
        
        # Single vectorized scatter_add across all batches and channels
        radial_spectrum.scatter_add_(2, bins_expanded, freq_flat)
        
        # Average by bin counts
        radial_spectrum = radial_spectrum / counts.view(1, 1, -1)
        
        return radial_spectrum

    def _compute_azimuthal_stats(
        self, 
        freq_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute azimuthal statistics (mean and std in each angular sector).
        
        Checkerboard artifacts from transposed convolutions create
        distinctive peaks at specific angles.
        
        Args:
            freq_mag: FFT magnitude [B, C, H, W_freq]
            
        Returns:
            azimuthal_stats: [B, C * num_azimuthal_bins * 2]
        """
        B, C, H, W_freq = freq_mag.shape
        
        # Flatten spatial dimensions
        freq_flat = freq_mag.view(B, C, -1)  # [B, C, H*W_freq]
        bins_flat = self._azimuthal_bins.view(-1)  # [H*W_freq]
        
        stats_list = []
        
        for bin_idx in range(self.num_azimuthal_bins):
            mask = (bins_flat == bin_idx)
            if mask.sum() > 0:
                sector_values = freq_flat[:, :, mask]  # [B, C, num_in_sector]
                sector_mean = sector_values.mean(dim=2)  # [B, C]
                sector_std = sector_values.std(dim=2)  # [B, C]
            else:
                sector_mean = torch.zeros(B, C, device=freq_mag.device, dtype=freq_mag.dtype)
                sector_std = torch.zeros(B, C, device=freq_mag.device, dtype=freq_mag.dtype)
            
            stats_list.extend([sector_mean, sector_std])
        
        # Concatenate all stats: [B, C * num_azimuthal_bins * 2]
        azimuthal_stats = torch.cat(stats_list, dim=1)
        
        return azimuthal_stats

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Analyze spectral characteristics comprehensively.
        
        Args:
            x: Input image [B, C, H, W] in [-1, 1] range
            
        Returns:
            output: Spectral features [B, hidden_dim]
            aux: Dictionary with intermediate features for visualization/debugging
        """
        B, C, H, W = x.shape
        
        # Compute FFT with AMP-safe float32
        with torch.amp.autocast('cuda', enabled=False):
            x_float = x.float()
            
            # Clamp input to prevent extreme values
            x_float = torch.clamp(x_float, -10.0, 10.0)
            
            # 2D FFT (rfft2 for real input - more efficient)
            freq = torch.fft.rfft2(x_float, dim=(-2, -1))
            
            # Magnitude (log-scale for better dynamic range)
            freq_mag = torch.abs(freq)
            # Clamp magnitude to prevent extreme values before log
            freq_mag = torch.clamp(freq_mag, min=1e-8, max=1e6)
            freq_mag_log = torch.log1p(freq_mag)
            # Additional clamp after log
            freq_mag_log = torch.clamp(freq_mag_log, min=-20.0, max=20.0)
            
            # Phase (normalized to [-1, 1])
            freq_phase = torch.angle(freq) / np.pi
            # Clamp phase (should already be in [-1, 1] but ensure)
            freq_phase = torch.clamp(freq_phase, -1.0, 1.0)
            
            # Initialize coordinate grids
            self._compute_grids(H, W, x.device)
            
            # Compute radial power spectrum
            radial_spectrum = self._compute_radial_spectrum(freq_mag)
            # Clamp radial spectrum
            radial_spectrum = torch.clamp(radial_spectrum, min=0.0, max=1e6)
            
            # Compute azimuthal statistics
            azimuthal_stats = self._compute_azimuthal_stats(freq_mag)
            # Replace NaN with zeros in azimuthal stats
            azimuthal_stats = torch.nan_to_num(azimuthal_stats, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Interpolate magnitude to original size for conv processing
        freq_mag_spatial = F.interpolate(
            freq_mag_log, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).to(x.dtype)
        
        # Interpolate phase similarly
        freq_phase_spatial = F.interpolate(
            freq_phase,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).to(x.dtype)
        
        # Extract magnitude features
        mag_features = self.mag_encoder(freq_mag_spatial)
        mag_pooled = F.adaptive_avg_pool2d(mag_features, 1).flatten(1)  # [B, hidden_dim//2]
        
        # Extract phase features
        phase_features = self.phase_encoder(freq_phase_spatial)
        phase_pooled = F.adaptive_avg_pool2d(phase_features, 1).flatten(1)  # [B, hidden_dim//4]
        
        # Analyze radial spectrum
        radial_features = self.radial_analyzer(radial_spectrum.to(x.dtype))  # [B, hidden_dim//4]
        
        # Combine all features
        combined = torch.cat([
            mag_pooled,
            phase_pooled,
            radial_features,
            azimuthal_stats.to(x.dtype)
        ], dim=1)
        
        # Safety: replace any NaN/Inf in combined features
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Project to output dimension
        output = self.projection(combined)
        
        # Final safety clamp
        output = torch.clamp(output, -100.0, 100.0)
        output = torch.nan_to_num(output, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Auxiliary outputs for analysis/debugging
        aux = {
            'freq_magnitude': freq_mag_spatial,
            'freq_phase': freq_phase_spatial,
            'radial_spectrum': radial_spectrum,
            'azimuthal_stats': azimuthal_stats,
            'mag_features': mag_features,
            'phase_features': phase_features,
        }
        
        return output, aux


class StatisticalMomentEstimator(nn.Module):
    """
    Estimates statistical moments and compares with natural image statistics.
    
    Uses learnable prototypes that accumulate statistics from real images during
    training (via EMA). At inference, the distance between input statistics and
    learned prototypes provides a signal about image authenticity.
    
    Natural images tend to have consistent statistical properties (texture
    statistics, color distributions, etc.) that GANs often fail to replicate
    exactly.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # Multi-scale feature extractor for richer statistics
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Statistics: mean + std + max + skewness approximation = 4 stats per channel
        # Output dimension: (feature_dim // 2) * 4
        stat_dim = (feature_dim // 2) * 4

        # Learnable prototypes for natural image statistics
        # These accumulate EMA of statistics from REAL images during training
        self.register_buffer('natural_prototypes', torch.zeros(stat_dim))
        self.register_buffer('prototypes_initialized', torch.tensor(False))
        self.register_buffer('prototype_update_count', torch.tensor(0))
        
        # Momentum warm-up parameters
        # Start with lower momentum (0.9) for faster initial learning
        # Gradually increase to 0.99 over warmup_steps for stability
        self.prototype_momentum_start = 0.9
        self.prototype_momentum_end = 0.99
        self.prototype_warmup_steps = 1000  # ~10 epochs with 100 batches each

        # Comparison network now takes:
        # - Original statistics: stat_dim
        # - Difference from prototypes: stat_dim
        # - Element-wise product (interaction): stat_dim
        # Total input: stat_dim * 3
        comparison_input_dim = stat_dim * 3
        
        self.comparison_net = nn.Sequential(
            nn.Linear(comparison_input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Learnable scale for prototype comparison
        self.prototype_scale = nn.Parameter(torch.ones(1))

    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor = None):
        """
        Update natural image statistics prototypes using EMA with momentum warm-up.
        
        Only updates from REAL images (label=1.0) if labels are provided.
        This ensures prototypes represent authentic image statistics.
        
        Momentum warm-up: starts at 0.9 and linearly increases to 0.99 over
        warmup_steps. This allows faster initial prototype learning while
        maintaining stability in later training.
        
        Args:
            features: Statistics tensor [B, stat_dim]
            labels: Optional labels [B, 1] - only updates from real images if provided
        """
        if not self.training:
            return
        
        # Filter to real images only if labels provided
        if labels is not None:
            real_mask = (labels.squeeze() == 1.0)
            if not real_mask.any():
                return  # No real images in batch
            features = features[real_mask]
        
        batch_mean = features.mean(dim=0)
        
        if not self.prototypes_initialized:
            self.natural_prototypes.copy_(batch_mean)
            self.prototypes_initialized.fill_(True)
            self.prototype_update_count += 1
            return
        
        # Compute momentum with warm-up schedule
        # Linear interpolation from start to end momentum over warmup_steps
        progress = min(1.0, self.prototype_update_count.item() / self.prototype_warmup_steps)
        current_momentum = (
            self.prototype_momentum_start + 
            (self.prototype_momentum_end - self.prototype_momentum_start) * progress
        )
        
        self.natural_prototypes.mul_(current_momentum).add_(
            batch_mean, alpha=1 - current_momentum
        )
        self.prototype_update_count += 1

    def _compute_statistics(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Compute comprehensive statistics from feature maps.
        
        Args:
            feat: Feature tensor [B, C, H, W]
            
        Returns:
            stats: Statistics tensor [B, C * 4]
        """
        B, C = feat.shape[:2]
        feat_flat = feat.view(B, C, -1)
        
        # Basic moments
        mean = feat_flat.mean(dim=2)
        std = feat_flat.std(dim=2) + 1e-6  # Numerical stability
        max_val, _ = feat_flat.max(dim=2)
        
        # Skewness approximation: (mean - median) / std
        # Using sorted middle value as median approximation
        median = feat_flat.median(dim=2).values
        skewness_approx = (mean - median) / std
        
        # Clamp skewness to prevent extreme values
        skewness_approx = torch.clamp(skewness_approx, -10.0, 10.0)
        
        all_stats = torch.cat([mean, std, max_val, skewness_approx], dim=1)
        
        # Safety: replace NaN/Inf
        all_stats = torch.nan_to_num(all_stats, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return all_stats

    def forward(
        self,
        x: torch.Tensor,
        update_prototypes: bool = False,
        labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Estimate statistical consistency with natural images.
        
        Computes statistics from input and compares with learned prototypes
        of natural image statistics.
        
        Args:
            x: Input image [B, C, H, W]
            update_prototypes: Whether to update prototypes (training only)
            labels: Optional labels for selective prototype update
            
        Returns:
            output: Feature output [B, feature_dim]
            aux: Auxiliary outputs including statistics and distances
        """
        # Extract features
        feat = self.feature_extractor(x)
        
        # Compute statistics
        all_stats = self._compute_statistics(feat)

        # Update prototypes if training (only from real images)
        if update_prototypes:
            self.update_prototypes(all_stats, labels)

        # Compute comparison with prototypes
        # Expand prototypes to batch size
        prototypes = self.natural_prototypes.unsqueeze(0).expand(x.size(0), -1)
        
        # Difference from prototypes (key signal for authenticity)
        diff = all_stats - prototypes
        diff_scaled = diff * self.prototype_scale
        
        # Interaction features (element-wise product captures correlation patterns)
        interaction = all_stats * prototypes
        
        # Combine all comparison features
        combined = torch.cat([all_stats, diff_scaled, interaction], dim=1)
        
        # Safety: clamp combined features
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Generate output through comparison network
        output = self.comparison_net(combined)
        
        # Final safety clamp
        output = torch.clamp(output, -100.0, 100.0)
        output = torch.nan_to_num(output, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Compute distance metrics for analysis
        l2_distance = torch.norm(diff, p=2, dim=1, keepdim=True)
        cosine_sim = F.cosine_similarity(
            all_stats, prototypes, dim=1
        ).unsqueeze(1)

        aux = {
            'statistics': all_stats,
            'prototypes': prototypes,
            'diff_from_prototypes': diff,
            'l2_distance': l2_distance,
            'cosine_similarity': cosine_sim,
            'prototype_update_count': self.prototype_update_count.item()
        }

        return output, aux
