"""
Component metrics for TIGAS.
Individual differentiable metrics that contribute to the final TIGAS score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class PerceptualDistance(nn.Module):
    """
    Perceptual distance component inspired by LPIPS.
    Measures perceptual similarity using deep features.
    """

    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual distance between x and y.

        Args:
            x: First image [B, C, H, W]
            y: Second image [B, C, H, W]

        Returns:
            distance: Perceptual distance [B]
        """
        # Extract features
        with torch.no_grad():
            feat_x = self.feature_extractor(x)
            feat_y = self.feature_extractor(y)

        # Compute L2 distance in feature space
        distances = []
        for fx, fy in zip(feat_x, feat_y):
            # Normalize features
            fx = F.normalize(fx, dim=1)
            fy = F.normalize(fy, dim=1)

            # Spatial L2 distance
            diff = (fx - fy) ** 2
            dist = diff.mean(dim=[1, 2, 3])  # [B]
            distances.append(dist)

        # Average across scales
        total_distance = torch.stack(distances, dim=1).mean(dim=1)

        return total_distance


class SpectralDivergence(nn.Module):
    """
    Spectral divergence metric.
    Measures difference in frequency domain characteristics.

    Natural images have characteristic spectral falloff (1/f^α).
    Generated images may deviate from this pattern.
    """

    def __init__(self):
        super().__init__()

    def compute_power_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Compute power spectral density."""
        # FFT
        freq = torch.fft.fft2(x, dim=(-2, -1))
        power = torch.abs(freq) ** 2

        # Shift to center
        power = torch.fft.fftshift(power, dim=(-2, -1))

        return power

    def compute_radial_profile(
        self,
        power: torch.Tensor,
        num_bins: int = 50
    ) -> torch.Tensor:
        """
        Compute radial average of power spectrum.

        Args:
            power: Power spectrum [B, C, H, W]
            num_bins: Number of radial bins

        Returns:
            profile: Radial profile [B, C, num_bins]
        """
        B, C, H, W = power.shape

        # Create radial coordinates
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(
            torch.arange(H, device=power.device) - center_h,
            torch.arange(W, device=power.device) - center_w,
            indexing='ij'
        )
        radius = torch.sqrt(x.float() ** 2 + y.float() ** 2)

        # Maximum radius
        max_radius = min(H, W) // 2

        # Bin edges
        bin_edges = torch.linspace(0, max_radius, num_bins + 1, device=power.device)

        # Compute profile for each bin
        profiles = []
        for b in range(B):
            channel_profiles = []
            for c in range(C):
                bin_values = []
                for i in range(num_bins):
                    mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
                    if mask.sum() > 0:
                        val = power[b, c][mask].mean()
                    else:
                        val = torch.tensor(0.0, device=power.device)
                    bin_values.append(val)

                channel_profiles.append(torch.stack(bin_values))
            profiles.append(torch.stack(channel_profiles))

        profile = torch.stack(profiles)  # [B, C, num_bins]

        return profile

    def forward(
        self,
        x: torch.Tensor,
        reference: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute spectral divergence.

        Args:
            x: Input image [B, C, H, W]
            reference: Reference image [B, C, H, W] (optional)

        Returns:
            divergence: Spectral divergence score [B]
            info: Additional information
        """
        # Compute power spectrum
        power_x = self.compute_power_spectrum(x)

        # Compute radial profile
        profile_x = self.compute_radial_profile(power_x)

        if reference is not None:
            # Compare with reference
            power_ref = self.compute_power_spectrum(reference)
            profile_ref = self.compute_radial_profile(power_ref)

            # KL divergence between profiles
            profile_x_norm = F.softmax(profile_x.flatten(1), dim=1)
            profile_ref_norm = F.softmax(profile_ref.flatten(1), dim=1)

            divergence = F.kl_div(
                profile_x_norm.log(),
                profile_ref_norm,
                reduction='none'
            ).sum(dim=1)  # [B]

        else:
            # Compare with expected natural spectrum (1/f decay)
            # In log-log space, natural images show linear decay
            log_profile = torch.log(profile_x + 1e-8)

            # Fit line and compute deviation
            # For simplicity, compute variance as irregularity measure
            divergence = log_profile.var(dim=[1, 2])  # [B]

        info = {
            'power_spectrum': power_x,
            'radial_profile': profile_x
        }

        return divergence, info


class StatisticalConsistency(nn.Module):
    """
    Statistical consistency metric.
    Measures how consistent image statistics are with natural images.

    Uses multiple statistical moments:
    - Mean, variance
    - Skewness, kurtosis
    - Entropy
    """

    def __init__(self):
        super().__init__()

    def compute_moments(self, x: torch.Tensor) -> dict:
        """
        Compute statistical moments.

        Args:
            x: Input [B, C, H, W]

        Returns:
            moments: Dictionary of statistical moments
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # [B, C, H*W]

        # First moment: mean
        mean = x_flat.mean(dim=2)  # [B, C]

        # Second moment: variance
        var = x_flat.var(dim=2)  # [B, C]

        # Third moment: skewness
        std = torch.sqrt(var + 1e-8)
        centered = (x_flat - mean.unsqueeze(2)) / std.unsqueeze(2)
        skewness = (centered ** 3).mean(dim=2)  # [B, C]

        # Fourth moment: kurtosis
        kurtosis = (centered ** 4).mean(dim=2)  # [B, C]

        # Entropy (approximated via histogram)
        entropy_vals = []
        for b in range(B):
            ch_entropy = []
            for c in range(C):
                hist = torch.histc(x[b, c], bins=256, min=-1.0, max=1.0)
                hist = hist / hist.sum()
                hist = hist[hist > 0]  # Remove zeros
                ent = -(hist * torch.log(hist)).sum()
                ch_entropy.append(ent)
            entropy_vals.append(torch.stack(ch_entropy))
        entropy = torch.stack(entropy_vals)  # [B, C]

        return {
            'mean': mean,
            'variance': var,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'entropy': entropy
        }

    def forward(
        self,
        x: torch.Tensor,
        reference_stats: dict = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute statistical consistency.

        Args:
            x: Input image [B, C, H, W]
            reference_stats: Reference statistics (optional)

        Returns:
            consistency: Consistency score [B]
            moments: Computed moments
        """
        moments = self.compute_moments(x)

        if reference_stats is not None:
            # Compute deviation from reference
            deviations = []
            for key in ['mean', 'variance', 'skewness', 'kurtosis', 'entropy']:
                if key in reference_stats:
                    ref = reference_stats[key]
                    curr = moments[key]
                    dev = torch.abs(curr - ref).mean(dim=1)  # [B]
                    deviations.append(dev)

            consistency = torch.stack(deviations, dim=1).mean(dim=1)  # [B]

        else:
            # Natural images typically have:
            # - Near-zero mean (after normalization)
            # - Moderate variance (0.1 - 0.3)
            # - Near-zero skewness
            # - Kurtosis around 3 (Gaussian)
            # - High entropy

            deviations = []

            # Mean deviation from zero
            mean_dev = torch.abs(moments['mean']).mean(dim=1)
            deviations.append(mean_dev)

            # Variance deviation from 0.2
            var_dev = torch.abs(moments['variance'] - 0.2).mean(dim=1)
            deviations.append(var_dev)

            # Skewness deviation from zero
            skew_dev = torch.abs(moments['skewness']).mean(dim=1)
            deviations.append(skew_dev)

            # Kurtosis deviation from 3
            kurt_dev = torch.abs(moments['kurtosis'] - 3.0).mean(dim=1)
            deviations.append(kurt_dev)

            consistency = torch.stack(deviations, dim=1).mean(dim=1)  # [B]

        return consistency, moments
