"""
TIGAS Model - Trained Image Generation Authenticity Score

Main model architecture that combines multiple analysis branches:
- Multi-scale perceptual features
- Spectral analysis
- Statistical consistency
- Local-global coherence

The model outputs a continuous score [0, 1] where:
- 1.0 = Natural/Real image
- 0.0 = Generated/Fake image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .feature_extractors import (
    MultiScaleFeatureExtractor,
    SpectralAnalyzer,
    StatisticalMomentEstimator
)
from .attention import CrossModalAttention, SelfAttention
from .layers import AdaptiveFeatureFusion


class TIGASModel(nn.Module):
    """
    Trained Image Generation Authenticity Score Model.

    Architecture:
    1. Multi-scale CNN backbone for perceptual features
    2. Spectral analyzer for frequency domain analysis
    3. Statistical moment estimator for distribution consistency
    4. Cross-modal attention for feature fusion
    5. Regression head for final score [0, 1]

    Key innovations:
    - Combines complementary analysis approaches
    - Fully differentiable (can be used as loss function)
    - Attention-based fusion of multi-modal features
    - Learnable statistics of natural images
    """

    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        base_channels: int = 32,
        feature_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        pretrained_backbone: bool = False
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # ==================== Feature Extraction Branches ====================

        # Branch 1: Multi-scale perceptual features
        self.perceptual_extractor = MultiScaleFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            stages=[2, 3, 4, 3]
        )

        # Branch 2: Spectral analyzer
        self.spectral_analyzer = SpectralAnalyzer(
            in_channels=in_channels,
            hidden_dim=feature_dim
        )

        # Branch 3: Statistical moment estimator
        self.statistical_estimator = StatisticalMomentEstimator(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

        # ==================== Feature Aggregation ====================

        # Aggregate multi-scale perceptual features
        perceptual_channels = sum(self.perceptual_extractor.out_channels)
        self.perceptual_aggregator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(perceptual_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # ==================== Cross-Modal Attention ====================

        # Spectral attending to perceptual
        self.spectral_to_perceptual_attn = CrossModalAttention(
            query_dim=feature_dim,
            key_value_dim=feature_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Statistical attending to perceptual
        self.stat_to_perceptual_attn = CrossModalAttention(
            query_dim=feature_dim,
            key_value_dim=feature_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Self-attention on fused features
        self.self_attention = SelfAttention(
            dim=feature_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # ==================== Adaptive Fusion ====================

        self.feature_fusion = AdaptiveFeatureFusion(
            num_streams=3,  # perceptual, spectral, statistical
            feature_dim=feature_dim
        )

        # ==================== Regression Head ====================

        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.LayerNorm(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # ==================== Auxiliary Heads (for training) ====================

        # Binary classification head (real vs fake)
        self.binary_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        update_prototypes: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W], normalized to [0, 1] or [-1, 1]
            return_features: Whether to return intermediate features
            update_prototypes: Whether to update statistical prototypes (training only)

        Returns:
            Dictionary containing:
                - 'score': TIGAS authenticity score [B, 1], range [0, 1]
                - 'logits': Binary classification logits [B, 2] (optional, for training)
                - 'features': Intermediate features (if return_features=True)
        """
        batch_size = x.size(0)

        # Ensure input is in valid range
        if x.min() < -1.0 or x.max() > 1.0:
            x = torch.clamp(x, -1.0, 1.0)

        # ==================== Feature Extraction ====================

        # 1. Multi-scale perceptual features
        perceptual_features = self.perceptual_extractor(x)  # List of 4 feature maps

        # Concatenate and aggregate
        perceptual_concat = torch.cat([
            F.adaptive_avg_pool2d(feat, 1).flatten(1)
            for feat in perceptual_features
        ], dim=1)
        perceptual_feat = self.perceptual_aggregator(perceptual_concat)  # [B, feature_dim]

        # 2. Spectral features
        spectral_feat, spectral_aux = self.spectral_analyzer(x)  # [B, feature_dim]

        # 3. Statistical features
        statistical_feat, stat_aux = self.statistical_estimator(
            x, update_prototypes=update_prototypes
        )  # [B, feature_dim]

        # ==================== Cross-Modal Attention ====================

        # Prepare for attention (add sequence dimension)
        perceptual_seq = perceptual_feat.unsqueeze(1)  # [B, 1, feature_dim]
        spectral_seq = spectral_feat.unsqueeze(1)
        statistical_seq = statistical_feat.unsqueeze(1)

        # Spectral attending to perceptual context
        spectral_attended = self.spectral_to_perceptual_attn(
            query=spectral_seq,
            key_value=perceptual_seq
        ).squeeze(1)  # [B, feature_dim]

        # Statistical attending to perceptual context
        stat_attended = self.stat_to_perceptual_attn(
            query=statistical_seq,
            key_value=perceptual_seq
        ).squeeze(1)  # [B, feature_dim]

        # ==================== Adaptive Feature Fusion ====================

        # Combine all features with learned weights
        fused_features = self.feature_fusion([
            perceptual_feat,
            spectral_attended,
            stat_attended
        ])  # [B, feature_dim]

        # Apply self-attention for refinement
        fused_refined = self.self_attention(
            fused_features.unsqueeze(1)
        ).squeeze(1)  # [B, feature_dim]

        # ==================== Output Heads ====================

        # Main output: TIGAS score [0, 1]
        tigas_score = self.regression_head(fused_refined)  # [B, 1]

        # Auxiliary output: binary classification (for training)
        class_logits = self.binary_classifier(fused_refined)  # [B, 2]

        # ==================== Prepare Outputs ====================

        outputs = {
            'score': tigas_score,
            'logits': class_logits
        }

        if return_features:
            outputs['features'] = {
                'perceptual': perceptual_feat,
                'spectral': spectral_feat,
                'statistical': statistical_feat,
                'spectral_attended': spectral_attended,
                'stat_attended': stat_attended,
                'fused': fused_refined,
                'spectral_aux': spectral_aux,
                'statistical_aux': stat_aux,
                'multi_scale': perceptual_features
            }

        return outputs

    def compute_tigas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to compute TIGAS score.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            score: TIGAS score [B, 1]
        """
        with torch.no_grad():
            outputs = self.forward(x, return_features=False)
        return outputs['score']

    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
        }


def create_tigas_model(
    img_size: int = 256,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> TIGASModel:
    """
    Factory function to create TIGAS model.

    Args:
        img_size: Input image size
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional arguments for TIGASModel

    Returns:
        model: TIGASModel instance
    """
    model = TIGASModel(img_size=img_size, **kwargs)

    if pretrained and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {checkpoint_path}")

    return model
