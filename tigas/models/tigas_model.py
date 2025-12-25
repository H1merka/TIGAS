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
from .constants import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_BASE_CHANNELS,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_STAGES,
    INPUT_MIN,
    INPUT_MAX,
    REGRESSION_HIDDEN_DIM_RATIO,
    REGRESSION_FINAL_DIM_RATIO,
    LINEAR_WEIGHT_STD
)


class TIGASModel(nn.Module):
    """
    Trained Image Generation Authenticity Score Model.
    
    OPTIMIZED for fast training with optional lightweight mode.
    """

    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        num_attention_heads: int = DEFAULT_ATTENTION_HEADS,
        dropout: float = 0.1,
        pretrained_backbone: bool = False,
        fast_mode: bool = True  # NEW: Enable fast training mode
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.fast_mode = fast_mode

        # ==================== Feature Extraction Branches ====================

        # Branch 1: Multi-scale perceptual features (ALWAYS used)
        self.perceptual_extractor = MultiScaleFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            stages=DEFAULT_STAGES
        )

        # Aggregate multi-scale perceptual features
        perceptual_channels = sum(self.perceptual_extractor.out_channels)
        self.perceptual_aggregator = nn.Sequential(
            nn.Linear(perceptual_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        if not fast_mode:
            # Full mode: all branches + attention
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

            # Cross-Modal Attention
            self.spectral_to_perceptual_attn = CrossModalAttention(
                query_dim=feature_dim,
                key_value_dim=feature_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )

            self.stat_to_perceptual_attn = CrossModalAttention(
                query_dim=feature_dim,
                key_value_dim=feature_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )

            self.self_attention = SelfAttention(
                dim=feature_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )

            self.feature_fusion = AdaptiveFeatureFusion(
                num_streams=3,
                feature_dim=feature_dim
            )
        else:
            # Fast mode: single branch with simple fusion
            # Lightweight auxiliary branch (much smaller)
            self.aux_branch = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, feature_dim // 2)
            )
            
            # Simple concat fusion
            self.fast_fusion = nn.Sequential(
                nn.Linear(feature_dim + feature_dim // 2, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        # ==================== Regression Head ====================
        self.regression_head = self._build_regression_head(feature_dim, dropout)

        # ==================== Auxiliary Heads (for training) ====================
        self.binary_classifier = self._build_classifier_head(feature_dim, dropout)

        # Initialize weights
        self._initialize_weights()

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Нормализовать входной тензор в диапазон [-1, 1].
        
        Поддерживает входы в диапазонах:
        - [0, 1]: преобразуется в [-1, 1]
        - [-1, 1]: остаётся без изменений
        - Другие: clamp в [-1, 1]
        """
        # Если вход в диапазоне [0, 1], преобразуем в [-1, 1]
        if x.min() >= 0 and x.max() <= 1:
            x = x * 2 - 1
        # Clamp для гарантии корректного диапазона
        x = torch.clamp(x, INPUT_MIN, INPUT_MAX)
        return x

    def _build_regression_head(
        self,
        feature_dim: int,
        dropout: float
    ) -> nn.Sequential:
        """Построить регрессионную голову."""
        hidden_dim = feature_dim // REGRESSION_HIDDEN_DIM_RATIO
        final_dim = feature_dim // REGRESSION_FINAL_DIM_RATIO

        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(final_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def _build_classifier_head(
        self,
        feature_dim: int,
        dropout: float
    ) -> nn.Sequential:
        """Построить классификационную голову."""
        hidden_dim = feature_dim // REGRESSION_HIDDEN_DIM_RATIO

        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

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
                nn.init.normal_(m.weight, 0, LINEAR_WEIGHT_STD)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        update_prototypes: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with fast_mode optimization.
        """
        # Normalize input
        x = self._normalize_input(x)

        if self.fast_mode:
            # FAST PATH: Single perceptual branch + lightweight aux
            return self._forward_fast(x, return_features)
        else:
            # FULL PATH: All branches + cross-modal attention
            return self._forward_full(x, return_features, update_prototypes)

    def _forward_fast(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Fast forward pass - perceptual only + lightweight aux."""
        # Extract perceptual features
        perceptual_features = self.perceptual_extractor(x)
        
        # Aggregate multi-scale features
        perceptual_concat = torch.cat([
            F.adaptive_avg_pool2d(feat, 1).flatten(1)
            for feat in perceptual_features
        ], dim=1)
        perceptual_feat = self.perceptual_aggregator(perceptual_concat)
        
        # Lightweight auxiliary features
        aux_feat = self.aux_branch(x)
        
        # Simple fusion
        fused = self.fast_fusion(torch.cat([perceptual_feat, aux_feat], dim=1))
        
        # Generate outputs
        tigas_score = self.regression_head(fused)
        tigas_score = torch.clamp(tigas_score, min=0.0, max=1.0)
        
        class_logits = self.binary_classifier(fused)
        class_logits = torch.clamp(class_logits, min=-10.0, max=10.0)

        outputs = {
            'score': tigas_score,
            'logits': class_logits
        }

        if return_features:
            outputs['features'] = {
                'perceptual': perceptual_feat,
                'fused': fused,
                'multi_scale': perceptual_features
            }

        return outputs

    def _forward_full(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        update_prototypes: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with all branches and attention."""
        # Extract features from all branches
        features = self._extract_features(x, update_prototypes)

        # Apply cross-modal attention
        attended_features = self._apply_cross_modal_attention(features)

        # Fuse features adaptively
        fused_features = self._fuse_features(attended_features)

        # Generate outputs
        outputs = self._generate_outputs(
            fused_features, features, attended_features, return_features
        )

        return outputs

    def _extract_features(
        self,
        x: torch.Tensor,
        update_prototypes: bool
    ) -> Dict[str, torch.Tensor]:
        """Извлечь признаки из всех ветвей."""
        # 1. Multi-scale perceptual features
        perceptual_features = self.perceptual_extractor(x)

        # Aggregate multi-scale features
        perceptual_concat = torch.cat([
            F.adaptive_avg_pool2d(feat, 1).flatten(1)
            for feat in perceptual_features
        ], dim=1)
        perceptual_feat = self.perceptual_aggregator(perceptual_concat)

        # 2. Spectral features
        spectral_feat, spectral_aux = self.spectral_analyzer(x)

        # 3. Statistical features
        statistical_feat, stat_aux = self.statistical_estimator(
            x, update_prototypes=update_prototypes
        )

        return {
            'perceptual': perceptual_feat,
            'perceptual_multi_scale': perceptual_features,
            'spectral': spectral_feat,
            'spectral_aux': spectral_aux,
            'statistical': statistical_feat,
            'statistical_aux': stat_aux
        }

    def _apply_cross_modal_attention(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Применить межмодальное внимание."""
        perceptual_feat = features['perceptual']
        spectral_feat = features['spectral']
        statistical_feat = features['statistical']

        # Prepare for attention (add sequence dimension)
        perceptual_seq = perceptual_feat.unsqueeze(1)
        spectral_seq = spectral_feat.unsqueeze(1)
        statistical_seq = statistical_feat.unsqueeze(1)

        # Spectral attending to perceptual context
        spectral_attended = self.spectral_to_perceptual_attn(
            query=spectral_seq,
            key_value=perceptual_seq
        ).squeeze(1)

        # Statistical attending to perceptual context
        stat_attended = self.stat_to_perceptual_attn(
            query=statistical_seq,
            key_value=perceptual_seq
        ).squeeze(1)

        return {
            'perceptual': perceptual_feat,
            'spectral_attended': spectral_attended,
            'stat_attended': stat_attended
        }

    def _fuse_features(
        self,
        attended_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Объединить признаки адаптивно."""
        # Combine all features with learned weights
        fused_features = self.feature_fusion([
            attended_features['perceptual'],
            attended_features['spectral_attended'],
            attended_features['stat_attended']
        ])

        # Apply self-attention for refinement
        fused_refined = self.self_attention(
            fused_features.unsqueeze(1)
        ).squeeze(1)

        return fused_refined

    def _generate_outputs(
        self,
        fused_features: torch.Tensor,
        extracted_features: Dict[str, torch.Tensor],
        attended_features: Dict[str, torch.Tensor],
        return_features: bool
    ) -> Dict[str, torch.Tensor]:
        """Сгенерировать выходы модели."""
        # Main output: TIGAS score [0, 1]
        tigas_score = self.regression_head(fused_features)
        
        # Clamp score to valid range [0, 1] to prevent NaN
        tigas_score = torch.clamp(tigas_score, min=0.0, max=1.0)

        # Auxiliary output: binary classification (for training)
        class_logits = self.binary_classifier(fused_features)
        
        # Clamp logits to prevent extreme values
        class_logits = torch.clamp(class_logits, min=-10.0, max=10.0)

        outputs = {
            'score': tigas_score,
            'logits': class_logits
        }

        if return_features:
            outputs['features'] = {
                'perceptual': extracted_features['perceptual'],
                'spectral': extracted_features['spectral'],
                'statistical': extracted_features['statistical'],
                'spectral_attended': attended_features['spectral_attended'],
                'stat_attended': attended_features['stat_attended'],
                'fused': fused_features,
                'spectral_aux': extracted_features['spectral_aux'],
                'statistical_aux': extracted_features['statistical_aux'],
                'multi_scale': extracted_features['perceptual_multi_scale']
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
    fast_mode: bool = True,  # Default to fast mode for training
    **kwargs
) -> TIGASModel:
    """
    Factory function to create TIGAS model.

    Args:
        img_size: Input image size
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to checkpoint file
        fast_mode: Use fast training mode (default: True)
        **kwargs: Additional arguments for TIGASModel

    Returns:
        model: TIGASModel instance
    """
    model = TIGASModel(img_size=img_size, fast_mode=fast_mode, **kwargs)

    if pretrained and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Handle both fast_mode and full model checkpoints
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as e:
            print(f"Warning: Partial weight loading due to architecture change: {e}")
        print(f"Loaded pretrained weights from {checkpoint_path}")

    return model
