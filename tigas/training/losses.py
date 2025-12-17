"""
Loss functions for TIGAS training.
Combines multiple objectives for robust training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TIGASLoss(nn.Module):
    """
    Main TIGAS loss function.

    Combines:
    1. Regression loss (MSE/Smooth L1) for continuous score prediction
    2. Binary classification loss (BCE) for real/fake classification
    3. Ranking loss (Margin Ranking) to ensure real > fake
    4. Regularization losses
    """

    def __init__(
        self,
        regression_weight: float = 1.0,
        classification_weight: float = 0.5,
        ranking_weight: float = 0.3,
        use_smooth_l1: bool = True,
        margin: float = 0.5
    ):
        super().__init__()

        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.ranking_weight = ranking_weight
        self.margin = margin

        # Regression loss
        if use_smooth_l1:
            self.regression_loss = nn.SmoothL1Loss()
        else:
            self.regression_loss = nn.MSELoss()

        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs dict with 'score' and 'logits'
            labels: Ground truth labels [B, 1], 1.0 for real, 0.0 for fake

        Returns:
            Dictionary with individual and total losses
        """
        scores = outputs['score']  # [B, 1]
        logits = outputs['logits']  # [B, 2]

        # Validate inputs for NaN/Inf before computing loss
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            import warnings
            warnings.warn(f"[TIGAS LOSS] NaN/Inf detected in scores. Values: min={scores.min().item():.6f}, max={scores.max().item():.6f}")
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            import warnings
            warnings.warn(f"[TIGAS LOSS] NaN/Inf detected in logits")

        # 1. Regression loss
        reg_loss = self.regression_loss(scores, labels)
        
        # CRITICAL: Stop training if NaN/Inf detected (don't mask it)
        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            raise RuntimeError(
                f"[TIGAS LOSS] NaN/Inf in Regression Loss detected!\n"
                f"Scores stats - min: {scores.min().item():.6f}, max: {scores.max().item():.6f}, "
                f"mean: {scores.mean().item():.6f}, std: {scores.std().item():.6f}\n"
                f"Labels stats - min: {labels.min().item():.6f}, max: {labels.max().item():.6f}, "
                f"mean: {labels.mean().item():.6f}, std: {labels.std().item():.6f}\n"
                f"This indicates a problematic batch (possibly corrupted images). "
                f"Run: python scripts/check_dataset.py --data_root <dataset>"
            )

        # 2. Classification loss
        class_labels = labels.squeeze(1).long()  # [B]
        cls_loss = self.classification_loss(logits, class_labels)
        
        # CRITICAL: Stop training if NaN/Inf detected (don't mask it)
        if torch.isnan(cls_loss) or torch.isinf(cls_loss):
            raise RuntimeError(
                f"[TIGAS LOSS] NaN/Inf in Classification Loss detected!\n"
                f"Logits stats - min: {logits.min().item():.6f}, max: {logits.max().item():.6f}, "
                f"mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}\n"
                f"Class labels: {class_labels.tolist()}\n"
                f"This indicates a problematic batch (possibly corrupted images). "
                f"Run: python scripts/check_dataset.py --data_root <dataset>"
            )

        # 3. Ranking loss (for paired samples)
        # Encourage real images to have higher scores than fake images
        real_mask = (labels == 1.0).squeeze(1)
        fake_mask = (labels == 0.0).squeeze(1)

        if real_mask.any() and fake_mask.any():
            real_scores = scores[real_mask]
            fake_scores = scores[fake_mask]

            # Sample pairs
            num_pairs = min(len(real_scores), len(fake_scores))
            real_sample = real_scores[:num_pairs].squeeze(1)  # [num_pairs, 1] -> [num_pairs]
            fake_sample = fake_scores[:num_pairs].squeeze(1)  # [num_pairs, 1] -> [num_pairs]

            # Margin ranking loss: real_score should be > fake_score + margin
            target = torch.ones(num_pairs, device=scores.device)
            rank_loss = F.margin_ranking_loss(
                real_sample, fake_sample, target, margin=self.margin
            )
            # CRITICAL: Stop training if NaN/Inf detected (don't mask it)
            if torch.isnan(rank_loss) or torch.isinf(rank_loss):
                raise RuntimeError(
                    f"[TIGAS LOSS] NaN/Inf in Ranking Loss detected!\n"
                    f"Real scores: min={real_sample.min().item():.6f}, max={real_sample.max().item():.6f}\n"
                    f"Fake scores: min={fake_sample.min().item():.6f}, max={fake_sample.max().item():.6f}\n"
                    f"This indicates a problematic batch (possibly corrupted images). "
                    f"Run: python scripts/check_dataset.py --data_root <dataset>"
                )
        else:
            rank_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # Total loss
        total_loss = (
            self.regression_weight * reg_loss +
            self.classification_weight * cls_loss +
            self.ranking_weight * rank_loss
        )

        return {
            'total': total_loss,
            'regression': reg_loss,
            'classification': cls_loss,
            'ranking': rank_loss
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning embeddings.
    Pulls real images together, pushes fake images away.
    """

    def __init__(self, margin: float = 1.0, temperature: float = 0.5):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: Feature embeddings [B, D]
            labels: Labels [B, 1]

        Returns:
            Contrastive loss
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute pairwise distances
        dist_matrix = torch.cdist(features, features, p=2)

        # Create label matrix
        labels = labels.squeeze(1)
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Positive pairs (same label)
        pos_mask = label_matrix.float()
        pos_mask.fill_diagonal_(0)  # Exclude self

        # Negative pairs (different label)
        neg_mask = (~label_matrix).float()

        # Contrastive loss
        pos_loss = (pos_mask * dist_matrix.pow(2)).sum() / (pos_mask.sum() + 1e-8)
        neg_loss = (neg_mask * F.relu(self.margin - dist_matrix).pow(2)).sum() / (neg_mask.sum() + 1e-8)

        return pos_loss + neg_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using feature extractor.
    Encourages learning meaningful perceptual representations.
    """

    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss between images.

        Args:
            pred_images: Predicted images [B, 3, H, W]
            target_images: Target images [B, 3, H, W]

        Returns:
            Perceptual loss
        """
        with torch.no_grad():
            pred_features = self.feature_extractor(pred_images)
            target_features = self.feature_extractor(target_images)

        # Compute L2 loss across all feature scales
        loss = 0
        for pf, tf in zip(pred_features, target_features):
            loss += F.mse_loss(pf, tf)

        return loss / len(pred_features)


class CombinedLoss(nn.Module):
    """
    Combined loss with all components.
    Highly configurable for different training strategies.
    """

    def __init__(
        self,
        use_tigas_loss: bool = True,
        use_contrastive: bool = False,
        use_regularization: bool = True,
        tigas_loss_config: Optional[dict] = None,
        contrastive_config: Optional[dict] = None,
        reg_weight: float = 1e-4
    ):
        super().__init__()

        self.use_tigas_loss = use_tigas_loss
        self.use_contrastive = use_contrastive
        self.use_regularization = use_regularization
        self.reg_weight = reg_weight

        # Initialize losses
        if use_tigas_loss:
            tigas_config = tigas_loss_config or {}
            self.tigas_loss = TIGASLoss(**tigas_config)

        if use_contrastive:
            contrastive_config = contrastive_config or {}
            self.contrastive_loss = ContrastiveLoss(**contrastive_config)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs
            labels: Ground truth labels
            model: Model (for regularization)

        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0

        # TIGAS loss
        if self.use_tigas_loss:
            tigas_losses = self.tigas_loss(outputs, labels)
            losses.update(tigas_losses)
            total_loss = total_loss + tigas_losses['total']

        # Contrastive loss
        if self.use_contrastive and 'features' in outputs:
            fused_features = outputs['features']['fused']
            contrast_loss = self.contrastive_loss(fused_features, labels)
            losses['contrastive'] = contrast_loss
            total_loss = total_loss + 0.1 * contrast_loss

        # L2 regularization
        if self.use_regularization and model is not None:
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            losses['l2_regularization'] = l2_reg
            total_loss = total_loss + self.reg_weight * l2_reg

        losses['combined_total'] = total_loss

        return losses


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Focuses on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Class logits [B, num_classes]
            labels: Labels [B]

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()
