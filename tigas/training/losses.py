"""
Loss functions for TIGAS training.
Combines multiple objectives for robust training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import warnings


class TIGASLoss(nn.Module):
    """
    Main TIGAS loss function.

    Combines:
    1. Regression loss (MSE/Smooth L1) for continuous score prediction
    2. Binary classification loss (BCE) for real/fake classification
    3. Ranking loss (Margin Ranking) to ensure real > fake with random pair sampling
    4. Regularization losses
    """

    def __init__(
        self,
        regression_weight: float = 1.0,
        classification_weight: float = 0.5,
        ranking_weight: float = 0.3,
        use_smooth_l1: bool = True,
        margin: float = 0.5,
        max_ranking_pairs: int = 64
    ):
        """
        Args:
            regression_weight: Weight for regression loss
            classification_weight: Weight for classification loss
            ranking_weight: Weight for ranking loss
            use_smooth_l1: Use Smooth L1 loss (True) or MSE (False) for regression
            margin: Margin for ranking loss
            max_ranking_pairs: Maximum number of random pairs for ranking loss
        """
        super().__init__()

        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.ranking_weight = ranking_weight
        self.margin = margin
        self.max_ranking_pairs = max_ranking_pairs
        
        # Iteration counter for reproducible random sampling
        self.register_buffer('_iteration', torch.tensor(0, dtype=torch.long))

        # Regression loss
        if use_smooth_l1:
            self.regression_loss = nn.SmoothL1Loss()
        else:
            self.regression_loss = nn.MSELoss()

        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()

    def _compute_ranking_loss(
        self, 
        real_scores: torch.Tensor, 
        fake_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking loss with REPRODUCIBLE random pair sampling.
        
        Uses CPU-based random generator with iteration-based seed for
        deterministic sampling across runs while still providing varied
        pairs during training.
        
        Args:
            real_scores: Scores for real images [N_real, 1]
            fake_scores: Scores for fake images [N_fake, 1]
            
        Returns:
            Margin ranking loss
        """
        n_real = len(real_scores)
        n_fake = len(fake_scores)
        
        # Calculate number of pairs to sample
        # Can create up to n_real * n_fake pairs, but limit for efficiency
        max_possible_pairs = n_real * n_fake
        num_pairs = min(self.max_ranking_pairs, max_possible_pairs, n_real, n_fake)
        
        if num_pairs == 0:
            return torch.tensor(0.0, device=real_scores.device, dtype=real_scores.dtype)
        
        # Reproducible random sampling using CPU generator with iteration seed
        # This ensures determinism while still varying samples across iterations
        generator = torch.Generator()  # CPU generator for reproducibility
        generator.manual_seed(42 + self._iteration.item())
        
        # Generate indices on CPU, then transfer to device
        real_indices = torch.randint(
            0, n_real, (num_pairs,), generator=generator
        ).to(real_scores.device)
        fake_indices = torch.randint(
            0, n_fake, (num_pairs,), generator=generator
        ).to(fake_scores.device)
        
        # Increment iteration counter for next call
        self._iteration += 1
        
        real_sample = real_scores[real_indices].squeeze(1)  # [num_pairs]
        fake_sample = fake_scores[fake_indices].squeeze(1)  # [num_pairs]
        
        # Margin ranking loss: real_score should be > fake_score + margin
        target = torch.ones(num_pairs, device=real_scores.device)
        
        rank_loss = F.margin_ranking_loss(
            real_sample, fake_sample, target, margin=self.margin
        )
        
        return rank_loss

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
            warnings.warn(
                f"[TIGAS LOSS] NaN/Inf detected in scores. "
                f"Values: min={scores.min().item():.6f}, max={scores.max().item():.6f}"
            )
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            warnings.warn("[TIGAS LOSS] NaN/Inf detected in logits")

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

        # 3. Ranking loss with random pair sampling
        real_mask = (labels == 1.0).squeeze(1)
        fake_mask = (labels == 0.0).squeeze(1)

        if real_mask.any() and fake_mask.any():
            real_scores = scores[real_mask]
            fake_scores = scores[fake_mask]

            # Use random sampling instead of deterministic slicing
            rank_loss = self._compute_ranking_loss(real_scores, fake_scores)
            
            # CRITICAL: Stop training if NaN/Inf detected (don't mask it)
            if torch.isnan(rank_loss) or torch.isinf(rank_loss):
                raise RuntimeError(
                    f"[TIGAS LOSS] NaN/Inf in Ranking Loss detected!\n"
                    f"Real scores: min={real_scores.min().item():.6f}, max={real_scores.max().item():.6f}\n"
                    f"Fake scores: min={fake_scores.min().item():.6f}, max={fake_scores.max().item():.6f}\n"
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
    Pulls images with same label together, pushes different labels apart.
    
    Uses InfoNCE-style loss which is more stable than margin-based contrastive.
    """

    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss using temperature-scaled cosine similarity.
        
        Args:
            features: Feature embeddings [B, D]
            labels: Labels [B, 1] - 1.0 for real, 0.0 for fake

        Returns:
            Contrastive loss scalar
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        
        # Normalize features for cosine similarity
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        
        # Create label matrix (same class = positive pair)
        labels_flat = labels.squeeze(1)
        label_matrix = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)
        
        # Positive pairs mask (same label, excluding self)
        pos_mask = label_matrix.float()
        pos_mask.fill_diagonal_(0)
        
        # For numerical stability
        sim_matrix_exp = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0])
        
        # Exclude self-similarity
        self_mask = torch.eye(features.size(0), device=features.device, dtype=torch.bool)
        sim_matrix_exp = sim_matrix_exp.masked_fill(self_mask, 0)
        
        # Compute loss: -log(sum of positive similarities / sum of all similarities)
        pos_sim = (sim_matrix_exp * pos_mask).sum(dim=1)
        all_sim = sim_matrix_exp.sum(dim=1)
        
        # Avoid log(0) by adding small epsilon
        loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8))
        
        # Only count samples that have positive pairs
        valid_mask = pos_mask.sum(dim=1) > 0
        if valid_mask.any():
            return loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using feature extractor.
    
    Computes multi-scale feature matching loss between predicted and target images.
    Can use any feature extractor that returns a list of feature maps.
    """

    def __init__(self, feature_extractor: nn.Module, weights: Optional[list] = None):
        """
        Args:
            feature_extractor: Network that returns list of feature maps
            weights: Optional per-scale weights (default: equal weighting)
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        self.weights = weights

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
            Perceptual loss (scalar)
        """
        with torch.no_grad():
            pred_features = self.feature_extractor(pred_images)
            target_features = self.feature_extractor(target_images)
        
        # Handle case where extractor returns single tensor
        if not isinstance(pred_features, (list, tuple)):
            pred_features = [pred_features]
            target_features = [target_features]

        # Compute weighted L2 loss across all feature scales
        num_scales = len(pred_features)
        weights = self.weights or [1.0 / num_scales] * num_scales
        
        loss = torch.tensor(0.0, device=pred_images.device, dtype=pred_images.dtype)
        for w, pf, tf in zip(weights, pred_features, target_features):
            loss = loss + w * F.mse_loss(pf, tf)

        return loss


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
        reg_weight: float = 1e-4,
        contrastive_weight: float = 0.1
    ):
        super().__init__()

        self.use_tigas_loss = use_tigas_loss
        self.use_contrastive = use_contrastive
        self.use_regularization = use_regularization
        self.reg_weight = reg_weight
        self.contrastive_weight = contrastive_weight

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
            total_loss = total_loss + self.contrastive_weight * contrast_loss

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
