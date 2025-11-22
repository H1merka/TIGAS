"""
Visualization utilities for TIGAS.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


def visualize_predictions(
    images: torch.Tensor,
    scores: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    num_images: int = 8,
    save_path: Optional[str] = None
):
    """
    Visualize predictions with TIGAS scores.

    Args:
        images: Images [B, C, H, W]
        scores: TIGAS scores [B, 1]
        labels: Ground truth labels [B, 1] (optional)
        num_images: Number of images to show
        save_path: Path to save figure
    """
    num_images = min(num_images, len(images))

    # Denormalize images
    images = images[:num_images].cpu()
    if images.min() < 0:
        images = images * 0.5 + 0.5  # From [-1, 1] to [0, 1]

    scores = scores[:num_images].cpu().numpy()
    if labels is not None:
        labels = labels[:num_images].cpu().numpy()

    # Create figure
    rows = int(np.ceil(num_images / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    for i in range(num_images):
        ax = axes[i]

        # Convert to numpy and transpose
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.axis('off')

        # Title with score and label
        score = scores[i][0]
        title = f'TIGAS: {score:.3f}'

        if labels is not None:
            label = labels[i][0]
            gt_text = 'Real' if label > 0.5 else 'Fake'
            title += f'\nGT: {gt_text}'

            # Color based on correctness
            pred_real = score > 0.5
            is_correct = (pred_real and label > 0.5) or (not pred_real and label < 0.5)
            color = 'green' if is_correct else 'red'
        else:
            color = 'blue'

        ax.set_title(title, color=color, fontsize=10)

    # Hide extra subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
    metrics: List[str] = ['total', 'regression', 'classification'],
    save_path: Optional[str] = None
):
    """
    Plot training history.

    Args:
        train_history: Training history
        val_history: Validation history
        metrics: Metrics to plot
        save_path: Path to save figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4))

    if num_metrics == 1:
        axes = [axes]

    epochs_train = list(range(len(train_history)))
    epochs_val = list(range(len(val_history)))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Extract metric values
        train_values = [h.get(metric, 0) for h in train_history]
        val_values = [h.get(metric, 0) for h in val_history]

        # Plot
        ax.plot(epochs_train, train_values, label='Train', marker='o', markersize=3)
        if val_values:
            ax.plot(epochs_val, val_values, label='Val', marker='s', markersize=3)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_score_distribution(
    real_scores: np.ndarray,
    fake_scores: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot distribution of TIGAS scores for real vs fake images.

    Args:
        real_scores: Scores for real images
        fake_scores: Scores for fake images
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    # Histograms
    plt.hist(real_scores, bins=50, alpha=0.5, label='Real', color='green', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.5, label='Fake', color='red', density=True)

    plt.xlabel('TIGAS Score')
    plt.ylabel('Density')
    plt.title('TIGAS Score Distribution: Real vs Fake')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add vertical line at 0.5 threshold
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
