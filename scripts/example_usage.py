"""
Example usage of TIGAS metric.
Demonstrates basic and advanced usage patterns.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from tigas import TIGAS, compute_nair_score
from tigas.models import create_tigas_model
from tigas.utils.visualization import visualize_predictions


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Initialize TIGAS
    nair = TIGAS(checkpoint_path=None, device='cpu')  # Use None for untrained model

    print("✓ TIGAS initialized")
    print(f"  Model info: {tigas.get_model_info()}")

    # Create dummy image
    dummy_image = torch.randn(1, 3, 256, 256)

    # Compute score
    score = nair(dummy_image)
    print(f"\n✓ TIGAS Score: {score.item():.4f}")

    if score.item() > 0.5:
        print("  Assessment: Likely REAL")
    else:
        print("  Assessment: Likely FAKE")


def example_batch_processing():
    """Batch processing example."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    nair = TIGAS(device='cpu')

    # Create batch of images
    batch_size = 8
    images = torch.randn(batch_size, 3, 256, 256)

    # Compute scores
    scores = nair(images)

    print(f"✓ Processed {batch_size} images")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Mean score: {scores.mean().item():.4f}")
    print(f"  Std score: {scores.std().item():.4f}")
    print(f"  Min score: {scores.min().item():.4f}")
    print(f"  Max score: {scores.max().item():.4f}")


def example_feature_extraction():
    """Feature extraction example."""
    print("\n" + "=" * 60)
    print("Example 3: Feature Extraction")
    print("=" * 60)

    nair = TIGAS(device='cpu')

    # Create image
    image = torch.randn(1, 3, 256, 256)

    # Get features
    outputs = nair(image, return_features=True)

    print("✓ Extracted features:")
    print(f"  Final score: {outputs['score'].item():.4f}")

    if 'features' in outputs:
        features = outputs['features']
        print(f"  Available features: {list(features.keys())}")
        print(f"  Fused features shape: {features['fused'].shape}")


def example_as_loss_function():
    """Using TIGAS as a loss function."""
    print("\n" + "=" * 60)
    print("Example 4: TIGAS as Loss Function")
    print("=" * 60)

    nair = TIGAS(device='cpu')

    # Simulated generator output
    generated_images = torch.randn(4, 3, 256, 256, requires_grad=True)

    # Compute TIGAS scores
    scores = nair(generated_images)

    # Loss: we want to maximize TIGAS score (make images more realistic)
    loss = 1.0 - scores.mean()

    print(f"✓ Generated images TIGAS score: {scores.mean().item():.4f}")
    print(f"  Loss (1 - score): {loss.item():.4f}")
    print(f"  Loss is differentiable: {loss.requires_grad}")

    # Backward pass
    loss.backward()
    print(f"✓ Gradients computed")
    print(f"  Gradient shape: {generated_images.grad.shape}")


def example_component_based():
    """Component-based metric (without trained model)."""
    print("\n" + "=" * 60)
    print("Example 5: Component-Based Metric")
    print("=" * 60)

    from tigas.metrics import TIGASMetric

    # Use component-based computation (no pretrained model needed)
    metric = TIGASMetric(use_model=False, device='cpu')

    images = torch.randn(4, 3, 256, 256)

    # Compute with components
    results = metric(images, return_components=True)

    print("✓ Component-based scores:")
    print(f"  Overall score: {results['score'].mean().item():.4f}")
    print(f"  Spectral score: {results['spectral_score'].mean().item():.4f}")
    print(f"  Statistical score: {results['statistical_score'].mean().item():.4f}")
    print(f"  Spatial score: {results['spatial_score'].mean().item():.4f}")


def example_model_creation():
    """Creating custom TIGAS models."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Model Creation")
    print("=" * 60)

    # Small model
    small_model = create_tigas_model(
        img_size=128,
        base_channels=16,
        feature_dim=128
    )
    small_info = small_model.get_model_size()

    print("✓ Small model:")
    print(f"  Parameters: {small_info['total_parameters']:,}")
    print(f"  Size: {small_info['model_size_mb']:.2f} MB")

    # Large model
    large_model = create_tigas_model(
        img_size=512,
        base_channels=64,
        feature_dim=512
    )
    large_info = large_model.get_model_size()

    print("\n✓ Large model:")
    print(f"  Parameters: {large_info['total_parameters']:,}")
    print(f"  Size: {large_info['model_size_mb']:.2f} MB")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TIGAS - Neural Authenticity and Realism Index")
    print("Usage Examples")
    print("=" * 60)

    try:
        example_basic_usage()
        example_batch_processing()
        example_feature_extraction()
        example_as_loss_function()
        example_component_based()
        example_model_creation()

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
