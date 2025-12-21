"""
TIGAS Usage Examples - Demonstrates various usage patterns

This script showcases different ways to use TIGAS for:
- Single image evaluation
- Batch processing
- Feature extraction
- Using as a differentiable loss function
- Directory processing
- Model configuration

Run: python scripts/example_usage.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tigas import TIGAS, compute_tigas_score
from tigas.models import create_tigas_model


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")


def example_1_basic_usage():
    """Example 1: Basic usage with single image evaluation."""
    print_section("Example 1: Basic Usage - Single Image")

    # Initialize TIGAS (will attempt auto-download if no checkpoint provided)
    tigas = TIGAS(device='cpu')
    print("✓ TIGAS initialized")

    # Get model info
    info = tigas.get_model_info()
    print(f"\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Create dummy image (in practice, load real image)
    dummy_image = torch.randn(1, 3, 256, 256)

    # Compute score
    outputs = tigas(dummy_image)
    score = outputs.item()

    print(f"\nTIGAS Score: {score:.4f}")

    # Interpretation
    if score > 0.7:
        print("Assessment: Likely REAL/Natural (High Confidence)")
    elif score > 0.5:
        print("Assessment: Probably REAL/Natural (Medium Confidence)")
    elif score > 0.3:
        print("Assessment: Probably FAKE/Generated (Medium Confidence)")
    else:
        print("Assessment: Likely FAKE/Generated (High Confidence)")


def example_2_batch_processing():
    """Example 2: Batch processing multiple images."""
    print_section("Example 2: Batch Processing")

    tigas = TIGAS(device='cpu')

    # Create batch of images
    batch_size = 8
    images = torch.randn(batch_size, 3, 256, 256)

    print(f"Processing batch of {batch_size} images...")

    # Compute scores
    scores = tigas(images)

    print(f"\n✓ Processed {batch_size} images")
    print(f"\nBatch Statistics:")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Mean score:   {scores.mean().item():.4f}")
    print(f"  Std score:    {scores.std().item():.4f}")
    print(f"  Min score:    {scores.min().item():.4f}")
    print(f"  Max score:    {scores.max().item():.4f}")

    # Individual scores
    print(f"\nIndividual Scores:")
    for i, score in enumerate(scores.squeeze(), 1):
        assessment = "REAL" if score > 0.5 else "FAKE"
        print(f"  Image {i}: {score:.4f} ({assessment})")


def example_3_feature_extraction():
    """Example 3: Extract intermediate features."""
    print_section("Example 3: Feature Extraction")

    tigas = TIGAS(device='cpu')

    # Create image
    image = torch.randn(1, 3, 256, 256)

    # Get features
    outputs = tigas(image, return_features=True)

    score = outputs['score']
    features = outputs['features']

    print(f"✓ Extracted features from model")
    print(f"\nFinal Score: {score.item():.4f}")

    print(f"\nAvailable Features:")
    for key, value in features.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: List of {len(value)} tensors")
            for i, tensor in enumerate(value):
                print(f"    [{i}]: {tensor.shape}")
        else:
            print(f"  {key}: {type(value)}")

    # Access specific features
    print(f"\nFused Features Shape: {features['fused'].shape}")
    print(f"Perceptual Features Shape: {features['perceptual'].shape}")
    print(f"Spectral Features Shape: {features['spectral'].shape}")
    print(f"Statistical Features Shape: {features['statistical'].shape}")


def example_4_as_loss_function():
    """Example 4: Use TIGAS as a differentiable loss function."""
    print_section("Example 4: TIGAS as Differentiable Loss")

    tigas = TIGAS(device='cpu')

    # Simulated generator output (with gradient tracking)
    generated_images = torch.randn(4, 3, 256, 256, requires_grad=True)

    print("Simulating generator training...")

    # Compute TIGAS scores
    scores = tigas(generated_images)

    # Loss: maximize TIGAS score (make images more realistic)
    # We minimize (1 - score) to make score closer to 1.0
    loss = 1.0 - scores.mean()

    print(f"\n✓ Generated images evaluated")
    print(f"\nScores for generated images:")
    for i, score in enumerate(scores.squeeze(), 1):
        print(f"  Image {i}: {score:.4f}")

    print(f"\nMean TIGAS Score: {scores.mean().item():.4f}")
    print(f"Loss (1 - score): {loss.item():.4f}")
    print(f"Loss requires_grad: {loss.requires_grad}")

    # Backward pass
    loss.backward()

    print(f"\n✓ Gradients computed")
    print(f"  Gradient shape: {generated_images.grad.shape}")
    print(f"  Gradient mean: {generated_images.grad.mean().item():.6f}")
    print(f"  Gradient std: {generated_images.grad.std().item():.6f}")

    print("\nIn a real training loop, you would:")
    print("  1. optimizer.zero_grad()")
    print("  2. generated_images = generator(noise)")
    print("  3. loss = 1.0 - tigas(generated_images).mean()")
    print("  4. loss.backward()")
    print("  5. optimizer.step()")


def example_5_directory_processing():
    """Example 5: Process entire directory (simulated)."""
    print_section("Example 5: Directory Processing")

    tigas = TIGAS(device='cpu')

    # Simulate directory processing
    print("Simulating directory processing...")
    print("(In practice, use: tigas.compute_directory('path/to/images/'))")

    # Simulate batch results
    num_images = 50
    simulated_scores = torch.rand(num_images) * 0.6 + 0.2  # Random scores [0.2, 0.8]

    print(f"\n✓ Processed {num_images} images (simulated)")

    # Compute statistics
    scores_np = simulated_scores.numpy()
    mean_score = scores_np.mean()
    std_score = scores_np.std()
    median_score = np.median(scores_np)

    print(f"\nStatistics:")
    print(f"  Mean:   {mean_score:.4f}")
    print(f"  Std:    {std_score:.4f}")
    print(f"  Median: {median_score:.4f}")
    print(f"  Min:    {scores_np.min():.4f}")
    print(f"  Max:    {scores_np.max():.4f}")

    # Count predictions
    real_count = (scores_np > 0.5).sum()
    fake_count = (scores_np <= 0.5).sum()

    print(f"\nPredictions (threshold=0.5):")
    print(f"  Real: {real_count} ({real_count/num_images*100:.1f}%)")
    print(f"  Fake: {fake_count} ({fake_count/num_images*100:.1f}%)")

    print(f"\nExample code:")
    print(f"  results = tigas.compute_directory(")
    print(f"      'path/to/images/',")
    print(f"      return_paths=True,")
    print(f"      batch_size=32")
    print(f"  )")


def example_6_model_configurations():
    """Example 6: Different model configurations (Fast vs Full mode)."""
    print_section("Example 6: Model Configurations")

    print("Creating different TIGAS model configurations...\n")

    # Fast Mode (default, optimized for speed)
    print("1. Fast Mode (default, optimized for speed):")
    fast_model = create_tigas_model(
        img_size=128,
        base_channels=32,
        feature_dim=256,
        fast_mode=True
    )
    fast_info = fast_model.get_model_size()
    print(f"   Input size: 128x128")
    print(f"   Parameters: {fast_info['total_parameters']:,}")
    print(f"   Model size: {fast_info['model_size_mb']:.2f} MB")
    print(f"   Use case: Fast training, limited VRAM (4-8 GB)")

    # Full Mode (all branches, higher accuracy)
    print("\n2. Full Mode (all branches, higher accuracy):")
    full_model = create_tigas_model(
        img_size=256,
        base_channels=32,
        feature_dim=256,
        fast_mode=False
    )
    full_info = full_model.get_model_size()
    print(f"   Input size: 256x256")
    print(f"   Parameters: {full_info['total_parameters']:,}")
    print(f"   Model size: {full_info['model_size_mb']:.2f} MB")
    print(f"   Use case: Maximum accuracy, 16+ GB VRAM")

    # Large model
    print("\n3. Large Full Mode (for maximum accuracy):")
    large_model = create_tigas_model(
        img_size=512,
        base_channels=64,
        feature_dim=512,
        fast_mode=False
    )
    large_info = large_model.get_model_size()
    print(f"   Input size: 512x512")
    print(f"   Parameters: {large_info['total_parameters']:,}")
    print(f"   Model size: {large_info['model_size_mb']:.2f} MB")
    print(f"   Use case: Research, 24+ GB VRAM (A100/RTX 4090)")

    print("\n" + "-"*60)
    print("Training commands:")
    print("  Fast Mode:  python scripts/train_script.py --img_size 128 --batch_size 16")
    print("  Full Mode:  python scripts/train_script.py --img_size 256 --batch_size 8 --full_mode")


def example_7_convenience_function():
    """Example 7: Using convenience function."""
    print_section("Example 7: Convenience Function")

    print("Using compute_tigas_score() convenience function...")

    # Create dummy image
    dummy_image = torch.randn(1, 3, 256, 256)

    # Compute score using convenience function
    score = compute_tigas_score(dummy_image, checkpoint_path=None, device='cpu')

    print(f"\n✓ Score computed: {score:.4f}")

    print("\nExample usage with file path:")
    print("  score = compute_tigas_score('image.jpg', checkpoint_path='model.pt')")
    print("  print(f'Score: {score:.4f}')")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TIGAS - Trained Image Generation Authenticity Score")
    print("Usage Examples & Tutorial")
    print("="*70)

    try:
        example_1_basic_usage()
        example_2_batch_processing()
        example_3_feature_extraction()
        example_4_as_loss_function()
        example_5_directory_processing()
        example_6_model_configurations()
        example_7_convenience_function()

        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)

        print("\nNext Steps:")
        print("  1. Try evaluating real images:")
        print("     python scripts/evaluate.py --image your_image.jpg --auto_download")
        print("\n  2. Train your own model:")
        print("     python scripts/train_script.py --data_root /path/to/data")
        print("\n  3. Check out the documentation in docs/")
        print()

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
