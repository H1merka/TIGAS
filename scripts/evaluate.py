"""
TIGAS Evaluation Script - Evaluate images for AI-generation detection

This script evaluates single images or directories of images using the TIGAS model,
providing authenticity scores [0.0-1.0] where:
    - 1.0 = Natural/Real image
    - 0.0 = AI-Generated/Fake image

Usage Examples:
    # Single image
    python scripts/evaluate.py --image test.jpg --checkpoint model.pt

    # Directory
    python scripts/evaluate.py --image_dir images/ --checkpoint model.pt --batch_size 32

    # Auto-download model from HuggingFace Hub
    python scripts/evaluate.py --image test.jpg --auto_download

    # Save results and plot distribution
    python scripts/evaluate.py --image_dir images/ --output results.json --plot
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import json
from tigas import TIGAS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate images with TIGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python scripts/evaluate.py --image test.jpg --checkpoint model.pt

  # Directory with auto-download
  python scripts/evaluate.py --image_dir images/ --auto_download

  # Save results
  python scripts/evaluate.py --image_dir images/ --output results.json --plot
        """
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to single image file"
    )
    input_group.add_argument(
        "--image_dir",
        type=str,
        help="Path to directory containing images"
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt file). If not specified, uses auto-download."
    )
    model_group.add_argument(
        "--auto_download",
        action="store_true",
        help="Automatically download default model from HuggingFace Hub"
    )
    model_group.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input image size (default: 256)"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON file"
    )
    output_group.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save score distribution plot (for directories)"
    )
    output_group.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Custom path for plot image (default: auto-generated)"
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for directory evaluation (default: 32)"
    )
    proc_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified."
    )

    # Display options
    display_group = parser.add_argument_group("Display Options")
    display_group.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    display_group.add_argument(
        "--show_top",
        type=int,
        default=None,
        help="Show top N most real/fake images (directory mode)"
    )

    return parser.parse_args()


def evaluate_single_image(tigas, image_path, verbose=False):
    """Evaluate a single image and return score with interpretation."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {image_path}")
        print(f"{'='*60}")

    score = tigas.compute_image(image_path)

    # Interpretation
    if score > 0.7:
        assessment = "Likely REAL/Natural"
        confidence = "High"
    elif score > 0.5:
        assessment = "Probably REAL/Natural"
        confidence = "Medium"
    elif score > 0.3:
        assessment = "Probably FAKE/Generated"
        confidence = "Medium"
    else:
        assessment = "Likely FAKE/Generated"
        confidence = "High"

    result = {
        'image': str(image_path),
        'score': float(score),
        'assessment': assessment,
        'confidence': confidence
    }

    # Print results
    print(f"\nTIGAS Score: {score:.4f}")
    print(f"Assessment:  {assessment}")
    print(f"Confidence:  {confidence}")

    return result


def evaluate_directory(
    tigas,
    image_dir,
    batch_size=32,
    verbose=False,
    show_top=None
):
    """Evaluate all images in a directory and return results with statistics."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating directory: {image_dir}")
        print(f"{'='*60}")

    # Compute scores
    results = tigas.compute_directory(
        image_dir,
        return_paths=True,
        batch_size=batch_size
    )

    if not results:
        print("[ERROR] No images found in directory!")
        return None

    # Convert to arrays for statistics
    scores = np.array(list(results.values()))
    paths = list(results.keys())

    # Compute statistics
    stats = {
        'total_images': len(results),
        'mean_score': float(scores.mean()),
        'std_score': float(scores.std()),
        'min_score': float(scores.min()),
        'max_score': float(scores.max()),
        'median_score': float(np.median(scores)),
        'q25_score': float(np.percentile(scores, 25)),
        'q75_score': float(np.percentile(scores, 75))
    }

    # Count predictions
    real_count = (scores > 0.5).sum()
    fake_count = (scores <= 0.5).sum()

    stats['predicted_real'] = int(real_count)
    stats['predicted_fake'] = int(fake_count)
    stats['real_percentage'] = float(real_count / len(scores) * 100)
    stats['fake_percentage'] = float(fake_count / len(scores) * 100)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images:    {stats['total_images']:,}")
    print(f"\nScore Statistics:")
    print(f"  Mean:          {stats['mean_score']:.4f}")
    print(f"  Std Dev:       {stats['std_score']:.4f}")
    print(f"  Min:           {stats['min_score']:.4f}")
    print(f"  Max:           {stats['max_score']:.4f}")
    print(f"  Median:        {stats['median_score']:.4f}")
    print(f"  25th percentile: {stats['q25_score']:.4f}")
    print(f"  75th percentile: {stats['q75_score']:.4f}")
    print(f"\nPredictions (threshold=0.5):")
    print(f"  Real:          {stats['predicted_real']} ({stats['real_percentage']:.1f}%)")
    print(f"  Fake:          {stats['predicted_fake']} ({stats['fake_percentage']:.1f}%)")

    # Show top N images if requested
    if show_top:
        print(f"\n{'='*60}")
        print(f"TOP {show_top} MOST REAL IMAGES:")
        print(f"{'='*60}")
        sorted_indices = np.argsort(scores)[::-1][:show_top]
        for i, idx in enumerate(sorted_indices, 1):
            print(f"{i}. {Path(paths[idx]).name}: {scores[idx]:.4f}")

        print(f"\n{'='*60}")
        print(f"TOP {show_top} MOST FAKE IMAGES:")
        print(f"{'='*60}")
        sorted_indices = np.argsort(scores)[:show_top]
        for i, idx in enumerate(sorted_indices, 1):
            print(f"{i}. {Path(paths[idx]).name}: {scores[idx]:.4f}")

    return {
        'statistics': stats,
        'results': {str(k): float(v) for k, v in results.items()}
    }


def plot_score_distribution(scores, save_path, title="TIGAS Score Distribution"):
    """Generate and save score distribution plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed. Cannot generate plot.")
        print("           Install with: pip install matplotlib")
        return

    plt.figure(figsize=(12, 6))

    # Histogram
    plt.hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

    # Add threshold line
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')

    # Add mean line
    mean_score = scores.mean()
    plt.axvline(x=mean_score, color='green', linestyle='-.', linewidth=2,
                label=f'Mean ({mean_score:.3f})')

    # Styling
    plt.xlabel('TIGAS Score', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text = (
        f"Total: {len(scores)}\n"
        f"Mean: {mean_score:.3f}\n"
        f"Std: {scores.std():.3f}\n"
        f"Real: {(scores > 0.5).sum()} ({(scores > 0.5).sum()/len(scores)*100:.1f}%)\n"
        f"Fake: {(scores <= 0.5).sum()} ({(scores <= 0.5).sum()/len(scores)*100:.1f}%)"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[PLOT] Saved distribution plot to: {save_path}")


def main():
    args = parse_args()

    # Initialize TIGAS
    print("\n" + "="*60)
    print("TIGAS - Trained Image Generation Authenticity Score")
    print("="*60)

    print("\n[INIT] Loading TIGAS model...")

    tigas = TIGAS(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        device=args.device,
        auto_download=args.auto_download or (args.checkpoint is None)
    )

    # Display model info
    if args.verbose:
        info = tigas.get_model_info()
        print(f"\n[INFO] Model configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # Single image evaluation
    if args.image:
        result = evaluate_single_image(tigas, args.image, args.verbose)

        # Save result if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n[OUTPUT] Results saved to: {args.output}")

    # Directory evaluation
    elif args.image_dir:
        result = evaluate_directory(
            tigas,
            args.image_dir,
            batch_size=args.batch_size,
            verbose=args.verbose,
            show_top=args.show_top
        )

        if result is None:
            sys.exit(1)

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n[OUTPUT] Results saved to: {args.output}")

        # Generate plot if requested
        if args.plot:
            scores = np.array(list(result['results'].values()))

            if args.plot_path:
                plot_path = args.plot_path
            else:
                plot_path = Path(args.image_dir).parent / "tigas_distribution.png"

            plot_score_distribution(scores, plot_path)

    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
