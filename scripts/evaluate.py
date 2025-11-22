"""
Evaluation script for TIGAS model.

Usage:
    # Evaluate single image
    python scripts/evaluate.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt

    # Evaluate directory
    python scripts/evaluate.py --image_dir path/to/images/ --checkpoint checkpoints/best_model.pt

    # Use as command-line tool (after pip install)
    nair path/to/image.jpg --checkpoint model.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tigas import TIGAS
from tigas.utils.visualization import plot_score_distribution


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate images with TIGAS")

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to single image")
    group.add_argument("--image_dir", type=str, help="Path to directory of images")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Input image size")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot score distribution (for directories)")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    # Batch processing
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for directory evaluation")

    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    # Initialize TIGAS
    print("Initializing TIGAS...")
    tigas = TIGAS(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        device=args.device
    )

    # Model info
    info = tigas.get_model_info()
    print(f"Model info: {info}")

    # Single image evaluation
    if args.image:
        print(f"\nEvaluating image: {args.image}")
        score = tigas.compute_image(args.image)
        print(f"\nTIGAS Score: {score:.4f}")

        if score > 0.7:
            print("Assessment: Likely REAL/Natural")
        elif score < 0.3:
            print("Assessment: Likely GENERATED/Fake")
        else:
            print("Assessment: Uncertain")

    # Directory evaluation
    elif args.image_dir:
        print(f"\nEvaluating directory: {args.image_dir}")
        results = tigas.compute_directory(
            args.image_dir,
            return_paths=True,
            batch_size=args.batch_size
        )

        if not results:
            print("No images found!")
            return

        # Statistics
        scores = np.array(list(results.values()))
        print(f"\nResults for {len(results)} images:")
        print(f"  Mean score: {scores.mean():.4f}")
        print(f"  Std dev: {scores.std():.4f}")
        print(f"  Min score: {scores.min():.4f}")
        print(f"  Max score: {scores.max():.4f}")
        print(f"  Median: {np.median(scores):.4f}")

        # Count predictions
        real_count = (scores > 0.5).sum()
        fake_count = (scores <= 0.5).sum()
        print(f"\nPredictions:")
        print(f"  Real: {real_count} ({real_count/len(scores)*100:.1f}%)")
        print(f"  Fake: {fake_count} ({fake_count/len(scores)*100:.1f}%)")

        # Save results
        if args.output:
            import json
            output_data = {
                'statistics': {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max()),
                    'median': float(np.median(scores))
                },
                'results': {str(k): float(v) for k, v in results.items()}
            }

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved results to {args.output}")

        # Plot distribution
        if args.plot:
            from tigas.utils.visualization import plot_score_distribution
            plot_path = Path(args.image_dir).parent / "nair_distribution.png"
            # For plotting, we'd need separate real/fake, so just plot all
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
            plt.xlabel('TIGAS Score')
            plt.ylabel('Count')
            plt.title('TIGAS Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {plot_path}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
