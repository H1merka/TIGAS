"""
Dataset validation script - checks image integrity before training.

Usage:
    python scripts/validate_dataset.py \
        --dataset_dir C:/Dev/TIGAS_dataset/TIGAS \
        --csv_file train.csv \
        --output_dir validation_report/ \
        --remove_corrupted \
        --update_csv

Features:
    - Validates all images in directory
    - Checks image integrity (not truncated, valid format)
    - Generates JSON report with statistics
    - Optionally removes corrupted files
    - Optionally regenerates CSV without corrupted entries
    - Detailed logging
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import json
from typing import List, Dict, Tuple, Any
import pandas as pd
from tqdm import tqdm
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_image(
    img_path: Path,
    max_pixels: int = 50_000_000,
    max_aspect_ratio: float = 10.0,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    Validate a single image file with detailed checks.
    
    Args:
        img_path: Path to image file
        max_pixels: Maximum allowed pixels (default: 50M)
        max_aspect_ratio: Maximum width/height ratio
        strict_mode: If True, any warning = invalid
        
    Returns:
        {
            'valid': bool,
            'error': str,           # critical error
            'warnings': [],         # non-critical issues
            'metadata': {
                'mode': str,
                'size': tuple,
                'pixels': int,
                'aspect_ratio': float,
                'file_size_mb': float
            }
        }
    """
    result = {
        'valid': True,
        'error': '',
        'warnings': [],
        'metadata': {
            'mode': None,
            'size': None,
            'pixels': None,
            'aspect_ratio': None,
            'file_size_mb': None
        }
    }
    
    try:
        if not img_path.exists():
            result['valid'] = False
            result['error'] = "File not found"
            return result
        
        # Get file size
        result['metadata']['file_size_mb'] = img_path.stat().st_size / (1024 * 1024)
        
        # Try to open and get metadata
        with Image.open(img_path) as img:
            # Get basic info
            result['metadata']['mode'] = img.mode
            result['metadata']['size'] = img.size
            
            width, height = img.size
            pixels = width * height
            result['metadata']['pixels'] = pixels
            
            # Check image mode
            if img.mode == 'P':
                result['warnings'].append("Palette image (mode='P') - should be RGB/RGBA")
            elif img.mode == 'L':
                result['warnings'].append("Grayscale image (mode='L') - should be RGB")
            elif img.mode == 'RGBA':
                result['warnings'].append("RGBA image with transparency - will be converted to RGB")
            elif img.mode not in ['RGB', '1']:
                result['warnings'].append(f"Unusual mode: {img.mode}")
            
            # Check size (decompression bomb)
            if pixels > max_pixels:
                result['warnings'].append(
                    f"Huge image: {pixels:,} pixels (>{max_pixels:,}) - potential DOS attack"
                )
            
            # Check aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            result['metadata']['aspect_ratio'] = aspect_ratio
            if aspect_ratio > max_aspect_ratio:
                result['warnings'].append(
                    f"Extreme aspect ratio: {aspect_ratio:.1f}:1 - suspicious"
                )
            
            # Verify image integrity
            img.verify()
        
        # Verify again with convert to RGB (actual use case)
        with Image.open(img_path) as img:
            import warnings as warn_module
            with warn_module.catch_warnings():
                warn_module.simplefilter("ignore")  # Suppress PIL warnings during validation
                img_rgb = img.convert('RGB')
                # Force load to catch truncation errors
                _ = img_rgb.tobytes()
        
        # Set validity based on strict mode
        if strict_mode and result['warnings']:
            result['valid'] = False
            result['error'] = f"Warnings in strict mode: {'; '.join(result['warnings'])}"
        
        return result
    
    except Image.UnidentifiedImageError:
        result['valid'] = False
        result['error'] = "Invalid image format"
        return result
    except OSError as e:
        result['valid'] = False
        result['error'] = f"OSError: {str(e)}"
        return result
    except Exception as e:
        result['valid'] = False
        result['error'] = f"Error: {str(e)}"
        return result


def scan_directory(
    dataset_dir: Path,
    verbose: bool = False,
    max_pixels: int = 50_000_000,
    max_aspect_ratio: float = 10.0,
    strict_mode: bool = False
) -> Dict:
    """
    Scan directory and validate all images with categorized results.
    
    Args:
        dataset_dir: Root dataset directory
        verbose: Print detailed progress
        max_pixels: Maximum allowed pixels
        max_aspect_ratio: Maximum aspect ratio
        strict_mode: Treat warnings as errors
        
    Returns:
        {
            'total': int,
            'valid': int,
            'issues': {
                'truncated': int,
                'palette': int,
                'huge': int,
                'aspect': int,
                'invalid_format': int,
                'grayscale': int,
                'rgba': int
            },
            'corrupted_files': [
                {
                    'path': str,
                    'relative_path': str,
                    'error': str,
                    'warnings': []
                }
            ]
        }
    """
    corrupted_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    # Find all image files
    all_files = []
    for ext in image_extensions:
        all_files.extend(dataset_dir.glob(f'**/*{ext}'))
        all_files.extend(dataset_dir.glob(f'**/*{ext.upper()}'))
    
    all_files = list(set(all_files))  # Remove duplicates
    
    if not all_files:
        print(f"[WARNING] No images found in {dataset_dir}")
        return {
            'total': 0,
            'valid': 0,
            'issues': {
                'truncated': 0,
                'palette': 0,
                'huge': 0,
                'aspect': 0,
                'invalid_format': 0,
                'grayscale': 0,
                'rgba': 0
            },
            'corrupted_files': []
        }
    
    # Validate each image
    valid_count = 0
    issues_count = {
        'truncated': 0,
        'palette': 0,
        'huge': 0,
        'aspect': 0,
        'invalid_format': 0,
        'grayscale': 0,
        'rgba': 0
    }
    
    print(f"\n[VALIDATION] Scanning {len(all_files)} files...")
    
    for img_path in tqdm(all_files, desc="Validating images"):
        validation = validate_image(
            img_path,
            max_pixels=max_pixels,
            max_aspect_ratio=max_aspect_ratio,
            strict_mode=strict_mode
        )
        
        if validation['valid'] and not validation['warnings']:
            valid_count += 1
        else:
            relative_path = img_path.relative_to(dataset_dir)
            
            # Categorize issues
            if validation['error']:
                if 'truncated' in validation['error'].lower():
                    issues_count['truncated'] += 1
                elif 'format' in validation['error'].lower():
                    issues_count['invalid_format'] += 1
                else:
                    issues_count['truncated'] += 1  # Default to truncated
            
            # Categorize warnings
            for warning in validation['warnings']:
                if 'Palette' in warning:
                    issues_count['palette'] += 1
                elif 'Huge' in warning:
                    issues_count['huge'] += 1
                elif 'aspect' in warning:
                    issues_count['aspect'] += 1
                elif 'Grayscale' in warning:
                    issues_count['grayscale'] += 1
                elif 'RGBA' in warning:
                    issues_count['rgba'] += 1
            
            corrupted_files.append({
                'path': str(img_path),
                'relative_path': str(relative_path),
                'error': validation['error'],
                'warnings': validation['warnings'],
                'metadata': validation['metadata']
            })
            
            if verbose:
                print(f"  ✗ {relative_path}")
                if validation['error']:
                    print(f"     Error: {validation['error']}")
                for warning in validation['warnings']:
                    print(f"     Warning: {warning}")
    
    return {
        'total': len(all_files),
        'valid': valid_count,
        'issues': issues_count,
        'corrupted_files': corrupted_files
    }


def update_csv(csv_path: Path, corrupted_paths: List[str], output_path: Path) -> int:
    """
    Update CSV by removing entries with corrupted files.
    
    Args:
        csv_path: Original CSV path
        corrupted_paths: List of absolute paths to corrupted files
        output_path: Output CSV path
        
    Returns:
        Number of rows removed
    """
    if not csv_path.exists():
        print(f"[WARNING] CSV file not found: {csv_path}")
        return 0
    
    # Normalize corrupted paths
    corrupted_normalized = {Path(p).resolve() for p in corrupted_paths}
    
    # Load CSV
    df = pd.read_csv(csv_path)
    initial_len = len(df)
    
    # Filter out corrupted files
    csv_dir = csv_path.parent
    
    def is_corrupted(row):
        img_path_str = str(row['image_path']).replace('\\', '/')
        img_path = Path(img_path_str)
        
        if not img_path.is_absolute():
            img_path = csv_dir / img_path
        
        return img_path.resolve() in corrupted_normalized
    
    df_clean = df[~df.apply(is_corrupted, axis=1)]
    removed = initial_len - len(df_clean)
    
    # Save updated CSV
    df_clean.to_csv(output_path, index=False)
    print(f"[CSV] Saved cleaned CSV: {output_path.name}")
    print(f"[CSV] Removed {removed} rows")
    
    return removed


def remove_corrupted_files(corrupted_paths: List[str]) -> int:
    """
    Remove problematic files from disk.
    
    Args:
        corrupted_paths: List of absolute paths
        
    Returns:
        Number of files removed
    """
    removed_count = 0
    
    for path_str in corrupted_paths:
        try:
            path = Path(path_str)
            if path.exists():
                path.unlink()
                removed_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to remove {path_str}: {e}")
    
    return removed_count


def generate_report(
    validation_result: Dict,
    output_dir: Path,
    dataset_dir: Path
) -> None:
    """Generate validation report files with categorized issues."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = validation_result['total']
    valid = validation_result['valid']
    issues = validation_result['issues']
    corrupted = validation_result['corrupted_files']
    
    # JSON report
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(validation_result, f, indent=2, default=str)
    print(f"[REPORT] Generated: {report_path.name}")
    
    # Summary text report
    txt_report_path = output_dir / 'validation_summary.txt'
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset Validation Report\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Validation Date: {pd.Timestamp.now()}\n\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"Total files: {total:,}\n")
        f.write(f"Valid files: {valid:,}\n")
        f.write(f"Problematic files: {len(corrupted):,}\n")
        
        if total > 0:
            f.write(f"Valid rate: {valid/total*100:.2f}%\n\n")
        
        f.write(f"ISSUES BREAKDOWN:\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"Critical errors:\n")
        f.write(f"  Truncated images: {issues['truncated']}\n")
        f.write(f"  Invalid format: {issues['invalid_format']}\n\n")
        
        f.write(f"Warnings (can affect training):\n")
        f.write(f"  Palette images: {issues['palette']}\n")
        f.write(f"  Huge images (>50M pixels): {issues['huge']}\n")
        f.write(f"  Extreme aspect ratio: {issues['aspect']}\n")
        f.write(f"  Grayscale images: {issues['grayscale']}\n")
        f.write(f"  RGBA with transparency: {issues['rgba']}\n")
    
    print(f"[REPORT] Generated: {txt_report_path.name}")
    
    # Detailed corrupted files list
    corrupted_path = output_dir / 'corrupted_files.txt'
    with open(corrupted_path, 'w', encoding='utf-8') as f:
        f.write(f"Detailed List of Problematic Files\n")
        f.write(f"=" * 80 + "\n\n")
        
        if not corrupted:
            f.write("All files are valid!\n")
        else:
            for idx, item in enumerate(corrupted, 1):
                f.write(f"{idx}. {item['relative_path']}\n")
                if item['error']:
                    f.write(f"   Error: {item['error']}\n")
                if item['warnings']:
                    f.write(f"   Warnings:\n")
                    for warning in item['warnings']:
                        f.write(f"     - {warning}\n")
                
                metadata = item.get('metadata', {})
                if metadata:
                    f.write(f"   Metadata:\n")
                    f.write(f"     Mode: {metadata.get('mode', 'N/A')}\n")
                    f.write(f"     Size: {metadata.get('size', 'N/A')}\n")
                    f.write(f"     Pixels: {metadata.get('pixels', 'N/A')}\n")
                    f.write(f"     Aspect ratio: {metadata.get('aspect_ratio', 'N/A')}\n")
                    f.write(f"     File size: {metadata.get('file_size_mb', 'N/A'):.2f} MB\n")
                f.write("\n")
    
    print(f"[REPORT] Generated: {corrupted_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate dataset images before training"
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Root dataset directory'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default=None,
        help='CSV file to update (optional)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='validation_report',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--remove_corrupted',
        action='store_true',
        help='Delete problematic files from disk'
    )
    parser.add_argument(
        '--update_csv',
        action='store_true',
        help='Update CSV to exclude problematic files'
    )
    parser.add_argument(
        '--max_pixels',
        type=int,
        default=50_000_000,
        help='Maximum allowed pixels for image (default: 50M)'
    )
    parser.add_argument(
        '--max_aspect_ratio',
        type=float,
        default=10.0,
        help='Maximum width/height aspect ratio (default: 10.0)'
    )
    parser.add_argument(
        '--strict_mode',
        action='store_true',
        help='Treat warnings as errors (remove images with warnings)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"TIGAS Dataset Validation")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Max pixels: {args.max_pixels:,}")
    print(f"Max aspect ratio: {args.max_aspect_ratio}")
    if args.strict_mode:
        print(f"Mode: STRICT (warnings = errors)")
    
    # Scan and validate
    validation_result = scan_directory(
        dataset_dir,
        verbose=args.verbose,
        max_pixels=args.max_pixels,
        max_aspect_ratio=args.max_aspect_ratio,
        strict_mode=args.strict_mode
    )
    
    # Print summary
    total = validation_result['total']
    valid = validation_result['valid']
    corrupted = len(validation_result['corrupted_files'])
    issues = validation_result['issues']
    
    print(f"\n[SUMMARY]")
    print(f"  Total files: {total:,}")
    print(f"  Valid files: {valid:,}")
    print(f"  Problematic files: {corrupted:,}")
    
    if total > 0:
        print(f"  Valid rate: {valid/total*100:.2f}%")
    
    if corrupted > 0:
        print(f"\n  Issues breakdown:")
        print(f"    Truncated/invalid: {issues['truncated'] + issues['invalid_format']}")
        print(f"    Palette images: {issues['palette']}")
        print(f"    Huge images: {issues['huge']}")
        print(f"    Extreme aspect: {issues['aspect']}")
        print(f"    Grayscale images: {issues['grayscale']}")
        print(f"    RGBA images: {issues['rgba']}")
    else:
        print(f"\n[SUCCESS] All images are valid! ✓")
    
    # Generate reports
    generate_report(validation_result, output_dir, dataset_dir)
    
    # Remove corrupted files if requested
    if args.remove_corrupted and corrupted > 0:
        print(f"\n[ACTION] Removing {corrupted} problematic files...")
        corrupted_paths = [f['path'] for f in validation_result['corrupted_files']]
        removed = remove_corrupted_files(corrupted_paths)
        print(f"[ACTION] Removed {removed} files")
    
    # Update CSV if requested
    if args.update_csv and args.csv_file:
        csv_path = Path(args.csv_file)
        if not csv_path.is_absolute():
            csv_path = dataset_dir / csv_path
        
        if csv_path.exists():
            print(f"\n[ACTION] Updating CSV: {csv_path.name}")
            
            # Generate output CSV path
            output_csv_path = output_dir / f"{csv_path.stem}_cleaned.csv"
            
            corrupted_paths = [f['path'] for f in validation_result['corrupted_files']]
            removed_rows = update_csv(csv_path, corrupted_paths, output_csv_path)
            
            print(f"[SUCCESS] Cleaned CSV saved to: {output_csv_path}")
        else:
            print(f"[WARNING] CSV file not found: {csv_path}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
