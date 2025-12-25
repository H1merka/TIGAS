#!/usr/bin/env python
"""
TIGAS Model Publishing Script - Upload model to HuggingFace Hub

This script uploads the contents of model_publish/ directory to HuggingFace Hub.
It handles authentication, repository creation, and file uploads with proper metadata.

Prerequisites:
    pip install huggingface-hub

Usage:
    # Interactive login (first time)
    huggingface-cli login

    # Upload model
    python scripts/publish_to_hf.py

    # Upload with custom repository
    python scripts/publish_to_hf.py --repo_id username/model-name

    # Create new repository if not exists
    python scripts/publish_to_hf.py --create_repo

    # Upload as private repository
    python scripts/publish_to_hf.py --private

    # Dry run (show what would be uploaded)
    python scripts/publish_to_hf.py --dry_run
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from huggingface_hub import (
        HfApi,
        create_repo,
        upload_file,
        upload_folder,
        login,
        whoami
    )
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


# Default configuration
DEFAULT_REPO_ID = "H1merka/TIGAS"
MODEL_PUBLISH_DIR = project_root / "model_publish"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload TIGAS model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # First time: login to HuggingFace
    huggingface-cli login

    # Upload model to default repository
    python scripts/publish_to_hf.py

    # Upload to custom repository
    python scripts/publish_to_hf.py --repo_id myuser/my-tigas-model

    # Create new repo and upload
    python scripts/publish_to_hf.py --repo_id myuser/new-model --create_repo

    # Preview what would be uploaded
    python scripts/publish_to_hf.py --dry_run
        """
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repository ID (default: {DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(MODEL_PUBLISH_DIR),
        help=f"Directory containing model files to upload (default: {MODEL_PUBLISH_DIR})"
    )
    parser.add_argument(
        "--create_repo",
        action="store_true",
        help="Create repository if it doesn't exist"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message (default: auto-generated)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, uses cached token if not provided)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def check_prerequisites():
    """Check if all prerequisites are met."""
    if not HF_HUB_AVAILABLE:
        print("=" * 60)
        print("[ERROR] huggingface-hub is not installed!")
        print("=" * 60)
        print("\nInstall with:")
        print("    pip install huggingface-hub")
        print("\nThen login with:")
        print("    huggingface-cli login")
        print("=" * 60)
        return False
    return True


def check_authentication(token=None):
    """Check if user is authenticated with HuggingFace."""
    try:
        user_info = whoami(token=token)
        return user_info
    except Exception as e:
        return None


def analyze_model_dir(model_dir: Path, verbose: bool = False):
    """
    Analyze model directory and return list of files to upload.
    
    Returns:
        list of (file_path, file_info) tuples
    """
    files = []
    total_size = 0
    
    for file_path in model_dir.iterdir():
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            files.append({
                'path': file_path,
                'name': file_path.name,
                'size': size,
                'size_mb': size / (1024 * 1024)
            })
    
    return files, total_size


def print_upload_plan(files, total_size, repo_id, dry_run=False):
    """Print what will be uploaded."""
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN - Upload Plan (no actual upload)")
    else:
        print("Upload Plan")
    print("=" * 60)
    print(f"\nTarget repository: {repo_id}")
    print(f"Files to upload: {len(files)}")
    print(f"Total size: {total_size / (1024 * 1024):.2f} MB")
    print("\nFiles:")
    
    for f in files:
        print(f"  • {f['name']:<30} {f['size_mb']:>8.2f} MB")
    
    print("=" * 60)


def upload_model(
    model_dir: Path,
    repo_id: str,
    create_repo_flag: bool = False,
    private: bool = False,
    commit_message: str = None,
    token: str = None,
    verbose: bool = False
):
    """
    Upload model directory to HuggingFace Hub.
    
    Args:
        model_dir: Path to directory containing model files
        repo_id: HuggingFace repository ID (format: username/repo-name)
        create_repo_flag: Create repository if it doesn't exist
        private: Make repository private
        commit_message: Custom commit message
        token: HuggingFace token
        verbose: Verbose output
    """
    api = HfApi()
    
    # Create repository if requested
    if create_repo_flag:
        print(f"\n[1/3] Creating repository: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,  # Don't fail if exists
                token=token
            )
            print(f"      ✓ Repository ready: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"      ⚠ Repository creation note: {e}")
    else:
        print(f"\n[1/3] Using existing repository: {repo_id}")
    
    # Generate commit message
    if commit_message is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Upload TIGAS model checkpoint ({timestamp})"
    
    print(f"\n[2/3] Uploading files...")
    print(f"      Commit message: {commit_message}")
    
    # Upload entire folder
    try:
        result = api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=token,
        )
        
        print(f"\n[3/3] Upload complete!")
        print(f"      ✓ Commit: {result}")
        print(f"\n" + "=" * 60)
        print(f"SUCCESS! Model published to HuggingFace Hub")
        print(f"=" * 60)
        print(f"\nView your model at:")
        print(f"    https://huggingface.co/{repo_id}")
        print(f"\nUsers can now use your model with:")
        print(f"    from tigas import TIGAS")
        print(f"    tigas = TIGAS(auto_download=True)")
        print(f"=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("TIGAS Model Publisher - HuggingFace Hub")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Check model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"\n[ERROR] Model directory not found: {model_dir}")
        print(f"Make sure you have model files in {model_dir}")
        sys.exit(1)
    
    # Analyze files to upload
    files, total_size = analyze_model_dir(model_dir, args.verbose)
    
    if not files:
        print(f"\n[ERROR] No files found in {model_dir}")
        sys.exit(1)
    
    # Check for required files
    file_names = [f['name'] for f in files]
    if 'README.md' not in file_names:
        print("\n[WARNING] README.md not found in model directory!")
        print("         A README.md is recommended for HuggingFace model cards.")
    
    if not any(f['name'].endswith('.pt') for f in files):
        print("\n[WARNING] No .pt checkpoint file found!")
        print("         Make sure to include your model checkpoint.")
    
    # Print upload plan
    print_upload_plan(files, total_size, args.repo_id, args.dry_run)
    
    # Dry run - stop here
    if args.dry_run:
        print("\n[DRY RUN] No files were uploaded.")
        print("Remove --dry_run flag to actually upload.")
        return
    
    # Check authentication
    print("\n[AUTH] Checking HuggingFace authentication...")
    user_info = check_authentication(args.token)
    
    if user_info is None:
        print("\n[ERROR] Not authenticated with HuggingFace!")
        print("=" * 60)
        print("\nPlease login first:")
        print("    huggingface-cli login")
        print("\nOr provide token via --token argument")
        print("=" * 60)
        sys.exit(1)
    
    print(f"       ✓ Authenticated as: {user_info.get('name', 'Unknown')}")
    
    # Confirm upload
    print(f"\n" + "-" * 60)
    response = input(f"Upload {len(files)} files ({total_size / (1024 * 1024):.1f} MB) to {args.repo_id}? [y/N]: ")
    
    if response.lower() not in ['y', 'yes']:
        print("\nUpload cancelled.")
        return
    
    # Upload
    success = upload_model(
        model_dir=model_dir,
        repo_id=args.repo_id,
        create_repo_flag=args.create_repo,
        private=args.private,
        commit_message=args.commit_message,
        token=args.token,
        verbose=args.verbose
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
