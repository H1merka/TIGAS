"""
Model Hub management for TIGAS.
Handles automatic model download from HuggingFace Hub with caching.
"""

from pathlib import Path
from typing import Optional
import sys
import warnings

try:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


# Configuration
DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'tigas' / 'models'
DEFAULT_MODEL_REPO = "H1merka/TIGAS"
DEFAULT_MODEL_FILE = "best_model.pt"


def get_cache_dir() -> Path:
    """Get TIGAS models cache directory."""
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def check_model_in_cache(model_filename: str = DEFAULT_MODEL_FILE) -> Optional[str]:
    """
    Check if model exists in local cache.
    
    Args:
        model_filename: Model filename to check
        
    Returns:
        Path to cached model if exists, None otherwise
    """
    cache_dir = get_cache_dir()
    
    # Прямой путь в корне кэша
    model_path = cache_dir / model_filename
    if model_path.exists():
        return str(model_path)
    
    # Поиск рекурсивно (HF Hub хранит в подпапках snapshots/)
    for found in cache_dir.glob(f'**/{model_filename}'):
        if found.is_file():
            return str(found)
    
    # Проверка через HF Hub API
    if HF_HUB_AVAILABLE:
        try:
            cached_path = try_to_load_from_cache(
                repo_id=DEFAULT_MODEL_REPO,
                filename=model_filename
            )
            if cached_path and Path(cached_path).exists():
                return str(cached_path)
        except Exception:
            pass
    
    return None


def download_default_model(
    model_filename: str = DEFAULT_MODEL_FILE,
    force_download: bool = False,
    show_progress: bool = True
) -> str:
    """
    Download default TIGAS model from HuggingFace Hub.
    
    Args:
        model_filename: Model filename to download
        force_download: Force re-download even if cached
        show_progress: Show download progress bar
        
    Returns:
        Path to downloaded model file
        
    Raises:
        ImportError: If huggingface_hub is not installed
        RuntimeError: If download fails
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to download models.\n"
            "Install with: pip install huggingface-hub\n"
            "Or manually download the model and pass checkpoint_path='/path/to/model.pt'"
        )
    
    # Check cache first unless force_download
    if not force_download:
        cached_path = check_model_in_cache(model_filename)
        if cached_path:
            if show_progress:
                print(f"[TIGAS] Using cached model: {cached_path}", file=sys.stderr)
            return cached_path
    
    # Download from HuggingFace Hub
    if show_progress:
        print(f"[TIGAS] Downloading {model_filename} from HuggingFace Hub...", file=sys.stderr)
        print(f"[TIGAS] Repository: {DEFAULT_MODEL_REPO}", file=sys.stderr)
    
    try:
        cache_dir = get_cache_dir()
        
        model_path = hf_hub_download(
            repo_id=DEFAULT_MODEL_REPO,
            filename=model_filename,
            cache_dir=str(cache_dir),
        )
        
        if show_progress:
            print(f"[TIGAS] ✓ Model downloaded successfully", file=sys.stderr)
            print(f"[TIGAS] Cached at: {model_path}", file=sys.stderr)
        
        return model_path
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model '{model_filename}' from HuggingFace Hub.\n"
            f"Repository: {DEFAULT_MODEL_REPO}\n"
            f"Error: {e}\n\n"
            f"You can:\n"
            f"1. Manually download from https://huggingface.co/{DEFAULT_MODEL_REPO}\n"
            f"2. Pass checkpoint_path='/path/to/model.pt' to TIGAS()\n"
            f"3. Check your internet connection and try again"
        )


def get_default_model_path(
    auto_download: bool = True,
    show_progress: bool = True
) -> Optional[str]:
    """
    Get path to default TIGAS model.
    Downloads automatically if not in cache and auto_download=True.
    
    Args:
        auto_download: Automatically download if not cached
        show_progress: Show download progress
        
    Returns:
        Path to model file, or None if not available and auto_download=False
    """
    # Check cache first
    cached_path = check_model_in_cache()
    if cached_path:
        return cached_path
    
    # Download if requested
    if auto_download:
        try:
            return download_default_model(show_progress=show_progress)
        except Exception as e:
            warnings.warn(
                f"Could not download default model: {e}\n"
                f"You can manually specify checkpoint_path when creating TIGAS instance.",
                RuntimeWarning
            )
            return None
    
    return None


def clear_cache():
    """Clear TIGAS model cache directory."""
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"[TIGAS] Cache cleared: {cache_dir}")
    else:
        print(f"[TIGAS] Cache directory does not exist: {cache_dir}")


def cache_info() -> dict:
    """Get information about cached models.
    
    Returns:
        Dictionary with cache information
    """
    cache_dir = get_cache_dir()
    
    info = {
        'cache_dir': str(cache_dir),
        'cache_exists': cache_dir.exists(),
        'models': [],
        'total_size_mb': 0.0
    }
    
    print(f"TIGAS Model Cache Information")
    print(f"=" * 60)
    print(f"Cache directory: {cache_dir}")
    print(f"Cache exists: {cache_dir.exists()}")
    
    if cache_dir.exists():
        # Ищем модели рекурсивно (HF Hub хранит в подпапках)
        models = list(cache_dir.glob('**/*.pt'))
        info['models'] = [str(m) for m in models]
        print(f"Cached models: {len(models)}")
        
        total_size = 0
        for model_path in models:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  - {model_path.name}: {size_mb:.2f} MB")
        
        info['total_size_mb'] = total_size
        print(f"Total cache size: {total_size:.2f} MB")
    else:
        print(f"No models cached yet")
    
    print(f"=" * 60)
    
    return info
