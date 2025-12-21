"""
Utility modules for TIGAS.
"""

from .config import load_config, save_config, get_default_config

# Lazy imports for optional dependencies
# visualization requires matplotlib (install with: pip install tigas-metric[vis])

__all__ = [
    "load_config",
    "save_config",
    "get_default_config",
]


def __getattr__(name):
    """Lazy import for visualization functions."""
    if name in ("visualize_predictions", "plot_training_history"):
        from .visualization import visualize_predictions, plot_training_history
        return {"visualize_predictions": visualize_predictions, 
                "plot_training_history": plot_training_history}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
