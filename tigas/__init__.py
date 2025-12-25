"""
TIGAS - Neural Authenticity and Realism Index

A novel differentiable metric for assessing the realism of generated images.
Combines perceptual, spectral, statistical, and structural analysis for
comprehensive image authenticity evaluation.

Authors: TIGAS Project Team
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.1"
__author__ = "TIGAS Project Team"

from .api import TIGAS, compute_tigas_score, load_tigas
from .metrics.tigas_metric import TIGASMetric
from .model_hub import (
    get_default_model_path,
    download_default_model,
    clear_cache,
    cache_info
)

__all__ = [
    "TIGAS",
    "compute_tigas_score",
    "load_tigas",
    "TIGASMetric",
    # Model hub utilities
    "get_default_model_path",
    "download_default_model",
    "clear_cache",
    "cache_info"
]
