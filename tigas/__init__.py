"""
TIGAS - Neural Authenticity and Realism Index

A novel differentiable metric for assessing the realism of generated images.
Combines perceptual, spectral, statistical, and structural analysis for
comprehensive image authenticity evaluation.

Authors: TIGAS Project Team
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "TIGAS Project Team"

from .api import TIGAS, compute_tigas_score
from .metrics.tigas_metric import TIGASMetric

__all__ = ["TIGAS", "compute_tigas_score", "TIGASMetric"]
