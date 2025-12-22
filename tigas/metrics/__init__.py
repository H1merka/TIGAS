"""
TIGAS metric computation modules.
"""

from .tigas_metric import TIGASMetric, compute_tigas_batch
from .components import (
    PerceptualDistance,
    SpectralDivergence,
    StatisticalConsistency
)

__all__ = [
    "TIGASMetric",
    "compute_tigas_batch",
    "PerceptualDistance",
    "SpectralDivergence",
    "StatisticalConsistency"
]
