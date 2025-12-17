"""
Constants for TIGAS model configuration.
Centralizes default values to avoid magic numbers.
"""

# Model architecture defaults
DEFAULT_FEATURE_DIM = 256
DEFAULT_BASE_CHANNELS = 32
DEFAULT_ATTENTION_HEADS = 8
DEFAULT_STAGES = [2, 3, 4, 3]

# Input normalization
INPUT_MIN = -1.0
INPUT_MAX = 1.0

# Regression head configuration
REGRESSION_HIDDEN_DIM_RATIO = 2
REGRESSION_FINAL_DIM_RATIO = 4

# Weight initialization
LINEAR_WEIGHT_STD = 0.02  # Increased from 0.01 for better numerical stability


