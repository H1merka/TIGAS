"""
Configuration management for TIGAS.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for TIGAS training.

    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'img_size': 256,
            'in_channels': 3,
            'base_channels': 32,
            'feature_dim': 256,
            'num_attention_heads': 8,
            'dropout': 0.1
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'use_amp': True,
            'early_stopping_patience': 10
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.1,
            'num_workers': 12,
            'augment_level': 'medium',
            'pin_memory': True
        },
        'loss': {
            'regression_weight': 1.0,
            'classification_weight': 0.5,
            'ranking_weight': 0.3,
            'use_smooth_l1': True,
            'margin': 0.5
        },
        'logging': {
            'log_interval': 50,
            'save_interval': 1,
            'validate_interval': 5,
            'use_tensorboard': False
        }
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Supports YAML and JSON formats.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load based on extension
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Merge with defaults
    default_config = get_default_config()
    merged_config = merge_configs(default_config, config)

    return merged_config


def save_config(config: Dict[str, Any], output_path: str, format: str = 'yaml'):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
        format: 'yaml' or 'json'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Saved config to {output_path}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


class ConfigManager:
    """Configuration manager with validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            self.config = get_default_config()
        else:
            self.config = config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = [
            'model.img_size',
            'training.num_epochs',
            'training.batch_size',
            'training.learning_rate'
        ]

        for key in required_keys:
            if self.get(key) is None:
                print(f"Missing required config: {key}")
                return False

        return True
