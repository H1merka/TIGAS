# TIGAS AI Coding Assistant Instructions

## Project Overview

**TIGAS** (Trained Image Generation Authenticity Score) is a differentiable neural metric for assessing image authenticity. It distinguishes real/natural images (score ≈ 1.0) from AI-generated/fake images (score ≈ 0.0).

## Architecture & Data Flow

### Multi-Branch Design
The system combines **4 complementary analysis branches**:
1. **Perceptual Features**: Multi-scale CNN backbone (ResNet-style) via `feature_extractors.MultiScaleFeatureExtractor`
2. **Spectral Analysis**: Frequency domain via `SpectralAnalyzer` (FFT-based)
3. **Statistical Moments**: Distribution consistency via `StatisticalMomentEstimator`
4. **Cross-Modal Attention**: Feature fusion via `CrossModalAttention`

All branches feed into a regression head → final score [0, 1] in [tigas/models/tigas_model.py](tigas/models/tigas_model.py).

### Key Components
- **Model**: [tigas/models/tigas_model.py](tigas/models/tigas_model.py) - `TIGASModel` (main architecture)
- **Metric**: [tigas/metrics/tigas_metric.py](tigas/metrics/tigas_metric.py) - `TIGASMetric` (supports model-based AND component-based modes)
- **Public API**: [tigas/api.py](tigas/api.py) - `TIGAS` class (high-level interface)
- **Input Processing**: [tigas/utils/input_processor.py](tigas/utils/input_processor.py) - handles str/Path/Tensor/PIL inputs

### Data Pipeline
- [tigas/data/dataset.py](tigas/data/dataset.py): Expects `root/{real,fake}/` directory structure
- [tigas/data/transforms.py](tigas/data/transforms.py): Inference transforms (256px resize, ImageNet normalization)
- [tigas/data/loaders.py](tigas/data/loaders.py): DataLoader builders (also supports CSV-based datasets)

### Training Pipeline
- [tigas/training/trainer.py](tigas/training/trainer.py): `TIGASTrainer` with mixed-precision, gradient accumulation, early stopping, TensorBoard logging
- [tigas/training/losses.py](tigas/training/losses.py): `CombinedLoss` (regression + classification + ranking)
- [tigas/training/optimizers.py](tigas/training/optimizers.py): AdamW + LR scheduling (warmup + cosine)

## Developer Workflows

### Running Evaluation
```bash
# Single image
python scripts/evaluate.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt

# Directory with visualization
python scripts/evaluate.py --image_dir images/ --output results.json --plot --batch_size 32
```

### Training from Scratch
```bash
# Basic training
python scripts/train_script.py --data_dir data/ --epochs 100 --batch_size 32 --checkpoint checkpoints/

# Advanced: with config file
python scripts/train_script.py --config config.yaml --data_dir data/ --output_dir checkpoints/
```

### Python API Usage
```python
from tigas import TIGAS, compute_tigas_score

# Load model and compute scores
tigas = TIGAS(checkpoint_path='model.pt', device='cuda')
score = tigas('image.jpg')  # Single
scores = tigas(torch.randn(4, 3, 256, 256))  # Batch tensor
scores = tigas.compute_directory('path/to/images/')  # Directory
```

## Project-Specific Patterns

### Model Output Handling
- All API outputs are **continuous scores [0, 1]** (single float or tensor)
- Batch operations return torch.Tensor; single operations return float
- Use `model.eval()` and `torch.no_grad()` for inference

### Configuration Management
- Default configs in [tigas/utils/config.py](tigas/utils/config.py) - `get_default_config()`
- Supports YAML + JSON config files (training scripts accept `--config`)
- Config keys: `model`, `training`, `data`, `loss`

### Device Handling
- Auto-detect CUDA availability as default; explicit `device` parameter available
- Mixed-precision training (`use_amp=True`) enabled by default in trainer
- All tensors should be moved to `self.device` (see trainer initialization pattern)

### Dual Operation Modes
`TIGASMetric` supports **two modes** (controlled by `use_model` flag):
- **Model-based** (default): Uses trained `TIGASModel` for end-to-end prediction
- **Component-based**: Weighted combination of individual metrics (no model needed)
- Switch modes in [tigas/metrics/tigas_metric.py](tigas/metrics/tigas_metric.py#L30)

## External Dependencies & Integration

### Core Dependencies
- **PyTorch 2.2+** (with torchvision 0.17+) - neural networks, model loading
- **NumPy/SciPy** - array operations, spectral analysis
- **scikit-learn** - statistical utilities
- **PIL/OpenCV** - image I/O
- **TensorBoard** - training visualization (via `torch.utils.tensorboard`)

### Checkpoint Management
- Checkpoints: `.pt` files (PyTorch native format)
- Load via `create_tigas_model(checkpoint_path=...)` in [tigas/models/tigas_model.py](tigas/models/tigas_model.py)
- Trainer auto-saves best model + periodic snapshots to `output_dir/`

## Critical Conventions

1. **Image Input**: Always 3-channel RGB, expects [0, 1] or [0, 255] (transforms handle normalization)
2. **Batch Dimension**: Tensors must be (B, 3, H, W); single images wrapped in batch
3. **Model Inference**: Always use context manager `with torch.no_grad():`
4. **Path Handling**: Use `pathlib.Path` consistently (avoid string paths for robustness)
5. **Logging**: Training uses TensorBoard; access via `tensorboard --logdir checkpoints/logs/`
6. **Windows Compatibility**: Scripts handle UTF-8 encoding explicitly (see [scripts/train_script.py](scripts/train_script.py#L6-L11))

## Cross-Component Communication

- **Feature extractors** → attention modules → regression head (sequential in TIGASModel)
- **Input processor** normalizes → transforms.py applies augmentations → model.forward()
- **Trainer** manages model/metric lifecycle; metric wraps model for loss computation
- **Scripts** orchestrate: config → dataloader → trainer → checkpoints
