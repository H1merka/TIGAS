# TIGAS - Trained Image Generation Authenticity Score

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)

**TIGAS** is a neural network metric for assessing the authenticity and realism of images, designed to distinguish between real/natural images and AI-generated/fake images.

## Description

TIGAS provides a continuous score in the range [0, 1]:
- **1.0** — natural/real image
- **0.0** — generated/fake image

### Key Features

- **Multi-Modal Analysis**: combines complementary analysis approaches
  - Perceptual features (multi-scale CNN)
  - Spectral analysis (frequency domain)
  - Statistical consistency (distribution analysis)
  - Local-global coherence

- **Fully Differentiable**: can be used as
  - Image quality assessment metric
  - Loss function for training generative models
  - Evaluation metric for image generation tasks

- **Flexible Deployment**:
  - Model-based computation (trained neural network)
  - Component-based computation (without trained model)

## Installation

### Basic Installation

```bash
git clone https://github.com/H1merka/TIGAS.git
cd TIGAS
pip install -r requirements.txt
pip install -e .
```

### With CUDA Support

```bash
pip install -r requirements_cuda.txt
```

### Dependencies

**Core Dependencies:**
- PyTorch >= 2.2.0
- torchvision >= 0.17.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- scikit-learn >= 1.3.0
- Pillow >= 10.0.0
- OpenCV >= 4.8.0
- pandas >= 2.0.0

## Quick Start

### Python API

```python
from tigas import TIGAS, compute_tigas_score
import torch

# Method 1: High-level function
score = compute_tigas_score('image.jpg', checkpoint_path='model.pt')
print(f"TIGAS Score: {score:.4f}")

# Method 2: Object-oriented API
tigas = TIGAS(checkpoint_path='model.pt', img_size=256, device='cuda')
score = tigas('image.jpg')  # Single image
scores = tigas(torch.randn(4, 3, 256, 256))  # Batch
scores = tigas.compute_directory('path/to/images/')  # Directory

# Method 3: Auto-download model from HuggingFace Hub
tigas = TIGAS(auto_download=True)  # Automatically downloads model from Hub
score = tigas('image.jpg')

# Method 4: As a loss function
generated_images = torch.randn(4, 3, 256, 256, requires_grad=True)
scores = tigas(generated_images)
loss = 1.0 - scores.mean()  # Maximize authenticity
loss.backward()
```

### Command Line

```bash
# Evaluate single image
python scripts/evaluate.py --image path/to/image.jpg --checkpoint model.pt

# Evaluate directory
python scripts/evaluate.py --image_dir path/to/images/ --checkpoint model.pt --batch_size 32

# With auto-download from HuggingFace Hub
python scripts/evaluate.py --image_dir images/ --auto_download

# With results saving and visualization
python scripts/evaluate.py --image_dir images/ --output results.json --plot
```

## Training

### Data Structure

**Directory mode:**
```
dataset/
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── fake/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

**CSV mode:**
```
dataset/
├── train/
│   ├── images/
│   └── annotations01.csv
├── val/
│   └── ...
└── test/
    └── ...
```

CSV format: `image_path,label` (1 — real, 0 — fake)

### Dataset Validation (Required Step)

**IMPORTANT**: Before training, you must validate dataset integrity:

```bash
python scripts/validate_dataset.py \
  --dataset_dir /path/to/data \
  --csv_file train.csv \
  --remove_corrupted \
  --update_csv
```

This will remove corrupted images and update CSV, preventing errors during training.

### Running Training

```bash
# Fast training (Fast Mode - default, optimized for speed)
python scripts/train_script.py \
  --data_root /path/to/data \
  --epochs 50 \
  --batch_size 16 \
  --img_size 128 \
  --lr 0.0003 \
  --use_amp \
  --output_dir ./checkpoints

# Full training (Full Mode - all branches, higher accuracy)
python scripts/train_script.py \
  --data_root /path/to/data \
  --epochs 50 \
  --batch_size 8 \
  --img_size 256 \
  --lr 0.0001 \
  --use_amp \
  --full_mode \
  --output_dir ./checkpoints

# Training with CSV annotations (recommended)
python scripts/train_script.py \
  --data_root /path/to/data \
  --use_csv \
  --epochs 50 \
  --use_amp

# Resume training from checkpoint (N more epochs)
python scripts/train_script.py \
  --data_root data/ \
  --resume checkpoints/best_model.pt \
  --epochs 10 \
  --lr 0.0001 \
  --reset_lr

# Resume with full LR and scheduler reset
python scripts/train_script.py \
  --data_root data/ \
  --resume checkpoints/best_model.pt \
  --epochs 10 \
  --lr 0.0003 \
  --reset_lr \
  --reset_scheduler
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_root` | Path to data | Required |
| `--epochs` | Number of epochs (on resume — N more epochs) | 50 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate | 0.0000125 |
| `--use_csv` | Use CSV annotations | False |
| `--img_size` | Image size | 256 |
| `--output_dir` | Checkpoint directory | ./checkpoints |
| `--device` | Device (cuda/cpu) | auto |
| `--num_workers` | DataLoader workers (0 for Windows) | 0 |
| `--use_amp` | Mixed Precision Training | False |
| `--fast_mode` | Fast architecture (optimized) | True |
| `--full_mode` | Full architecture (all branches) | False |
| `--resume` | Path to checkpoint for resuming | None |
| `--reset_lr` | Reset LR on resume | False |
| `--reset_scheduler` | Reset scheduler on resume | False |

## Architecture

### TIGASModel

Multi-branch neural network including:

1. **Multi-Scale Feature Extractor**
   - 4-stage CNN backbone (1/2, 1/4, 1/8, 1/16 resolutions)
   - Preserves high-frequency details for artifact detection
   - EfficientNet-inspired design

2. **Spectral Analyzer**
   - FFT-based frequency domain analysis
   - Detection of GAN artifacts (checkerboard patterns, unnatural spectra)
   - Radial profile extraction from power spectrum

3. **Statistical Moment Estimator**
   - Distribution consistency analysis
   - Learnable natural image statistics
   - Moment matching against natural image priors

4. **Attention Mechanisms**
   - Self-Attention for capturing long-range dependencies
   - Cross-Modal Attention for fusing features from different modalities

5. **Adaptive Feature Fusion**
   - Learnable weighting of 3 feature streams
   - Combines perceptual, spectral, and statistical features

## Project Structure

```
TIGAS/
├── tigas/                          # Main package
│   ├── __init__.py                # Initialization and exports
│   ├── api.py                     # High-level API (TIGAS class)
│   │
│   ├── models/                    # Neural network architectures
│   │   ├── tigas_model.py        # Main TIGASModel architecture
│   │   ├── feature_extractors.py # Feature extractors
│   │   ├── attention.py          # Attention mechanisms
│   │   ├── layers.py             # Custom layers
│   │   └── constants.py          # Configuration constants
│   │
│   ├── metrics/                   # Metric computation modules
│   │   ├── tigas_metric.py       # Main metric calculator
│   │   └── components.py         # Metric components
│   │
│   ├── data/                      # Data loading and preprocessing
│   │   ├── dataset.py            # Dataset classes
│   │   ├── loaders.py            # DataLoader creation
│   │   └── transforms.py         # Augmentations and transforms
│   │
│   ├── training/                  # Training infrastructure
│   │   ├── trainer.py            # Main trainer class
│   │   ├── losses.py             # Loss functions
│   │   └── optimizers.py         # Optimizers and schedulers
│   │
│   └── utils/                     # Utilities
│       ├── config.py             # Configuration management
│       ├── input_processor.py    # Input data processing
│       └── visualization.py      # Visualization
│
├── scripts/                       # Executable scripts
│   ├── evaluate.py              # Evaluation/inference script
│   ├── example_usage.py          # Usage examples
│   └── train_script.py           # Training script
│
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies
├── requirements_cuda.txt        # CUDA dependencies
└── LICENSE                      # MIT License
```

## Usage Examples

### 1. Basic Usage

```python
from tigas import TIGAS

tigas = TIGAS(checkpoint_path='model.pt')
score = tigas('test_image.jpg')
print(f"Authenticity score: {score:.4f}")
```
import torch

tigas = TIGAS(checkpoint_path='model.pt')
image = torch.randn(1, 3, 256, 256)
outputs = tigas(image, return_features=True)

score = outputs['score']
features = outputs['features']
print(f"Score: {score.item():.4f}")
print(f"Available features: {list(features.keys())}")
print(f"Fused features shape: {features['fused']
from tigas import TIGAS
import torch

tigas = TIGAS(checkpoint_path='model.pt', device='cuda')
images = torch.randn(8, 3, 256, 256)
scores = tigas(images)
print(f"Mean score: {scores.mean():.4f}")
```

### 3. Directory Processing

```python
from tigas import TIGAS

tigas = TIGAS(checkpoint_path='model.pt')

# Get scores for all images
results = tigas.compute_directory(
    'path/to/images/',
    return_paths=True,
    batch_size=32
)

for img_path, score in results.items():
    print(f"{img_path}: {score:.4f}")
print(f"Feature dimension: {features.shape}")
```

### 4. Using as Loss Function

```python
from tigas import TIGAS

tigas = TIGAS(checkpoint_path='model.pt')

# In generator training loop
generated_images = generator(noise)
authenticity_score = tigas(generated_images)
loss = 1.0 - authenticity_score.mean()
loss.backward()
```

### 5. Component-Based Metric

```python
from tigas.metrics import TIGASMetric

metric = TIGASMetric(use_model=False)
score = metric.compute(image_tensor)
```

## Image Requirements

- **Formats**: JPG, JPEG, PNG, BMP
- **Resolution**: default 256x256 (configurable)
- Images are automatically resized if needed
- Normalized to [-1, 1] range

## Training Features

- **Mixed Precision Training**: accelerated training with AMP
- **Gradient Accumulation**: for larger effective batch sizes
- **Learning Rate Scheduling**: cosine annealing, warmup
- **Early Stopping**: automatic stopping on overfitting
- **TensorBoard Logging**: training process visualization
- **Checkpoint Management**: model saving and loading

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- Dmitrij Morgenshtern

## Links

- [GitHub Repository](https://github.com/H1merka/TIGAS)