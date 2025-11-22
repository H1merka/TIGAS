# TIGAS - Trained Image Generation Authenticity Score

**A novel, differentiable metric for assessing the realism and authenticity of generated images.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

**TIGAS** is a state-of-the-art deep learning metric designed to evaluate the authenticity and realism of images, particularly for assessing generative models (GANs, Diffusion Models, etc.). Unlike existing metrics that focus on specific aspects, TIGAS combines multiple complementary analysis approaches:

- **Perceptual Analysis**: Multi-scale deep features for semantic understanding
- **Spectral Coherence**: Frequency domain analysis to detect GAN artifacts
- **Statistical Consistency**: Comparison with natural image statistics
- **Multi-Modal Fusion**: Attention-based integration of different signals

### Key Features

✅ **Fully Differentiable** - Can be used as a loss function for training generative models
✅ **Comprehensive** - Combines perceptual, spectral, and statistical analysis
✅ **Easy to Use** - Simple API: `score = tigas(image)`
✅ **Modular Design** - Clean, extensible architecture following best practices
✅ **Research-Ready** - Includes training pipeline, evaluation tools, and visualization
✅ **Production-Ready** - Optimized with mixed precision, gradient accumulation, etc.

---

## 📊 TIGAS Metric Innovation

### What Makes TIGAS Different?

| Metric | Perceptual | Spectral | Statistical | Differentiable | Single Image |
|--------|-----------|----------|-------------|----------------|--------------|
| **TIGAS** | ✅ | ✅ | ✅ | ✅ | ✅ |
| LPIPS | ✅ | ❌ | ❌ | ✅ | ❌ (needs reference) |
| FID | ❌ | ❌ | ✅ | ❌ | ❌ (distribution) |
| SSIM | ❌ | ❌ | ❌ | ✅ | ❌ (needs reference) |
| IS | ❌ | ❌ | ❌ | ❌ | ❌ (distribution) |

### Architecture Highlights

```
Input Image → Multi-Branch Analysis → Attention Fusion → TIGAS Score [0, 1]
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
  Perceptual    Spectral      Statistical
   Features      Analysis     Consistency
   (4 scales)  (FFT/DCT)    (Moments)
        ↓             ↓             ↓
        └─────→ Cross-Attention ←───┘
                      ↓
              Adaptive Fusion
                      ↓
              Score: 1.0 = Real
                     0.0 = Fake
```

---

## 🚀 Installation

### From Source (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/H1merka/TIGAS.git
cd TIGAS

# Install in editable mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[dev,vis,training]"
```

### Using pip (After Publishing)

```bash
pip install tigas-metric
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA (optional, for GPU acceleration)

---

## 📖 Quick Start

### Basic Usage

```python
from tigas import TIGAS

# Initialize TIGAS (with pretrained model)
tigas = TIGAS(checkpoint_path='checkpoints/best_model.pt')

# Evaluate single image
score = tigas('path/to/image.jpg')
print(f"TIGAS Score: {score:.3f}")
# Output: TIGAS Score: 0.856  (likely real)

# Evaluate directory
scores = tigas.compute_directory('path/to/images/')
print(f"Mean score: {scores.mean():.3f}")
```

### Command-Line Interface

```bash
# Evaluate single image
nair path/to/image.jpg --checkpoint checkpoints/best_model.pt

# Evaluate directory with statistics
nair --image_dir path/to/images/ --checkpoint model.pt --plot

# Output:
# TIGAS Score: 0.923
# Assessment: Likely REAL/Natural
```

### PyTorch Integration

```python
import torch
from tigas import TIGAS

# Initialize
tigas = TIGAS(checkpoint_path='model.pt', device='cuda')

# Batch processing
images = torch.randn(16, 3, 256, 256).cuda()
scores = tigas(images)  # [16, 1]

# Use as differentiable loss
loss = 1.0 - nair(generated_images).mean()
loss.backward()
```

---

## 🏋️ Training Your Own Model

### 1. Prepare Dataset

Organize your data:

```
data/
├── real/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── fake/
    ├── gan_001.jpg
    ├── diffusion_001.jpg
    └── ...
```

### 2. Configure Training

Edit `configs/training_config.yaml` or use defaults:

```yaml
model:
  img_size: 256
  base_channels: 32
  feature_dim: 256

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.0001

data:
  train_split: 0.8
  augment_level: medium
```

### 3. Start Training

```bash
# Using script
python scripts/train.py \
    --data_root data/ \
    --config configs/training_config.yaml \
    --output_dir checkpoints/

# Or using Python API
from tigas.training import TIGASTrainer
from tigas.models import create_tigas_model
from tigas.data import create_dataloaders

# Create components
model = create_tigas_model(img_size=256)
dataloaders = create_dataloaders('data/', batch_size=32)

# Train
trainer = TIGASTrainer(tigas_model, dataloaders['train'], dataloaders['val'])
trainer.train(num_epochs=100)
```

### 4. Monitor Training

```bash
# TensorBoard
tensorboard --logdir checkpoints/logs
```

---

## 🔬 Advanced Usage

### Custom Loss Function for GANs

```python
from tigas import TIGAS

# Initialize TIGAS
tigas = TIGAS(checkpoint_path='model.pt')

# In your GAN training loop
def generator_loss(fake_images):
    # Standard GAN loss
    gan_loss = adversarial_loss(fake_images)

    # TIGAS realism loss (higher is better)
    tigas_scores = tigas(fake_images)
    realism_loss = -tigas_scores.mean()  # Maximize score

    # Combined loss
    total_loss = gan_loss + 0.1 * realism_loss
    return total_loss
```

### Feature Extraction

```python
# Get intermediate features for analysis
outputs = tigas(images, return_features=True)

score = outputs['score']  # Final TIGAS score
features = outputs['features']  # Dict of intermediate features

# Available features:
# - 'perceptual': Multi-scale perceptual features
# - 'spectral': Frequency domain features
# - 'statistical': Statistical consistency features
# - 'fused': Final fused representation
```

### Batch Evaluation with Custom Metrics

```python
from tigas.metrics import TIGASMetric

metric = TIGASMetric(model, use_model=True)

# Compute with component breakdown
results = metric(images, return_components=True)

print(f"Overall score: {results['score'].mean():.3f}")
print(f"Spectral score: {results['spectral_score'].mean():.3f}")
print(f"Statistical score: {results['statistical_score'].mean():.3f}")
```

---

## 📁 Project Structure

```
TIGAS/
├── tigas/                          # Main package
│   ├── models/                    # Neural network models
│   │   ├── tigas_model.py         # Main TIGAS model
│   │   ├── feature_extractors.py # Multi-scale, spectral, statistical
│   │   ├── attention.py          # Cross-modal attention
│   │   └── layers.py             # Custom layers
│   ├── metrics/                   # Metric computation
│   │   ├── tigas_metric.py        # Main metric class
│   │   └── components.py         # Individual metric components
│   ├── data/                      # Data handling
│   │   ├── dataset.py            # Dataset classes
│   │   ├── transforms.py         # Augmentations
│   │   └── loaders.py            # Data loaders
│   ├── training/                  # Training infrastructure
│   │   ├── trainer.py            # Main trainer
│   │   ├── losses.py             # Loss functions
│   │   └── optimizers.py         # Optimizer configs
│   ├── utils/                     # Utilities
│   │   ├── config.py             # Configuration management
│   │   └── visualization.py      # Plotting and visualization
│   └── api.py                     # Public API
├── scripts/                       # Executable scripts
│   ├── train.py                  # Training script
│   └── evaluate.py               # Evaluation script
├── configs/                       # Configuration files
│   ├── model_config.yaml
│   └── training_config.yaml
├── tests/                         # Unit tests
├── setup.py                       # Package installation
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🎓 Technical Details

### Model Architecture

**TIGAS Model** consists of:

1. **Multi-Scale Feature Extractor**
   - Custom CNN backbone (EfficientNet-inspired)
   - 4-scale pyramid: {1/2, 1/4, 1/8, 1/16}
   - Gated residual blocks with CBAM attention

2. **Spectral Analyzer**
   - 2D FFT/DCT for frequency analysis
   - Radial profile computation
   - Detects checkerboard and spectral artifacts

3. **Statistical Moment Estimator**
   - Computes 5 moments: mean, variance, skewness, kurtosis, entropy
   - Learnable prototypes of natural image statistics
   - Multi-scale local statistics

4. **Cross-Modal Fusion**
   - Cross-attention between modalities
   - Adaptive feature weighting
   - Self-attention refinement

5. **Regression Head**
   - 3-layer MLP with dropout
   - Sigmoid activation → [0, 1]
   - Auxiliary binary classification head

### Training Methodology

- **Loss Function**: Combined regression + classification + ranking
- **Optimizer**: AdamW with warmup and cosine annealing
- **Augmentation**: Heavy augmentation to prevent overfitting
- **Regularization**: Weight decay, dropout, gradient clipping
- **Mixed Precision**: Automatic mixed precision (AMP) for efficiency
- **Early Stopping**: Validation-based with patience

### Performance Optimizations

- ✅ Mixed precision training (FP16)
- ✅ Gradient accumulation
- ✅ Multi-GPU support (DataParallel/DistributedDataParallel ready)
- ✅ Efficient data loading with prefetching
- ✅ TensorBoard integration

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project builds upon insights from:

- **LPIPS** - Perceptual similarity metrics
- **FID** - Frechet Inception Distance
- **StyleGAN** - High-quality image generation
- **Shift-Tolerant LPIPS** - Robustness to transformations

Special thanks to the open-source community for providing foundational tools and datasets.

---

## 📞 Contact

- **Project Lead**: Dmitrij Morgenshtern
- **GitHub**: [H1merka/TIGAS](https://github.com/H1merka/TIGAS)
- **Issues**: [GitHub Issues](https://github.com/H1merka/TIGAS/issues)

---

**Made with ❤️ by the TIGAS Research Team**
