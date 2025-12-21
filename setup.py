"""
TIGAS - Trained Image Generation Authenticity Score
Setup script for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="tigas-metric",
    version="0.1.0",
    author="TIGAS Project Team",
    author_email="tigas@example.com",
    description="Trained Image Generation Authenticity Score - A novel metric for assessing image realism",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/H1merka/TIGAS",
    packages=find_packages(exclude=["tests", "scripts", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements + [
        "torch>=2.2.0",  # PyTorch core dependency
        "torchvision>=0.17.0",  # For image transforms
        "huggingface-hub>=0.19.0",  # For automatic model download
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "vis": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "training": [
            "tensorboard>=2.9.0",
            "wandb>=0.12.0",
        ]
    },
    zip_safe=False,
    keywords=[
        "deep learning",
        "computer vision",
        "image quality",
        "generative models",
        "gan evaluation",
        "image authenticity",
        "pytorch",
        "metric learning"
    ],
)
