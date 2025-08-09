"""
DDPM Package

This package implements Denoising Diffusion Probabilistic Models (DDPM) for image generation.

Main components:
- models: Core model implementations (UNet, GaussianDiffusion)
- training: Training utilities and components (EMA, training utils)
- utils: Utility functions and script helpers

Reference:
"Denoising Diffusion Probabilistic Models" by Ho et al. (2020)
https://arxiv.org/abs/2006.11239
"""

# Main imports for easy access
from .models import GaussianDiffusion, generate_linear_schedule, generate_cosine_schedule, UNet
from .training import EMA
from .training.utils import extract

__version__ = "1.0.0"
__author__ = "DDPM Implementation"
