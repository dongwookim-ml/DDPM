"""
DDPM Package

This package implements Denoising Diffusion Probabilistic Models (DDPM) for image generation.

Main components:
- diffusion: Core diffusion process implementation
- unet: U-Net architecture for noise prediction
- ema: Exponential Moving Average for stable training
- utils: Utility functions
- script_utils: Helper functions for training and inference scripts

Reference:
"Denoising Diffusion Probabilistic Models" by Ho et al. (2020)
https://arxiv.org/abs/2006.11239
"""

# Main imports for easy access
from .diffusion import GaussianDiffusion, generate_linear_schedule, generate_cosine_schedule
from .unet import UNet
from .ema import EMA
from .utils import extract

__version__ = "1.0.0"
__author__ = "DDPM Implementation"
