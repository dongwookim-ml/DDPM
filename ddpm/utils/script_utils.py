"""
Script Utilities for DDPM Training and Inference

This module provides utility functions and configurations for training and running
DDPM models. It includes data loading helpers, argument parsing utilities, and
model creation functions.
"""

import argparse
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from ..models.unet import UNet
from ..models.diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl: torch.utils.data.DataLoader) -> Iterator[Any]:
    """
    Create an infinite generator from a DataLoader.
    
    This is useful for training loops where you want to continuously
    sample from the dataset without worrying about epoch boundaries.
    
    Args:
        dl: PyTorch DataLoader
        
    Yields:
        Batches from the DataLoader indefinitely
        
    Source: https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data


def get_transform() -> transforms.Compose:
    """
    Get the image preprocessing transform for DDPM training.
    
    The transform:
    1. Converts PIL Images to tensors with values in [0, 1]
    2. Rescales to [-1, 1] range (standard for diffusion models)
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    class RescaleChannels:
        """Custom transform to rescale image values from [0,1] to [-1,1]"""
        def __call__(self, sample: torch.Tensor) -> torch.Tensor:
            return 2 * sample - 1

    return transforms.Compose([
        transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v: Any) -> bool:
    """
    Convert string argument to boolean value.
    
    This is needed for argparse to properly handle boolean command line arguments.
    
    Args:
        v: String value to convert
        
    Returns:
        Boolean value
        
    Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser: argparse.ArgumentParser, default_dict: Dict[str, Any]) -> None:
    """
    Add dictionary of default values as command line arguments to parser.
    
    This utility function automatically creates command line arguments from
    a dictionary of default values, inferring the appropriate type for each argument.
    
    Args:
        parser: argparse.ArgumentParser instance
        default_dict: Dictionary with argument names as keys and default values
        
    Source: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        # Determine argument type based on default value
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults() -> Dict[str, Any]:
    """
    Get default hyperparameters for diffusion model training.
    
    These defaults are based on the DDPM paper and common practices
    for training on datasets like CIFAR-10.
    
    Returns:
        dict: Dictionary of default hyperparameters
    """
    defaults = dict(
        # Diffusion process parameters
        num_timesteps=1000,  # Number of diffusion timesteps
        schedule="linear",   # Beta schedule type ("linear" or "cosine")
        loss_type="l2",      # Loss function ("l1" or "l2")
        use_labels=False,    # Whether to use class conditioning
        
        # Additional schedule parameters (for linear schedule)
        schedule_low=1e-4,   # Starting beta value
        schedule_high=0.02,  # Ending beta value

        # U-Net architecture parameters
        base_channels=128,           # Base number of channels
        channel_mults=(1, 2, 2, 2), # Channel multipliers per resolution
        num_res_blocks=2,            # Residual blocks per resolution
        time_emb_dim=128 * 4,        # Time embedding dimension
        norm="gn",                   # Normalization type
        dropout=0.1,                 # Dropout rate
        activation="silu",           # Activation function
        attention_resolutions=(1,),  # Resolutions to apply attention
        
        # EMA parameters
        ema_decay=0.9999,     # EMA decay rate
        ema_update_rate=1,    # EMA update frequency
    )

    return defaults


def get_diffusion_from_args(args: argparse.Namespace) -> GaussianDiffusion:
    """
    Create a diffusion model from command line arguments.
    
    This function builds the complete diffusion model (U-Net + GaussianDiffusion)
    using the hyperparameters specified in the command line arguments.
    
    Args:
        args: Parsed command line arguments containing model hyperparameters
        
    Returns:
        GaussianDiffusion: Complete diffusion model ready for training or inference
    """
    # Map activation function names to actual functions
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    # Create U-Net model
    model = UNet(
        img_channels=3,  # RGB images

        # Architecture parameters
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,

        # Conditioning parameters
        num_classes=None if not args.use_labels else 10,  # CIFAR-10 has 10 classes
        initial_pad=0,  # No padding needed for 32x32 images
    )

    # Create beta schedule based on type
    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        # Linear schedule with scaling for proper range
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    # Create complete diffusion model
    diffusion = GaussianDiffusion(
        model, 
        (32, 32),  # CIFAR-10 image size
        3,         # RGB channels
        10,        # Number of classes in CIFAR-10
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,  # Start EMA after 2000 steps
        loss_type=args.loss_type,
    )

    return diffusion