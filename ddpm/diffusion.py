"""
Gaussian Diffusion Model Implementation

This module implements the core diffusion process for DDPM (Denoising Diffusion Probabilistic Models).
It contains the GaussianDiffusion class which handles the forward noising process and reverse denoising
process, along with utility functions for generating beta schedules.

Key concepts:
- Forward process: Gradually adds Gaussian noise to data
- Reverse process: Learns to denoise data step by step
- Beta schedule: Controls the amount of noise added at each timestep
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .ema import EMA  # Exponential Moving Average for model weights
from .utils import extract  # Utility function for tensor indexing

class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(
        self,
        model,
        img_size,
        img_channels,
        num_classes,
        betas,
        loss_type="l2",
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
    ):
        super().__init__()

        # Store the main model and create a copy for Exponential Moving Average (EMA)
        # EMA provides more stable and better-performing sampling
        self.model = model
        self.ema_model = deepcopy(model)

        # Initialize EMA helper class for tracking moving averages of model parameters
        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start  # Number of steps before starting EMA
        self.ema_update_rate = ema_update_rate  # Frequency of EMA updates
        self.step = 0  # Training step counter

        # Store image properties for validation and tensor creation
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        # Validate loss type
        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        # Convert beta schedule to alpha values
        # Alpha represents the fraction of signal retained at each step
        alphas = 1.0 - betas
        # Cumulative product of alphas - represents total signal retention from start to step t
        alphas_cumprod = np.cumprod(alphas)

        # Helper function to convert numpy arrays to PyTorch tensors
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Register all diffusion parameters as buffers (they won't be updated during training)
        # These are the core mathematical components of the diffusion process
        
        # Beta schedule - controls noise addition at each step
        self.register_buffer("betas", to_torch(betas))
        # Alpha values - signal retention at each step
        self.register_buffer("alphas", to_torch(alphas))
        # Cumulative alphas - total signal retention from start to step t
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # Precomputed terms for efficient forward process (adding noise)
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        
        # Precomputed terms for efficient reverse process (removing noise)
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        
        # Standard deviation for sampling during reverse process
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        """
        Update the Exponential Moving Average (EMA) model weights.
        
        EMA helps stabilize training and often produces better results during sampling
        by maintaining a running average of model parameters that changes more slowly
        than the main model parameters.
        """
        self.step += 1
        # Only update EMA at specified intervals
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                # Before EMA start, just copy the current model weights
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                # After EMA start, use exponential moving average
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True):
        """
        Remove noise from a noisy image at timestep t using the trained model.
        
        This implements the reverse diffusion step, predicting and subtracting noise
        to get a cleaner image. Uses the reparameterization from DDPM paper.
        
        Args:
            x: Noisy image tensor of shape (batch_size, channels, height, width)
            t: Timestep tensor of shape (batch_size,)
            y: Class labels (optional) of shape (batch_size,)
            use_ema: Whether to use EMA model weights for better quality
            
        Returns:
            Denoised image tensor of the same shape as x
        """
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        """
        Generate new images by sampling from the diffusion model.
        
        This implements the full reverse diffusion process, starting from pure noise
        and gradually denoising to produce realistic images.
        
        Args:
            batch_size: Number of images to generate
            device: Device to run computation on (CPU/GPU)
            y: Optional class labels for conditional generation
            use_ema: Whether to use EMA model weights for better quality
            
        Returns:
            Generated images tensor of shape (batch_size, channels, height, width)
        """
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        # Start with pure noise from a standard Gaussian distribution
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        # Reverse diffusion: iterate from maximum noise (T-1) to no noise (0)
        for t in range(self.num_timesteps - 1, -1, -1):
            # Create timestep tensor for the current batch
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            # Remove predicted noise at this timestep
            x = self.remove_noise(x, t_batch, y, use_ema)

            # Add noise for next iteration (except at final step)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        """
        Generate new images and return the full diffusion sequence for visualization.
        
        This is similar to sample() but returns the intermediate steps of the denoising process,
        useful for creating animations or understanding how the model gradually denoises.
        
        Args:
            batch_size: Number of images to generate
            device: Device to run computation on (CPU/GPU)
            y: Optional class labels for conditional generation
            use_ema: Whether to use EMA model weights for better quality
            
        Returns:
            List of image tensors showing the denoising progression
        """
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        # Start with pure noise
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        # Record each step of the denoising process
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        """
        Apply the forward diffusion process - add noise to clean images.
        
        This implements the reparameterization trick from the DDPM paper, allowing
        us to sample from q(x_t | x_0) directly instead of iteratively.
        
        Args:
            x: Clean image tensor of shape (batch_size, channels, height, width)
            t: Timestep tensor of shape (batch_size,)
            noise: Random noise tensor of the same shape as x
            
        Returns:
            Noisy image tensor x_t
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x, t, y):
        """
        Calculate the training loss for the diffusion model.
        
        This implements the loss function from the DDPM paper, which trains the model
        to predict the noise that was added to the clean image.
        
        Args:
            x: Clean image tensor
            t: Timestep tensor
            y: Class labels (optional)
            
        Returns:
            Loss scalar tensor
        """
        # Generate random noise to add to the clean image
        noise = torch.randn_like(x)

        # Apply forward diffusion process to get noisy image
        perturbed_x = self.perturb_x(x, t, noise)
        # Let the model predict the noise
        estimated_noise = self.model(perturbed_x, t, y)

        # Calculate loss between predicted and actual noise
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def forward(self, x, y=None):
        """
        Forward pass for training - applies random timestep noise and calculates loss.
        
        Args:
            x: Clean image batch of shape (batch_size, channels, height, width)
            y: Optional class labels for conditional training
            
        Returns:
            Training loss (scalar tensor)
        """
        b, c, h, w = x.shape
        device = x.device

        # Validate input dimensions
        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")
        
        # Sample random timesteps for each image in the batch
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, y)


def generate_cosine_schedule(T, s=0.008):
    """
    Generate a cosine beta schedule for the diffusion process.
    
    This schedule starts with very small noise amounts and gradually increases,
    but avoids the saturation issues of linear schedules at the end of the
    diffusion process.
    
    Args:
        T: Total number of timesteps
        s: Small offset parameter to prevent beta from being 0 at t=0
        
    Returns:
        np.array: Beta values for each timestep
    """
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)


def generate_linear_schedule(T, low, high):
    """
    Generate a linear beta schedule for the diffusion process.
    
    This is the simplest schedule that linearly increases the noise amount
    from low to high over T timesteps.
    
    Args:
        T: Total number of timesteps
        low: Starting beta value (small noise)
        high: Ending beta value (large noise)
        
    Returns:
        np.array: Beta values for each timestep
    """
    return np.linspace(low, high, T)