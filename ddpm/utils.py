"""
Utility Functions for DDPM

This module contains helper functions used throughout the DDPM implementation.
"""

def extract(a, t, x_shape):
    """
    Extract values from tensor 'a' at indices 't' and reshape for broadcasting.
    
    This is a key utility function used to index into precomputed diffusion coefficients
    (like alpha, beta values) at specific timesteps and reshape them to broadcast
    correctly with image tensors.
    
    Args:
        a: 1D tensor containing values (e.g., alpha_cumprod, beta values)
        t: tensor of timestep indices to extract from 'a'
        x_shape: shape of the tensor we want to broadcast to (typically image shape)
        
    Returns:
        Extracted values reshaped to broadcast with tensors of shape x_shape
        
    Example:
        If a = [0.1, 0.2, 0.3, 0.4] and t = [1, 3], x_shape = (2, 3, 32, 32)
        Returns tensor of shape (2, 1, 1, 1) with values [0.2, 0.4]
        This can then be multiplied with images of shape (2, 3, 32, 32)
    """
    b, *_ = t.shape  # Get batch size from timestep tensor
    out = a.gather(-1, t)  # Extract values at indices t
    # Reshape to (batch_size, 1, 1, ...) for broadcasting with x_shape
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))