"""
U-Net Architecture for DDPM

This module implements the U-Net neural network architecture used in DDPM for noise prediction.
The U-Net consists of an encoder (downsampling path) and decoder (upsampling path) with
skip connections, along with time embedding and optional class conditioning.

Key components:
- PositionalEmbedding: Encodes timestep information
- ResidualBlock: Core building block with time/class conditioning
- AttentionBlock: Self-attention for better global context
- Downsample/Upsample: For changing spatial resolution
- UNet: Main model combining all components
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.normalization import GroupNorm


def get_norm(norm, num_channels, num_groups):
    """
    Factory function to create normalization layers.
    
    Args:
        norm: Type of normalization ("in", "bn", "gn", or None)
        num_channels: Number of input channels
        num_groups: Number of groups for GroupNorm
        
    Returns:
        Appropriate normalization layer
    """
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


class PositionalEmbedding(nn.Module):
    """
    Computes a positional embedding of timesteps using sinusoidal encoding.
    
    This is similar to the positional encoding used in Transformers, but adapted
    for encoding the diffusion timestep. The embedding helps the model understand
    which step of the diffusion process it's currently processing.

    Input:
        x: tensor of shape (N) containing timestep values
    Output:
        tensor of shape (N, dim) containing positional embeddings
    Args:
        dim (int): embedding dimension (must be even)
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0  # Dimension must be even for sin/cos pairs
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        """
        Generate sinusoidal positional embeddings for timesteps.
        
        Uses alternating sin and cos functions at different frequencies
        to create unique embeddings for each timestep.
        """
        device = x.device
        half_dim = self.dim // 2
        # Create frequency scaling factors
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Apply frequencies to scaled timesteps
        emb = torch.outer(x * self.scale, emb)
        # Concatenate sin and cos components
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    """
    Downsamples a given tensor by a factor of 2 using strided convolution.
    
    This is used in the encoder path of the U-Net to reduce spatial resolution
    while increasing the receptive field. Assumes even height and width.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored (for compatibility with ResidualBlock interface)
        y: ignored (for compatibility with ResidualBlock interface)
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()
        # Use strided convolution for downsampling
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb, y):
        # Validate input dimensions for downsampling
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    """
    Upsamples a given tensor by a factor of 2 using resize convolution.
    
    Uses nearest neighbor upsampling followed by convolution to avoid
    checkerboard artifacts that can occur with transposed convolution.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored (for compatibility with ResidualBlock interface)
        y: ignored (for compatibility with ResidualBlock interface)
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()
        # Use resize + convolution to avoid checkerboard artifacts
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, time_emb, y):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    """
    Applies QKV self-attention with a residual connection.
    
    Self-attention allows the model to capture long-range dependencies
    in the feature maps, which is especially useful for generating
    coherent global structure in images.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    """
    
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        # Single convolution to generate Q, K, V simultaneously
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        # Output projection
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # Generate Q, K, V and split them
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        # Reshape for attention computation
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)  # (b, hw, c)
        k = k.view(b, c, h * w)  # (b, c, hw)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)  # (b, hw, c)

        # Compute attention scores with scaling
        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        # Apply softmax to get attention weights
        attention = torch.softmax(dot_products, dim=-1)
        # Apply attention to values
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        
        # Reshape back to spatial dimensions
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        # Apply output projection and residual connection
        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    """
    Applies two conv blocks with residual connection and conditioning.
    
    This is the core building block of the U-Net. It performs convolution
    operations while incorporating time embedding and optional class conditioning.
    The residual connection helps with gradient flow during training.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None
        y: classes tensor of shape (N) or None
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        dropout: dropout probability for regularization
        time_emb_dim (int or None): time embedding dimension or None. Default: None
        num_classes (int or None): number of classes or None. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): normalization type. Default: "gn"
        num_groups (int): number of groups for group normalization. Default: 32
        use_attention (bool): whether to apply attention after convolutions. Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        time_emb_dim=None,
        num_classes=None,
        activation=F.relu,
        norm="gn",
        num_groups=32,
        use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        # First convolution block
        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Second convolution block with dropout
        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # Conditioning layers
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        # Residual connection (adjust channels if needed)
        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        # Optional attention
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
    
    def forward(self, x, time_emb=None, y=None):
        # First convolution block
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        # Add time conditioning if specified
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            # Add time bias with spatial broadcasting
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        # Add class conditioning if specified
        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")
            # Add class bias with spatial broadcasting
            out += self.class_bias(y)[:, :, None, None]

        # Second convolution block with residual connection
        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        
        # Apply attention if specified
        out = self.attention(out)

        return out


class UNet(nn.Module):
    """
    UNet model used to estimate noise in DDPM.
    
    This is the main neural network architecture that learns to predict
    the noise added to images during the diffusion process. It consists of:
    - Encoder path: downsampling to capture context
    - Bottleneck: processing at the lowest resolution
    - Decoder path: upsampling with skip connections from encoder
    
    The model can be conditioned on timesteps and optionally on class labels.

    Input:
        x: tensor of shape (N, in_channels, H, W) - noisy images
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None
        y: classes tensor of shape (N) or None
    Output:
        tensor of shape (N, out_channels, H, W) - predicted noise
    Args:
        img_channels (int): number of image channels (e.g., 3 for RGB)
        base_channels (int): number of base channels after first convolution
        channel_mults (tuple): channel multipliers for each resolution. Default: (1, 2, 4, 8)
        num_res_blocks (int): number of residual blocks per resolution. Default: 2
        time_emb_dim (int or None): time embedding dimension. Default: None
        time_emb_scale (float): linear scale for timesteps. Default: 1.0
        num_classes (int or None): number of classes for conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        dropout (float): dropout rate for regularization
        attention_resolutions (tuple): resolutions at which to apply attention. Default: ()
        norm (string or None): normalization type. Default: "gn"
        num_groups (int): number of groups for group normalization. Default: 32
        initial_pad (int): initial padding for non-power-of-2 dimensions. Default: 0
    """

    def __init__(
        self,
        img_channels,
        base_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=None,
        time_emb_scale=1.0,
        num_classes=None,
        activation=F.relu,
        dropout=0.1,
        attention_resolutions=(),
        norm="gn",
        num_groups=32,
        initial_pad=0,
    ):
        super().__init__()

        self.activation = activation
        self.initial_pad = initial_pad

        # Store class conditioning info
        self.num_classes = num_classes
        
        # Time embedding network (converts timestep to rich representation)
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None
    
        # Initial convolution to process input images
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Encoder and decoder paths
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Track channel progression for skip connections
        channels = [base_channels]
        now_channels = base_channels

        # Build encoder path (downsampling)
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            # Add residual blocks at this resolution
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)
            
            # Add downsampling (except at the deepest level)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        
        # Bottleneck processing at lowest resolution
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,  # Always use attention in bottleneck
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])

        # Build decoder path (upsampling with skip connections)
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            # Add residual blocks at this resolution
            # Note: +1 because we have one extra block that processes skip connection
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,  # Concatenate skip connection
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
            
            # Add upsampling (except at the final level)
            if i != 0:
                self.ups.append(Upsample(now_channels))
        
        # Ensure all skip connections were used
        assert len(channels) == 0
        
        # Final output layers
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)
    
    def forward(self, x, time=None, y=None):
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input noisy images
            time: Timestep values for time conditioning
            y: Class labels for class conditioning
            
        Returns:
            Predicted noise with same shape as input
        """
        # Apply initial padding if needed (for non-power-of-2 dimensions)
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        # Process time embedding
        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            time_emb = self.time_mlp(time)
        else:
            time_emb = None
        
        # Validate class conditioning
        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")
        
        # Initial convolution
        x = self.init_conv(x)

        # Store skip connections for decoder
        skips = [x]

        # Encoder path
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
        
        # Bottleneck processing
        for layer in self.mid:
            x = layer(x, time_emb, y)
        
        # Decoder path with skip connections
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                # Concatenate skip connection before residual block
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)

        # Final output processing
        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)
        
        # Remove initial padding if it was applied
        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x