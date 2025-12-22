"""
Attention mechanisms for TIGAS model.
Implements self-attention and cross-modal attention for feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Captures long-range dependencies in feature maps.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C] where N is sequence length

        Returns:
            output: Attention output [B, N, C]
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Clamp attention scores to prevent overflow
        attn = torch.clamp(attn, min=-1e4, max=1e4)
        
        attn = F.softmax(attn, dim=-1)
        
        # Replace any NaN from softmax with uniform attention
        if torch.isnan(attn).any():
            import warnings
            warnings.warn("[ATTENTION] NaN detected in attention weights, using uniform distribution")
            attn = torch.ones_like(attn) / attn.shape[-1]
        
        attn = self.attn_dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing different feature modalities.
    Allows one modality to attend to another (e.g., spatial attending to frequency).
    """

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert query_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for query and key-value
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)

        self.out_proj = nn.Linear(query_dim, query_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor [B, N_q, C_q]
            key_value: Key-Value tensor [B, N_kv, C_kv]
            mask: Optional attention mask

        Returns:
            output: Cross-attended features [B, N_q, C_q]
        """
        B, N_q, C_q = query.shape
        N_kv = key_value.shape[1]

        # Project Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim)

        # Transpose for attention
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, N_q, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N_q, N_kv]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Clamp before softmax to prevent overflow
        attn = torch.clamp(attn, min=-1e4, max=1e4)
        
        attn = F.softmax(attn, dim=-1)
        
        # Safety check for NaN
        if torch.isnan(attn).any():
            import warnings
            warnings.warn("[CROSSMODAL ATTN] NaN in attention, using uniform")
            attn = torch.ones_like(attn) / attn.shape[-1]
        
        attn = self.attn_dropout(attn)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C_q)

        # Output projection
        x = self.out_proj(x)
        x = self.proj_dropout(x)

        return x


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for emphasizing important regions.
    Helps focus on areas with potential artifacts.
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // reduction, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            attended: Spatially attended features [B, C, H, W]
            attention_map: Spatial attention map [B, 1, H, W]
        """
        # Generate attention map
        attn = F.relu(self.conv1(x), inplace=True)
        attn = F.relu(self.conv2(attn), inplace=True)
        attn = torch.sigmoid(self.conv3(attn))  # [B, 1, H, W]

        # Apply attention
        attended = x * attn

        return attended, attn


class ChannelAttention(nn.Module):
    """
    Channel attention (Squeeze-and-Excitation).
    Recalibrates channel-wise feature responses.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            attended: Channel-attended features [B, C, H, W]
        """
        B, C, _, _ = x.shape

        # Average and max pooling
        avg_pool = self.avg_pool(x).view(B, C)
        max_pool = self.max_pool(x).view(B, C)

        # Channel attention
        avg_attn = self.fc(avg_pool)
        max_attn = self.fc(max_pool)

        # Combine and apply sigmoid
        attn = torch.sigmoid(avg_attn + max_attn).view(B, C, 1, 1)

        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines channel and spatial attention.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            out: Attended features [B, C, H, W]
            spatial_map: Spatial attention map [B, 1, H, W]
        """
        # Channel attention
        x = self.channel_attn(x)

        # Spatial attention
        x, spatial_map = self.spatial_attn(x)

        return x, spatial_map


class PositionalEncoding(nn.Module):
    """
    Positional encoding for spatial features.
    Adds position information to feature maps.
    """

    def __init__(self, channels: int, height: int, width: int):
        super().__init__()

        # Create 2D positional encoding
        position_h = torch.arange(height).unsqueeze(1).repeat(1, width)
        position_w = torch.arange(width).unsqueeze(0).repeat(height, 1)

        # Sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels)
        )

        pe = torch.zeros(channels, height, width)
        pe[0::2, :, :] = torch.sin(position_h.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::2, :, :] = torch.cos(position_w.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            output: Input with positional encoding [B, C, H, W]
        """
        return x + self.pe.unsqueeze(0)
