"""Model utilities: RoPE, RMSNorm, activations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Used in Qwen3, LLaMA, and many modern LLMs.
    More efficient than LayerNorm as it doesn't subtract mean.
    
    Args:
        hidden_size: Size of the hidden dimension
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional fused residual addition.
        
        Args:
            hidden_states: Input tensor [..., hidden_size]
            residual: Optional residual tensor to add before normalization
            
        Returns:
            Tuple of (normalized_output, new_residual)
            If residual is provided, returns (norm(hidden_states + residual), hidden_states + residual)
            If residual is None, returns (norm(hidden_states), hidden_states)
        """
        input_dtype = hidden_states.dtype
        
        # Add residual if provided (before normalization)
        if residual is not None:
            # Save the sum as new residual
            new_residual = hidden_states + residual
            hidden_states = new_residual.to(torch.float32)
        else:
            # No residual, use hidden_states as is
            new_residual = hidden_states
            hidden_states = hidden_states.to(torch.float32)
        
        # Compute RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # Apply weight and convert back to input dtype
        hidden_states = self.weight * hidden_states.to(input_dtype)
        
        return hidden_states, new_residual


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    Applies rotary embeddings to query and key tensors.
    Used in Qwen3, LLaMA, GPT-NeoX, and many modern LLMs.
    
    Args:
        dim: Dimension of the rotary embedding (typically head_dim)
        max_position_embeddings: Maximum sequence length
        base: Base for the geometric progression (theta)
        scaling_factor: Scaling factor for extended context
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache cos/sin values for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cached cos/sin values if needed."""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = max(seq_len, self._seq_len_cached)
            
            # Generate position indices
            t = torch.arange(
                self._seq_len_cached,
                device=device,
                dtype=self.inv_freq.dtype,
            )
            t = t / self.scaling_factor
            
            # Compute frequencies
            freqs = torch.outer(t, self.inv_freq)
            # Different from paper, but it uses a different permutation
            # to get the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key.
        
        Args:
            positions: Position indices [batch_size] or [batch_size, seq_len]
            query: Query tensor [batch_size, num_heads, seq_len, head_dim] or
                                [batch_size, num_heads, head_dim]
            key: Key tensor, same shape as query
            
        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        # Get max position for cache
        max_pos = positions.max().item() + 1
        self._update_cos_sin_cache(max_pos, positions.device, query.dtype)
        
        # Get cos/sin for the given positions
        cos = self._cos_cached[positions]  # [batch_size, ...] or [batch_size, seq_len, dim]
        sin = self._sin_cached[positions]
        
        # Apply rotary embedding
        query = self._apply_rotary_emb(query, cos, sin)
        key = self._apply_rotary_emb(key, cos, sin)
        
        return query, key
    
    @staticmethod
    def _apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding to input tensor.
        
        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim] or [batch, num_heads, head_dim]
            cos: Cosine values [batch, seq_len, dim] or [batch, dim]
            sin: Sine values [batch, seq_len, dim] or [batch, dim]
            
        Returns:
            Rotated tensor with same shape as x
        """
        # Split x into two halves and apply rotation
        # x_rotated = [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        
        # Ensure cos/sin have the right shape to broadcast with x
        # x: [batch, num_heads, seq_len, head_dim] or [batch, num_heads, head_dim]
        # cos/sin: [batch, seq_len, dim] or [batch, dim]
        
        # Need to add num_heads dimension to cos/sin
        if x.dim() == 4:  # [batch, num_heads, seq_len, head_dim]
            # cos/sin: [batch, seq_len, dim] -> [batch, 1, seq_len, dim]
            if cos.dim() == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        elif x.dim() == 3:  # [batch, num_heads, head_dim]
            # cos/sin: [batch, dim] -> [batch, 1, dim]
            if cos.dim() == 2:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        
        # Split cos/sin to match x1, x2
        cos1 = cos[..., : cos.shape[-1] // 2]
        cos2 = cos[..., cos.shape[-1] // 2 :]
        sin1 = sin[..., : sin.shape[-1] // 2]
        sin2 = sin[..., sin.shape[-1] // 2 :]
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos1 - x2 * sin1,
            x1 * sin2 + x2 * cos2,
        ], dim=-1)
        
        return rotated


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation function.
    
    Also known as Swish. Used in Qwen3 MLP.
    SiLU(x) = x * sigmoid(x)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class SiLUAndMul(nn.Module):
    """Fused SiLU and element-wise multiplication.
    
    Used in gated MLPs. Given input [x, y], computes:
        SiLU(x) * y
    
    This is more memory efficient than separate operations.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [..., 2 * hidden_size]
               Will be split into two halves: gate and up
               
        Returns:
            Output tensor [..., hidden_size]
        """
        # Split input into gate and up projections
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

