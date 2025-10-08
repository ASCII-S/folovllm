"""Attention layer implementation."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from folovllm.attention.backends import AttentionBackend, TorchNaiveBackend
from folovllm.attention.ops import create_causal_mask
from folovllm.model_executor.models.utils import RotaryEmbedding


class Attention(nn.Module):
    """Attention layer with KV cache and RoPE support.
    
    This is a general attention layer that can be used in various models.
    It handles:
    - Grouped Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - KV cache management
    - Different attention backends
    
    Args:
        hidden_size: Hidden dimension size
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension of each head (defaults to hidden_size // num_heads)
        max_position_embeddings: Maximum sequence length
        rope_theta: Base for RoPE
        rope_scaling: Scaling factor for RoPE
        bias: Whether to use bias in projections
        backend: Attention backend to use (defaults to TorchNaiveBackend)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[float] = None,
        bias: bool = False,
        backend: Optional[AttentionBackend] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        
        # Validate dimensions
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        
        # QKV projections
        self.qkv_proj = nn.Linear(
            hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=bias,
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.q_size,
            hidden_size,
            bias=bias,
        )
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=rope_scaling or 1.0,
        )
        
        # Attention backend
        self.backend = backend or TorchNaiveBackend()
        
        # KV cache (will be set externally by model runner)
        self.kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            positions: Position indices [batch_size] or [batch_size, seq_len]
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            kv_cache: Optional KV cache tuple
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)
        
        # For decode phase (seq_len=1), we pass k/v as 3D tensors to backend
        # For prefill phase (seq_len>1), we pass k/v as 4D tensors
        if seq_len == 1:
            # Decode: remove seq_len dimension
            k = k.squeeze(2)  # [batch_size, num_kv_heads, head_dim]
            v = v.squeeze(2)
        
        # Create causal mask for prefill
        attn_mask = None
        if seq_len > 1:
            # Prefill phase: need causal mask
            cache_len = 0
            if kv_cache is not None and kv_cache[0].numel() > 0:
                cache_len = kv_cache[0].shape[2]
            
            total_len = cache_len + seq_len
            attn_mask = create_causal_mask(
                seq_len, total_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        
        # Run attention
        output, kv_cache = self.backend.forward(
            q, k, v, kv_cache, self.scaling, attn_mask
        )
        
        # Reshape output: [batch_size, num_heads, seq_len, head_dim] 
        # -> [batch_size, seq_len, num_heads * head_dim]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.q_size)
        
        # Output projection
        output = self.o_proj(output)
        
        # Update cache attribute for next iteration
        self.kv_cache = kv_cache
        
        return output

