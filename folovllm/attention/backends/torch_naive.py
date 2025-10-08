"""Naive PyTorch attention backend (M1)."""

from typing import Optional, Tuple
import torch

from folovllm.attention.backends.abstract import AttentionBackend
from folovllm.attention.ops import naive_attention, reshape_and_cache_kv


class TorchNaiveBackend(AttentionBackend):
    """Naive attention implementation using pure PyTorch.
    
    This is the baseline implementation for M1:
    - No kernel fusion
    - No memory optimization
    - Simple and readable
    
    Will be replaced by optimized backends in future milestones.
    """
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with naive attention.
        
        Args:
            query: [batch_size, num_heads, seq_len_q, head_dim]
            key: [batch_size, num_kv_heads, head_dim] or 
                 [batch_size, num_kv_heads, seq_len_k, head_dim]
            value: Same shape as key
            kv_cache: Tuple of (key_cache, value_cache) or None
            scale: Attention scale (typically 1/sqrt(head_dim))
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, updated_kv_cache)
        """
        # Initialize cache if not provided
        if kv_cache is None:
            key_cache = torch.empty(0, device=query.device, dtype=query.dtype)
            value_cache = torch.empty(0, device=query.device, dtype=query.dtype)
        else:
            key_cache, value_cache = kv_cache
        
        # If key is 3D (decode phase: [batch, num_kv_heads, head_dim]),
        # we need to add it to cache
        if key.dim() == 3:
            key_cache, value_cache = reshape_and_cache_kv(
                key, value, (key_cache, value_cache)
            )
            # Use full cache for attention
            key = key_cache
            value = value_cache
        elif key.dim() == 4:
            # Prefill phase: key is already [batch, num_kv_heads, seq_len, head_dim]
            # Update cache with all keys/values
            if key_cache.numel() == 0:
                key_cache = key
                value_cache = value
            else:
                # Append new keys/values
                key_cache = torch.cat([key_cache, key], dim=2)
                value_cache = torch.cat([value_cache, value], dim=2)
            key = key_cache
            value = value_cache
        else:
            raise ValueError(f"Invalid key shape: {key.shape}")
        
        # Run attention
        output = naive_attention(query, key, value, scale, attn_mask)
        
        return output, (key_cache, value_cache)
    
    def get_name(self) -> str:
        """Get backend name."""
        return "torch_naive"

