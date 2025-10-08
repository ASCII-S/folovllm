"""Attention operations with KV cache support."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def reshape_and_cache_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    slot_mapping: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reshape and cache key/value tensors.
    
    For M1, we use simple contiguous memory allocation.
    In M3, this will be replaced with paged attention.
    
    Args:
        key: Key tensor of shape [batch_size, num_kv_heads, head_dim]
        value: Value tensor of shape [batch_size, num_kv_heads, head_dim]
        kv_cache: Tuple of (key_cache, value_cache) tensors
        slot_mapping: Optional slot mapping for cache (used in M3+)
        
    Returns:
        Updated (key_cache, value_cache) tuple
    """
    key_cache, value_cache = kv_cache
    
    # For M1: simple concatenation along sequence dimension
    # key/value shape: [batch_size, num_kv_heads, head_dim]
    # We need to append to the cache
    
    if key_cache.numel() == 0:
        # First token: initialize cache
        # Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        key_cache = key.unsqueeze(2)
        value_cache = value.unsqueeze(2)
    else:
        # Append new key/value to cache
        # key_cache shape: [batch_size, num_kv_heads, past_seq_len, head_dim]
        # key shape: [batch_size, num_kv_heads, head_dim]
        key = key.unsqueeze(2)  # [batch_size, num_kv_heads, 1, head_dim]
        value = value.unsqueeze(2)
        key_cache = torch.cat([key_cache, key], dim=2)
        value_cache = torch.cat([value_cache, value], dim=2)
    
    return key_cache, value_cache


def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Naive attention implementation using PyTorch operations.
    
    This is a reference implementation without optimizations.
    Used as baseline in M1, will be replaced by Flash Attention in M4.
    
    Args:
        query: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        key: Key tensor [batch_size, num_kv_heads, seq_len_k, head_dim]
        value: Value tensor [batch_size, num_kv_heads, seq_len_k, head_dim]
        scale: Scaling factor (typically 1/sqrt(head_dim))
        attn_mask: Optional attention mask [batch_size, 1, seq_len_q, seq_len_k]
        
    Returns:
        Output tensor [batch_size, num_heads, seq_len_q, head_dim]
    """
    # Handle Grouped Query Attention (GQA)
    # If num_heads > num_kv_heads, repeat key/value
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    _, num_kv_heads, seq_len_k, _ = key.shape
    
    if num_heads > num_kv_heads:
        # Repeat key/value heads to match query heads
        num_repeats = num_heads // num_kv_heads
        # Expand along head dimension
        key = key.unsqueeze(2).expand(
            batch_size, num_kv_heads, num_repeats, seq_len_k, head_dim
        ).reshape(batch_size, num_heads, seq_len_k, head_dim)
        value = value.unsqueeze(2).expand(
            batch_size, num_kv_heads, num_repeats, seq_len_k, head_dim
        ).reshape(batch_size, num_heads, seq_len_k, head_dim)
    
    # Compute attention scores: Q @ K^T
    # [batch_size, num_heads, seq_len_q, head_dim] @ [batch_size, num_heads, head_dim, seq_len_k]
    # -> [batch_size, num_heads, seq_len_q, seq_len_k]
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply attention mask (for causal attention)
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    # Apply softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # Compute output: attn_weights @ V
    # [batch_size, num_heads, seq_len_q, seq_len_k] @ [batch_size, num_heads, seq_len_k, head_dim]
    # -> [batch_size, num_heads, seq_len_q, head_dim]
    output = torch.matmul(attn_weights, value)
    
    return output


def create_causal_mask(
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create causal attention mask.
    
    Args:
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length  
        device: Device to create mask on
        dtype: Data type for mask
        
    Returns:
        Causal mask of shape [1, 1, seq_len_q, seq_len_k]
        Values are 0 for allowed positions, -inf for masked positions
    """
    # Create a mask where position i can only attend to positions <= i
    # For generation: seq_len_q=1 (current token), seq_len_k=total_len (all past tokens)
    # For prefill: seq_len_q=seq_len_k=prompt_len
    
    mask = torch.ones(seq_len_q, seq_len_k, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=seq_len_k - seq_len_q + 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    # Add batch and head dimensions
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, seq_len_k]
    
    return mask

