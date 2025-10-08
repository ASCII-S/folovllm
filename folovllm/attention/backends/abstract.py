"""Abstract attention backend interface."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class AttentionBackend(ABC):
    """Abstract base class for attention implementations.
    
    Different backends can implement optimized attention:
    - M1: TorchNaiveBackend (pure PyTorch)
    - M3: PagedAttentionBackend (paged attention with block manager)
    - M4: FlashAttentionBackend (Flash Attention 2)
    """
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of attention.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            key: Key tensor [batch_size, num_kv_heads, head_dim] (for decode)
                           or [batch_size, num_kv_heads, seq_len_k, head_dim] (for prefill)
            value: Value tensor, same shape as key
            kv_cache: Optional tuple of (key_cache, value_cache)
            scale: Attention scale factor
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output, updated_kv_cache)
            - output: [batch_size, num_heads, seq_len_q, head_dim]
            - updated_kv_cache: Tuple of updated (key_cache, value_cache)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_name(self) -> str:
        """Get backend name for debugging."""
        raise NotImplementedError

