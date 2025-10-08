"""Unit tests for M1 attention implementations."""

import pytest
import torch

from folovllm.attention.ops import (
    naive_attention,
    reshape_and_cache_kv,
    create_causal_mask,
)
from folovllm.attention.backends import TorchNaiveBackend


class TestAttentionOps:
    """Test attention operations."""
    
    def test_create_causal_mask(self):
        """Test causal mask creation."""
        # Test square mask
        mask = create_causal_mask(4, 4, device="cpu")
        assert mask.shape == (1, 1, 4, 4)
        
        # Check that mask is causal (lower triangular with 0s, upper with -inf)
        # Position i can attend to positions 0..i
        expected = torch.tensor([
            [0, float('-inf'), float('-inf'), float('-inf')],
            [0, 0, float('-inf'), float('-inf')],
            [0, 0, 0, float('-inf')],
            [0, 0, 0, 0],
        ])
        assert torch.allclose(mask[0, 0], expected, equal_nan=True)
        
        # Test decode case (query_len=1, key_len=5)
        mask = create_causal_mask(1, 5, device="cpu")
        assert mask.shape == (1, 1, 1, 5)
        # Should be all zeros (can attend to all past tokens)
        assert torch.all(mask == 0)
    
    def test_reshape_and_cache_kv(self):
        """Test KV cache storage."""
        batch_size, num_heads, head_dim = 2, 4, 64
        
        # First token: initialize cache
        key = torch.randn(batch_size, num_heads, head_dim)
        value = torch.randn(batch_size, num_heads, head_dim)
        key_cache = torch.empty(0)
        value_cache = torch.empty(0)
        
        key_cache, value_cache = reshape_and_cache_kv(
            key, value, (key_cache, value_cache)
        )
        
        assert key_cache.shape == (batch_size, num_heads, 1, head_dim)
        assert value_cache.shape == (batch_size, num_heads, 1, head_dim)
        
        # Second token: append to cache
        key2 = torch.randn(batch_size, num_heads, head_dim)
        value2 = torch.randn(batch_size, num_heads, head_dim)
        
        key_cache, value_cache = reshape_and_cache_kv(
            key2, value2, (key_cache, value_cache)
        )
        
        assert key_cache.shape == (batch_size, num_heads, 2, head_dim)
        assert value_cache.shape == (batch_size, num_heads, 2, head_dim)
    
    def test_naive_attention(self):
        """Test naive attention computation."""
        batch_size = 2
        num_heads = 4
        seq_len_q = 3
        seq_len_k = 5
        head_dim = 64
        scale = head_dim ** -0.5
        
        query = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len_k, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len_k, head_dim)
        
        output = naive_attention(query, key, value, scale)
        
        assert output.shape == (batch_size, num_heads, seq_len_q, head_dim)
        
        # Test with causal mask
        mask = create_causal_mask(seq_len_q, seq_len_k, device=query.device)
        output_masked = naive_attention(query, key, value, scale, mask)
        
        assert output_masked.shape == (batch_size, num_heads, seq_len_q, head_dim)
    
    def test_naive_attention_gqa(self):
        """Test naive attention with Grouped Query Attention."""
        batch_size = 2
        num_heads = 8
        num_kv_heads = 2  # 4x groups
        seq_len = 4
        head_dim = 64
        scale = head_dim ** -0.5
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        
        output = naive_attention(query, key, value, scale)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)


class TestTorchNaiveBackend:
    """Test TorchNaiveBackend."""
    
    def test_backend_forward_prefill(self):
        """Test backend forward pass in prefill phase."""
        backend = TorchNaiveBackend()
        
        batch_size = 2
        num_heads = 4
        num_kv_heads = 2
        seq_len = 5
        head_dim = 64
        scale = head_dim ** -0.5
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        
        mask = create_causal_mask(seq_len, seq_len, device=query.device)
        
        output, kv_cache = backend.forward(query, key, value, None, scale, mask)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert kv_cache is not None
        assert kv_cache[0].shape == (batch_size, num_kv_heads, seq_len, head_dim)
        assert kv_cache[1].shape == (batch_size, num_kv_heads, seq_len, head_dim)
    
    def test_backend_forward_decode(self):
        """Test backend forward pass in decode phase."""
        backend = TorchNaiveBackend()
        
        batch_size = 2
        num_heads = 4
        num_kv_heads = 2
        head_dim = 64
        scale = head_dim ** -0.5
        
        # Initialize cache from prefill
        seq_len_prefill = 5
        key_cache = torch.randn(batch_size, num_kv_heads, seq_len_prefill, head_dim)
        value_cache = torch.randn(batch_size, num_kv_heads, seq_len_prefill, head_dim)
        
        # Decode: single token
        query = torch.randn(batch_size, num_heads, 1, head_dim)
        key = torch.randn(batch_size, num_kv_heads, head_dim)
        value = torch.randn(batch_size, num_kv_heads, head_dim)
        
        output, kv_cache = backend.forward(
            query, key, value, (key_cache, value_cache), scale, None
        )
        
        assert output.shape == (batch_size, num_heads, 1, head_dim)
        assert kv_cache[0].shape == (batch_size, num_kv_heads, seq_len_prefill + 1, head_dim)
        assert kv_cache[1].shape == (batch_size, num_kv_heads, seq_len_prefill + 1, head_dim)
    
    def test_backend_name(self):
        """Test backend name."""
        backend = TorchNaiveBackend()
        assert backend.get_name() == "torch_naive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

