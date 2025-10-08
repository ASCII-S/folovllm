"""Unit tests for M1 model components."""

import pytest
import torch

from folovllm.model_executor.models.utils import RMSNorm, RotaryEmbedding, SiLUAndMul


class TestRMSNorm:
    """Test RMSNorm layer."""
    
    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        hidden_size = 128
        batch_size = 2
        seq_len = 10
        
        norm = RMSNorm(hidden_size)
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        out, residual = norm(x)
        
        assert out.shape == (batch_size, seq_len, hidden_size)
        assert residual.shape == (batch_size, seq_len, hidden_size)
        
        # Check that output has approximately unit variance
        variance = out.pow(2).mean(-1)
        assert torch.allclose(variance, torch.ones_like(variance), atol=0.1)
    
    def test_rmsnorm_with_residual(self):
        """Test RMSNorm with residual addition."""
        hidden_size = 128
        batch_size = 2
        seq_len = 10
        
        norm = RMSNorm(hidden_size)
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        residual = torch.randn(batch_size, seq_len, hidden_size)
        
        out, new_residual = norm(x, residual)
        
        assert out.shape == (batch_size, seq_len, hidden_size)
        # Residual should be x + old_residual
        assert torch.allclose(new_residual, x + residual)


class TestRotaryEmbedding:
    """Test Rotary Position Embedding."""
    
    def test_rope_initialization(self):
        """Test RoPE initialization."""
        dim = 64
        max_pos = 2048
        
        rope = RotaryEmbedding(dim, max_position_embeddings=max_pos)
        
        assert rope.dim == dim
        assert rope.max_position_embeddings == max_pos
        assert rope.inv_freq.shape == (dim // 2,)
    
    def test_rope_forward(self):
        """Test RoPE forward pass."""
        dim = 64
        batch_size = 2
        num_heads = 4
        seq_len = 10
        
        rope = RotaryEmbedding(dim)
        
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        q = torch.randn(batch_size, num_heads, seq_len, dim)
        k = torch.randn(batch_size, num_heads, seq_len, dim)
        
        q_rot, k_rot = rope(positions, q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        
        # Check that rotation was applied (values should be different)
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)
    
    def test_rope_decode_phase(self):
        """Test RoPE in decode phase (single token)."""
        dim = 64
        batch_size = 2
        num_heads = 4
        
        rope = RotaryEmbedding(dim)
        
        # Decode phase: single position
        positions = torch.tensor([[5], [5]])  # Position 5 for both sequences
        q = torch.randn(batch_size, num_heads, dim)
        k = torch.randn(batch_size, num_heads, dim)
        
        q_rot, k_rot = rope(positions, q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestSiLUAndMul:
    """Test SiLU and multiplication."""
    
    def test_silu_and_mul(self):
        """Test SiLU + multiplication."""
        batch_size = 2
        seq_len = 10
        hidden_size = 256
        
        act = SiLUAndMul()
        
        # Input should be 2 * hidden_size (gate and up projections)
        x = torch.randn(batch_size, seq_len, 2 * hidden_size)
        
        out = act(x)
        
        assert out.shape == (batch_size, seq_len, hidden_size)
        
        # Manually compute to verify
        gate, up = x.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up
        
        assert torch.allclose(out, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

