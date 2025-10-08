"""Unit tests for M1 sampling."""

import pytest
import torch

from folovllm.sampling_params import SamplingParams, SamplingType
from folovllm.sample import Sampler
from folovllm.sample.ops import (
    apply_top_k_filtering,
    apply_top_p_filtering,
    apply_min_p_filtering,
)


class TestSamplingOps:
    """Test sampling operations."""
    
    def test_top_k_filtering(self):
        """Test top-k filtering."""
        # Create logits with known order
        logits = torch.tensor([[5.0, 3.0, 1.0, 4.0, 2.0]])
        
        # Keep top 3
        filtered = apply_top_k_filtering(logits, top_k=3)
        
        # Should keep 5.0, 4.0, 3.0 and mask others
        assert not torch.isinf(filtered[0, 0])  # 5.0
        assert not torch.isinf(filtered[0, 1])  # 3.0
        assert torch.isinf(filtered[0, 2])      # 1.0 (masked)
        assert not torch.isinf(filtered[0, 3])  # 4.0
        assert torch.isinf(filtered[0, 4])      # 2.0 (masked)
    
    def test_top_p_filtering(self):
        """Test top-p (nucleus) filtering."""
        # Create logits
        logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
        
        # Apply top-p
        filtered = apply_top_p_filtering(logits, top_p=0.8)
        
        # Should keep tokens until cumulative prob >= 0.8
        # At least first token should not be masked
        assert not torch.isinf(filtered[0, 0])
    
    def test_min_p_filtering(self):
        """Test min-p filtering."""
        # Create logits with one dominant token
        logits = torch.tensor([[10.0, 5.0, 1.0, 0.5]])
        
        # Apply min-p (keep tokens with prob >= 0.1 * max_prob)
        filtered = apply_min_p_filtering(logits, min_p=0.1)
        
        # First token should not be masked
        assert not torch.isinf(filtered[0, 0])


class TestSampler:
    """Test Sampler class."""
    
    def test_greedy_sampling(self):
        """Test greedy sampling (temperature=0)."""
        sampler = Sampler()
        
        # Create logits where token 2 has highest value
        logits = torch.tensor([[1.0, 2.0, 5.0, 3.0]])
        
        sampling_params = SamplingParams(temperature=0.0)
        
        tokens, log_probs = sampler.sample(logits, sampling_params)
        
        # Should always select argmax (token 2)
        assert tokens[0].item() == 2
        assert sampling_params.sampling_type == SamplingType.GREEDY
    
    def test_random_sampling(self):
        """Test random sampling with temperature."""
        sampler = Sampler()
        
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        
        sampling_params = SamplingParams(temperature=1.0, seed=42)
        
        tokens, _ = sampler.sample(logits, sampling_params)
        
        # Should sample from distribution (token ID should be valid)
        assert 0 <= tokens[0].item() < 4
        assert sampling_params.sampling_type == SamplingType.RANDOM
    
    def test_top_k_sampling(self):
        """Test top-k sampling."""
        sampler = Sampler()
        
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        sampling_params = SamplingParams(
            temperature=1.0,
            top_k=2,
            seed=42,
        )
        
        tokens, _ = sampler.sample(logits, sampling_params)
        
        # Should only sample from top 2 tokens (3: 5.0 and 4: 4.0)
        assert tokens[0].item() in [3, 4]
    
    def test_top_p_sampling(self):
        """Test top-p sampling."""
        sampler = Sampler()
        
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.5,
            seed=42,
        )
        
        tokens, _ = sampler.sample(logits, sampling_params)
        
        # Should sample from nucleus
        assert 0 <= tokens[0].item() < 5
    
    def test_check_stop_conditions_max_tokens(self):
        """Test stop condition: max_tokens."""
        sampler = Sampler()
        
        sampling_params = SamplingParams(max_tokens=5)
        
        # Test with fewer tokens
        should_stop, reason = sampler.check_stop_conditions(
            [1, 2, 3], "", sampling_params
        )
        assert not should_stop
        
        # Test with exactly max_tokens
        should_stop, reason = sampler.check_stop_conditions(
            [1, 2, 3, 4, 5], "", sampling_params
        )
        assert should_stop
        assert reason == "length"
    
    def test_check_stop_conditions_eos(self):
        """Test stop condition: EOS token."""
        sampler = Sampler()
        
        sampling_params = SamplingParams(max_tokens=100)
        eos_token_id = 2
        
        # Test without EOS
        should_stop, reason = sampler.check_stop_conditions(
            [1, 3, 4], "", sampling_params, eos_token_id
        )
        assert not should_stop
        
        # Test with EOS
        should_stop, reason = sampler.check_stop_conditions(
            [1, 3, 2], "", sampling_params, eos_token_id
        )
        assert should_stop
        assert reason == "stop"
    
    def test_check_stop_conditions_stop_strings(self):
        """Test stop condition: stop strings."""
        sampler = Sampler()
        
        sampling_params = SamplingParams(
            max_tokens=100,
            stop=["STOP", "END"],
        )
        
        # Test without stop string
        should_stop, reason = sampler.check_stop_conditions(
            [1, 2, 3], "Hello world", sampling_params
        )
        assert not should_stop
        
        # Test with stop string
        should_stop, reason = sampler.check_stop_conditions(
            [1, 2, 3], "Hello STOP world", sampling_params
        )
        assert should_stop
        assert reason == "stop"


class TestSamplingParams:
    """Test SamplingParams class."""
    
    def test_default_params(self):
        """Test default parameters."""
        params = SamplingParams()
        
        assert params.n == 1
        assert params.best_of == 1
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.max_tokens == 16
    
    def test_greedy_detection(self):
        """Test greedy sampling detection."""
        params = SamplingParams(temperature=0.0)
        assert params.sampling_type == SamplingType.GREEDY
        
        params = SamplingParams(temperature=1.0)
        assert params.sampling_type == SamplingType.RANDOM
    
    def test_validation(self):
        """Test parameter validation."""
        # Test invalid n
        with pytest.raises(ValueError):
            SamplingParams(n=0)
        
        # Test invalid best_of
        with pytest.raises(ValueError):
            SamplingParams(n=2, best_of=1)
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            SamplingParams(temperature=-1.0)
        
        # Test invalid top_p
        with pytest.raises(ValueError):
            SamplingParams(top_p=1.5)
        
        # Test beam search not supported
        with pytest.raises(NotImplementedError):
            SamplingParams(use_beam_search=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

