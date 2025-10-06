"""Unit tests for SamplingParams (Milestone 0)."""

import pytest

from folovllm.sampling_params import SamplingParams, SamplingType


class TestSamplingParams:
    """Test SamplingParams class."""
    
    def test_default_values(self):
        """Test default sampling params."""
        params = SamplingParams()
        assert params.n == 1
        assert params.best_of == 1
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.max_tokens == 16
    
    def test_custom_values(self):
        """Test custom sampling params."""
        params = SamplingParams(
            n=2,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            max_tokens=100,
        )
        assert params.n == 2
        assert params.temperature == 0.8
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 100
    
    def test_best_of_defaults_to_n(self):
        """Test that best_of defaults to n."""
        params = SamplingParams(n=3)
        assert params.best_of == 3
        
        params = SamplingParams(n=2, best_of=5)
        assert params.best_of == 5
    
    def test_validation_n(self):
        """Test n validation."""
        with pytest.raises(ValueError, match="n must be at least 1"):
            SamplingParams(n=0)
    
    def test_validation_best_of(self):
        """Test best_of validation."""
        with pytest.raises(ValueError, match="best_of .* must be >= n"):
            SamplingParams(n=3, best_of=2)
    
    def test_validation_temperature(self):
        """Test temperature validation."""
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            SamplingParams(temperature=-0.1)
    
    def test_validation_top_p(self):
        """Test top_p validation."""
        with pytest.raises(ValueError, match="top_p must be in"):
            SamplingParams(top_p=0.0)
        
        with pytest.raises(ValueError, match="top_p must be in"):
            SamplingParams(top_p=1.5)
    
    def test_validation_top_k(self):
        """Test top_k validation."""
        with pytest.raises(ValueError, match="top_k must be -1"):
            SamplingParams(top_k=0)
        
        # Valid values
        params = SamplingParams(top_k=-1)  # Disabled
        assert params.top_k == -1
        
        params = SamplingParams(top_k=50)
        assert params.top_k == 50
    
    def test_validation_max_tokens(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            SamplingParams(max_tokens=0)
    
    def test_sampling_type_greedy(self):
        """Test greedy sampling type detection."""
        params = SamplingParams(temperature=0.0)
        assert params.sampling_type == SamplingType.GREEDY
    
    def test_sampling_type_random(self):
        """Test random sampling type detection."""
        params = SamplingParams(temperature=0.8)
        assert params.sampling_type == SamplingType.RANDOM
    
    def test_stop_conditions(self):
        """Test stop conditions."""
        params = SamplingParams(
            stop=["</s>", "\n\n"],
            stop_token_ids=[2, 3],
        )
        assert params.stop == ["</s>", "\n\n"]
        assert params.stop_token_ids == [2, 3]
    
    def test_beam_search_not_implemented(self):
        """Test that beam search raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Beam search is not supported"):
            SamplingParams(use_beam_search=True)

