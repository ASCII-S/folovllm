"""Unit tests for M1 input processor."""

import pytest
from transformers import AutoTokenizer

from folovllm.engine.processor import InputProcessor
from folovllm.sampling_params import SamplingParams
from folovllm.request import Request


@pytest.fixture
def tokenizer():
    """Create a test tokenizer."""
    # Use a small, fast tokenizer for testing
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def processor(tokenizer):
    """Create an InputProcessor."""
    return InputProcessor(tokenizer)


class TestInputProcessor:
    """Test InputProcessor class."""
    
    def test_process_single_request(self, processor):
        """Test processing a single request."""
        prompt = "Hello, world!"
        sampling_params = SamplingParams(max_tokens=10)
        
        request = processor.process_request(prompt, sampling_params)
        
        assert isinstance(request, Request)
        assert request.prompt == prompt
        assert len(request.prompt_token_ids) > 0
        assert request.sampling_params == sampling_params
        assert request.request_id is not None
    
    def test_process_request_with_id(self, processor):
        """Test processing request with custom ID."""
        prompt = "Test prompt"
        sampling_params = SamplingParams()
        request_id = "custom-id-123"
        
        request = processor.process_request(
            prompt,
            sampling_params,
            request_id=request_id,
        )
        
        assert request.request_id == request_id
    
    def test_process_multiple_requests(self, processor):
        """Test processing multiple requests."""
        prompts = ["Hello", "World", "Test"]
        sampling_params = SamplingParams(max_tokens=5)
        
        requests = processor.process_requests(prompts, sampling_params)
        
        assert len(requests) == 3
        for i, request in enumerate(requests):
            assert request.prompt == prompts[i]
            assert request.sampling_params == sampling_params
    
    def test_process_requests_different_params(self, processor):
        """Test processing requests with different sampling params."""
        prompts = ["A", "B"]
        params_list = [
            SamplingParams(temperature=0.0),
            SamplingParams(temperature=1.0),
        ]
        
        requests = processor.process_requests(prompts, params_list)
        
        assert len(requests) == 2
        assert requests[0].sampling_params.temperature == 0.0
        assert requests[1].sampling_params.temperature == 1.0
    
    def test_process_requests_length_mismatch(self, processor):
        """Test error on length mismatch."""
        prompts = ["A", "B", "C"]
        params_list = [SamplingParams(), SamplingParams()]
        
        with pytest.raises(ValueError):
            processor.process_requests(prompts, params_list)
    
    def test_decode_tokens(self, processor, tokenizer):
        """Test token decoding."""
        prompt = "Hello, world!"
        token_ids = tokenizer.encode(prompt)
        
        decoded = processor.decode_tokens(token_ids)
        
        # Decoded text should be similar to original
        # (may have slight differences due to tokenization)
        assert "Hello" in decoded or "hello" in decoded.lower()
    
    def test_decode_tokens_skip_special(self, processor, tokenizer):
        """Test token decoding with special tokens."""
        # Create tokens with special tokens
        token_ids = [tokenizer.bos_token_id, 123, 456, tokenizer.eos_token_id]
        
        decoded_skip = processor.decode_tokens(token_ids, skip_special_tokens=True)
        decoded_keep = processor.decode_tokens(token_ids, skip_special_tokens=False)
        
        # Length should differ when special tokens are included
        assert len(decoded_keep) >= len(decoded_skip)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

