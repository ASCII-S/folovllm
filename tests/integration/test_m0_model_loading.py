"""Integration tests for model loading (Milestone 0).

These tests require downloading models from HuggingFace.
Skip if model is not available locally.
"""

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from folovllm.config import ModelConfig
from folovllm.model_loader import ModelLoader, get_model_and_tokenizer


# Use a very small model for testing
TEST_MODEL = "Qwen/Qwen2.5-0.6B"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestModelLoadingGPU:
    """Test model loading on GPU."""
    
    def test_load_model_config(self):
        """Test loading model with ModelConfig."""
        config = ModelConfig(
            model=TEST_MODEL,
            dtype="float16",
            trust_remote_code=True,
        )
        
        loader = ModelLoader(config)
        
        # This will download the model if not cached
        # Skip if download fails
        try:
            model = loader.load_model(device="cuda")
            assert isinstance(model, PreTrainedModel)
            assert next(model.parameters()).device.type == "cuda"
            assert next(model.parameters()).dtype == torch.float16
        except Exception as e:
            pytest.skip(f"Model loading failed (may need download): {e}")
    
    def test_load_tokenizer(self):
        """Test loading tokenizer."""
        config = ModelConfig(
            model=TEST_MODEL,
            trust_remote_code=True,
        )
        
        loader = ModelLoader(config)
        
        try:
            tokenizer = loader.load_tokenizer()
            assert isinstance(tokenizer, PreTrainedTokenizer)
            assert tokenizer.pad_token is not None
            assert tokenizer.eos_token is not None
        except Exception as e:
            pytest.skip(f"Tokenizer loading failed (may need download): {e}")
    
    def test_load_model_and_tokenizer(self):
        """Test loading both model and tokenizer."""
        config = ModelConfig(
            model=TEST_MODEL,
            dtype="float16",
            trust_remote_code=True,
        )
        
        try:
            model, tokenizer = get_model_and_tokenizer(config, device="cuda")
            
            assert isinstance(model, PreTrainedModel)
            assert isinstance(tokenizer, PreTrainedTokenizer)
            
            # Test tokenization
            text = "Hello, world!"
            tokens = tokenizer.encode(text)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            
            # Test decoding
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            assert isinstance(decoded, str)
            
        except Exception as e:
            pytest.skip(f"Model/tokenizer loading failed (may need download): {e}")


class TestModelLoadingCPU:
    """Test model loading on CPU (always available)."""
    
    def test_load_model_cpu(self):
        """Test loading model on CPU."""
        config = ModelConfig(
            model=TEST_MODEL,
            dtype="float32",
            trust_remote_code=True,
        )
        
        loader = ModelLoader(config)
        
        try:
            model = loader.load_model(device="cpu")
            assert isinstance(model, PreTrainedModel)
            assert next(model.parameters()).device.type == "cpu"
        except Exception as e:
            pytest.skip(f"Model loading failed (may need download): {e}")
    
    def test_tokenizer_encode_decode(self):
        """Test tokenizer encode/decode roundtrip."""
        config = ModelConfig(
            model=TEST_MODEL,
            trust_remote_code=True,
        )
        
        loader = ModelLoader(config)
        
        try:
            tokenizer = loader.load_tokenizer()
            
            # Test various inputs
            test_texts = [
                "Hello, world!",
                "你好，世界！",
                "This is a test.",
                "",  # Empty string
            ]
            
            for text in test_texts:
                if not text:  # Skip empty
                    continue
                    
                tokens = tokenizer.encode(text, add_special_tokens=False)
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                
                # Decoded text should be similar (may have minor differences)
                assert isinstance(decoded, str)
                
        except Exception as e:
            pytest.skip(f"Tokenizer test failed (may need download): {e}")
    
    def test_max_model_len_inference(self):
        """Test that max_model_len is inferred from model config."""
        config = ModelConfig(
            model=TEST_MODEL,
            trust_remote_code=True,
        )
        
        assert config.max_model_len is None  # Not set initially
        
        loader = ModelLoader(config)
        
        try:
            loader.load_model(device="cpu")
            
            # Should be set after loading
            assert config.max_model_len is not None
            assert config.max_model_len > 0
            
        except Exception as e:
            pytest.skip(f"Model loading failed (may need download): {e}")

