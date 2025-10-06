"""Unit tests for configuration classes (Milestone 0)."""

import pytest
import torch

from folovllm.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_basic_creation(self):
        """Test basic model config creation."""
        config = ModelConfig(model="Qwen/Qwen2.5-0.6B")
        assert config.model == "Qwen/Qwen2.5-0.6B"
        assert config.tokenizer == "Qwen/Qwen2.5-0.6B"  # Should default to model
        assert config.tokenizer_mode == "auto"
        assert config.trust_remote_code is False
    
    def test_tokenizer_default(self):
        """Test that tokenizer defaults to model path."""
        config = ModelConfig(model="test_model")
        assert config.tokenizer == "test_model"
    
    def test_tokenizer_custom(self):
        """Test custom tokenizer path."""
        config = ModelConfig(model="model_path", tokenizer="tokenizer_path")
        assert config.tokenizer == "tokenizer_path"
    
    def test_dtype_conversion(self):
        """Test dtype string to torch.dtype conversion."""
        # Test auto
        config = ModelConfig(model="test", dtype="auto")
        assert config.torch_dtype is None
        
        # Test float16
        config = ModelConfig(model="test", dtype="float16")
        assert config.torch_dtype == torch.float16
        
        # Test half (same as float16)
        config = ModelConfig(model="test", dtype="half")
        assert config.torch_dtype == torch.float16
        
        # Test bfloat16
        config = ModelConfig(model="test", dtype="bfloat16")
        assert config.torch_dtype == torch.bfloat16
        
        # Test float32
        config = ModelConfig(model="test", dtype="float32")
        assert config.torch_dtype == torch.float32


class TestCacheConfig:
    """Test CacheConfig class."""
    
    def test_default_values(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.block_size == 16
        assert config.gpu_memory_utilization == 0.9
        assert config.swap_space == 4.0
        assert config.enable_prefix_caching is False
    
    def test_custom_values(self):
        """Test custom cache config values."""
        config = CacheConfig(
            block_size=32,
            gpu_memory_utilization=0.8,
            swap_space=8.0,
            enable_prefix_caching=True,
        )
        assert config.block_size == 32
        assert config.gpu_memory_utilization == 0.8
        assert config.swap_space == 8.0
        assert config.enable_prefix_caching is True
    
    def test_validation_block_size(self):
        """Test block_size validation."""
        with pytest.raises(ValueError, match="block_size must be positive"):
            CacheConfig(block_size=0)
        
        with pytest.raises(ValueError, match="block_size must be positive"):
            CacheConfig(block_size=-1)
    
    def test_validation_gpu_memory_utilization(self):
        """Test gpu_memory_utilization validation."""
        with pytest.raises(ValueError, match="gpu_memory_utilization must be in"):
            CacheConfig(gpu_memory_utilization=0.0)
        
        with pytest.raises(ValueError, match="gpu_memory_utilization must be in"):
            CacheConfig(gpu_memory_utilization=1.5)


class TestSchedulerConfig:
    """Test SchedulerConfig class."""
    
    def test_default_values(self):
        """Test default scheduler config values."""
        config = SchedulerConfig()
        assert config.max_num_batched_tokens is None
        assert config.max_num_seqs == 256
        assert config.max_model_len is None
        assert config.enable_chunked_prefill is False


class TestEngineConfig:
    """Test EngineConfig class."""
    
    def test_basic_creation(self):
        """Test basic engine config creation."""
        model_config = ModelConfig(model="test_model")
        config = EngineConfig(model_config=model_config)
        
        assert config.model_config == model_config
        assert isinstance(config.cache_config, CacheConfig)
        assert isinstance(config.scheduler_config, SchedulerConfig)
    
    def test_custom_sub_configs(self):
        """Test engine config with custom sub-configs."""
        model_config = ModelConfig(model="test_model", max_model_len=4096)
        cache_config = CacheConfig(block_size=32)
        scheduler_config = SchedulerConfig(max_num_seqs=128)
        
        config = EngineConfig(
            model_config=model_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
        )
        
        assert config.cache_config.block_size == 32
        assert config.scheduler_config.max_num_seqs == 128
    
    def test_max_model_len_sync(self):
        """Test that max_model_len is synced from model_config to scheduler_config."""
        model_config = ModelConfig(model="test_model", max_model_len=2048)
        config = EngineConfig(model_config=model_config)
        
        assert config.scheduler_config.max_model_len == 2048

