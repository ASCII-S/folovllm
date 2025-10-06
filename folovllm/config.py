"""Configuration classes (aligned with vLLM's config system)."""

import torch
from dataclasses import dataclass, field
from typing import Literal, Optional


ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
TokenizerMode = Literal["auto", "slow"]


@dataclass
class ModelConfig:
    """Model configuration.
    
    Aligned with vLLM's ModelConfig, simplified for Milestone 0.
    
    Args:
        model: Name or path of the HuggingFace model.
        tokenizer: Name or path of the tokenizer. If None, uses model path.
        tokenizer_mode: Tokenizer mode ('auto' or 'slow').
        trust_remote_code: Whether to trust remote code from HuggingFace.
        dtype: Data type for model weights and activations.
        max_model_len: Maximum sequence length supported by the model.
        seed: Random seed for reproducibility.
    """
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: TokenizerMode = "auto"
    trust_remote_code: bool = False
    dtype: ModelDType = "auto"
    max_model_len: Optional[int] = None
    seed: int = 0
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # If tokenizer is not specified, use model path
        if self.tokenizer is None:
            self.tokenizer = self.model
            
        # Convert string dtype to torch dtype
        if isinstance(self.dtype, str):
            dtype_map = {
                "auto": None,  # Will be inferred from model config
                "half": torch.float16,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float": torch.float32,
                "float32": torch.float32,
            }
            self.torch_dtype = dtype_map.get(self.dtype)
        else:
            self.torch_dtype = self.dtype
    

@dataclass  
class CacheConfig:
    """KV Cache configuration.
    
    Aligned with vLLM's CacheConfig, simplified for Milestone 0.
    
    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory for KV cache.
        swap_space: CPU swap space per GPU (in GiB).
        enable_prefix_caching: Whether to enable prefix caching (M6).
    """
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4.0  # GiB
    enable_prefix_caching: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
    

@dataclass
class SchedulerConfig:
    """Scheduler configuration.
    
    Aligned with vLLM's SchedulerConfig, simplified for Milestone 0.
    Reserved for M2 (Continuous Batching).
    
    Args:
        max_num_batched_tokens: Maximum tokens per iteration.
        max_num_seqs: Maximum sequences per iteration.
        max_model_len: Maximum sequence length.
        enable_chunked_prefill: Whether to enable chunked prefill (M5).
    """
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    enable_chunked_prefill: bool = False
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Will be set from model config if not specified
        pass
    

@dataclass
class EngineConfig:
    """Unified engine configuration.
    
    Combines all configuration classes for the engine.
    """
    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    def __post_init__(self):
        """Synchronize configurations."""
        # Sync max_model_len across configs
        if self.scheduler_config.max_model_len is None:
            self.scheduler_config.max_model_len = self.model_config.max_model_len

