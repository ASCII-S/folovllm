"""GPU executor for managing GPU workers."""

import torch
from typing import Optional

from folovllm.config import ModelConfig
from folovllm.worker import GPUWorker


class GPUExecutor:
    """Executor for GPU-based inference.
    
    Manages GPU workers and provides a unified interface for execution.
    For M1: single GPU support.
    For M6+: multi-GPU support with tensor parallelism.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        device: Optional[str] = None,
    ):
        """Initialize GPU executor.
        
        Args:
            model_config: Model configuration
            device: Device string (e.g., 'cuda:0')
        """
        self.model_config = model_config
        self.device = device
        
        # Create single GPU worker
        self.worker = GPUWorker(model_config, device)
    
    def execute_model(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Execute model forward pass.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            start_pos: Starting position
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        return self.worker.execute_model(token_ids, start_pos)
    
    def get_next_token_logits(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Get logits for next token.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            start_pos: Starting position
            
        Returns:
            Logits [batch_size, vocab_size]
        """
        return self.worker.get_next_token_logits(token_ids, start_pos)
    
    def clear_kv_caches(self):
        """Clear KV caches in all workers."""
        self.worker.clear_kv_caches()

