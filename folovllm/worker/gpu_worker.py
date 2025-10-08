"""GPU worker for model execution."""

import torch
from typing import Dict, Optional

from folovllm.config import ModelConfig
from folovllm.model_loader import ModelLoader
from folovllm.worker.model_runner import ModelRunner
from folovllm.worker.input_batch import InputBatch


class GPUWorker:
    """Worker that executes models on GPU.
    
    Handles:
    - Model loading and initialization
    - Device management
    - Model execution via ModelRunner
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        device: Optional[str] = None,
    ):
        """Initialize GPU worker.
        
        Args:
            model_config: Model configuration
            device: Device string ('cuda', 'cuda:0', etc.). If None, auto-detect.
        """
        self.model_config = model_config
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Load model
        print(f"Initializing GPU worker on {self.device}...")
        loader = ModelLoader(model_config)
        self.model = loader.load_model(device=str(self.device))
        
        # Create model runner
        self.model_runner = ModelRunner(
            model=self.model,
            model_config=model_config,
            device=self.device,
        )
        
        print(f"GPU worker initialized successfully!")
    
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
        # Move to device if needed
        if token_ids.device != self.device:
            token_ids = token_ids.to(self.device)
        
        return self.model_runner.execute_model(token_ids, start_pos)
    
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
        # Move to device if needed
        if token_ids.device != self.device:
            token_ids = token_ids.to(self.device)
        
        return self.model_runner.get_next_token_logits(token_ids, start_pos)
    
    def clear_kv_caches(self):
        """Clear KV caches."""
        self.model_runner.clear_kv_caches()
    
    def execute_model_batch(
        self,
        input_batch: InputBatch,
    ) -> Dict[str, torch.Tensor]:
        """Execute model for a batch of requests (M2).
        
        Args:
            input_batch: Batch of inputs
            
        Returns:
            Dict mapping req_id -> logits for next token
        """
        return self.model_runner.execute_model_batch(input_batch)
    
    def free_request_cache(self, req_id: str):
        """Free KV cache for a finished request.
        
        Args:
            req_id: Request ID to free cache for
        """
        self.model_runner.free_request_cache(req_id)

