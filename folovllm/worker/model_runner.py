"""Model runner for executing model forward passes."""

import torch
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel

from folovllm.config import ModelConfig
from folovllm.worker.input_batch import InputBatch


class ModelRunner:
    """Runs the model forward pass with batched inputs.
    
    Handles:
    - Input preparation (token IDs, positions)
    - KV cache management
    - Model execution
    - Logits extraction
    
    For M1, we support single-request execution.
    M2 will add batching support.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        model_config: ModelConfig,
        device: torch.device,
    ):
        """Initialize model runner.
        
        Args:
            model: The loaded model
            model_config: Model configuration
            device: Device to run on
        """
        self.model = model
        self.model_config = model_config
        self.device = device
        
        # KV caches for each layer (will be initialized on first forward)
        # M1: Single cache for single request
        self.kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.past_key_values = None  # For HuggingFace models
        
        # M2: Per-request caches for batching
        # req_id -> past_key_values
        self.request_caches: Dict[str, any] = {}
        
        # Get number of layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.num_layers = len(self.model.model.layers)
        else:
            self.num_layers = 0
    
    def initialize_kv_caches(self, batch_size: int = 1):
        """Initialize KV caches for all layers.
        
        For M1: simple contiguous memory allocation.
        For M3: will be replaced with paged attention.
        
        Args:
            batch_size: Batch size
        """
        # Get model dtype (check different attributes for compatibility)
        if hasattr(self.model, 'dtype'):
            model_dtype = self.model.dtype
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'torch_dtype'):
            model_dtype = self.model.config.torch_dtype
        else:
            # Infer from first parameter
            model_dtype = next(self.model.parameters()).dtype
        
        # Initialize empty caches for each layer
        self.kv_caches = []
        for _ in range(self.num_layers):
            # Start with empty tensors
            key_cache = torch.empty(0, device=self.device, dtype=model_dtype)
            value_cache = torch.empty(0, device=self.device, dtype=model_dtype)
            self.kv_caches.append((key_cache, value_cache))
    
    def clear_kv_caches(self):
        """Clear KV caches."""
        self.kv_caches = None
        self.past_key_values = None
        self.request_caches.clear()
    
    def prepare_inputs(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare model inputs.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            start_pos: Starting position (for decode phase)
            
        Returns:
            Tuple of (input_ids, positions)
        """
        batch_size, seq_len = token_ids.shape
        
        # Create position indices
        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=self.device,
            dtype=torch.long,
        )
        # Expand for batch
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        return token_ids, positions
    
    @torch.no_grad()
    def execute_model(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Execute model forward pass.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            start_pos: Starting position in sequence
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Prepare inputs
        input_ids, positions = self.prepare_inputs(token_ids, start_pos)
        
        # Initialize caches if needed (for our custom models)
        if self.kv_caches is None and self.num_layers > 0:
            # Only initialize for our custom models that need it
            try:
                self.initialize_kv_caches(batch_size=input_ids.shape[0])
            except:
                pass  # HF models manage their own cache
        
        # Check if this is a HuggingFace model or our custom model
        if hasattr(self.model, 'forward') and 'position_ids' in str(self.model.forward.__code__.co_varnames):
            # HuggingFace model - use past_key_values for KV cache
            outputs = self.model(
                input_ids=input_ids,
                position_ids=positions,
                past_key_values=self.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits
            # Update cached past_key_values for next iteration
            self.past_key_values = outputs.past_key_values
        elif hasattr(self.model, '__call__'):
            # Try to call with positions parameter (our custom model)
            try:
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=self.kv_caches,
                )
                
                # Update caches from attention layers
                if self.kv_caches is not None:
                    for layer_idx, layer in enumerate(self.model.model.layers):
                        if hasattr(layer.self_attn, 'attn') and hasattr(layer.self_attn.attn, 'kv_cache'):
                            self.kv_caches[layer_idx] = layer.self_attn.attn.kv_cache
                
                # Compute logits
                logits = self.model.compute_logits(hidden_states)
            except TypeError:
                # Fallback: HF model without position_ids but with past_key_values
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits
                self.past_key_values = outputs.past_key_values
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")
        
        return logits
    
    def get_next_token_logits(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Get logits for next token prediction.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            start_pos: Starting position
            
        Returns:
            Logits for next token [batch_size, vocab_size]
        """
        # Execute model
        logits = self.execute_model(token_ids, start_pos)
        
        # Return logits for last position
        return logits[:, -1, :]
    
    @torch.no_grad()
    def execute_model_batch(
        self,
        input_batch: InputBatch,
    ) -> Dict[str, torch.Tensor]:
        """Execute model for a batch of requests.
        
        This is the M2 batching method that handles multiple requests
        simultaneously, including mixed prefill and decode phases.
        
        Args:
            input_batch: Batch of inputs with variable-length sequences
            
        Returns:
            Dict mapping req_id -> logits for next token [vocab_size]
        """
        if input_batch.batch_size == 0:
            return {}
        
        # Convert to padded tensors
        token_ids, attention_mask, positions = input_batch.to_tensors(
            device=self.device,
            pad_token_id=0,  # Padding token
        )
        
        # For M2, we process each request separately to manage KV caches
        # M3+ will use PagedAttention for efficient batched execution
        results: Dict[str, torch.Tensor] = {}
        
        for i, req_id in enumerate(input_batch.req_ids):
            # Extract this request's inputs
            req_token_ids = token_ids[i:i+1]  # [1, seq_len]
            req_positions = positions[i:i+1]  # [1, seq_len]
            req_attention_mask = attention_mask[i:i+1]  # [1, seq_len]
            
            # Remove padding
            seq_len = req_attention_mask[0].sum().item()
            req_token_ids = req_token_ids[:, :seq_len]
            req_positions = req_positions[:, :seq_len]
            
            # Get or initialize cache for this request
            if req_id not in self.request_caches:
                self.request_caches[req_id] = None
            
            past_key_values = self.request_caches[req_id]
            
            # Execute model for this request
            if hasattr(self.model, 'forward') and 'position_ids' in str(self.model.forward.__code__.co_varnames):
                # HuggingFace model
                outputs = self.model(
                    input_ids=req_token_ids,
                    position_ids=req_positions,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits
                # Update cache for this request
                self.request_caches[req_id] = outputs.past_key_values
            else:
                # Custom model or fallback
                try:
                    outputs = self.model(
                        input_ids=req_token_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    logits = outputs.logits
                    self.request_caches[req_id] = outputs.past_key_values
                except:
                    # Fallback to no cache
                    outputs = self.model(
                        input_ids=req_token_ids,
                        return_dict=True,
                    )
                    logits = outputs.logits
            
            # Extract logits for the last token
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            results[req_id] = next_token_logits
        
        return results
    
    def free_request_cache(self, req_id: str):
        """Free KV cache for a finished request.
        
        Args:
            req_id: Request ID to free cache for
        """
        if req_id in self.request_caches:
            del self.request_caches[req_id]

