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
        self.kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.past_key_values = None  # For HuggingFace models
        
        # M2: Per-sequence KV caches for batched execution
        # Maps seq_id -> past_key_values for each sequence
        self.seq_kv_caches: Dict[str, any] = {}
        
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
        self.seq_kv_caches.clear()
    
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
    def execute_batch(
        self,
        input_batch: InputBatch,
    ) -> torch.Tensor:
        """Execute model on a batch of requests (M2).
        
        Supports both prefill and decode phases with proper KV cache management.
        
        Args:
            input_batch: InputBatch containing multiple requests.
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        if input_batch.is_prefill:
            return self._execute_batch_prefill(input_batch)
        else:
            return self._execute_batch_decode(input_batch)
    
    def _execute_batch_prefill(
        self,
        input_batch: InputBatch,
    ) -> torch.Tensor:
        """Execute prefill for a batch of requests.
        
        Prefill processes the entire prompt and initializes KV caches.
        
        Args:
            input_batch: InputBatch in prefill phase.
            
        Returns:
            Logits [batch_size, max_seq_len, vocab_size]
        """
        input_ids = input_batch.token_ids
        position_ids = input_batch.position_ids
        attention_mask = input_batch.attention_mask
        
        # Execute model (HuggingFace style)
        # For prefill, we don't use past_key_values (start fresh)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        # Store KV caches for each sequence
        for i, seq_id in enumerate(input_batch.seq_ids):
            # Extract KV cache for this sequence
            # For batch processing, we need to slice the batch dimension
            seq_past_kv = []
            for layer_kv in past_key_values:
                # layer_kv is (key_cache, value_cache)
                # Each has shape [batch_size, num_heads, seq_len, head_dim]
                seq_layer_kv = (
                    layer_kv[0][i:i+1],  # key for this sequence
                    layer_kv[1][i:i+1],  # value for this sequence
                )
                seq_past_kv.append(seq_layer_kv)
            
            self.seq_kv_caches[seq_id] = tuple(seq_past_kv)
        
        return logits
    
    def _execute_batch_decode(
        self,
        input_batch: InputBatch,
    ) -> torch.Tensor:
        """Execute decode for a batch of requests.
        
        Decode generates one token at a time using cached KV values.
        
        Args:
            input_batch: InputBatch in decode phase.
            
        Returns:
            Logits [batch_size, 1, vocab_size]
        """
        input_ids = input_batch.token_ids
        position_ids = input_batch.position_ids
        
        # Gather past_key_values for all sequences in the batch
        batch_past_kv = []
        num_layers = len(self.seq_kv_caches[input_batch.seq_ids[0]])
        
        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            
            for seq_id in input_batch.seq_ids:
                if seq_id not in self.seq_kv_caches:
                    raise ValueError(f"No KV cache found for sequence {seq_id}")
                
                seq_past_kv = self.seq_kv_caches[seq_id]
                key, value = seq_past_kv[layer_idx]
                layer_keys.append(key)
                layer_values.append(value)
            
            # Concatenate along batch dimension
            batch_key = torch.cat(layer_keys, dim=0)
            batch_value = torch.cat(layer_values, dim=0)
            batch_past_kv.append((batch_key, batch_value))
        
        batch_past_kv = tuple(batch_past_kv)
        
        # Execute model with past KV caches
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=batch_past_kv,
            use_cache=True,
            return_dict=True,
        )
        
        logits = outputs.logits
        new_past_kv = outputs.past_key_values
        
        # Update KV caches for each sequence
        for i, seq_id in enumerate(input_batch.seq_ids):
            seq_past_kv = []
            for layer_kv in new_past_kv:
                seq_layer_kv = (
                    layer_kv[0][i:i+1],
                    layer_kv[1][i:i+1],
                )
                seq_past_kv.append(seq_layer_kv)
            
            self.seq_kv_caches[seq_id] = tuple(seq_past_kv)
        
        return logits
    
    def remove_seq_kv_cache(self, seq_id: str) -> None:
        """Remove KV cache for a finished sequence.
        
        Args:
            seq_id: Sequence ID to remove.
        """
        self.seq_kv_caches.pop(seq_id, None)

