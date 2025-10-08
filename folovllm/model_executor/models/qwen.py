"""Qwen3 model implementation for FoloVLLM.

This is a simplified version of Qwen3 for single-GPU inference.
Based on the reference implementation but without tensor parallelism.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import Qwen2Config  # Qwen3 uses Qwen2Config

from folovllm.model_executor.layers.attention import Attention
from folovllm.model_executor.models.utils import RMSNorm, SiLUAndMul


class Qwen3Attention(nn.Module):
    """Qwen3 attention layer with RoPE and GQA.
    
    Args:
        config: Qwen2Config from transformers
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', None) or self.hidden_size // self.num_heads
        
        # Use our general Attention layer
        self.attn = Attention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
            bias=getattr(config, 'attention_bias', True),
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            positions: Position indices [batch_size] or [batch_size, seq_len]
            hidden_states: Input [batch_size, seq_len, hidden_size]
            kv_cache: Optional KV cache
            
        Returns:
            Output [batch_size, seq_len, hidden_size]
        """
        return self.attn(positions, hidden_states, kv_cache)


class Qwen3MLP(nn.Module):
    """Qwen3 MLP with gated activation (SwiGLU).
    
    Args:
        config: Qwen2Config from transformers
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate and up projections (merged for efficiency)
        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
        )
        
        # Down projection
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )
        
        # Activation (SwiGLU: SiLU + elementwise mul)
        self.act_fn = SiLUAndMul()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            hidden_states: Input [batch_size, seq_len, hidden_size]
            
        Returns:
            Output [batch_size, seq_len, hidden_size]
        """
        # Gate and up projection
        gate_up = self.gate_up_proj(hidden_states)
        
        # Apply SwiGLU activation
        hidden_states = self.act_fn(gate_up)
        
        # Down projection
        hidden_states = self.down_proj(hidden_states)
        
        return hidden_states


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 transformer decoder layer.
    
    Args:
        config: Qwen2Config from transformers
        layer_idx: Layer index
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Attention
        self.self_attn = Qwen3Attention(config)
        
        # MLP
        self.mlp = Qwen3MLP(config)
        
        # Layer norms
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            positions: Position indices
            hidden_states: Input tensor
            residual: Residual from previous layer (or None for first layer)
            kv_cache: Optional KV cache for this layer
            
        Returns:
            Tuple of (hidden_states, residual)
        """
        # Attention block with pre-norm
        if residual is None:
            residual = hidden_states
            hidden_states, _ = self.input_layernorm(hidden_states, residual=None)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states, kv_cache)
        
        # MLP block with pre-norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 model (without LM head).
    
    Args:
        config: Qwen2Config from transformers
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len]
            kv_caches: Optional list of KV caches, one per layer
            
        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize residual
        residual = None
        
        # Pass through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            kv_cache = None
            if kv_caches is not None and layer_idx < len(kv_caches):
                kv_cache = kv_caches[layer_idx]
            
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_cache,
            )
        
        # Final norm
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 for causal language modeling.
    
    Args:
        config: Qwen2Config from transformers
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        
        # Base model
        self.model = Qwen3Model(config)
        
        # LM head
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass to get hidden states.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len]
            kv_caches: Optional list of KV caches
            
        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        return self.model(input_ids, positions, kv_caches)
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits from hidden states.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        print("Qwen3ForCausalLM.compute_logits")
        return self.lm_head(hidden_states)
    
    @torch.no_grad()
    def generate_token(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate logits for next token.
        
        Convenience method that combines forward and compute_logits.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            positions: Position indices [batch_size, seq_len]
            kv_caches: Optional list of KV caches
            
        Returns:
            Tuple of (logits, hidden_states)
            - logits: [batch_size, seq_len, vocab_size]
            - hidden_states: [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.forward(input_ids, positions, kv_caches)
        logits = self.compute_logits(hidden_states)
        return logits, hidden_states

