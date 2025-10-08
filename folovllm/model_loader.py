"""Model loader for HuggingFace models (aligned with vLLM)."""

import os
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from folovllm.config import ModelConfig


class ModelLoader:
    """Loads HuggingFace models and tokenizers.
    
    Simplified version of vLLM's model loading logic for Milestone 0.
    """
    
    def __init__(self, model_config: ModelConfig):
        """Initialize the model loader.
        
        Args:
            model_config: Model configuration.
        """
        self.model_config = model_config
    
    def load_model(self, device: str = "cuda") -> PreTrainedModel:
        """Load the model from HuggingFace.
        
        For M1, we use HuggingFace models directly for compatibility.
        
        Args:
            device: Device to load the model on ('cuda' or 'cpu').
            
        Returns:
            Loaded model (wrapped with compute_logits method).
        """
        model_path = self.model_config.model
        
        # Determine dtype
        dtype = self._get_dtype()
        
        # Load model config first
        hf_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        
        # Update max_model_len from config if not set
        if self.model_config.max_model_len is None:
            # Try to get max position embeddings from config
            if hasattr(hf_config, "max_position_embeddings"):
                self.model_config.max_model_len = hf_config.max_position_embeddings
            else:
                # Default fallback
                self.model_config.max_model_len = 2048
        
        print(f"Loading model from {model_path}...")
        print(f"  - dtype: {dtype}")
        print(f"  - max_model_len: {self.model_config.max_model_len}")
        print(f"  - device: {device}")
        
        # Load model using HuggingFace AutoModel
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=hf_config,
            torch_dtype=dtype,
            trust_remote_code=self.model_config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        
        # Move to device
        model = model.to(device=device)
        model.eval()
        
        # Wrap model to add compute_logits method if it doesn't exist
        model = self._wrap_model_for_folovllm(model)
        
        print(f"Model loaded successfully!")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Number of parameters: {self._count_parameters(model):,}")
        
        return model
    
    def _wrap_model_for_folovllm(self, model: PreTrainedModel) -> PreTrainedModel:
        """Wrap HuggingFace model to add FoloVLLM-specific methods.
        
        Args:
            model: HuggingFace model
            
        Returns:
            Wrapped model with compute_logits method
        """
        # Add compute_logits method if it doesn't exist
        if not hasattr(model, 'compute_logits'):
            def compute_logits(hidden_states):
                """Compute logits from hidden states."""
                return model.lm_head(hidden_states)
            
            # Bind the method to the model
            import types
            model.compute_logits = types.MethodType(compute_logits, model)
        
        return model
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer from HuggingFace.
        
        Returns:
            Loaded tokenizer.
        """
        tokenizer_path = self.model_config.tokenizer or self.model_config.model
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        
        # Determine tokenizer mode
        use_fast = self.model_config.tokenizer_mode == "auto"
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=use_fast,
            trust_remote_code=self.model_config.trust_remote_code,
            padding_side="left",  # For batch generation
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Fallback: add a new pad token
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        print(f"Tokenizer loaded successfully!")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        print(f"  - Special tokens: pad={tokenizer.pad_token}, eos={tokenizer.eos_token}")
        
        return tokenizer
    
    def load_model_and_tokenizer(
        self, device: str = "cuda"
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load both model and tokenizer.
        
        Args:
            device: Device to load the model on.
            
        Returns:
            Tuple of (model, tokenizer).
        """
        model = self.load_model(device)
        tokenizer = self.load_tokenizer()
        return model, tokenizer
    
    def _get_dtype(self) -> torch.dtype:
        """Get the torch dtype based on config."""
        if self.model_config.torch_dtype is not None:
            return self.model_config.torch_dtype
        
        # Default: use float16 for GPU, float32 for CPU
        if torch.cuda.is_available():
            return torch.float16
        else:
            return torch.float32
    
    @staticmethod
    def _count_parameters(model: PreTrainedModel) -> int:
        """Count the number of parameters in the model."""
        return sum(p.numel() for p in model.parameters())


def get_model_and_tokenizer(
    model_config: ModelConfig, device: str = "cuda"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Convenience function to load model and tokenizer.
    
    Args:
        model_config: Model configuration.
        device: Device to load the model on.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    loader = ModelLoader(model_config)
    return loader.load_model_and_tokenizer(device)

