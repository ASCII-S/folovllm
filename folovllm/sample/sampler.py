"""Sampler for token generation."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from folovllm.sampling_params import SamplingParams, SamplingType
from folovllm.sample.ops import (
    apply_top_k_filtering,
    apply_top_p_filtering,
    apply_min_p_filtering,
)


class Sampler:
    """Sampler for generating next tokens.
    
    Supports various sampling strategies:
    - Greedy sampling (temperature=0)
    - Random sampling with temperature
    - Top-k filtering
    - Top-p (nucleus) filtering
    - Min-p filtering
    
    Also handles stop conditions and probability tracking.
    """
    
    def __init__(self):
        """Initialize the sampler."""
        self._generator = None
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample next tokens from logits.
        
        Args:
            logits: Logits tensor [batch_size, vocab_size]
            sampling_params: Sampling parameters
            
        Returns:
            Tuple of (sampled_tokens, log_probs)
            - sampled_tokens: [batch_size]
            - log_probs: Optional [batch_size] (if logprobs requested)
        """
        # Set random seed if specified
        if sampling_params.seed is not None:
            if self._generator is None:
                self._generator = torch.Generator(device=logits.device)
            self._generator.manual_seed(sampling_params.seed)
        
        # Apply temperature
        if sampling_params.temperature > 0:
            logits = logits / sampling_params.temperature
        
        # Apply filtering
        if sampling_params.min_p > 0:
            logits = apply_min_p_filtering(logits, sampling_params.min_p)
        
        if sampling_params.top_k > 0:
            logits = apply_top_k_filtering(logits, sampling_params.top_k)
        
        if sampling_params.top_p < 1.0:
            logits = apply_top_p_filtering(logits, sampling_params.top_p)
        
        # Sample tokens
        if sampling_params.sampling_type == SamplingType.GREEDY:
            # Greedy: always pick argmax
            sampled_tokens = torch.argmax(logits, dim=-1)
        else:
            # Random sampling
            probs = F.softmax(logits, dim=-1)
            sampled_tokens = torch.multinomial(
                probs,
                num_samples=1,
                generator=self._generator,
            ).squeeze(-1)
        
        # Compute log probabilities if requested
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            # Get log prob of sampled token
            log_probs = log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        
        return sampled_tokens, log_probs
    
    def check_stop_conditions(
        self,
        token_ids: List[int],
        token_text: str,
        sampling_params: SamplingParams,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check if generation should stop.
        
        Args:
            token_ids: Generated token IDs so far
            token_text: Decoded text so far
            sampling_params: Sampling parameters
            eos_token_id: EOS token ID
            
        Returns:
            Tuple of (should_stop, finish_reason)
            - should_stop: True if generation should stop
            - finish_reason: Reason for stopping ('stop', 'length', or None)
        """
        # Check max length
        if sampling_params.max_tokens is not None:
            if len(token_ids) >= sampling_params.max_tokens:
                return True, "length"
        
        # Check EOS token (unless ignore_eos is True)
        if not sampling_params.ignore_eos and eos_token_id is not None:
            if len(token_ids) > 0 and token_ids[-1] == eos_token_id:
                return True, "stop"
        
        # Check stop token IDs
        if sampling_params.stop_token_ids:
            if len(token_ids) > 0 and token_ids[-1] in sampling_params.stop_token_ids:
                return True, "stop"
        
        # Check stop strings
        if sampling_params.stop:
            for stop_str in sampling_params.stop:
                if stop_str in token_text:
                    return True, "stop"
        
        return False, None
    
    def apply_penalties(
        self,
        logits: torch.Tensor,
        token_ids: List[int],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Apply penalties to logits (reserved for future milestones).
        
        Args:
            logits: Logits tensor [batch_size, vocab_size]
            token_ids: Previously generated tokens
            frequency_penalty: Frequency penalty coefficient
            presence_penalty: Presence penalty coefficient
            repetition_penalty: Repetition penalty coefficient
            
        Returns:
            Penalized logits
        """
        # M1: Not implemented yet
        # Will be added in future milestones
        return logits


def sample_tokens(
    logits: torch.Tensor,
    sampling_params: SamplingParams,
    sampler: Optional[Sampler] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Convenience function for sampling.
    
    Args:
        logits: Logits tensor [batch_size, vocab_size]
        sampling_params: Sampling parameters
        sampler: Optional Sampler instance (creates new one if None)
        
    Returns:
        Tuple of (sampled_tokens, log_probs)
    """
    if sampler is None:
        sampler = Sampler()
    return sampler.sample(logits, sampling_params)

