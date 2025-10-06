"""Sampling parameters for text generation (aligned with vLLM)."""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional


class SamplingType(IntEnum):
    """Type of sampling to use."""
    GREEDY = 0  # Deterministic, always pick argmax
    RANDOM = 1  # Random sampling with temperature/top-k/top-p


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.
    
    Aligned with vLLM's SamplingParams, simplified for Milestone 0-1.
    
    Args:
        n: Number of output sequences to generate.
        best_of: Number of sequences to generate and return the best one.
        temperature: Randomness of sampling. 0.0 = greedy, higher = more random.
        top_p: Nucleus sampling - only sample from top tokens with cumulative probability >= top_p.
        top_k: Only sample from top k tokens.
        min_p: Minimum probability threshold.
        use_beam_search: Whether to use beam search (not supported in M0-M1).
        length_penalty: Exponential penalty to the length (for beam search).
        early_stopping: Stop beam search when at least `best_of` sequences are finished.
        stop: List of stop strings. Generation stops when any is produced.
        stop_token_ids: List of stop token IDs.
        include_stop_str_in_output: Whether to include stop string in output.
        ignore_eos: Whether to ignore EOS token.
        max_tokens: Maximum number of tokens to generate.
        min_tokens: Minimum number of tokens to generate.
        logprobs: Number of log probabilities to return (not implemented in M0-M1).
        prompt_logprobs: Number of prompt log probabilities to return.
        skip_special_tokens: Whether to skip special tokens in output.
        spaces_between_special_tokens: Whether to add spaces between special tokens.
        seed: Random seed for reproducibility.
    """
    
    # Number of outputs
    n: int = 1
    best_of: Optional[int] = None
    
    # Sampling parameters
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    min_p: float = 0.0
    
    # Beam search (M0-M1: not supported, reserved for future)
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False
    
    # Stop conditions
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    
    # Length constraints
    max_tokens: Optional[int] = 16  # Default max output length
    min_tokens: int = 0
    
    # Logprobs (reserved for future milestones)
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    
    # Output formatting
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    
    # Randomness
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate and normalize parameters."""
        # Set best_of to n if not specified
        if self.best_of is None:
            self.best_of = self.n
        
        # Validation
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")
        
        if self.best_of < self.n:
            raise ValueError(
                f"best_of ({self.best_of}) must be >= n ({self.n})"
            )
        
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}"
            )
        
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disabled) or >= 1, got {self.top_k}"
            )
        
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}"
            )
        
        if self.use_beam_search:
            # For M0-M1, beam search is not supported
            raise NotImplementedError(
                "Beam search is not supported in M0-M1. "
                "It will be implemented in future milestones."
            )
        
        # Normalize stop conditions
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []
    
    @property
    def sampling_type(self) -> SamplingType:
        """Get the sampling type based on parameters."""
        if self.temperature == 0.0:
            return SamplingType.GREEDY
        else:
            return SamplingType.RANDOM
    
    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, temperature={self.temperature}, "
            f"top_p={self.top_p}, top_k={self.top_k}, max_tokens={self.max_tokens})"
        )

