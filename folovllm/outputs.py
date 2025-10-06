"""Output definitions (aligned with vLLM v1)."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CompletionOutput:
    """Output of a single completion sequence."""
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[List[float]] = None  # Reserved for future milestones
    finish_reason: Optional[str] = None  # 'stop', 'length', or None
    
    def finished(self) -> bool:
        """Check if this completion is finished."""
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    """Output of a request.
    
    Aligned with vLLM v1 output structure.
    
    Args:
        request_id: Unique request ID.
        prompt: The input prompt text.
        prompt_token_ids: Tokenized prompt.
        outputs: List of completion outputs (length = n).
        finished: Whether all sequences are finished.
    """
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool
    
    # Metrics (reserved for future milestones)
    metrics: Optional[dict] = None
    
    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt[:50]}..., "
            f"outputs={len(self.outputs)}, finished={self.finished})"
        )

