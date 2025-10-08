"""Scheduler output dataclasses (aligned with vLLM v1).

These dataclasses define the information passed from the scheduler
to the model runner and back.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from folovllm.sampling_params import SamplingParams


@dataclass
class NewRequestData:
    """Data for a request being scheduled for the first time.
    
    When a request is first scheduled, we send all its information
    to the workers so they can cache it. This avoids re-sending
    the full prompt every iteration.
    
    Attributes:
        req_id: Unique request identifier
        prompt_token_ids: Tokenized prompt
        sampling_params: Sampling configuration
        num_computed_tokens: Number of tokens already processed (0 for new)
        block_ids: KV cache block IDs (M3+, empty for M2)
    """
    req_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    num_computed_tokens: int = 0
    # M3+: KV cache block allocation
    block_ids: List[int] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
            f"NewRequestData("
            f"req_id={self.req_id}, "
            f"prompt_len={len(self.prompt_token_ids)}, "
            f"num_computed={self.num_computed_tokens})"
        )


@dataclass
class CachedRequestData:
    """Data for requests that have been scheduled before.
    
    For continuing requests, we only send incremental updates
    (new tokens) rather than the full request data, which is
    already cached in the workers.
    
    Attributes:
        req_ids: List of request IDs in this batch
        new_token_ids: New tokens generated for each request
        num_computed_tokens: Total tokens computed per request
        num_output_tokens: Output tokens generated per request
        new_block_ids: New KV cache blocks (M3+, None for M2)
    """
    req_ids: List[str]
    new_token_ids: List[List[int]]  # One list per request
    num_computed_tokens: List[int]
    num_output_tokens: List[int]
    # M3+: New KV cache blocks allocated
    new_block_ids: List[Optional[List[int]]] = field(default_factory=list)
    
    @property
    def num_reqs(self) -> int:
        """Get number of requests in this batch."""
        return len(self.req_ids)
    
    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        """Create an empty CachedRequestData.
        
        Used when there are no continuing requests in the iteration.
        """
        return cls(
            req_ids=[],
            new_token_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
            new_block_ids=[],
        )
    
    def __repr__(self) -> str:
        return (
            f"CachedRequestData("
            f"num_reqs={self.num_reqs}, "
            f"req_ids={self.req_ids})"
        )


@dataclass
class SchedulerOutput:
    """Output of the scheduler for one iteration.
    
    This contains all information needed by the model runner to:
    1. Build the input batch
    2. Execute the model
    3. Update request states
    
    Attributes:
        scheduled_new_reqs: Requests being scheduled for first time
        scheduled_cached_reqs: Continuing requests
        num_scheduled_tokens: Tokens to process per request
        total_num_scheduled_tokens: Sum of all tokens in this iteration
        finished_req_ids: Requests that finished in previous step
    """
    # Requests being scheduled
    scheduled_new_reqs: List[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    
    # Token scheduling information
    num_scheduled_tokens: Dict[str, int]  # req_id -> num_tokens
    total_num_scheduled_tokens: int
    
    # Finished requests (need to notify workers to free resources)
    finished_req_ids: Set[str]
    
    # M3+: Additional scheduling metadata
    # num_common_prefix_blocks: int = 0  # For prefix caching
    
    @property
    def num_new_reqs(self) -> int:
        """Number of new requests in this iteration."""
        return len(self.scheduled_new_reqs)
    
    @property
    def num_cached_reqs(self) -> int:
        """Number of continuing requests in this iteration."""
        return self.scheduled_cached_reqs.num_reqs
    
    @property
    def total_num_reqs(self) -> int:
        """Total number of requests in this iteration."""
        return self.num_new_reqs + self.num_cached_reqs
    
    @property
    def is_empty(self) -> bool:
        """Check if this iteration has no scheduled requests."""
        return self.total_num_reqs == 0
    
    def __repr__(self) -> str:
        return (
            f"SchedulerOutput("
            f"new_reqs={self.num_new_reqs}, "
            f"cached_reqs={self.num_cached_reqs}, "
            f"total_tokens={self.total_num_scheduled_tokens}, "
            f"finished={len(self.finished_req_ids)})"
        )

