"""Scheduler output dataclass."""

from dataclasses import dataclass, field
from typing import List

from folovllm.request import Request


@dataclass
class SchedulerOutput:
    """Output of the scheduler's schedule() method.
    
    Contains information about which requests are scheduled for this iteration
    and metadata needed by the model runner.
    
    Simplified version for M2. M3+ will add more fields (block tables, etc.).
    """
    
    # List of requests scheduled for this iteration
    scheduled_requests: List[Request] = field(default_factory=list)
    
    # Total number of sequences scheduled
    num_scheduled_seqs: int = 0
    
    # Total number of tokens to process in this iteration
    num_scheduled_tokens: int = 0
    
    # Whether this is a prefill batch (True) or decode batch (False)
    # M2: We keep prefill and decode separate
    # M5: Will support mixed batches (chunked prefill)
    is_prefill: bool = False
    
    # List of request IDs that finished in the previous iteration
    # and should be cleaned up
    finished_request_ids: List[str] = field(default_factory=list)
    
    def is_empty(self) -> bool:
        """Check if there are no scheduled requests."""
        return len(self.scheduled_requests) == 0
    
    def __repr__(self) -> str:
        return (
            f"SchedulerOutput("
            f"num_reqs={len(self.scheduled_requests)}, "
            f"num_seqs={self.num_scheduled_seqs}, "
            f"num_tokens={self.num_scheduled_tokens}, "
            f"is_prefill={self.is_prefill})"
        )

