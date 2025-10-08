"""Input batch preparation for batched model execution.

This module handles the preparation of batched inputs from scheduler output,
including padding and creating attention masks for variable-length sequences.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

from folovllm.core.sched.output import SchedulerOutput


@dataclass
class InputBatch:
    """Batch of inputs for model execution.
    
    Handles variable-length sequences with padding and masking.
    
    Attributes:
        req_ids: List of request IDs in the batch
        token_ids: List of token ID sequences (ragged)
        start_positions: Starting position for each sequence
        is_prefill: Whether each request is in prefill (True) or decode (False)
        prompt_lens: Length of prompt for new requests (for prefill)
    """
    req_ids: List[str]
    token_ids: List[List[int]]  # Ragged list
    start_positions: List[int]
    is_prefill: List[bool]
    prompt_lens: List[int]  # For prefill requests
    
    def to_tensors(
        self,
        device: torch.device,
        pad_token_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to padded tensors for model execution.
        
        Args:
            device: Device to place tensors on
            pad_token_id: Token ID to use for padding
            
        Returns:
            Tuple of:
            - token_ids: [batch_size, max_seq_len]
            - attention_mask: [batch_size, max_seq_len]
            - positions: [batch_size, max_seq_len]
        """
        batch_size = len(self.token_ids)
        max_len = max(len(tokens) for tokens in self.token_ids)
        
        # Initialize padded tensors
        padded_token_ids = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )
        positions = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )
        
        # Fill in actual tokens and create masks
        for i, tokens in enumerate(self.token_ids):
            seq_len = len(tokens)
            padded_token_ids[i, :seq_len] = torch.tensor(
                tokens,
                dtype=torch.long,
                device=device,
            )
            attention_mask[i, :seq_len] = 1
            
            # Create position indices
            start_pos = self.start_positions[i]
            positions[i, :seq_len] = torch.arange(
                start_pos,
                start_pos + seq_len,
                dtype=torch.long,
                device=device,
            )
        
        return padded_token_ids, attention_mask, positions
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return len(self.req_ids)
    
    def __repr__(self) -> str:
        return (
            f"InputBatch(batch_size={self.batch_size}, "
            f"req_ids={self.req_ids[:3]}{'...' if len(self.req_ids) > 3 else ''})"
        )


def prepare_inputs_from_scheduler_output(
    scheduler_output: SchedulerOutput,
    requests_dict: Optional[Dict] = None,
) -> InputBatch:
    """Prepare batched inputs from scheduler output.
    
    This function builds an InputBatch from the scheduler's decisions about
    which requests to process and how many tokens for each.
    
    Args:
        scheduler_output: Output from the scheduler
        requests_dict: Optional dict of req_id -> Request for validation
        
    Returns:
        InputBatch ready for model execution
    """
    req_ids: List[str] = []
    token_ids: List[List[int]] = []
    start_positions: List[int] = []
    is_prefill: List[bool] = []
    prompt_lens: List[int] = []
    
    # Process new requests (prefill phase)
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_ids.append(new_req_data.req_id)
        # For prefill, we process the entire prompt
        token_ids.append(new_req_data.prompt_token_ids)
        start_positions.append(0)  # Start from position 0
        is_prefill.append(True)
        prompt_lens.append(len(new_req_data.prompt_token_ids))
    
    # Process cached requests (decode phase)
    cached_reqs = scheduler_output.scheduled_cached_reqs
    for i, req_id in enumerate(cached_reqs.req_ids):
        req_ids.append(req_id)
        # For decode, we only process the last token
        # The new_token_ids contains the token to process
        token_ids.append(cached_reqs.new_token_ids[i])
        
        # Start position is the number of already computed tokens
        # For decode, this is prompt_len + num_output_tokens - 1
        # (we're about to add one more token)
        num_computed = cached_reqs.num_computed_tokens[i]
        start_positions.append(num_computed)
        
        is_prefill.append(False)
        prompt_lens.append(0)  # Not applicable for decode
    
    return InputBatch(
        req_ids=req_ids,
        token_ids=token_ids,
        start_positions=start_positions,
        is_prefill=is_prefill,
        prompt_lens=prompt_lens,
    )

