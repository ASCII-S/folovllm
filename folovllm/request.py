"""Request and Sequence definitions (aligned with vLLM v1)."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from folovllm.sampling_params import SamplingParams


class RequestStatus(Enum):
    """Status of a request in the scheduler."""
    WAITING = "waiting"              # In waiting queue
    RUNNING = "running"              # Being processed
    SWAPPED = "swapped"              # Swapped to CPU (M2+)
    FINISHED_STOPPED = "finished_stopped"               # Stopped by stop condition
    FINISHED_LENGTH_CAPPED = "finished_length_capped"   # Reached max length
    FINISHED_ABORTED = "finished_aborted"               # Aborted by user


class SequenceStatus(Enum):
    """Status of a sequence in generation."""
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"
    FINISHED_IGNORED = "finished_ignored"
    
    def is_finished(self) -> bool:
        """Check if the sequence is finished."""
        return self in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]


@dataclass
class SequenceData:
    """Data associated with a sequence.
    
    This contains the actual tokens and their state.
    """
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    
    def get_len(self) -> int:
        """Get total length of the sequence."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)
    
    def get_prompt_len(self) -> int:
        """Get length of the prompt."""
        return len(self.prompt_token_ids)
    
    def get_output_len(self) -> int:
        """Get length of the output."""
        return len(self.output_token_ids)
    
    def get_token_ids(self) -> List[int]:
        """Get all token IDs (prompt + output)."""
        return self.prompt_token_ids + self.output_token_ids
    
    def get_last_token_id(self) -> int:
        """Get the last token ID."""
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]
    
    def add_token_id(self, token_id: int):
        """Add a new output token."""
        self.output_token_ids.append(token_id)


@dataclass
class Sequence:
    """A sequence in generation.
    
    Aligned with vLLM's Sequence class.
    
    Args:
        seq_id: Unique sequence ID.
        request_id: The request this sequence belongs to.
        data: The sequence data (tokens).
        sampling_params: Sampling parameters for this sequence.
    """
    seq_id: str
    request_id: str
    data: SequenceData
    sampling_params: SamplingParams
    status: SequenceStatus = SequenceStatus.WAITING
    
    # M3: KV cache blocks will be added here
    block_ids: List[int] = field(default_factory=list)
    
    def get_len(self) -> int:
        """Get total length of the sequence."""
        return self.data.get_len()
    
    def get_prompt_len(self) -> int:
        """Get length of the prompt."""
        return self.data.get_prompt_len()
    
    def get_output_len(self) -> int:
        """Get length of the output."""
        return self.data.get_output_len()
    
    def get_token_ids(self) -> List[int]:
        """Get all token IDs."""
        return self.data.get_token_ids()
    
    def get_last_token_id(self) -> int:
        """Get the last token ID."""
        return self.data.get_last_token_id()
    
    def add_token_id(self, token_id: int):
        """Add a new output token."""
        self.data.add_token_id(token_id)
    
    def is_finished(self) -> bool:
        """Check if the sequence is finished."""
        return self.status.is_finished()
    
    def fork(self, new_seq_id: str) -> "Sequence":
        """Fork this sequence for beam search or parallel sampling.
        
        Reserved for future milestones (M1+).
        """
        new_data = SequenceData(
            prompt_token_ids=self.data.prompt_token_ids.copy(),
            output_token_ids=self.data.output_token_ids.copy(),
        )
        return Sequence(
            seq_id=new_seq_id,
            request_id=self.request_id,
            data=new_data,
            sampling_params=self.sampling_params,
            status=self.status,
            block_ids=self.block_ids.copy(),
        )


@dataclass
class Request:
    """A request for inference.
    
    Aligned with vLLM v1 request structure.
    """
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.time)
    
    # Sequences generated for this request (for n > 1)
    sequences: Dict[str, Sequence] = field(default_factory=dict)
    
    # Status
    status: RequestStatus = RequestStatus.WAITING
    
    # M2: Add scheduling metadata
    # M3: Add KV cache blocks
    
    def __post_init__(self):
        """Initialize sequences for this request."""
        if not self.sequences:
            # Create sequences based on sampling_params.best_of
            for i in range(self.sampling_params.best_of):
                seq_id = f"{self.request_id}-{i}"
                seq_data = SequenceData(prompt_token_ids=self.prompt_token_ids.copy())
                seq = Sequence(
                    seq_id=seq_id,
                    request_id=self.request_id,
                    data=seq_data,
                    sampling_params=self.sampling_params,
                )
                self.sequences[seq_id] = seq
    
    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        """Get sequences, optionally filtered by status."""
        if status is None:
            return list(self.sequences.values())
        return [seq for seq in self.sequences.values() if seq.status == status]
    
    def get_num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        """Get number of sequences, optionally filtered by status."""
        return len(self.get_seqs(status))
    
    def is_finished(self) -> bool:
        """Check if all sequences are finished."""
        return all(seq.is_finished() for seq in self.sequences.values())

