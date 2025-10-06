"""Unit tests for Request and Sequence (Milestone 0)."""

import pytest

from folovllm.request import (
    Request,
    RequestStatus,
    Sequence,
    SequenceData,
    SequenceStatus,
)
from folovllm.sampling_params import SamplingParams


class TestSequenceData:
    """Test SequenceData class."""
    
    def test_creation(self):
        """Test sequence data creation."""
        data = SequenceData(prompt_token_ids=[1, 2, 3])
        assert data.prompt_token_ids == [1, 2, 3]
        assert data.output_token_ids == []
    
    def test_get_len(self):
        """Test get_len method."""
        data = SequenceData(prompt_token_ids=[1, 2, 3])
        assert data.get_len() == 3
        assert data.get_prompt_len() == 3
        assert data.get_output_len() == 0
        
        data.add_token_id(4)
        assert data.get_len() == 4
        assert data.get_output_len() == 1
    
    def test_get_token_ids(self):
        """Test get_token_ids method."""
        data = SequenceData(prompt_token_ids=[1, 2, 3])
        data.add_token_id(4)
        data.add_token_id(5)
        
        assert data.get_token_ids() == [1, 2, 3, 4, 5]
    
    def test_get_last_token_id(self):
        """Test get_last_token_id method."""
        data = SequenceData(prompt_token_ids=[1, 2, 3])
        assert data.get_last_token_id() == 3
        
        data.add_token_id(4)
        assert data.get_last_token_id() == 4


class TestSequence:
    """Test Sequence class."""
    
    def test_creation(self):
        """Test sequence creation."""
        data = SequenceData(prompt_token_ids=[1, 2, 3])
        params = SamplingParams()
        
        seq = Sequence(
            seq_id="seq-0",
            request_id="req-0",
            data=data,
            sampling_params=params,
        )
        
        assert seq.seq_id == "seq-0"
        assert seq.request_id == "req-0"
        assert seq.status == SequenceStatus.WAITING
    
    def test_is_finished(self):
        """Test is_finished method."""
        data = SequenceData(prompt_token_ids=[1, 2, 3])
        params = SamplingParams()
        seq = Sequence(
            seq_id="seq-0",
            request_id="req-0",
            data=data,
            sampling_params=params,
        )
        
        assert not seq.is_finished()
        
        seq.status = SequenceStatus.FINISHED_STOPPED
        assert seq.is_finished()
    
    def test_fork(self):
        """Test sequence forking."""
        data = SequenceData(prompt_token_ids=[1, 2, 3])
        data.add_token_id(4)
        params = SamplingParams()
        
        seq = Sequence(
            seq_id="seq-0",
            request_id="req-0",
            data=data,
            sampling_params=params,
        )
        
        forked = seq.fork("seq-1")
        assert forked.seq_id == "seq-1"
        assert forked.request_id == "req-0"
        assert forked.get_token_ids() == [1, 2, 3, 4]
        
        # Ensure deep copy
        seq.add_token_id(5)
        assert seq.get_token_ids() == [1, 2, 3, 4, 5]
        assert forked.get_token_ids() == [1, 2, 3, 4]


class TestRequest:
    """Test Request class."""
    
    def test_creation(self):
        """Test request creation."""
        params = SamplingParams()
        request = Request(
            request_id="req-0",
            prompt="Hello",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
        )
        
        assert request.request_id == "req-0"
        assert request.prompt == "Hello"
        assert request.status == RequestStatus.WAITING
    
    def test_sequences_initialization(self):
        """Test that sequences are initialized based on best_of."""
        params = SamplingParams(best_of=3)
        request = Request(
            request_id="req-0",
            prompt="Hello",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
        )
        
        assert len(request.sequences) == 3
        assert "req-0-0" in request.sequences
        assert "req-0-1" in request.sequences
        assert "req-0-2" in request.sequences
    
    def test_get_seqs(self):
        """Test get_seqs method."""
        params = SamplingParams(best_of=3)
        request = Request(
            request_id="req-0",
            prompt="Hello",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
        )
        
        # All sequences
        assert len(request.get_seqs()) == 3
        
        # Filter by status
        request.sequences["req-0-0"].status = SequenceStatus.RUNNING
        running_seqs = request.get_seqs(status=SequenceStatus.RUNNING)
        assert len(running_seqs) == 1
        assert running_seqs[0].seq_id == "req-0-0"
    
    def test_is_finished(self):
        """Test is_finished method."""
        params = SamplingParams(best_of=2)
        request = Request(
            request_id="req-0",
            prompt="Hello",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params,
        )
        
        assert not request.is_finished()
        
        # Finish one sequence
        request.sequences["req-0-0"].status = SequenceStatus.FINISHED_STOPPED
        assert not request.is_finished()
        
        # Finish all sequences
        request.sequences["req-0-1"].status = SequenceStatus.FINISHED_STOPPED
        assert request.is_finished()

