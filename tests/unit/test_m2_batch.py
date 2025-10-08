"""Unit tests for M2 batch processing."""

import torch
import pytest
from folovllm.worker.input_batch import InputBatch, prepare_inputs_from_scheduler_output
from folovllm.core.sched.output import (
    SchedulerOutput,
    NewRequestData,
    CachedRequestData,
)
from folovllm.sampling_params import SamplingParams


def test_input_batch_creation():
    """Test creating an InputBatch."""
    batch = InputBatch(
        req_ids=["req-1", "req-2"],
        token_ids=[[1, 2, 3], [4, 5]],
        start_positions=[0, 0],
        is_prefill=[True, True],
        prompt_lens=[3, 2],
    )
    
    assert batch.batch_size == 2
    assert len(batch.req_ids) == 2


def test_input_batch_to_tensors():
    """Test converting InputBatch to tensors."""
    batch = InputBatch(
        req_ids=["req-1", "req-2"],
        token_ids=[[1, 2, 3], [4, 5]],  # Different lengths
        start_positions=[0, 0],
        is_prefill=[True, True],
        prompt_lens=[3, 2],
    )
    
    device = torch.device("cpu")
    token_ids, attention_mask, positions = batch.to_tensors(device)
    
    # Check shapes
    assert token_ids.shape == (2, 3)  # Batch size 2, max length 3
    assert attention_mask.shape == (2, 3)
    assert positions.shape == (2, 3)
    
    # Check padding
    assert attention_mask[0].sum() == 3  # First sequence: all valid
    assert attention_mask[1].sum() == 2  # Second sequence: 2 valid + 1 padding
    
    # Check positions
    assert positions[0].tolist() == [0, 1, 2]
    assert positions[1, :2].tolist() == [0, 1]


def test_prepare_inputs_new_requests():
    """Test preparing inputs from scheduler output with new requests."""
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="req-1",
                prompt_token_ids=[1, 2, 3, 4],
                sampling_params=SamplingParams(),
                num_computed_tokens=0,
            ),
            NewRequestData(
                req_id="req-2",
                prompt_token_ids=[5, 6, 7],
                sampling_params=SamplingParams(),
                num_computed_tokens=0,
            ),
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req-1": 4, "req-2": 3},
        total_num_scheduled_tokens=7,
        finished_req_ids=set(),
    )
    
    batch = prepare_inputs_from_scheduler_output(scheduler_output)
    
    assert batch.batch_size == 2
    assert batch.req_ids == ["req-1", "req-2"]
    assert batch.token_ids == [[1, 2, 3, 4], [5, 6, 7]]
    assert batch.start_positions == [0, 0]
    assert batch.is_prefill == [True, True]


def test_prepare_inputs_cached_requests():
    """Test preparing inputs from scheduler output with cached requests."""
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["req-1", "req-2"],
            new_token_ids=[[10], [20]],  # Decode phase: 1 token each
            num_computed_tokens=[10, 15],
            num_output_tokens=[5, 8],
        ),
        num_scheduled_tokens={"req-1": 1, "req-2": 1},
        total_num_scheduled_tokens=2,
        finished_req_ids=set(),
    )
    
    batch = prepare_inputs_from_scheduler_output(scheduler_output)
    
    assert batch.batch_size == 2
    assert batch.req_ids == ["req-1", "req-2"]
    assert batch.token_ids == [[10], [20]]
    assert batch.start_positions == [10, 15]  # Continue from computed tokens
    assert batch.is_prefill == [False, False]


def test_prepare_inputs_mixed():
    """Test preparing inputs with both new and cached requests."""
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="new-1",
                prompt_token_ids=[1, 2, 3],
                sampling_params=SamplingParams(),
                num_computed_tokens=0,
            ),
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["cached-1"],
            new_token_ids=[[10]],
            num_computed_tokens=[5],
            num_output_tokens=[2],
        ),
        num_scheduled_tokens={"new-1": 3, "cached-1": 1},
        total_num_scheduled_tokens=4,
        finished_req_ids=set(),
    )
    
    batch = prepare_inputs_from_scheduler_output(scheduler_output)
    
    assert batch.batch_size == 2
    # New requests come first
    assert batch.req_ids[0] == "new-1"
    assert batch.req_ids[1] == "cached-1"
    assert batch.is_prefill == [True, False]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

