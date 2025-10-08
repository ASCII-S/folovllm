"""Unit tests for M2 scheduler components."""

import pytest
from folovllm.core.sched import (
    Scheduler,
    FCFSRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from folovllm.config import ModelConfig, SchedulerConfig
from folovllm.request import Request, RequestStatus, SequenceStatus
from folovllm.sampling_params import SamplingParams


def test_fcfs_queue_basic():
    """Test basic FCFS queue operations."""
    queue = FCFSRequestQueue()
    
    # Create test requests
    req1 = Request(
        request_id="req-1",
        prompt="Hello",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    req2 = Request(
        request_id="req-2",
        prompt="World",
        prompt_token_ids=[4, 5, 6],
        sampling_params=SamplingParams(),
    )
    
    # Test adding
    queue.add_request(req1)
    queue.add_request(req2)
    assert len(queue) == 2
    
    # Test FIFO order
    popped = queue.pop_request()
    assert popped.request_id == "req-1"
    assert len(queue) == 1
    
    popped = queue.pop_request()
    assert popped.request_id == "req-2"
    assert len(queue) == 0


def test_fcfs_queue_prepend():
    """Test prepending to FCFS queue."""
    queue = FCFSRequestQueue()
    
    req1 = Request("req-1", "A", [1], SamplingParams())
    req2 = Request("req-2", "B", [2], SamplingParams())
    req3 = Request("req-3", "C", [3], SamplingParams())
    
    queue.add_request(req1)
    queue.add_request(req2)
    queue.prepend_request(req3)
    
    # req3 should be first
    assert queue.pop_request().request_id == "req-3"
    assert queue.pop_request().request_id == "req-1"
    assert queue.pop_request().request_id == "req-2"


def test_queue_factory():
    """Test queue factory function."""
    fcfs_queue = create_request_queue(SchedulingPolicy.FCFS)
    assert isinstance(fcfs_queue, FCFSRequestQueue)
    
    # Priority queue should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        create_request_queue(SchedulingPolicy.PRIORITY)


def test_scheduler_add_request():
    """Test adding requests to scheduler."""
    model_config = ModelConfig(model="test")
    scheduler_config = SchedulerConfig(max_num_seqs=4)
    
    scheduler = Scheduler(model_config, scheduler_config)
    
    req = Request(
        request_id="test-req",
        prompt="Test",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(),
    )
    
    scheduler.add_request(req)
    
    # Check request is in waiting queue
    assert len(scheduler.waiting) == 1
    assert scheduler.get_num_unfinished_requests() == 1
    assert req.status == RequestStatus.WAITING


def test_scheduler_schedule_new_request():
    """Test scheduling a new request."""
    model_config = ModelConfig(model="test")
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=100,
    )
    
    scheduler = Scheduler(model_config, scheduler_config)
    
    # Add request
    req = Request(
        request_id="test-req",
        prompt="Test",
        prompt_token_ids=[1, 2, 3, 4, 5],  # 5 tokens
        sampling_params=SamplingParams(),
    )
    scheduler.add_request(req)
    
    # Schedule
    output = scheduler.schedule()
    
    # Should schedule the full prompt
    assert len(output.scheduled_new_reqs) == 1
    new_req = output.scheduled_new_reqs[0]
    assert new_req.req_id == "test-req"
    assert len(new_req.prompt_token_ids) == 5
    assert output.total_num_scheduled_tokens == 5
    
    # Request should now be running
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0


def test_scheduler_schedule_multiple_requests():
    """Test scheduling multiple requests."""
    model_config = ModelConfig(model="test")
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=100,
    )
    
    scheduler = Scheduler(model_config, scheduler_config)
    
    # Add multiple requests
    for i in range(3):
        req = Request(
            request_id=f"req-{i}",
            prompt=f"Prompt {i}",
            prompt_token_ids=list(range(i*10, (i+1)*10)),  # 10 tokens each
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(req)
    
    # Schedule
    output = scheduler.schedule()
    
    # Should schedule all 3 requests
    assert len(output.scheduled_new_reqs) == 3
    assert output.total_num_scheduled_tokens == 30  # 3 * 10 tokens
    assert len(scheduler.running) == 3


def test_scheduler_token_budget():
    """Test that scheduler respects token budget."""
    model_config = ModelConfig(model="test")
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=25,  # Limited budget
    )
    
    scheduler = Scheduler(model_config, scheduler_config)
    
    # Add requests that exceed budget
    for i in range(5):
        req = Request(
            request_id=f"req-{i}",
            prompt=f"Prompt {i}",
            prompt_token_ids=list(range(20)),  # 20 tokens each
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(req)
    
    # Schedule
    output = scheduler.schedule()
    
    # Should only schedule 1 request (20 tokens < 25 budget)
    # Cannot fit 2 requests (40 tokens > 25 budget)
    assert len(output.scheduled_new_reqs) == 1
    assert output.total_num_scheduled_tokens == 20
    assert len(scheduler.waiting) == 4  # 4 still waiting


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

