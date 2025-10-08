"""Scheduler implementation (aligned with vLLM v1).

The scheduler manages request queues and decides which requests
to process in each iteration (continuous batching).
"""

import time
from typing import Dict, Iterable, List, Optional, Set, Union

from folovllm.config import ModelConfig, SchedulerConfig
from folovllm.request import Request, RequestStatus, SequenceStatus
from folovllm.outputs import RequestOutput, CompletionOutput
from folovllm.core.sched.interface import SchedulerInterface
from folovllm.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from folovllm.core.sched.output import (
    NewRequestData,
    CachedRequestData,
    SchedulerOutput,
)


class Scheduler(SchedulerInterface):
    """Continuous batching scheduler for M2.
    
    Implements iteration-level scheduling where:
    - New requests get scheduled for prefill (process full prompt)
    - Running requests get scheduled for decode (process 1 token)
    - Finished requests are tracked and resources freed
    
    For M2: Simple scheduling without preemption or swapping
    For M3+: Will add KV cache block management, preemption, swapping
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
    ):
        """Initialize the scheduler.
        
        Args:
            model_config: Model configuration
            scheduler_config: Scheduler configuration
        """
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        
        # Scheduling constraints
        self.max_num_seqs = scheduler_config.max_num_seqs
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        self.max_model_len = scheduler_config.max_model_len or 2048
        
        # Request storage: req_id -> Request
        self.requests: Dict[str, Request] = {}
        
        # Request queues
        self.policy = SchedulingPolicy.FCFS  # M2: Only FCFS
        self.waiting: RequestQueue = create_request_queue(self.policy)
        self.running: List[Request] = []  # Requests currently being processed
        
        # Finished request tracking
        # These are requests that finished in the previous step and need
        # to be communicated to workers for resource cleanup
        self.finished_req_ids: Set[str] = set()
        
        # M3+ will add:
        # - KV cache manager
        # - Block allocator
        # - Swapping manager
    
    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue.
        
        Args:
            request: The request to add
        """
        if request.request_id in self.requests:
            # Request already exists, ignore
            return
        
        request.status = RequestStatus.WAITING
        for seq in request.get_seqs():
            seq.status = SequenceStatus.WAITING
        
        self.requests[request.request_id] = request
        self.waiting.add_request(request)
    
    def schedule(self) -> SchedulerOutput:
        """Schedule requests for the next iteration.
        
        Scheduling logic for M2:
        1. Try to move waiting requests to running (up to max_num_seqs)
        2. For new requests: schedule full prompt (prefill)
        3. For running requests: schedule 1 token (decode)
        4. Ensure total tokens <= max_num_batched_tokens
        
        Returns:
            SchedulerOutput with scheduled requests
        """
        scheduled_new_reqs: List[NewRequestData] = []
        scheduled_cached_req_ids: List[str] = []
        scheduled_cached_tokens: List[List[int]] = []
        scheduled_cached_computed: List[int] = []
        scheduled_cached_output: List[int] = []
        
        num_scheduled_tokens: Dict[str, int] = {}
        total_tokens = 0
        
        # Step 1: Try to admit new requests from waiting queue
        while self.waiting and len(self.running) < self.max_num_seqs:
            request = self.waiting.peek_request()
            
            # Get the first (and for M2, only) sequence
            seq = request.get_seqs()[0]
            prompt_len = seq.get_prompt_len()
            
            # Check if we have budget for this request
            if total_tokens + prompt_len > self.max_num_batched_tokens:
                # Cannot fit this request, stop admitting
                break
            
            # Admit the request
            request = self.waiting.pop_request()
            request.status = RequestStatus.RUNNING
            seq.status = SequenceStatus.RUNNING
            self.running.append(request)
            
            # Schedule full prompt for prefill
            new_req_data = NewRequestData(
                req_id=request.request_id,
                prompt_token_ids=seq.data.prompt_token_ids,
                sampling_params=request.sampling_params,
                num_computed_tokens=0,
                block_ids=[],  # M3+: Will allocate KV cache blocks
            )
            scheduled_new_reqs.append(new_req_data)
            num_scheduled_tokens[request.request_id] = prompt_len
            total_tokens += prompt_len
        
        # Step 2: Schedule decode for running requests
        for request in self.running:
            # Skip if this request was just added (already scheduled above)
            if request.request_id in [r.req_id for r in scheduled_new_reqs]:
                continue
            
            seq = request.get_seqs()[0]
            
            # Check if we have budget for 1 more token
            if total_tokens + 1 > self.max_num_batched_tokens:
                # Out of budget, cannot schedule this request
                # M3+: This is where we'd implement preemption
                continue
            
            # Schedule 1 token for decode
            # Get the last generated token (or last prompt token if no output yet)
            if seq.get_output_len() > 0:
                last_token = seq.data.output_token_ids[-1]
            else:
                # First decode step after prefill
                last_token = seq.data.prompt_token_ids[-1]
            
            scheduled_cached_req_ids.append(request.request_id)
            scheduled_cached_tokens.append([last_token])
            scheduled_cached_computed.append(seq.get_len())
            scheduled_cached_output.append(seq.get_output_len())
            
            num_scheduled_tokens[request.request_id] = 1
            total_tokens += 1
        
        # Build CachedRequestData
        cached_req_data = CachedRequestData(
            req_ids=scheduled_cached_req_ids,
            new_token_ids=scheduled_cached_tokens,
            num_computed_tokens=scheduled_cached_computed,
            num_output_tokens=scheduled_cached_output,
            new_block_ids=[None] * len(scheduled_cached_req_ids),  # M3+
        )
        
        # Build SchedulerOutput
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_tokens,
            finished_req_ids=self.finished_req_ids.copy(),
        )
        
        # Clear finished_req_ids (they've been communicated)
        self.finished_req_ids.clear()
        
        return scheduler_output
    
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: Dict[str, int],  # req_id -> next_token_id
    ) -> Dict[str, RequestOutput]:
        """Update scheduler state based on model output.
        
        Args:
            scheduler_output: The output from schedule()
            model_output: Dict mapping req_id to generated token_id
            
        Returns:
            Dict of req_id -> RequestOutput for updated/finished requests
        """
        outputs: Dict[str, RequestOutput] = {}
        finished_requests: List[Request] = []
        
        # Update all scheduled requests with new tokens
        all_req_ids = (
            [r.req_id for r in scheduler_output.scheduled_new_reqs] +
            scheduler_output.scheduled_cached_reqs.req_ids
        )
        
        for req_id in all_req_ids:
            if req_id not in model_output:
                continue
            
            request = self.requests[req_id]
            seq = request.get_seqs()[0]
            next_token_id = model_output[req_id]
            
            # Add the new token
            seq.add_token_id(next_token_id)
            
            # Check stop conditions
            should_stop, finish_reason = self._check_stop_conditions(request, seq)
            
            if should_stop:
                # Mark as finished
                if finish_reason == "stop":
                    seq.status = SequenceStatus.FINISHED_STOPPED
                    request.status = RequestStatus.FINISHED_STOPPED
                elif finish_reason == "length":
                    seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                
                finished_requests.append(request)
            
            # Build output for this request (even if not finished, for streaming)
            output = self._build_request_output(request)
            outputs[req_id] = output
        
        # Remove finished requests from running queue
        for request in finished_requests:
            self.running.remove(request)
            self.finished_req_ids.add(request.request_id)
            # M3+: Free KV cache blocks here
        
        return outputs
    
    def _check_stop_conditions(
        self,
        request: Request,
        seq,
    ) -> tuple[bool, Optional[str]]:
        """Check if a sequence should stop generation.
        
        Args:
            request: The request
            seq: The sequence to check
            
        Returns:
            Tuple of (should_stop, finish_reason)
        """
        sampling_params = request.sampling_params
        
        # Check max tokens
        if sampling_params.max_tokens is not None:
            if seq.get_output_len() >= sampling_params.max_tokens:
                return True, "length"
        
        # Check EOS token (if not ignored)
        if not sampling_params.ignore_eos:
            # We'll check EOS in the actual token checking
            # For now, this is a placeholder
            pass
        
        # Check stop token IDs
        if sampling_params.stop_token_ids:
            if seq.get_last_token_id() in sampling_params.stop_token_ids:
                return True, "stop"
        
        # M2: We don't decode tokens here, so can't check stop strings
        # The engine will handle stop string checking after decoding
        
        return False, None
    
    def _build_request_output(self, request: Request) -> RequestOutput:
        """Build RequestOutput for a request.
        
        Args:
            request: The request
            
        Returns:
            RequestOutput
        """
        # Build completion outputs for all sequences
        completion_outputs = []
        for idx, seq in enumerate(request.get_seqs()):
            finish_reason = None
            if seq.status == SequenceStatus.FINISHED_STOPPED:
                finish_reason = "stop"
            elif seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED:
                finish_reason = "length"
            
            completion_output = CompletionOutput(
                index=idx,
                text="",  # Will be decoded by engine
                token_ids=seq.data.output_token_ids.copy(),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
            completion_outputs.append(completion_output)
        
        finished = request.is_finished()
        
        return RequestOutput(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            outputs=completion_outputs,
            finished=finished,
        )
    
    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Mark requests as finished.
        
        Args:
            request_ids: Single ID or iterable of IDs
            finished_status: The status to assign
        """
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        
        for req_id in request_ids:
            if req_id not in self.requests:
                continue
            
            request = self.requests[req_id]
            request.status = finished_status
            
            # Update sequence status
            seq_status_map = {
                RequestStatus.FINISHED_STOPPED: SequenceStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED: SequenceStatus.FINISHED_LENGTH_CAPPED,
                RequestStatus.FINISHED_ABORTED: SequenceStatus.FINISHED_ABORTED,
            }
            seq_status = seq_status_map.get(
                finished_status,
                SequenceStatus.FINISHED_ABORTED
            )
            for seq in request.get_seqs():
                seq.status = seq_status
            
            # Remove from queues
            self.waiting.remove_request(request)
            if request in self.running:
                self.running.remove(request)
            
            self.finished_req_ids.add(req_id)
    
    def get_num_unfinished_requests(self) -> int:
        """Get number of unfinished requests."""
        return len(self.waiting) + len(self.running)
    
    def has_finished_requests(self) -> bool:
        """Check if there are finished requests pending notification."""
        return len(self.finished_req_ids) > 0
    
    def get_request_counts(self) -> tuple[int, int]:
        """Get request counts.
        
        Returns:
            (num_running, num_waiting)
        """
        return len(self.running), len(self.waiting)

