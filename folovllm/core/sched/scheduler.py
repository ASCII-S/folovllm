"""Main scheduler implementation for continuous batching."""

from typing import List, Optional

from folovllm.config import SchedulerConfig
from folovllm.core.sched.interface import SchedulerInterface
from folovllm.core.sched.output import SchedulerOutput
from folovllm.core.sched.request_queue import RequestQueue
from folovllm.request import Request, RequestStatus, SequenceStatus


class Scheduler(SchedulerInterface):
    """Scheduler for continuous batching.
    
    Manages request lifecycle and decides which requests to process at each iteration.
    Implements iteration-level scheduling with separate prefill and decode batches.
    
    For M2:
    - FCFS scheduling policy
    - Separate prefill and decode batches (no mixing)
    - Basic preemption support
    - No paged attention yet (continuous KV cache)
    
    M3+ will add:
    - Paged attention and block management
    - More sophisticated preemption
    - Prefix caching
    """
    
    def __init__(self, scheduler_config: SchedulerConfig):
        """Initialize the scheduler.
        
        Args:
            scheduler_config: Scheduler configuration.
        """
        self.scheduler_config = scheduler_config
        
        # Scheduling constraints
        self.max_num_seqs = scheduler_config.max_num_seqs
        self.max_num_batched_tokens = (
            scheduler_config.max_num_batched_tokens or 2048
        )
        self.max_model_len = scheduler_config.max_model_len or 2048
        
        # Request queues (FCFS for M2)
        self.waiting: RequestQueue = RequestQueue()  # Waiting to be scheduled
        self.running: RequestQueue = RequestQueue()  # Currently being processed
        self.finished: List[Request] = []            # Finished in last iteration
        
        # Track requests by ID for quick lookup
        self._request_map: dict[str, Request] = {}
    
    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue.
        
        Args:
            request: The request to add.
        """
        request.status = RequestStatus.WAITING
        # Mark all sequences as waiting
        for seq in request.get_seqs():
            seq.status = SequenceStatus.WAITING
        
        self.waiting.add_request(request)
        self._request_map[request.request_id] = request
    
    def abort_request(self, request_id: str) -> None:
        """Abort a request by its ID.
        
        Args:
            request_id: The ID of the request to abort.
        """
        # Try to find and remove from waiting queue
        if self.waiting.remove_request_by_id(request_id):
            if request_id in self._request_map:
                req = self._request_map.pop(request_id)
                req.status = RequestStatus.FINISHED_ABORTED
            return
        
        # Try to find in running queue
        request = self._request_map.get(request_id)
        if request and request in self.running:
            self.running.remove_request(request)
            request.status = RequestStatus.FINISHED_ABORTED
            for seq in request.get_seqs():
                seq.status = SequenceStatus.FINISHED_ABORTED
            self._request_map.pop(request_id)
    
    def schedule(self) -> SchedulerOutput:
        """Schedule requests for the next iteration.
        
        M2 scheduling logic:
        1. First, try to schedule waiting requests (prefill)
        2. If no waiting requests, schedule running requests (decode)
        3. Keep prefill and decode separate (no mixing)
        
        Returns:
            SchedulerOutput with scheduled requests.
        """
        # Clear finished requests from last iteration
        self.finished.clear()
        
        # Try prefill first (waiting requests)
        scheduler_output = self._schedule_prefill()
        if not scheduler_output.is_empty():
            return scheduler_output
        
        # If no prefill, do decode (running requests)
        return self._schedule_decode()
    
    def _schedule_prefill(self) -> SchedulerOutput:
        """Schedule waiting requests for prefill.
        
        Prefill processes the entire prompt in one forward pass.
        
        Returns:
            SchedulerOutput for prefill batch.
        """
        scheduled_requests = []
        num_seqs = 0
        num_tokens = 0
        
        # Try to schedule as many waiting requests as possible
        while self.waiting:
            if num_seqs >= self.max_num_seqs:
                break
            
            request = self.waiting.peek_request()
            
            # Get the first sequence (M2 only supports best_of=1)
            seqs = request.get_seqs()
            if not seqs:
                # No sequences, skip this request
                self.waiting.pop_request()
                continue
            
            seq = seqs[0]
            prompt_len = seq.get_prompt_len()
            
            # Check if we can fit this request
            if num_tokens + prompt_len > self.max_num_batched_tokens:
                # Cannot fit, stop scheduling
                break
            
            # Check max_model_len constraint
            if prompt_len > self.max_model_len:
                # Prompt too long, skip and mark as finished
                self.waiting.pop_request()
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                for s in seqs:
                    s.status = SequenceStatus.FINISHED_LENGTH_CAPPED
                continue
            
            # Can schedule this request
            self.waiting.pop_request()
            request.status = RequestStatus.RUNNING
            for s in seqs:
                s.status = SequenceStatus.RUNNING
            
            self.running.add_request(request)
            scheduled_requests.append(request)
            num_seqs += len(seqs)
            num_tokens += prompt_len
        
        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            num_scheduled_seqs=num_seqs,
            num_scheduled_tokens=num_tokens,
            is_prefill=True,
            finished_request_ids=[],
        )
    
    def _schedule_decode(self) -> SchedulerOutput:
        """Schedule running requests for decode.
        
        Decode generates one token at a time for each sequence.
        
        Returns:
            SchedulerOutput for decode batch.
        """
        scheduled_requests = []
        num_seqs = 0
        
        # Schedule running requests
        # For decode, each sequence generates 1 token
        for request in list(self.running):
            seqs = request.get_seqs(status=SequenceStatus.RUNNING)
            if not seqs:
                # No running sequences, skip
                continue
            
            if num_seqs + len(seqs) > self.max_num_seqs:
                # Cannot fit more sequences
                break
            
            scheduled_requests.append(request)
            num_seqs += len(seqs)
        
        # For decode, we process 1 token per sequence
        num_tokens = num_seqs
        
        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            num_scheduled_seqs=num_seqs,
            num_scheduled_tokens=num_tokens,
            is_prefill=False,
            finished_request_ids=[req.request_id for req in self.finished],
        )
    
    def finish_request(self, request: Request, status: RequestStatus) -> None:
        """Mark a request as finished and remove it from running queue.
        
        Args:
            request: The request to finish.
            status: The finished status.
        """
        request.status = status
        
        # Update sequence status
        seq_status_map = {
            RequestStatus.FINISHED_STOPPED: SequenceStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED: SequenceStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_ABORTED: SequenceStatus.FINISHED_ABORTED,
        }
        seq_status = seq_status_map.get(status, SequenceStatus.FINISHED_STOPPED)
        
        for seq in request.get_seqs():
            seq.status = seq_status
        
        # Remove from running queue
        if request in self.running:
            self.running.remove_request(request)
        
        # Add to finished list
        self.finished.append(request)
        
        # Remove from request map
        self._request_map.pop(request.request_id, None)
    
    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests.
        
        Returns:
            True if there are requests in waiting or running queues.
        """
        return len(self.waiting) > 0 or len(self.running) > 0
    
    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests.
        
        Returns:
            Total number of requests in waiting and running queues.
        """
        return len(self.waiting) + len(self.running)
    
    def get_num_waiting_requests(self) -> int:
        """Get the number of waiting requests."""
        return len(self.waiting)
    
    def get_num_running_requests(self) -> int:
        """Get the number of running requests."""
        return len(self.running)

