"""Scheduler interface (aligned with vLLM v1).

Defines the abstract interface that all scheduler implementations must follow.
This allows for different scheduler strategies while maintaining a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Union

from folovllm.request import Request, RequestStatus


class SchedulerInterface(ABC):
    """Abstract interface for schedulers.
    
    The scheduler is responsible for:
    1. Managing request queues (waiting, running, finished)
    2. Deciding which requests to process in each iteration
    3. Managing KV cache allocation (M3+)
    4. Handling preemption and swapping (M3+)
    """
    
    @abstractmethod
    def schedule(self):
        """Schedule requests for the next iteration.
        
        This is the core method called by the engine at each iteration.
        It decides which requests to process and how many tokens to
        process for each request.
        
        The scheduler produces:
        - New requests entering the batch (prefill phase)
        - Continuing requests (decode phase)
        - Total token budget for the iteration
        
        Returns:
            SchedulerOutput: Information about scheduled requests.
        """
        raise NotImplementedError
    
    @abstractmethod
    def update_from_output(self, scheduler_output, model_output):
        """Update scheduler state based on model output.
        
        Called after the model has processed the scheduled requests.
        Updates sequences with new tokens, checks stop conditions,
        and moves finished requests out of the running queue.
        
        Args:
            scheduler_output: The output from schedule()
            model_output: Output from the model runner
            
        Returns:
            Dict[str, RequestOutput]: Outputs for each request
        """
        raise NotImplementedError
    
    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a new request to the scheduler.
        
        The request is added to the waiting queue and will be
        scheduled when resources are available.
        
        Args:
            request: The request to add
        """
        raise NotImplementedError
    
    @abstractmethod
    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Mark requests as finished.
        
        This can be called when:
        - A request is aborted by the user
        - A stop string is detected
        - Any other external finish condition
        
        Args:
            request_ids: Single request ID or iterable of IDs
            finished_status: The final status to assign
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests.
        
        Returns:
            Number of requests in waiting + running queues
        """
        raise NotImplementedError
    
    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests.
        
        Returns:
            True if there are requests in waiting or running queues
        """
        return self.get_num_unfinished_requests() > 0
    
    @abstractmethod
    def has_finished_requests(self) -> bool:
        """Check if there are finished requests to be communicated.
        
        The scheduler maintains a list of requests finished in the
        previous step. This list needs to be sent to workers to
        free cached states.
        
        Returns:
            True if there are finished requests pending notification
        """
        raise NotImplementedError
    
    def has_requests(self) -> bool:
        """Check if there are any requests at all.
        
        Returns:
            True if there are unfinished or pending-notification requests
        """
        return self.has_unfinished_requests() or self.has_finished_requests()
    
    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """Get counts of requests in different queues.
        
        Returns:
            Tuple of (num_running_reqs, num_waiting_reqs)
        """
        raise NotImplementedError
    
    # M3+ Methods - placeholders for future milestones
    
    def reset_prefix_cache(self) -> None:
        """Reset the prefix cache (M3+).
        
        This will be used when prefix caching is enabled
        and the cache needs to be invalidated.
        """
        # M3+: Will implement prefix cache reset
        pass
    
    def update_draft_token_ids(self, draft_token_ids) -> None:
        """Update draft tokens for speculative decoding (M5+).
        
        Args:
            draft_token_ids: Draft tokens from speculative decoding
        """
        # M5+: Will implement speculative decoding support
        pass
    
    def shutdown(self) -> None:
        """Shutdown the scheduler and clean up resources.
        
        For M2: Simple cleanup
        For M3+: May need to handle KV cache cleanup, etc.
        """
        pass

