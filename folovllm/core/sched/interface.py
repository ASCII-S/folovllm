"""Scheduler interface (aligned with vLLM v1)."""

from abc import ABC, abstractmethod
from typing import List, Optional

from folovllm.request import Request


class SchedulerInterface(ABC):
    """Abstract interface for schedulers.
    
    Defines the core scheduling operations that all schedulers must implement.
    This interface aligns with vLLM v1's SchedulerInterface.
    """
    
    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a new request to the scheduler's queue.
        
        Args:
            request: The new request to add.
        """
        raise NotImplementedError
    
    @abstractmethod
    def abort_request(self, request_id: str) -> None:
        """Abort a request by its ID.
        
        Args:
            request_id: The ID of the request to abort.
        """
        raise NotImplementedError
    
    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """Schedule requests for the next iteration.
        
        This is the core scheduling method called at each iteration.
        It decides which requests to process in this step.
        
        Returns:
            SchedulerOutput containing scheduled requests and metadata.
        """
        raise NotImplementedError
    
    @abstractmethod
    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests.
        
        Returns:
            True if there are requests waiting or running.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests.
        
        Returns:
            Number of requests in waiting or running state.
        """
        raise NotImplementedError

