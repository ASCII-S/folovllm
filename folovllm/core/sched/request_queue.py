"""Request queue implementations (aligned with vLLM v1).

This module provides queue implementations for different scheduling policies.
For M2, we implement FCFS (First-Come-First-Served) policy.
M3+ will add priority-based scheduling.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Iterator
from enum import Enum

from folovllm.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"  # Reserved for M3+


class RequestQueue(ABC):
    """Abstract base class for request queues.
    
    Defines the interface that all queue implementations must follow.
    This abstraction allows for different scheduling policies (FCFS, priority, etc.).
    """
    
    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass
    
    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass
    
    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass
    
    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue.
        
        Used for preempted requests that need to resume quickly.
        """
        pass
    
    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this queue."""
        pass
    
    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass
    
    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass
    
    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass
    
    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque, RequestQueue):
    """First-Come-First-Served queue using deque.
    
    This is the simplest scheduling policy where requests are processed
    in the order they arrive. It provides O(1) operations for adding
    and removing from both ends.
    
    For M2: This is the primary queue implementation.
    For M3+: Priority queue will be added for more sophisticated scheduling.
    """
    
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0
    
    def add_request(self, request: Request) -> None:
        """Add a request to the end of the queue."""
        self.append(request)
    
    def pop_request(self) -> Request:
        """Pop a request from the front of the queue."""
        return self.popleft()
    
    def peek_request(self) -> Request:
        """Peek at the next request without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]
    
    def prepend_request(self, request: Request) -> None:
        """Add a request to the front of the queue.
        
        Used when a request is preempted and needs to resume quickly.
        """
        self.appendleft(request)
    
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue.
        
        Requests are added in reverse order to maintain their relative ordering.
        """
        for request in reversed(requests):
            self.prepend_request(request)
    
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        try:
            self.remove(request)
        except ValueError:
            # Request not in queue, ignore
            pass
    
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        for request in requests:
            self.remove_request(request)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Factory function to create a request queue based on policy.
    
    Args:
        policy: The scheduling policy to use.
        
    Returns:
        A RequestQueue instance implementing the specified policy.
        
    Raises:
        ValueError: If the policy is not supported.
    """
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.PRIORITY:
        # Reserved for M3+
        raise NotImplementedError(
            "Priority scheduling is not yet implemented. "
            "It will be added in M3+."
        )
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")

