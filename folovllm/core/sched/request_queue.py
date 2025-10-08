"""Request queue implementations."""

from collections import deque
from typing import Iterator, List

from folovllm.request import Request


class RequestQueue:
    """A FCFS (First-Come-First-Served) request queue.
    
    Uses deque for efficient O(1) operations at both ends.
    Aligned with vLLM v1's FCFSRequestQueue.
    
    For M2, we use simple FCFS scheduling.
    M3+ may add priority-based scheduling.
    """
    
    def __init__(self):
        """Initialize an empty request queue."""
        self._queue: deque[Request] = deque()
    
    def add_request(self, request: Request) -> None:
        """Add a request to the back of the queue.
        
        Args:
            request: The request to add.
        """
        self._queue.append(request)
    
    def pop_request(self) -> Request:
        """Remove and return the request at the front of the queue.
        
        Returns:
            The request at the front.
            
        Raises:
            IndexError: If the queue is empty.
        """
        return self._queue.popleft()
    
    def peek_request(self) -> Request:
        """Return the request at the front without removing it.
        
        Returns:
            The request at the front.
            
        Raises:
            IndexError: If the queue is empty.
        """
        if not self._queue:
            raise IndexError("peek from an empty queue")
        return self._queue[0]
    
    def prepend_request(self, request: Request) -> None:
        """Add a request to the front of the queue.
        
        Used for request preemption.
        
        Args:
            request: The request to prepend.
        """
        self._queue.appendleft(request)
    
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue.
        
        Args:
            request: The request to remove.
            
        Raises:
            ValueError: If the request is not in the queue.
        """
        self._queue.remove(request)
    
    def remove_request_by_id(self, request_id: str) -> bool:
        """Remove a request by its ID.
        
        Args:
            request_id: The ID of the request to remove.
            
        Returns:
            True if the request was found and removed, False otherwise.
        """
        for req in self._queue:
            if req.request_id == request_id:
                self._queue.remove(req)
                return True
        return False
    
    def __len__(self) -> int:
        """Return the number of requests in the queue."""
        return len(self._queue)
    
    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return len(self._queue) > 0
    
    def __iter__(self) -> Iterator[Request]:
        """Iterate over requests in FCFS order."""
        return iter(self._queue)
    
    def __contains__(self, request: Request) -> bool:
        """Check if a request is in the queue."""
        return request in self._queue
    
    def clear(self) -> None:
        """Remove all requests from the queue."""
        self._queue.clear()
    
    def get_all_requests(self) -> List[Request]:
        """Get a list of all requests in the queue.
        
        Returns:
            List of all requests in FCFS order.
        """
        return list(self._queue)

