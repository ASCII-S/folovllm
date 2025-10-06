"""Common utility functions for FoloVLLM."""

import random
import uuid
from typing import List, Optional

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_request_id() -> str:
    """Generate a unique request ID.
    
    Returns:
        Unique request ID string.
    """
    return str(uuid.uuid4())


def get_gpu_memory_info(device: int = 0) -> dict:
    """Get GPU memory information.
    
    Args:
        device: GPU device index.
        
    Returns:
        Dictionary with memory info (allocated, reserved, free, total in GB).
    """
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    free = total - allocated
    
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
    }


def print_gpu_memory_info(device: int = 0) -> None:
    """Print GPU memory information.
    
    Args:
        device: GPU device index.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    info = get_gpu_memory_info(device)
    print(f"GPU Memory (Device {device}):")
    print(f"  Allocated: {info['allocated_gb']:.2f} GB")
    print(f"  Reserved:  {info['reserved_gb']:.2f} GB")
    print(f"  Free:      {info['free_gb']:.2f} GB")
    print(f"  Total:     {info['total_gb']:.2f} GB")


def is_cuda_available() -> bool:
    """Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device.
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto).
        
    Returns:
        Torch device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def move_to_device(
    tensors: List[torch.Tensor], device: torch.device
) -> List[torch.Tensor]:
    """Move tensors to device.
    
    Args:
        tensors: List of tensors to move.
        device: Target device.
        
    Returns:
        List of tensors on the target device.
    """
    return [t.to(device) for t in tensors]

