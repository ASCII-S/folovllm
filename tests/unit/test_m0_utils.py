"""Unit tests for utility functions (Milestone 0)."""

import torch

from folovllm.utils.common import (
    generate_request_id,
    get_device,
    get_gpu_memory_info,
    is_cuda_available,
    move_to_device,
    set_random_seed,
)


class TestUtils:
    """Test utility functions."""
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        id1 = generate_request_id()
        id2 = generate_request_id()
        
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2  # Should be unique
    
    def test_set_random_seed(self):
        """Test setting random seed."""
        set_random_seed(42)
        val1 = torch.rand(1).item()
        
        set_random_seed(42)
        val2 = torch.rand(1).item()
        
        assert val1 == val2  # Should be reproducible
    
    def test_is_cuda_available(self):
        """Test CUDA availability check."""
        result = is_cuda_available()
        assert isinstance(result, bool)
        assert result == torch.cuda.is_available()
    
    def test_get_device(self):
        """Test get_device function."""
        # Auto selection
        device = get_device()
        assert isinstance(device, torch.device)
        
        # Explicit CPU
        device = get_device("cpu")
        assert device.type == "cpu"
        
        # Explicit CUDA (if available)
        if torch.cuda.is_available():
            device = get_device("cuda")
            assert device.type == "cuda"
    
    def test_get_gpu_memory_info(self):
        """Test GPU memory info retrieval."""
        info = get_gpu_memory_info()
        
        if torch.cuda.is_available():
            assert "allocated_gb" in info
            assert "reserved_gb" in info
            assert "free_gb" in info
            assert "total_gb" in info
            assert all(isinstance(v, (int, float)) for v in info.values())
        else:
            assert info == {}
    
    def test_move_to_device(self):
        """Test moving tensors to device."""
        tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        device = torch.device("cpu")
        
        moved = move_to_device(tensors, device)
        assert len(moved) == 2
        assert all(t.device.type == "cpu" for t in moved)

