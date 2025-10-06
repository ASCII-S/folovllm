"""
FoloVLLM: A Lightweight LLM Inference Framework

Inspired by vLLM, this framework implements modern LLM inference optimizations
in a progressive, educational manner.
"""

__version__ = "0.1.0"
__author__ = "FoloVLLM Contributors"

# M0: Core data structures and configuration
from folovllm.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
)
from folovllm.outputs import CompletionOutput, RequestOutput
from folovllm.request import (
    Request,
    RequestStatus,
    Sequence,
    SequenceData,
    SequenceStatus,
)
from folovllm.sampling_params import SamplingParams, SamplingType

# M0: Model loading
from folovllm.model_loader import ModelLoader, get_model_and_tokenizer

# M0: Utilities
from folovllm.utils.common import (
    generate_request_id,
    get_device,
    get_gpu_memory_info,
    is_cuda_available,
    set_random_seed,
)

# Will be implemented in future milestones
# from folovllm.engine import LLM

__all__ = [
    "__version__",
    # Configuration
    "ModelConfig",
    "CacheConfig",
    "SchedulerConfig",
    "EngineConfig",
    # Requests and Sequences
    "Request",
    "RequestStatus",
    "Sequence",
    "SequenceData",
    "SequenceStatus",
    # Sampling
    "SamplingParams",
    "SamplingType",
    # Outputs
    "RequestOutput",
    "CompletionOutput",
    # Model Loading
    "ModelLoader",
    "get_model_and_tokenizer",
    # Utilities
    "generate_request_id",
    "set_random_seed",
    "get_device",
    "is_cuda_available",
    "get_gpu_memory_info",
]

