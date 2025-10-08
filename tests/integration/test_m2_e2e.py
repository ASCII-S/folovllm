"""Integration test for M2 batch inference.

Tests end-to-end batch generation with continuous batching.
"""

import pytest
from folovllm import LLMEngine, ModelConfig, SchedulerConfig, SamplingParams


@pytest.fixture(scope="module")
def engine():
    """Create an LLM engine for testing."""
    model_config = ModelConfig(
        model="Qwen/Qwen2.5-0.5B",
        dtype="auto",
        trust_remote_code=True,
    )
    
    scheduler_config = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=512,
    )
    
    engine = LLMEngine(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )
    return engine


def test_batch_inference_basic(engine):
    """Test basic batch inference with multiple prompts."""
    prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Tell me a joke.",
    ]
    
    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0,  # Greedy for reproducibility
    )
    
    # Run batch inference
    outputs = engine.generate_batch(prompts, sampling_params)
    
    # Check that we got outputs for all prompts
    assert len(outputs) == len(prompts)
    
    # Check that each output is valid
    for req_id, output in outputs.items():
        assert output.finished
        assert len(output.outputs) == 1
        completion = output.outputs[0]
        assert len(completion.token_ids) > 0
        assert len(completion.text) > 0
        assert completion.finish_reason in ["stop", "length"]


def test_batch_inference_different_lengths(engine):
    """Test batch inference with prompts of different lengths."""
    prompts = [
        "Hi",  # Very short
        "This is a medium length prompt with more words",  # Medium
        "A" * 50,  # Long
    ]
    
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
    )
    
    outputs = engine.generate_batch(prompts, sampling_params)
    
    assert len(outputs) == len(prompts)
    for output in outputs.values():
        assert output.finished


def test_batch_inference_single_prompt(engine):
    """Test that batch inference works with a single prompt."""
    prompts = ["Hello, world!"]
    
    sampling_params = SamplingParams(
        max_tokens=15,
        temperature=0.0,
    )
    
    outputs = engine.generate_batch(prompts, sampling_params)
    
    assert len(outputs) == 1
    output = list(outputs.values())[0]
    assert output.finished
    assert len(output.outputs[0].token_ids) > 0


def test_batch_inference_max_tokens(engine):
    """Test that max_tokens is respected in batch inference."""
    prompts = [
        "Count to 100:",
        "List all countries:",
        "Tell me everything:",
    ]
    
    max_tokens = 5
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )
    
    outputs = engine.generate_batch(prompts, sampling_params)
    
    for output in outputs.values():
        assert output.finished
        # Should finish due to length limit
        assert len(output.outputs[0].token_ids) <= max_tokens
        assert output.outputs[0].finish_reason in ["stop", "length"]


def test_batch_vs_sequential_consistency(engine):
    """Test that batch and sequential inference produce same results.
    
    With greedy sampling (temperature=0), batch and sequential should
    produce identical outputs.
    """
    prompts = [
        "What is the capital of France?",
        "How does gravity work?",
    ]
    
    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0,  # Greedy
        seed=42,
    )
    
    # Batch inference
    batch_outputs = engine.generate_batch(prompts, sampling_params)
    
    # Sequential inference
    sequential_outputs = {}
    for prompt in prompts:
        output = engine.generate(prompt, sampling_params)
        sequential_outputs[output.request_id] = output
    
    # Compare outputs (token IDs should match with greedy sampling)
    batch_tokens = [
        output.outputs[0].token_ids
        for output in batch_outputs.values()
    ]
    sequential_tokens = [
        output.outputs[0].token_ids
        for output in sequential_outputs.values()
    ]
    
    # With continuous batching, the order might be different
    # but the token sequences should be the same for each prompt
    # Sort both lists for comparison
    batch_tokens_sorted = sorted([tuple(t) for t in batch_tokens])
    sequential_tokens_sorted = sorted([tuple(t) for t in sequential_tokens])
    
    assert batch_tokens_sorted == sequential_tokens_sorted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

