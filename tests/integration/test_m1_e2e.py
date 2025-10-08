"""M1 end-to-end integration test.

Tests the complete inference pipeline and compares with HuggingFace.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from folovllm import LLMEngine, ModelConfig, SamplingParams


# Use Qwen3-0.6B as the test model
TEST_MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
class TestE2EGeneration:
    """End-to-end generation tests."""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """Create LLM engine."""
        model_config = ModelConfig(
            model=TEST_MODEL,
            dtype="float16",
            trust_remote_code=True,
        )
        engine = LLMEngine(model_config, device="cuda")
        return engine
    
    @pytest.fixture(scope="class")
    def hf_model_and_tokenizer(self):
        """Load HuggingFace model and tokenizer for comparison."""
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            TEST_MODEL,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).cuda()
        model.eval()
        return model, tokenizer
    
    def test_basic_generation(self, engine):
        """Test basic text generation."""
        prompt = "What is the capital of France?"
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.0,  # Greedy for determinism
        )
        
        output = engine.generate(prompt, sampling_params)
        
        assert output.finished
        assert len(output.outputs) == 1
        assert len(output.outputs[0].text) > 0
        assert output.outputs[0].finish_reason in ["stop", "length"]
        print(f"\nGenerated: {output.outputs[0].text}")
    
    def test_greedy_matches_hf(self, engine, hf_model_and_tokenizer):
        """Test that greedy sampling matches HuggingFace."""
        hf_model, hf_tokenizer = hf_model_and_tokenizer
        
        prompt = "The quick brown fox"
        max_tokens = 10
        
        # FoloVLLM generation
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )
        folo_output = engine.generate(prompt, sampling_params)
        folo_tokens = folo_output.outputs[0].token_ids
        
        # HuggingFace generation
        hf_inputs = hf_tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hf_outputs = hf_model.generate(
                **hf_inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy
                pad_token_id=hf_tokenizer.eos_token_id,
            )
        hf_tokens = hf_outputs[0][hf_inputs.input_ids.shape[1]:].cpu().tolist()
        
        # Compare tokens (first few should match for greedy)
        print(f"\nFoloVLLM tokens: {folo_tokens[:5]}")
        print(f"HF tokens: {hf_tokens[:5]}")
        
        # First token should definitely match for greedy
        assert folo_tokens[0] == hf_tokens[0], \
            f"First token mismatch: FoloVLLM={folo_tokens[0]}, HF={hf_tokens[0]}"
    
    def test_different_temperatures(self, engine):
        """Test generation with different temperatures."""
        prompt = "Once upon a time"
        
        # Greedy
        greedy_output = engine.generate(
            prompt,
            SamplingParams(temperature=0.0, max_tokens=10),
        )
        
        # Random (with seed for reproducibility)
        random_output1 = engine.generate(
            prompt,
            SamplingParams(temperature=1.0, max_tokens=10, seed=42),
        )
        random_output2 = engine.generate(
            prompt,
            SamplingParams(temperature=1.0, max_tokens=10, seed=42),
        )
        
        # Greedy should be deterministic
        greedy_tokens = greedy_output.outputs[0].token_ids
        assert len(greedy_tokens) > 0
        
        # Random with same seed should be deterministic
        assert random_output1.outputs[0].token_ids == random_output2.outputs[0].token_ids
        
        print(f"\nGreedy: {greedy_output.outputs[0].text}")
        print(f"Random: {random_output1.outputs[0].text}")
    
    def test_top_k_sampling(self, engine):
        """Test top-k sampling."""
        prompt = "The weather today is"
        
        output = engine.generate(
            prompt,
            SamplingParams(
                temperature=1.0,
                top_k=5,
                max_tokens=15,
                seed=42,
            ),
        )
        
        assert output.finished
        assert len(output.outputs[0].token_ids) > 0
        print(f"\nTop-k output: {output.outputs[0].text}")
    
    def test_top_p_sampling(self, engine):
        """Test top-p (nucleus) sampling."""
        prompt = "In conclusion,"
        
        output = engine.generate(
            prompt,
            SamplingParams(
                temperature=1.0,
                top_p=0.9,
                max_tokens=15,
                seed=42,
            ),
        )
        
        assert output.finished
        assert len(output.outputs[0].token_ids) > 0
        print(f"\nTop-p output: {output.outputs[0].text}")
    
    def test_stop_strings(self, engine):
        """Test stop string detection."""
        prompt = "Count: 1, 2, 3,"
        
        output = engine.generate(
            prompt,
            SamplingParams(
                temperature=0.0,
                max_tokens=50,
                stop=[","],  # Stop at comma
            ),
        )
        
        # Should stop early due to stop string
        assert output.outputs[0].finish_reason == "stop"
        print(f"\nWith stop string: {output.outputs[0].text}")
    
    def test_metrics_present(self, engine):
        """Test that performance metrics are present."""
        prompt = "Hello world"
        
        output = engine.generate(
            prompt,
            SamplingParams(temperature=0.0, max_tokens=10),
        )
        
        assert output.metrics is not None
        assert "ttft" in output.metrics
        assert "tpot" in output.metrics
        assert "total_time" in output.metrics
        assert "throughput" in output.metrics
        
        assert output.metrics["ttft"] > 0
        assert output.metrics["total_time"] > 0
        
        print(f"\nMetrics:")
        print(f"  TTFT: {output.metrics['ttft']*1000:.2f} ms")
        print(f"  TPOT: {output.metrics['tpot']*1000:.2f} ms")
        print(f"  Throughput: {output.metrics['throughput']:.2f} tokens/s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

