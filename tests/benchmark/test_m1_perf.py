"""M1 performance benchmark.

Measures baseline performance metrics:
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- Throughput (tokens/s)
- GPU memory usage

Compares with HuggingFace baseline.
"""

import time
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

from folovllm import LLMEngine, ModelConfig, SamplingParams


# Test configuration
TEST_MODEL = "Qwen/Qwen3-0.6B"
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "The quick brown fox jumps over the lazy dog.",
]
MAX_TOKENS = 50


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_folovllm():
    """Benchmark FoloVLLM."""
    print("=" * 80)
    print("Benchmarking FoloVLLM")
    print("=" * 80)
    
    # Initialize engine
    model_config = ModelConfig(
        model=TEST_MODEL,
        dtype="float16",
        trust_remote_code=True,
    )
    
    init_start = time.time()
    engine = LLMEngine(model_config, device="cuda")
    init_time = time.time() - init_start
    
    print(f"Initialization time: {init_time:.2f}s")
    print(f"GPU memory after init: {get_gpu_memory_mb():.2f} MB")
    print()
    
    # Benchmark generation
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for consistent results
        max_tokens=MAX_TOKENS,
    )
    
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"[{i+1}/{len(TEST_PROMPTS)}] Generating for: {prompt[:50]}...")
        
        output = engine.generate(prompt, sampling_params)
        
        metrics = output.metrics
        results.append({
            "prompt": prompt,
            "output": output.outputs[0].text,
            "num_tokens": len(output.outputs[0].token_ids),
            "ttft_ms": metrics["ttft"] * 1000,
            "tpot_ms": metrics["tpot"] * 1000,
            "total_time": metrics["total_time"],
            "throughput": metrics["throughput"],
        })
        
        print(f"  Tokens: {results[-1]['num_tokens']}")
        print(f"  TTFT: {results[-1]['ttft_ms']:.2f} ms")
        print(f"  TPOT: {results[-1]['tpot_ms']:.2f} ms")
        print(f"  Throughput: {results[-1]['throughput']:.2f} tokens/s")
        print()
    
    # Aggregate metrics
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    avg_tpot = sum(r["tpot_ms"] for r in results) / len(results)
    avg_throughput = sum(r["throughput"] for r in results) / len(results)
    
    print("=" * 80)
    print("FoloVLLM Aggregate Results:")
    print("=" * 80)
    print(f"Average TTFT: {avg_ttft:.2f} ms")
    print(f"Average TPOT: {avg_tpot:.2f} ms")
    print(f"Average Throughput: {avg_throughput:.2f} tokens/s")
    print(f"Peak GPU memory: {get_gpu_memory_mb():.2f} MB")
    print()
    
    return results, {
        "init_time": init_time,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "avg_throughput": avg_throughput,
        "gpu_memory_mb": get_gpu_memory_mb(),
    }


def benchmark_huggingface():
    """Benchmark HuggingFace baseline."""
    print("=" * 80)
    print("Benchmarking HuggingFace Baseline")
    print("=" * 80)
    
    # Load model
    init_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda()
    model.eval()
    init_time = time.time() - init_start
    
    print(f"Initialization time: {init_time:.2f}s")
    print(f"GPU memory after init: {get_gpu_memory_mb():.2f} MB")
    print()
    
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"[{i+1}/{len(TEST_PROMPTS)}] Generating for: {prompt[:50]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.eos_token_id,
            )
        total_time = time.time() - start_time
        
        output_tokens = outputs[0][prompt_len:].cpu().tolist()
        num_tokens = len(output_tokens)
        
        # Rough estimates (HF doesn't provide per-token timing)
        # Assume first token takes 50% of total time, rest divided equally
        est_ttft = total_time * 0.5
        est_tpot = (total_time - est_ttft) / max(num_tokens - 1, 1) if num_tokens > 1 else 0
        throughput = num_tokens / total_time if total_time > 0 else 0
        
        results.append({
            "prompt": prompt,
            "num_tokens": num_tokens,
            "ttft_ms": est_ttft * 1000,
            "tpot_ms": est_tpot * 1000,
            "total_time": total_time,
            "throughput": throughput,
        })
        
        print(f"  Tokens: {num_tokens}")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Throughput: {throughput:.2f} tokens/s")
        print()
    
    # Aggregate metrics
    avg_throughput = sum(r["throughput"] for r in results) / len(results)
    
    print("=" * 80)
    print("HuggingFace Aggregate Results:")
    print("=" * 80)
    print(f"Average Throughput: {avg_throughput:.2f} tokens/s")
    print(f"Peak GPU memory: {get_gpu_memory_mb():.2f} MB")
    print()
    
    return results, {
        "init_time": init_time,
        "avg_throughput": avg_throughput,
        "gpu_memory_mb": get_gpu_memory_mb(),
    }


def main():
    """Run benchmarks."""
    if not torch.cuda.is_available():
        print("GPU not available, skipping benchmark")
        return
    
    print("M1 Performance Benchmark")
    print(f"Model: {TEST_MODEL}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print(f"Max tokens: {MAX_TOKENS}")
    print()
    
    # Benchmark FoloVLLM
    folo_results, folo_summary = benchmark_folovllm()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Benchmark HuggingFace
    hf_results, hf_summary = benchmark_huggingface()
    
    # Comparison
    print("=" * 80)
    print("Comparison: FoloVLLM vs HuggingFace")
    print("=" * 80)
    print(f"FoloVLLM throughput: {folo_summary['avg_throughput']:.2f} tokens/s")
    print(f"HuggingFace throughput: {hf_summary['avg_throughput']:.2f} tokens/s")
    
    if hf_summary['avg_throughput'] > 0:
        ratio = folo_summary['avg_throughput'] / hf_summary['avg_throughput']
        print(f"Relative performance: {ratio:.2f}x")
    
    print()
    print(f"FoloVLLM GPU memory: {folo_summary['gpu_memory_mb']:.2f} MB")
    print(f"HuggingFace GPU memory: {hf_summary['gpu_memory_mb']:.2f} MB")
    print()
    
    print("Note: This is a baseline benchmark for M1.")
    print("Future milestones will add optimizations:")
    print("  - M2: Continuous batching")
    print("  - M3: Paged attention")
    print("  - M4: Flash Attention")
    print("=" * 80)


if __name__ == "__main__":
    main()

