"""M2: Batch inference with continuous batching.

This example demonstrates M2's continuous batching capabilities:
- Processing multiple prompts concurrently
- Dynamic request scheduling
- Improved throughput compared to sequential processing
"""

import argparse
import time
from folovllm import LLMEngine, ModelConfig, SchedulerConfig, SamplingParams


# Default prompts for testing
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about artificial intelligence.",
    "What are the benefits of exercise?",
    "How does photosynthesis work?",
]


def main():
    parser = argparse.ArgumentParser(
        description="FoloVLLM M2: Batch Inference with Continuous Batching"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    
    # Batch arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="List of prompts (default: use built-in prompts)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to use from defaults (default: 5)",
    )
    
    # Generation arguments
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling",
    )
    
    # Scheduler arguments
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences in batch",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per iteration",
    )
    
    # Comparison mode
    parser.add_argument(
        "--compare-sequential",
        action="store_true",
        help="Also run sequential inference for comparison",
    )
    
    args = parser.parse_args()
    
    # Determine prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = DEFAULT_PROMPTS[:args.num_prompts]
    
    print("=" * 80)
    print("FoloVLLM M2: Batch Inference with Continuous Batching")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")
    print(f"Max batch size: {args.max_num_seqs}")
    print(f"Max batched tokens: {args.max_num_batched_tokens}")
    print("=" * 80)
    print()
    
    # Initialize configurations
    model_config = ModelConfig(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    
    scheduler_config = SchedulerConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    # Initialize engine
    print("Initializing LLM Engine with M2 continuous batching...")
    start_init = time.time()
    engine = LLMEngine(
        model_config=model_config,
        scheduler_config=scheduler_config,
        device=args.device,
    )
    init_time = time.time() - start_init
    print(f"Engine initialized in {init_time:.2f}s")
    print()
    
    # =========================================================================
    # Batch Inference (M2 - Continuous Batching)
    # =========================================================================
    print("=" * 80)
    print("Running BATCH inference (M2 - Continuous Batching)")
    print("=" * 80)
    print()
    
    # Display prompts
    print("Prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  [{i}] {prompt}")
    print()
    
    print("Generating (batched)...")
    print("-" * 80)
    
    batch_start = time.time()
    outputs = engine.generate_batch(prompts, sampling_params)
    batch_time = time.time() - batch_start
    
    print("-" * 80)
    print()
    
    # Display results
    print("Results:")
    print("=" * 80)
    total_tokens_batch = 0
    for i, (req_id, output) in enumerate(outputs.items()):
        completion = output.outputs[0]
        print(f"\n[{i}] Prompt: {output.prompt[:60]}...")
        print(f"    Generated: {completion.text}")
        print(f"    Tokens: {len(completion.token_ids)}")
        print(f"    Finish reason: {completion.finish_reason}")
        total_tokens_batch += len(completion.token_ids)
    
    print()
    print("=" * 80)
    print("Batch Inference Metrics:")
    print("-" * 80)
    print(f"  Total time: {batch_time:.2f}s")
    print(f"  Total tokens: {total_tokens_batch}")
    print(f"  Throughput: {total_tokens_batch/batch_time:.2f} tokens/s")
    print(f"  Average latency: {batch_time/len(prompts):.2f}s per request")
    print("=" * 80)
    print()
    
    # =========================================================================
    # Sequential Inference (M1 - for comparison)
    # =========================================================================
    if args.compare_sequential:
        print()
        print("=" * 80)
        print("Running SEQUENTIAL inference (M1 - for comparison)")
        print("=" * 80)
        print()
        
        sequential_start = time.time()
        total_tokens_sequential = 0
        
        for i, prompt in enumerate(prompts):
            print(f"Generating [{i+1}/{len(prompts)}]...")
            output = engine.generate(prompt, sampling_params)
            total_tokens_sequential += len(output.outputs[0].token_ids)
        
        sequential_time = time.time() - sequential_start
        
        print()
        print("=" * 80)
        print("Sequential Inference Metrics:")
        print("-" * 80)
        print(f"  Total time: {sequential_time:.2f}s")
        print(f"  Total tokens: {total_tokens_sequential}")
        print(f"  Throughput: {total_tokens_sequential/sequential_time:.2f} tokens/s")
        print(f"  Average latency: {sequential_time/len(prompts):.2f}s per request")
        print("=" * 80)
        print()
        
        # Comparison
        speedup = sequential_time / batch_time
        throughput_improvement = (
            (total_tokens_batch/batch_time) / (total_tokens_sequential/sequential_time)
        )
        
        print()
        print("=" * 80)
        print("COMPARISON: Batch vs Sequential")
        print("=" * 80)
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Throughput improvement: {throughput_improvement:.2f}x")
        print(f"  Time saved: {sequential_time - batch_time:.2f}s")
        print("=" * 80)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()

