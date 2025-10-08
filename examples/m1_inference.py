"""M1: Basic offline inference example.

This example demonstrates the basic inference capabilities of FoloVLLM M1:
- Single prompt generation
- Various sampling strategies (greedy, top-k, top-p, temperature)
- Performance metrics (TTFT, TPOT, throughput)
"""

import argparse
import time
from folovllm import LLMEngine, ModelConfig, SamplingParams


def main():
    parser = argparse.ArgumentParser(
        description="FoloVLLM M1: Basic Inference Example"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu)",
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)",
    )
    
    # Sampling arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (0.0 = greedy, default: 0.6)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (-1 = disabled, default: 20)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("FoloVLLM M1: Basic Offline Inference")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    print()
    
    # Initialize model config
    model_config = ModelConfig(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    
    # Initialize engine
    print("Initializing LLM Engine...")
    start_init = time.time()
    engine = LLMEngine(model_config, device=args.device)
    init_time = time.time() - start_init
    print(f"Engine initialized in {init_time:.2f}s")
    print()
    
    # Create sampling params
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    
    # Generate
    print("Generating...")
    print("-" * 80)
    output = engine.generate(args.prompt, sampling_params)
    print("-" * 80)
    print()
    
    # Print results
    print("Results:")
    print("=" * 80)
    print(f"Prompt: {output.prompt}")
    print()
    print("Generated text:")
    for completion in output.outputs:
        print(f"  [{completion.index}] {completion.text}")
        print(f"      Tokens: {len(completion.token_ids)}")
        print(f"      Finish reason: {completion.finish_reason}")
    print()
    
    # Print metrics
    if output.metrics:
        print("Performance Metrics:")
        print("-" * 80)
        print(f"  Time to First Token (TTFT): {output.metrics['ttft']*1000:.2f} ms")
        print(f"  Time per Output Token (TPOT): {output.metrics['tpot']*1000:.2f} ms")
        print(f"  Total time: {output.metrics['total_time']:.2f} s")
        print(f"  Throughput: {output.metrics['throughput']:.2f} tokens/s")
        print()
    
    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()

