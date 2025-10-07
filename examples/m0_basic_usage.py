"""
Milestone 0 示例：基础配置和模型加载

这个示例展示了如何使用 M0 完成的功能：
1. 创建配置
2. 加载模型和 tokenizer
3. 创建请求和序列
4. 使用工具函数
"""

import torch

from folovllm import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    Request,
    SamplingParams,
    SchedulerConfig,
)
from folovllm.model_loader import get_model_and_tokenizer
from folovllm.utils import (
    is_cuda_available,
    get_gpu_memory_info,
    get_device,
    set_random_seed,
    generate_request_id,
)

def main():
    print("=" * 60)
    print("FoloVLLM Milestone 0 - 基础使用示例")
    print("=" * 60)
    print()
    
    # 1. 检查 CUDA 可用性
    print("1. 检查设备")
    print(f"   CUDA 可用: {is_cuda_available()}")
    device = "cuda" if is_cuda_available() else "cpu"
    print(f"   使用设备: {device}")
    
    if is_cuda_available():
        mem_info = get_gpu_memory_info()
        print(f"   GPU 显存: {mem_info['total_gb']:.2f} GB (总)")
        print(f"            {mem_info['free_gb']:.2f} GB (可用)")
    print()
    
    # 2. 创建配置
    print("2. 创建配置")
    
    # 模型配置
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        dtype="float16" if device == "cuda" else "float32",
        trust_remote_code=True,
        max_model_len=2048,
    )
    print(f"   ModelConfig: {model_config.model}")
    print(f"   - dtype: {model_config.dtype}")
    print(f"   - max_model_len: {model_config.max_model_len}")
    
    # 缓存配置
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
    )
    print(f"   CacheConfig: block_size={cache_config.block_size}")
    
    # 调度器配置
    scheduler_config = SchedulerConfig(
        max_num_seqs=256,
    )
    print(f"   SchedulerConfig: max_num_seqs={scheduler_config.max_num_seqs}")
    
    # 引擎配置
    engine_config = EngineConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
    )
    print(f"   EngineConfig 创建成功")
    print()
    
    # 3. 加载模型和 tokenizer（可选，需要下载模型）
    print("3. 加载模型和 Tokenizer")
    print("   注意: 首次运行会下载模型（约 1.2GB）")
    
    try:
        model, tokenizer = get_model_and_tokenizer(model_config, device=device)
        print(f"   ✓ 模型加载成功")
        print(f"   ✓ Tokenizer 加载成功")
        print(f"   - Vocab size: {tokenizer.vocab_size}")
        
        # 测试 tokenization
        test_text = "你好，世界！"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"   - 测试编码: '{test_text}' -> {tokens[:10]}...")
        print(f"   - 测试解码: {tokens[:10]}... -> '{decoded}'")
        
    except Exception as e:
        print(f"   ⚠ 模型加载失败（可能需要下载）: {e}")
        print(f"   跳过模型加载步骤")
        
        # 创建模拟 tokenizer 用于后续演示
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3, 4, 5]
        
        tokenizer = MockTokenizer()
    
    print()
    
    # 4. 创建采样参数
    print("4. 创建采样参数")
    
    # Greedy sampling
    greedy_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
    )
    print(f"   Greedy: temperature={greedy_params.temperature}")
    
    # Random sampling with top-p
    random_params = SamplingParams(
        n=1,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        max_tokens=100,
    )
    print(f"   Random: temperature={random_params.temperature}, "
          f"top_p={random_params.top_p}, top_k={random_params.top_k}")
    
    # Multiple sequences
    multi_params = SamplingParams(
        n=3,
        best_of=5,
        temperature=0.7,
        max_tokens=80,
    )
    print(f"   Multi: n={multi_params.n}, best_of={multi_params.best_of}")
    print()
    
    # 5. 创建请求
    print("5. 创建请求")
    
    prompt = "你好，请介绍一下自己"
    prompt_token_ids = tokenizer.encode(prompt)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 生成请求 ID
    request_id = generate_request_id()
    print(f"   Request ID: {request_id}")
    
    # 创建请求
    request = Request(
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        sampling_params=multi_params,
    )
    
    print(f"   Prompt: '{request.prompt}'")
    print(f"   Prompt tokens: {len(request.prompt_token_ids)} tokens")
    print(f"   Status: {request.status.value}")
    print(f"   Sequences: {len(request.sequences)} (best_of={multi_params.best_of})")
    print()
    
    # 6. 序列操作
    print("6. 序列操作")
    
    # 获取所有序列
    sequences = request.get_seqs()
    print(f"   总序列数: {len(sequences)}")
    
    # 访问第一个序列
    seq = sequences[0]
    print(f"   序列 0:")
    print(f"   - seq_id: {seq.seq_id}")
    print(f"   - 长度: {seq.get_len()} tokens")
    print(f"   - 状态: {seq.status.value}")
    
    # 模拟添加 output token
    seq.add_token_id(100)
    seq.add_token_id(101)
    print(f"   - 添加 2 个 token 后长度: {seq.get_len()}")
    print(f"   - 最后一个 token: {seq.get_last_token_id()}")
    
    # 序列 fork（用于 beam search）
    forked_seq = seq.fork(new_seq_id=f"{seq.seq_id}-fork")
    print(f"   - Fork 序列: {forked_seq.seq_id}")
    print(f"   - Fork 后长度: {forked_seq.get_len()}")
    print()
    
    # 7. 总结
    print("=" * 60)
    print("✅ Milestone 0 基础功能演示完成！")
    print("=" * 60)
    print()
    print("已实现的功能：")
    print("  ✓ 配置系统 (ModelConfig, CacheConfig, SchedulerConfig)")
    print("  ✓ 采样参数 (SamplingParams)")
    print("  ✓ 请求和序列管理 (Request, Sequence)")
    print("  ✓ 模型加载 (HuggingFace)")
    print("  ✓ 工具函数 (随机种子、设备管理、显存监控)")
    print()
    print("下一步: Milestone 1 - 基础离线推理")
    print("  - 实现 LLMEngine")
    print("  - 实现 Token 生成（Greedy, Top-k, Top-p）")
    print("  - 实现简单 KV Cache")
    print("  - 端到端推理流程")
    print()


if __name__ == "__main__":
    main()

