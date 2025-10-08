# FoloVLLM - 轻量级 LLM 推理框架

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

一个模仿 vLLM 设计的轻量级大语言模型推理框架，专注于教学和理解现代 LLM 推理优化技术。

## 🎯 项目目标

FoloVLLM 旨在通过渐进式开发，实现一个**可理解、可复现**的 LLM 推理框架，涵盖以下核心技术：

- ✅ **离线推理** - 基础推理流程
- ✅ **连续批处理** (Continuous Batching) - 动态批处理调度  
- ✅ **Paged KV Cache** - PagedAttention 内存优化
- ✅ **Flash Attention** - 高效 attention 计算
- ✅ **Chunked Prefill** - 分块预填充
- ✅ **前缀复用** (Prefix Caching) - 共享前缀优化
- ✅ **GPTQ 量化** - 4-bit 量化推理

## 🚀 快速开始

### 1. 环境设置

**一键设置（推荐）:**

```bash
# Linux / macOS
bash setup_env.sh

# Windows
setup_env.bat
```

**手动设置:**

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate.bat  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

📖 详细说明: [环境设置指南](ENVIRONMENT_SETUP.md)

### 2. 运行示例

```bash
# M0 基础功能演示
python examples/m0_basic_usage.py

# M1 推理示例
python examples/m1_inference.py \
    --model Qwen/Qwen2.5-0.5B \
    --prompt "What is the capital of France?" \
    --max-tokens 50

# M2 批量推理示例（连续批处理）
python examples/m2_inference.py \
    --model Qwen/Qwen2.5-0.5B \
    --num-prompts 5 \
    --max-tokens 64 \
    --compare-sequential

# 运行测试
pytest tests/unit/test_m0_*.py -v
pytest tests/unit/test_m1_*.py -v
pytest tests/unit/test_m2_*.py -v
pytest tests/integration/test_m2_e2e.py -v
```

### 3. 基础使用

```python
from folovllm import LLMEngine, ModelConfig, SamplingParams

# 创建配置
config = ModelConfig(
    model="Qwen/Qwen3-0.6B",
    dtype="float16",
    trust_remote_code=True
)

# 初始化引擎
engine = LLMEngine(config, device="cuda")

# M1: 单请求生成
sampling_params = SamplingParams(
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    max_tokens=100
)
output = engine.generate("你好，请介绍一下自己", sampling_params)
print(output.outputs[0].text)

# M2: 批量生成（连续批处理）
from folovllm import SchedulerConfig

scheduler_config = SchedulerConfig(
    max_num_seqs=256,
    max_num_batched_tokens=2048
)
engine = LLMEngine(config, scheduler_config=scheduler_config, device="cuda")

prompts = [
    "What is the capital of France?",
    "Explain quantum computing.",
    "Write a haiku about AI.",
]
outputs = engine.generate_batch(prompts, sampling_params)

for req_id, output in outputs.items():
    print(f"{output.prompt} -> {output.outputs[0].text}")
```

## 📚 开发路线

本项目采用**渐进式开发**，每个阶段都是上一阶段的超集：

| 阶段   | 功能            | 状态     | 文档                                                                                                                                                                   |
| ------ | --------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **M0** | 项目初始化      | ✅ 已完成 | [开发日志](docs/dev/milestone_0.md)                                                                                                                                    |
| **M1** | 基础离线推理    | ✅ 已完成 | [📖 总览](docs/milestone_1_index.md) · [学习笔记](docs/learn/milestone_1.md) · [口述展示](docs/presentation/milestone_1.md) · [面试指南](docs/interview/milestone_1.md) |
| **M2** | 连续批处理      | ✅ 已完成 | [开发日志](docs/dev/milestone_2.md)                                                                                                                                    |
| **M3** | Paged KV Cache  | ⏳ 待开始 | [学习笔记](docs/learn/03_paged_kv_cache.md)                                                                                                                            |
| **M4** | Flash Attention | ⏳ 待开始 | [学习笔记](docs/learn/04_flash_attention.md)                                                                                                                           |
| **M5** | Chunked Prefill | ⏳ 待开始 | [学习笔记](docs/learn/05_chunked_prefill.md)                                                                                                                           |
| **M6** | 前缀复用        | ⏳ 待开始 | [学习笔记](docs/learn/06_prefix_caching.md)                                                                                                                            |
| **M7** | GPTQ 量化       | ⏳ 待开始 | [学习笔记](docs/learn/07_gptq_quantization.md)                                                                                                                         |

📖 **完整开发计划**: [development_plan.md](docs/development_plan.md)

## 🏗️ 项目结构

> **设计原则**: 项目结构与 [vLLM v1](https://github.com/vllm-project/vllm) 源码完全对齐，便于学习和参考

```
folovllm/
├── folovllm/                  # 核心包（对齐 vllm.v1）
│   ├── request.py            # 请求定义
│   ├── outputs.py            # 输出格式
│   ├── config.py             # 配置管理
│   ├── core/                 # 核心组件
│   │   ├── sched/           # 调度器
│   │   ├── block_pool.py    # Block Pool
│   │   └── kv_cache_manager.py  # KV Cache 管理
│   ├── engine/              # 推理引擎
│   ├── model_executor/      # 模型执行
│   ├── attention/           # Attention 实现
│   ├── sample/              # 采样
│   ├── worker/              # Worker
│   └── executor/            # 执行器
├── tests/                   # 测试
├── docs/                    # 文档
│   ├── project_structure.md # 📋 结构详解
│   ├── learn/              # 学习笔记
│   └── dev/                # 开发日志
└── examples/               # 示例代码
```

📖 详细结构说明: [project_structure.md](docs/project_structure.md)

## 💡 核心技术

### 1. Continuous Batching
动态调度多个请求，实现高吞吐量推理。

### 2. PagedAttention
使用分页内存管理 KV Cache，内存利用率提升至接近 100%。

### 3. Flash Attention
优化的 attention 算法，降低 HBM 访问，提升计算效率。

### 4. Chunked Prefill
将长 prefill 分块处理，平衡首token延迟和吞吐量。

### 5. Prefix Caching
自动检测和复用共享前缀，加速 few-shot 和多轮对话。

### 6. GPTQ Quantization
4-bit 权重量化，降低显存占用，提升推理速度。

## 📊 性能指标

### 目标性能 (Qwen3-0.6B on A100)

| 优化阶段            | 吞吐量     | 延迟 (TTFT) | 显存占用 |
| ------------------- | ---------- | ----------- | -------- |
| M1: 基础推理        | 基线       | 基线        | 基线     |
| M2: 连续批处理      | 3-5x ↑     | -           | -        |
| M3: Paged KV        | -          | -           | 2x ↓     |
| M4: Flash Attn      | 1.5-2x ↑   | 20-30% ↓    | -        |
| M5: Chunked Prefill | -          | 显著改善    | -        |
| M6: Prefix Cache    | -          | 3-10x ↓     | -        |
| M7: GPTQ            | 1.2-1.5x ↑ | -           | 4x ↓     |

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/

# 性能测试
pytest tests/benchmark/
```

## 📖 学习资源

每个阶段都包含详细的学习笔记，涵盖：
- ✨ 技术原理讲解
- 🔧 实现细节分析
- 💼 面试常见问题
- 📚 参考资料链接

查看 [docs/learn/](docs/learn/) 目录获取完整内容。

## 🔗 参考资料

### 论文
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

### 代码
- [vLLM Official Repository](https://github.com/vllm-project/vllm)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

## 🤝 贡献

本项目主要用于学习和教学目的。欢迎提出问题和建议！

## 📝 License

Apache 2.0 License

## 🙏 致谢

感谢 vLLM 团队的开源工作，为本项目提供了宝贵的参考。

---

**Current Status**: 🔄 Milestone 0 - 项目初始化中

查看 [开发计划](docs/development_plan.md) 了解详细进度。
