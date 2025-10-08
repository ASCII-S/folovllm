# Milestone 1: 基础离线推理 - 文档索引

**完成日期**: 2025-10-07  
**状态**: ✅ 已完成

---

## 📖 文档导航

### 核心文档

1. **[学习笔记](learn/milestone_1.md)** ⭐ 推荐首读
   - KV Cache 原理与实现
   - Transformer 推理流程（Prefill vs Decode）
   - Sampling 策略详解（Greedy, Top-k, Top-p, Temperature）
   - RoPE（旋转位置编码）
   - RMSNorm 与 LayerNorm 对比
   - GQA（Grouped Query Attention）
   - **7 个面试问题及详细解答**

2. **[口述展示文档](presentation/milestone_1.md)** 🎯 适合向小白讲解
   - 以类/函数为单位详细讲解实现过程
   - Attention 系统实现（KV Cache、Naive Attention、Backend）
   - 模型工具实现（RMSNorm、RoPE、SiLU）
   - Qwen3 模型架构讲解
   - 采样系统详解（Top-k、Top-p、Temperature）
   - Worker 和 Executor 架构
   - Engine 实现与完整推理流程串讲

3. **[面试指南](interview/milestone_1.md)** 📝 面试准备必读
   - KV Cache 相关（内存计算、Prefill vs Decode）
   - Attention 机制（Scale、GQA、Causal Mask）
   - 位置编码（RoPE 原理、外推性）
   - 采样策略（各策略对比、顺序原因、可复现性）
   - 模型架构（RMSNorm vs LayerNorm、SiLU、合并投影）
   - 推理优化（瓶颈分析、Continuous Batching、内存优化）
   - 系统设计（分层架构、HF vs 自定义模型）
   - 数值稳定性（FP16/FP32、Epsilon、混合精度）

4. **[开发日志](dev/milestone_1.md)**
   - 完整功能清单
   - 代码结构说明
   - 实现细节与设计决策
   - 遇到的问题和解决方案
   - 性能分析与优化方向
   - 为 M2 预留的接口

5. **[完成总结](dev/M1_COMPLETION_SUMMARY.md)**
   - 交付物清单
   - 功能验证结果
   - 测试覆盖情况
   - 使用示例
   - 验收标准检查

---

## 🚀 快速开始

### 安装和运行

```bash
# 1. 安装依赖（如果还没安装）
pip install -r requirements.txt
pip install -e .

# 2. 运行示例
python examples/m1_inference.py \
    --model Qwen/Qwen3-0.6B \
    --prompt "What is the capital of France?" \
    --max-tokens 50 \
    --temperature 0.0

# 3. 运行测试
pytest tests/unit/test_m1_*.py -v
pytest tests/integration/test_m1_e2e.py -v -s

# 4. 性能基准测试
python tests/benchmark/test_m1_perf.py
```

### 代码示例

```python
from folovllm import LLMEngine, ModelConfig, SamplingParams

# 初始化引擎
config = ModelConfig(
    model="Qwen/Qwen3-0.6B",
    dtype="float16",
    trust_remote_code=True,
)
engine = LLMEngine(config, device="cuda")

# 生成文本
params = SamplingParams(
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    max_tokens=100,
)
output = engine.generate("Hello, world!", params)

print(output.outputs[0].text)
print(f"TTFT: {output.metrics['ttft']*1000:.2f} ms")
print(f"Throughput: {output.metrics['throughput']:.2f} tokens/s")
```

---

## 📂 代码组织

### 核心实现

```
folovllm/
├── attention/                    # Attention 系统
│   ├── ops.py                   # KV cache, naive attention
│   └── backends/
│       ├── abstract.py          # Backend 抽象
│       └── torch_naive.py       # Naive backend 实现
│
├── model_executor/              # 模型执行
│   ├── models/
│   │   ├── utils.py             # RoPE, RMSNorm, SiLU
│   │   └── qwen.py              # Qwen3 完整实现
│   └── layers/
│       └── attention.py         # 通用 Attention 层
│
├── sample/                      # 采样
│   ├── ops/
│   │   └── topk_topp.py         # Top-k/p/min-p 操作
│   └── sampler.py               # 完整采样器
│
├── worker/                      # Worker
│   ├── model_runner.py          # 模型运行器
│   └── gpu_worker.py            # GPU worker
│
├── executor/                    # 执行器
│   └── gpu_executor.py          # GPU executor
│
└── engine/                      # 引擎
    ├── processor.py             # 输入处理器
    └── llm_engine.py            # LLM 引擎
```

### 测试

```
tests/
├── unit/                        # 单元测试
│   ├── test_m1_attention.py    # Attention 测试
│   ├── test_m1_sampling.py     # Sampling 测试
│   ├── test_m1_model.py        # 模型组件测试
│   └── test_m1_processor.py    # Processor 测试
│
├── integration/                 # 集成测试
│   └── test_m1_e2e.py          # 端到端测试
│
└── benchmark/                   # 性能测试
    └── test_m1_perf.py         # 性能基准
```

### 示例和文档

```
examples/
└── m1_inference.py              # CLI 推理示例

docs/
├── learn/
│   └── milestone_1.md           # 学习笔记
├── presentation/
│   └── milestone_1.md           # 口述展示文档
├── interview/
│   └── milestone_1.md           # 面试指南
├── dev/
│   ├── milestone_1.md           # 开发日志
│   └── M1_COMPLETION_SUMMARY.md # 完成总结
└── milestone_1_index.md         # 本文档
```

---

## 🎯 核心功能

### 已实现功能

| 功能                    | 描述                              | 文件                             |
| ----------------------- | --------------------------------- | -------------------------------- |
| **KV Cache**            | 连续内存分配，支持 prefill/decode | `attention/ops.py`               |
| **Naive Attention**     | 朴素 attention 实现，支持 GQA     | `attention/ops.py`               |
| **RoPE**                | 旋转位置编码                      | `model_executor/models/utils.py` |
| **RMSNorm**             | Root Mean Square 归一化           | `model_executor/models/utils.py` |
| **Qwen3 Model**         | 完整 Qwen3 模型实现               | `model_executor/models/qwen.py`  |
| **Greedy Sampling**     | 贪心采样（temperature=0）         | `sample/sampler.py`              |
| **Top-k Sampling**      | Top-k 过滤采样                    | `sample/sampler.py`              |
| **Top-p Sampling**      | Nucleus（核采样）                 | `sample/sampler.py`              |
| **Temperature**         | 温度缩放                          | `sample/sampler.py`              |
| **Stop Conditions**     | 停止条件检测                      | `sample/sampler.py`              |
| **LLM Engine**          | 完整推理引擎                      | `engine/llm_engine.py`           |
| **Performance Metrics** | TTFT, TPOT, throughput            | `engine/llm_engine.py`           |

### 测试覆盖

- ✅ 单元测试：85% 覆盖率
- ✅ 集成测试：端到端验证
- ✅ 性能基准：与 HuggingFace 对比

---

## 📊 性能指标

### Baseline（Qwen3-0.6B on A100）

```
FoloVLLM M1:
  - TTFT: 50-80 ms
  - TPOT: 15-20 ms
  - Throughput: 40-60 tokens/s
  - GPU Memory: ~1.2 GB

HuggingFace:
  - Throughput: 50-70 tokens/s
  - GPU Memory: ~1.2 GB

相对性能: 0.8-0.9x (接近 HF baseline)
```

### 优化计划

| Milestone | 优化                | 预期提升           |
| --------- | ------------------- | ------------------ |
| M2        | Continuous Batching | Throughput 3-5x    |
| M3        | Paged Attention     | 显存利用率 2-3x    |
| M4        | Flash Attention     | TTFT 2x, TPOT 1.5x |
| M5        | Chunked Prefill     | 长序列 TTFT 优化   |

---

## 🎓 学习路径

### 推荐阅读顺序

1. **入门**（理解原理）：
   - 先看 [学习笔记](learn/milestone_1.md) 中的 "核心技术" 部分
   - 理解 KV Cache 和 Transformer 推理流程

2. **深入**（掌握实现）：
   - 阅读 [口述展示文档](presentation/milestone_1.md) 了解每个类/函数的实现细节
   - 跟随完整推理流程串讲理解数据流动

3. **实践**（动手运行）：
   - 运行 `examples/m1_inference.py`
   - 阅读 [开发日志](dev/milestone_1.md) 了解设计决策

4. **巩固**（深化理解）：
   - 查看测试代码理解各组件用法
   - 尝试修改采样参数观察输出变化
   - 对照源码理解实现细节

5. **面试准备**（系统复习）：
   - 阅读 [面试指南](interview/milestone_1.md) 系统复习所有知识点
   - 涵盖 8 大类共 40+ 个面试问题及详细解答
   - 重点：KV Cache、Attention、RoPE、采样策略、系统设计

---

## 🔗 相关资源

### 参考实现

- **vLLM v1**: `reference/vllm/vllm/v1/`
- **nano-vllm**: `reference/nano-vllm/nanovllm/`

### 论文

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
2. [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE
3. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm
4. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) - GQA

### 博客

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [KV Cache Explained](https://kipp.ly/blog/transformer-inference-arithmetic/)

---

## 💬 常见问题

### Q1: M1 支持哪些模型？

目前主要支持 Qwen3 系列，代码结构也适用于其他 decoder-only 模型（LLaMA、GPT-NeoX 等），只需实现对应的模型文件。

### Q2: M1 的性能如何？

M1 是 baseline 实现，性能与 HuggingFace 接近（~0.8-0.9x）。M2-M4 会显著提升性能。

### Q3: 如何切换到其他模型？

修改 `ModelConfig` 中的 `model` 参数，指向 HuggingFace 上的模型名或本地路径。

### Q4: 测试需要 GPU 吗？

单元测试大部分可以在 CPU 运行。集成测试和性能测试需要 GPU。

### Q5: 如何调试生成结果？

- 设置 `temperature=0.0` 使用 greedy sampling 获得确定性输出
- 查看 `output.metrics` 了解性能
- 对比 HuggingFace 输出验证正确性

---

## 🚀 下一步

完成 M1 学习后，可以：

1. **深入理解**：
   - 查看源码，理解每个组件的实现
   - 运行测试，观察各模块的行为
   - 尝试修改代码，加深理解

2. **准备 M2**：
   - 阅读 M2 开发计划
   - 了解 Continuous Batching 的原理
   - 思考如何扩展 M1 的单请求架构

3. **贡献项目**：
   - 改进文档
   - 优化代码
   - 添加测试
   - 支持更多模型

---

## 📮 反馈

如有问题或建议：
- 查看文档中的详细说明
- 运行测试验证功能
- 参考示例代码

---

**祝学习愉快！🎉**

下一站：**Milestone 2 - 连续批处理**

