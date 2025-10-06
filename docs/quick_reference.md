# FoloVLLM 快速参考卡片

## 🚀 一分钟快速开始

```bash
# 安装
pip install -e .

# 基础推理
from folovllm import LLM
llm = LLM(model="Qwen/Qwen2.5-0.6B")
output = llm.generate("你好")[0]
print(output.text)
```

---

## 📊 Milestone 速查表

| 阶段   | 功能            | 核心文件（对齐 vLLM v1）                | 关键概念           | 性能提升         |
| ------ | --------------- | --------------------------------------- | ------------------ | ---------------- |
| **M0** | 项目初始化      | `config.py`, `request.py`               | 配置管理, 请求定义 | -                |
| **M1** | 基础推理        | `engine/llm_engine.py`                  | KV Cache, Sampling | Baseline         |
| **M2** | 连续批处理      | `core/sched/scheduler.py`               | Dynamic Batching   | 吞吐 3-5x ↑      |
| **M3** | Paged KV        | `core/block_pool.py`                    | PagedAttention     | 显存利用率 100%  |
| **M4** | Flash Attn      | `attention/backends/flash_attn.py`      | IO-aware           | 速度 1.5-2x ↑    |
| **M5** | Chunked Prefill | `core/sched/scheduler.py`               | 混合调度           | TTFT 显著 ↓      |
| **M6** | 前缀复用        | `core/kv_cache_manager.py`              | Trie, COW          | 缓存命中 3-10x ↓ |
| **M7** | GPTQ            | `model_executor/layers/quantization.py` | 4-bit 量化         | 显存 75% ↓       |

---

## 🔧 核心 API

### LLM 初始化

```python
from folovllm import LLM
from folovllm.config import EngineConfig

config = EngineConfig(
    # M3: Paged KV
    enable_paged_kv=True,
    block_size=16,
    
    # M4: Flash Attention
    attention_backend="flash",
    
    # M5: Chunked Prefill
    enable_chunked_prefill=True,
    max_chunk_size=512,
    
    # M6: Prefix Caching
    enable_prefix_caching=True,
    
    # M7: GPTQ
    quantization="gptq"
)

llm = LLM(model="Qwen/Qwen2.5-0.6B", engine_config=config)
```

### Sampling 参数

```python
from folovllm.sampling_params import SamplingParams

params = SamplingParams(
    temperature=0.7,     # 随机性 (0=确定, 1=随机)
    top_p=0.9,          # nucleus sampling
    top_k=50,           # top-k sampling
    max_tokens=100,     # 最大生成 token 数
    repetition_penalty=1.1,  # 重复惩罚
)

output = llm.generate(prompt, params)
```

### 批量推理

```python
prompts = ["问题1", "问题2", "问题3"]
outputs = llm.generate(prompts, params)

for i, output in enumerate(outputs):
    print(f"Prompt {i}: {output.prompt}")
    print(f"Output {i}: {output.text}")
```

---

## 📈 性能优化速查

### 提升吞吐量

```python
config = EngineConfig(
    enable_paged_kv=True,      # 允许更大 batch
    max_batch_size=64,         # 大 batch
    attention_backend="flash", # 加速计算
    max_chunk_size=1024,       # 大 chunk (prefill 吞吐)
)
```

### 降低延迟

```python
config = EngineConfig(
    attention_backend="flash",     # 加速
    enable_chunked_prefill=True,   # 减少阻塞
    max_chunk_size=256,            # 小 chunk (TTFT)
    enable_prefix_caching=True,    # 缓存加速
    max_batch_size=4,              # 小 batch (减少等待)
)
```

### 节省显存

```python
config = EngineConfig(
    enable_paged_kv=True,    # 零碎片
    block_size=16,           # 灵活分配
    quantization="gptq",     # 4-bit 权重
    max_model_len=2048,      # 限制长度
)
```

---

## 🔍 常用命令

### 开发

```bash
# 格式化
make format

# 检查
make lint

# 测试
make test

# 覆盖率
make coverage
```

### 性能测试

```bash
# 延迟测试
python tests/benchmark/latency_test.py --model Qwen/Qwen2.5-0.6B

# 吞吐量测试
python tests/benchmark/throughput_test.py --batch-size 16

# 对比不同 milestone
python tests/benchmark/compare.py --milestones m1,m2,m3
```

### 运行示例

```bash
# 基础推理
python examples/basic_inference.py

# 批量推理
python examples/batch_inference.py

# Few-shot
python examples/few_shot.py
```

---

## 🐛 调试技巧

### 详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

llm = LLM(model="...", log_level="DEBUG")
```

### 性能分析

```python
import torch.profiler as profiler

with profiler.profile(activities=[
    profiler.ProfilerActivity.CPU,
    profiler.ProfilerActivity.CUDA
]) as prof:
    llm.generate(prompt)

print(prof.key_averages().table())
```

### 显存追踪

```python
import torch

torch.cuda.reset_peak_memory_stats()
llm.generate(prompt)
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_mem:.2f} GB")
```

---

## 💡 关键数据结构

### Request

```python
@dataclass
class Request:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    arrival_time: float
```

### Sequence

```python
@dataclass
class Sequence:
    seq_id: str
    request_id: str
    token_ids: List[int]
    kv_blocks: List[KVBlock]
    status: SequenceStatus  # WAITING/RUNNING/FINISHED
```

### KVBlock

```python
@dataclass
class KVBlock:
    block_id: int          # 物理 block ID
    ref_count: int         # 引用计数
    block_hash: Optional[int]  # 前缀 hash (M6)
```

---

## 🎯 性能指标

### 延迟指标

- **TTFT**: Time to First Token (首 token 延迟)
- **TPOT**: Time Per Output Token (平均每 token 时间)
- **E2E**: End-to-End Latency (总延迟)

### 吞吐量指标

- **Tokens/s**: 每秒处理 token 数
- **Requests/s**: 每秒完成请求数

### 资源指标

- **Memory**: 显存占用
- **GPU Util**: GPU 利用率

---

## 📚 文档导航

### 规划文档
- [开发计划](development_plan.md) - 完整开发路线图
- [技术路线图](roadmap.md) - 技术演进路径
- [里程碑检查清单](milestone_checklist.md) - 完成标准

### 学习文档
- [学习笔记](learn/) - 技术原理深度讲解
- [面试准备](interview_guide.md) - 面试问答汇总
- [技术对比](technical_comparison.md) - 性能对比分析

### 使用文档
- [快速开始](getting_started.md) - 安装和使用指南
- [贡献指南](../CONTRIBUTING.md) - 开发规范
- [项目总结](project_summary.md) - 项目概览

### 开发文档
- [开发日志](dev/) - 各阶段实现细节
- [API 文档](api/) - 自动生成的 API 文档

---

## 🔑 关键概念速记

### KV Cache
- **作用**: 避免重复计算历史 token
- **实现**: 存储 Key/Value，增量更新
- **问题**: 显存占用大

### PagedAttention
- **思想**: 虚拟内存管理
- **优势**: 零碎片，高利用率
- **实现**: Block Pool + Block Table

### Continuous Batching
- **原理**: Iteration-level scheduling
- **优势**: 动态批处理，高吞吐
- **实现**: 动态添加/移除序列

### Flash Attention
- **优化**: IO-aware, Tiling
- **效果**: 减少 HBM 访问，2-4x 快
- **实现**: Kernel fusion, Recomputation

### Chunked Prefill
- **目的**: 平衡 TTFT 和吞吐
- **方法**: Prefill 分块，与 Decode 混合
- **权衡**: Chunk size 选择

### Prefix Caching
- **应用**: Few-shot, 多轮对话
- **实现**: Trie 匹配 + COW
- **效果**: 缓存命中 10x 快

### GPTQ
- **目标**: 4-bit 量化，保持精度
- **算法**: Hessian-based
- **效果**: 显存 75% ↓, 精度损失 < 1%

---

## 🚨 常见问题快速解决

| 问题           | 原因            | 解决方案                            |
| -------------- | --------------- | ----------------------------------- |
| **CUDA OOM**   | 显存不足        | ↓batch size / 启用 Paged KV / GPTQ  |
| **速度慢**     | 计算效率低      | 启用 Flash Attn / ↑batch size       |
| **TTFT 高**    | Prefill 阻塞    | 启用 Chunked Prefill / Prefix Cache |
| **输出质量差** | 参数不当        | 调整 temperature / top_p            |
| **缓存不生效** | 未启用/前缀不同 | 检查配置 / 确认 token 序列          |

---

## ✅ 开发 Checklist

### 开始新 Milestone
- [ ] 阅读开发计划和上一阶段日志
- [ ] 创建功能分支
- [ ] 设计接口和数据结构
- [ ] 编写测试用例

### 完成 Milestone
- [ ] 所有测试通过
- [ ] 性能测试完成
- [ ] 学习笔记编写
- [ ] 开发日志记录
- [ ] README 更新
- [ ] 提交 PR

---

## 🎓 推荐学习路径

1. **基础知识** (1-2 天)
   - Transformer 架构
   - Attention 机制
   - 自回归生成

2. **M0-M1** (3-5 天)
   - 项目初始化
   - 基础推理流程
   - KV Cache 原理

3. **M2-M3** (7-10 天)
   - 动态批处理
   - PagedAttention
   - 内存管理

4. **M4-M5** (5-7 天)
   - Flash Attention
   - Chunked Prefill
   - 性能优化

5. **M6-M7** (5-7 天)
   - 前缀复用
   - GPTQ 量化
   - 综合优化

**总计**: ~6 周

---

## 📞 获取帮助

- **文档**: 查看 `docs/` 目录
- **示例**: 运行 `examples/` 代码
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**持续更新中... 🚀**

