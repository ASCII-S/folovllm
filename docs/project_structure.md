# FoloVLLM 项目结构说明

> **设计原则**: 项目结构与 vLLM v1 源码完全对齐，便于学习和参考

---

## 📁 目录结构

```
folovllm/
├── folovllm/                    # 核心包（对齐 vllm.v1）
│   ├── __init__.py
│   ├── request.py              # 请求和序列定义
│   ├── outputs.py              # 输出格式定义
│   ├── config.py               # 配置类
│   │
│   ├── core/                   # 核心组件
│   │   ├── __init__.py
│   │   ├── block_pool.py       # M3: Block Pool 管理
│   │   ├── kv_cache_manager.py # M3: KV Cache 管理器
│   │   ├── kv_cache_utils.py   # M3: KV Cache 工具
│   │   └── sched/              # M2: 调度器
│   │       ├── __init__.py
│   │       ├── scheduler.py    # 主调度器
│   │       ├── request_queue.py # 请求队列
│   │       ├── interface.py    # 调度接口
│   │       └── output.py       # 调度输出
│   │
│   ├── engine/                 # 推理引擎
│   │   ├── __init__.py
│   │   ├── llm_engine.py       # M1: LLM 引擎（用户接口）
│   │   ├── core.py             # M2: 核心引擎逻辑
│   │   └── processor.py        # M1: 输入处理器
│   │
│   ├── model_executor/         # 模型执行
│   │   ├── __init__.py
│   │   ├── models/             # 模型实现
│   │   │   ├── __init__.py
│   │   │   ├── qwen.py         # M1: Qwen 模型
│   │   │   └── utils.py        # 模型工具
│   │   └── layers/             # 模型层
│   │       ├── __init__.py
│   │       ├── attention.py    # M1: Attention 层
│   │       └── quantization.py # M7: 量化层
│   │
│   ├── attention/              # Attention 实现
│   │   ├── __init__.py
│   │   ├── ops.py              # M1: Attention 操作
│   │   └── backends/           # Attention 后端
│   │       ├── __init__.py
│   │       ├── abstract.py     # 抽象接口
│   │       ├── torch_naive.py  # M1: 朴素实现
│   │       ├── paged.py        # M3: PagedAttention
│   │       └── flash_attn.py   # M4: Flash Attention
│   │
│   ├── sample/                 # 采样
│   │   ├── __init__.py
│   │   ├── sampler.py          # M1: 采样器
│   │   ├── ops/                # 采样操作
│   │   │   ├── __init__.py
│   │   │   ├── topk_topp.py    # Top-k/Top-p
│   │   │   └── penalties.py    # 惩罚项
│   │   └── logits_processor/   # Logits 处理
│   │       ├── __init__.py
│   │       └── interface.py    # 处理器接口
│   │
│   ├── worker/                 # Worker 实现
│   │   ├── __init__.py
│   │   ├── worker_base.py      # Worker 基类
│   │   ├── gpu_worker.py       # M1: GPU Worker
│   │   ├── model_runner.py     # M1: 模型运行器
│   │   └── input_batch.py      # M2: 输入批处理
│   │
│   ├── executor/               # 执行器
│   │   ├── __init__.py
│   │   └── gpu_executor.py     # M1: GPU 执行器
│   │
│   ├── metrics/                # 指标统计
│   │   ├── __init__.py
│   │   └── stats.py            # 性能统计
│   │
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       └── common.py           # 通用工具
│
├── tests/                      # 测试
│   ├── unit/                   # 单元测试
│   │   ├── test_m1_*.py
│   │   ├── test_m2_*.py
│   │   └── ...
│   ├── integration/            # 集成测试
│   │   └── test_e2e.py
│   └── benchmark/              # 性能测试
│       └── run_benchmark.py
│
├── examples/                   # 示例代码
│   ├── basic_inference.py
│   ├── batch_inference.py
│   └── advanced_usage.py
│
├── docs/                       # 文档
│   ├── learn/                  # 学习笔记
│   ├── dev/                    # 开发日志
│   └── api/                    # API 文档
│
└── reference/                  # vLLM 参考代码
    └── vllm/
```

---

## 🔗 与 vLLM v1 的对应关系

### 核心模块映射

| FoloVLLM                            | vLLM v1                            | 说明           |
| ----------------------------------- | ---------------------------------- | -------------- |
| `folovllm/request.py`               | `vllm/v1/request.py`               | 请求定义       |
| `folovllm/outputs.py`               | `vllm/v1/outputs.py`               | 输出格式       |
| `folovllm/config.py`                | `vllm/config.py`                   | 配置管理       |
| `folovllm/core/sched/scheduler.py`  | `vllm/v1/core/sched/scheduler.py`  | 调度器         |
| `folovllm/core/block_pool.py`       | `vllm/v1/core/block_pool.py`       | Block Pool     |
| `folovllm/core/kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | KV Cache 管理  |
| `folovllm/engine/llm_engine.py`     | `vllm/v1/engine/llm_engine.py`     | LLM 引擎       |
| `folovllm/worker/gpu_worker.py`     | `vllm/v1/worker/gpu_worker.py`     | GPU Worker     |
| `folovllm/sample/sampler.py`        | `vllm/v1/sample/sampler.py`        | 采样器         |
| `folovllm/attention/backends/`      | `vllm/v1/attention/backends/`      | Attention 后端 |

---

## 📝 各阶段文件开发计划

### M0: 项目初始化

**创建文件**:
- ✅ `folovllm/request.py` - 请求定义
- ✅ `folovllm/outputs.py` - 输出定义
- ✅ `folovllm/config.py` - 配置类
- ⏳ `folovllm/utils/common.py` - 通用工具

**参考**:
- `vllm/v1/request.py`
- `vllm/v1/outputs.py`
- `vllm/config.py`

---

### M1: 基础离线推理

**创建文件**:
- `folovllm/engine/llm_engine.py` - LLM 引擎
- `folovllm/engine/processor.py` - 输入处理
- `folovllm/model_executor/models/qwen.py` - Qwen 模型
- `folovllm/model_executor/layers/attention.py` - Attention 层
- `folovllm/attention/ops.py` - Attention 操作
- `folovllm/attention/backends/torch_naive.py` - 朴素实现
- `folovllm/sample/sampler.py` - 采样器
- `folovllm/sample/ops/topk_topp.py` - Top-k/p 采样
- `folovllm/worker/gpu_worker.py` - GPU Worker
- `folovllm/worker/model_runner.py` - 模型运行器
- `folovllm/executor/gpu_executor.py` - GPU 执行器

**参考**:
- `vllm/v1/engine/llm_engine.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/sample/sampler.py`

---

### M2: 连续批处理

**创建文件**:
- `folovllm/core/sched/scheduler.py` - 主调度器
- `folovllm/core/sched/request_queue.py` - 请求队列
- `folovllm/core/sched/interface.py` - 调度接口
- `folovllm/core/sched/output.py` - 调度输出
- `folovllm/engine/core.py` - 核心引擎
- `folovllm/worker/input_batch.py` - 输入批处理

**参考**:
- `vllm/v1/core/sched/scheduler.py` (68K 行，重点参考)
- `vllm/v1/core/sched/request_queue.py`
- `vllm/v1/engine/core.py`

---

### M3: Paged KV Cache

**创建文件**:
- `folovllm/core/block_pool.py` - Block Pool 管理
- `folovllm/core/kv_cache_manager.py` - KV Cache 管理器
- `folovllm/core/kv_cache_utils.py` - KV Cache 工具
- `folovllm/attention/backends/paged.py` - PagedAttention

**参考**:
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_utils.py` (53K 行，核心实现)

---

### M4: Flash Attention

**创建文件**:
- `folovllm/attention/backends/flash_attn.py` - Flash Attention 后端
- `folovllm/attention/backends/abstract.py` - 后端抽象接口

**参考**:
- `vllm/attention/backends/flash_attn.py`
- `vllm/attention/layer.py`

---

### M5: Chunked Prefill

**修改文件**:
- `folovllm/core/sched/scheduler.py` - 添加 chunked prefill 逻辑
- `folovllm/engine/core.py` - 支持混合调度

**参考**:
- `vllm/v1/core/sched/scheduler.py` 中的 chunked prefill 部分

---

### M6: 前缀复用

**创建文件**:
- `folovllm/core/prefix_cache.py` - 前缀缓存管理

**修改文件**:
- `folovllm/core/kv_cache_manager.py` - 添加前缀复用逻辑

**参考**:
- `vllm/core/block_manager_v2.py` 中的 prefix caching
- vLLM 设计文档: `docs/design/prefix_caching.md`

---

### M7: GPTQ 量化

**创建文件**:
- `folovllm/model_executor/layers/quantization.py` - 量化层
- `folovllm/model_executor/layers/linear.py` - 量化 Linear 层

**参考**:
- `vllm/model_executor/layers/quantization/gptq.py`
- `vllm/model_executor/layers/linear.py`

---

## 🔍 开发时的参考策略

### 1. 文件级对照

开发每个文件时：
```bash
# 1. 打开对应的 vLLM 源文件
code reference/vllm/vllm/v1/core/sched/scheduler.py

# 2. 在另一个窗口编辑 FoloVLLM 文件
code folovllm/core/sched/scheduler.py

# 3. 对照理解和简化实现
```

### 2. 接口对齐

确保关键接口与 vLLM 一致：
- 类名相同或相似
- 方法签名兼容
- 数据结构对齐

### 3. 简化原则

- **保留核心逻辑**: 调度算法、内存管理等
- **简化功能**: 移除多模态、LoRA、投机解码等
- **注释说明**: 标注简化的部分和原因

---

## 📚 学习路径

### 边开发边学习

1. **M1 阶段**: 
   - 重点看 `engine/llm_engine.py`
   - 理解基础推理流程

2. **M2 阶段**:
   - 精读 `core/sched/scheduler.py`
   - 理解调度算法

3. **M3 阶段**:
   - 深入 `core/block_pool.py`
   - 深入 `core/kv_cache_manager.py`
   - 理解 PagedAttention

4. **M4 阶段**:
   - 研究 `attention/backends/`
   - 理解不同 backend 的设计

---

## ✅ 优势

### 1. 易于参考

```python
# 开发时可以直接对照
# vLLM:
from vllm.v1.core.sched.scheduler import Scheduler

# FoloVLLM:
from folovllm.core.sched.scheduler import Scheduler
```

### 2. 结构清晰

- 目录对应，快速定位
- 文件对应，直接参考
- 接口对齐，易于理解

### 3. 深度学习

- 逐个模块对照学习
- 理解设计思想
- 掌握实现细节

---

## 📖 文档更新

所有文档中的文件路径已更新为新结构：
- ✅ [开发计划](development_plan.md)
- ✅ [技术路线图](roadmap.md)
- ✅ [快速参考](quick_reference.md)
- ✅ [里程碑检查清单](milestone_checklist.md)

---

**结构已完全对齐 vLLM v1，可以开始愉快地开发和学习了！** 🚀

