# FoloVLLM 渐进式开发计划

## 项目概述

FoloVLLM 是一个轻量级的大语言模型推理框架，模仿 vLLM 的核心设计思想，以 Qwen3-0.6B 模型为例实现完整的推理流程。项目采用渐进式开发，每个阶段都是上一阶段的超集。

## 技术栈

- **框架**: PyTorch 2.0+
- **推理优化**: CUDA, Flash Attention, PagedAttention
- **量化**: GPTQ
- **模型**: Qwen3-0.6B

---

## 开发阶段规划

### 阶段 0: 项目初始化 (Milestone 0)

**目标**: 搭建项目基础架构和开发环境

**核心任务**:
1. 项目结构设计
   - 创建核心模块目录结构
   - 配置管理系统
   - 日志系统
   
2. 模型加载基础设施
   - HuggingFace 模型加载器
   - 权重转换和管理
   - Qwen3-0.6B 模型适配

3. 基础数据结构
   - Request/Sequence 抽象
   - SamplingParams
   - 输出格式定义

**交付物**:
- 基础项目结构
- 模型加载代码
- 配置文件示例
- 开发文档: `docs/dev/milestone_0.md`

**测试**:
- 单元测试: 模型加载、配置解析
- 集成测试: 端到端模型加载验证

---

### 阶段 1: 基础离线推理 (Milestone 1)

**目标**: 实现最简单的单请求、单批次推理流程

**核心功能**:
1. **模型推理引擎**
   - 基础 LLMEngine 类
   - 单请求处理流程
   - Transformer forward pass
   
2. **Token 生成**
   - Greedy sampling
   - Top-k/Top-p sampling
   - Temperature scaling
   
3. **KV Cache 管理 (简单版)**
   - 连续内存分配
   - 基础的 cache 读写

**推理流程**:
```
Input Text -> Tokenization -> Model Forward -> Sampling -> Detokenization -> Output
```

**交付物**:
- 基础推理引擎
- 简单的 KV Cache 实现
- CLI 推理工具
- 学习笔记: `docs/learn/01_basic_inference.md`
- 开发日志: `docs/dev/milestone_1.md`

**测试**:
- 单元测试: Sampling 策略、Tokenization
- 集成测试: 端到端推理验证
- 性能基准: 单请求延迟和吞吐量

---

### 阶段 2: 连续批处理 (Milestone 2)

**目标**: 实现 Continuous Batching，支持动态请求处理

**核心功能**:
1. **调度器 (Scheduler)**
   - 请求队列管理
   - 动态批处理调度
   - 请求优先级处理
   
2. **批处理引擎**
   - 动态 batch 组装
   - 不同长度序列的 padding/masking
   - 完成序列的移除和新序列的添加
   
3. **请求生命周期管理**
   - Request 状态机 (WAITING -> RUNNING -> FINISHED)
   - 抢占和恢复机制 (基础版)

**关键优化**:
- Iteration-level batching
- 动态 batch size 调整
- 内存利用率优化

**交付物**:
- Scheduler 实现
- 连续批处理引擎
- 学习笔记: `docs/learn/02_continuous_batching.md`
- 开发日志: `docs/dev/milestone_2.md`

**测试**:
- 单元测试: Scheduler 逻辑、批处理组装
- 集成测试: 多请求并发处理
- 性能测试: 对比单批次和连续批处理的吞吐量

---

### 阶段 3: Paged KV Cache (Milestone 3)

**目标**: 实现 PagedAttention，大幅提升内存利用率

**核心功能**:
1. **Block Pool Manager**
   - 固定大小的 KV block 分配
   - Block 引用计数
   - 内存池管理
   
2. **KV Cache Manager**
   - 逻辑 block 到物理 block 的映射
   - Block 分配和回收
   - Copy-on-write 机制
   
3. **PagedAttention 算子**
   - 修改 attention 计算支持分页 KV
   - Block table 管理
   - 高效的 block 访问

**内存布局**:
```
逻辑视图: [seq1_kv][seq2_kv][seq3_kv]
物理视图: [block0][block2][block5][block1][block3]...
```

**交付物**:
- Block Pool 实现
- Paged KV Cache Manager
- PagedAttention 算子
- 学习笔记: `docs/learn/03_paged_kv_cache.md`
- 开发日志: `docs/dev/milestone_3.md`

**测试**:
- 单元测试: Block 分配/回收、映射表
- 集成测试: 端到端验证正确性
- 性能测试: 内存利用率提升、吞吐量对比

---

### 阶段 4: Flash Attention (Milestone 4)

**目标**: 集成 Flash Attention 2，优化 attention 计算

**核心功能**:
1. **Flash Attention 集成**
   - Flash Attention 2 库集成
   - 适配 Paged KV Cache
   - Prefill 和 Decode 阶段的统一处理
   
2. **Attention Backend 抽象**
   - 统一的 Attention 接口
   - 支持多种 backend (naive, flash, paged)
   - Backend 自动选择

**优化点**:
- 降低 HBM 访问
- 提升 attention 计算效率
- 支持长上下文推理

**交付物**:
- Flash Attention 集成代码
- Attention Backend 抽象层
- 学习笔记: `docs/learn/04_flash_attention.md`
- 开发日志: `docs/dev/milestone_4.md`

**测试**:
- 单元测试: Attention 计算正确性
- 集成测试: 不同 backend 结果一致性
- 性能测试: Attention 计算延迟和吞吐量

---

### 阶段 5: Chunked Prefill (Milestone 5)

**目标**: 实现分块预填充，平衡 prefill 和 decode

**核心功能**:
1. **Chunked Prefill 调度**
   - Prefill chunk size 控制
   - Prefill 和 Decode 混合调度
   - 动态 chunk size 调整
   
2. **两阶段处理**
   - Prefill 阶段分块处理
   - Decode 阶段正常处理
   - 状态转换管理
   
3. **调度策略优化**
   - TTFT (Time to First Token) 优化
   - 吞吐量和延迟平衡

**调度示例**:
```
Iteration 1: [Prefill_req1_chunk1, Decode_req2, Decode_req3]
Iteration 2: [Prefill_req1_chunk2, Decode_req2, Decode_req3]
Iteration 3: [Decode_req1, Decode_req2, Decode_req3]
```

**交付物**:
- Chunked Prefill 实现
- 优化的调度器
- 学习笔记: `docs/learn/05_chunked_prefill.md`
- 开发日志: `docs/dev/milestone_5.md`

**测试**:
- 单元测试: Chunk 分割逻辑
- 集成测试: 混合调度验证
- 性能测试: TTFT 和吞吐量指标

---

### 阶段 6: 前缀复用 (Milestone 6)

**目标**: 实现 Prefix Caching，复用共享前缀

**核心功能**:
1. **Prefix Hash 和匹配**
   - Token 序列哈希
   - 前缀树 (Trie) 结构
   - 最长前缀匹配
   
2. **Block 共享机制**
   - Copy-on-write 完善
   - Block 引用计数
   - 自动前缀检测
   
3. **缓存淘汰策略**
   - LRU 淘汰
   - 引用计数管理
   - 缓存命中率统计

**应用场景**:
- Few-shot prompting
- 多轮对话
- 批量请求共享 system prompt

**交付物**:
- Prefix Caching 实现
- Hash 和匹配逻辑
- 学习笔记: `docs/learn/06_prefix_caching.md`
- 开发日志: `docs/dev/milestone_6.md`

**测试**:
- 单元测试: Hash、匹配、淘汰逻辑
- 集成测试: 端到端前缀复用验证
- 性能测试: 缓存命中率、TTFT 改善

---

### 阶段 7: GPTQ 量化 (Milestone 7)

**目标**: 支持 GPTQ 4-bit 量化，降低显存占用

**核心功能**:
1. **GPTQ 权重加载**
   - GPTQ 格式权重解析
   - 量化参数加载
   - 反量化逻辑
   
2. **量化算子集成**
   - GPTQ CUDA kernel
   - AutoGPTQ 库集成
   - Linear 层替换
   
3. **端到端量化推理**
   - 量化模型加载
   - 推理流程适配
   - 精度验证

**交付物**:
- GPTQ 量化支持
- 量化模型加载器
- 学习笔记: `docs/learn/07_gptq_quantization.md`
- 开发日志: `docs/dev/milestone_7.md`

**测试**:
- 单元测试: 量化算子、权重加载
- 集成测试: 量化模型推理验证
- 性能测试: 显存占用、推理速度、精度对比

---

## 项目结构

> **设计原则**: 项目结构与 vLLM v1 源码完全对齐，便于学习和参考

```
folovllm/
├── folovllm/                       # 核心包（对齐 vllm.v1）
│   ├── request.py                 # 请求和序列定义
│   ├── outputs.py                 # 输出格式定义
│   ├── config.py                  # 配置管理
│   │
│   ├── core/                      # 核心组件
│   │   ├── block_pool.py          # M3: Block Pool 管理
│   │   ├── kv_cache_manager.py    # M3: KV Cache 管理器
│   │   └── sched/                 # M2: 调度器
│   │       ├── scheduler.py       # 主调度器
│   │       ├── request_queue.py   # 请求队列
│   │       └── interface.py       # 调度接口
│   │
│   ├── engine/                    # 推理引擎
│   │   ├── llm_engine.py          # M1: LLM 引擎
│   │   ├── core.py                # M2: 核心引擎逻辑
│   │   └── processor.py           # M1: 输入处理器
│   │
│   ├── model_executor/            # 模型执行
│   │   ├── models/                # 模型实现
│   │   │   └── qwen.py            # M1: Qwen 模型
│   │   └── layers/                # 模型层
│   │       ├── attention.py       # M1: Attention 层
│   │       └── quantization.py    # M7: 量化层
│   │
│   ├── attention/                 # Attention 实现
│   │   ├── ops.py                 # M1: Attention 操作
│   │   └── backends/              # Attention 后端
│   │       ├── abstract.py        # 抽象接口
│   │       ├── torch_naive.py     # M1: 朴素实现
│   │       ├── paged.py           # M3: PagedAttention
│   │       └── flash_attn.py      # M4: Flash Attention
│   │
│   ├── sample/                    # 采样
│   │   ├── sampler.py             # M1: 采样器
│   │   ├── ops/                   # 采样操作
│   │   └── logits_processor/      # Logits 处理
│   │
│   ├── worker/                    # Worker 实现
│   │   ├── gpu_worker.py          # M1: GPU Worker
│   │   ├── model_runner.py        # M1: 模型运行器
│   │   └── input_batch.py         # M2: 输入批处理
│   │
│   ├── executor/                  # 执行器
│   │   └── gpu_executor.py        # M1: GPU 执行器
│   │
│   ├── metrics/                   # 指标统计
│   │   └── stats.py               # 性能统计
│   │
│   └── utils/                     # 工具函数
│       └── common.py              # 通用工具
│
├── tests/                         # 测试
│   ├── unit/                      # 单元测试
│   ├── integration/               # 集成测试
│   └── benchmark/                 # 性能测试
│
├── docs/                          # 文档
│   ├── project_structure.md       # 📋 结构说明（新增）
│   ├── learn/                     # 学习笔记
│   ├── dev/                       # 开发日志
│   └── api/                       # API 文档
│
├── examples/                      # 示例代码
├── reference/vllm/                # vLLM 参考代码
├── requirements.txt               # 依赖
└── README.md                      # 项目说明
```

📖 **详细结构说明**: 参见 [project_structure.md](project_structure.md)

---

## 开发准则

### 每个 Milestone 的交付标准

1. **代码实现**
   - 功能完整，通过所有测试
   - 代码注释清晰
   - 符合代码规范

2. **学习笔记** (`docs/learn/`)
   - 技术原理详解
   - 设计思路说明
   - 面试常见问题汇总
   - 参考资料链接

3. **开发日志** (`docs/dev/`)
   - 功能清单
   - 实现细节
   - 遇到的问题和解决方案
   - 下一阶段的接口预留

4. **测试覆盖**
   - 单元测试 (coverage > 80%)
   - 集成测试 (端到端验证)
   - 性能测试 (有优化时必须提供)

5. **文档更新**
   - README 更新
   - API 文档
   - 使用示例

---

## 技术深度要求

### 学习笔记要包含

1. **原理讲解**
   - 为什么需要这个优化？
   - 解决了什么问题？
   - 核心算法/数据结构

2. **实现细节**
   - 关键代码讲解
   - 与 vLLM 的对比
   - 简化点和保留点

3. **面试准备**
   - 技术点提问示例
   - 追问方向
   - 回答要点

### 性能测试指标

- **延迟**: TTFT, TPOT (Time Per Output Token), E2E Latency
- **吞吐量**: Tokens/s, Requests/s
- **资源**: GPU 显存占用, GPU 利用率
- **质量**: 准确率 (与原模型对比)

---

## 开发时间线估算

| 阶段                | 预计时间 | 累计时间 |
| ------------------- | -------- | -------- |
| M0: 项目初始化      | 2-3天    | 3天      |
| M1: 基础推理        | 3-5天    | 8天      |
| M2: 连续批处理      | 4-6天    | 14天     |
| M3: Paged KV Cache  | 5-7天    | 21天     |
| M4: Flash Attention | 3-4天    | 25天     |
| M5: Chunked Prefill | 4-5天    | 30天     |
| M6: 前缀复用        | 4-6天    | 36天     |
| M7: GPTQ 量化       | 3-5天    | 41天     |

**总计**: 约 6 周

---

## 参考资源

### 论文
- [vLLM: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)

### 代码
- [vLLM Official Repo](https://github.com/vllm-project/vllm)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

### 博客
- vLLM Blog Posts
- Flash Attention 解析
- PagedAttention 原理

---

## 验收标准

项目完成时应满足:

1. ✅ 所有 7 个阶段功能实现
2. ✅ 测试覆盖率 > 80%
3. ✅ 完整的学习笔记和开发日志
4. ✅ 性能提升可验证
5. ✅ 可运行的 Demo 和示例
6. ✅ 清晰的项目文档

---

## 下一步行动

1. **立即开始**: 阶段 0 - 项目初始化
2. **创建必要目录和文件**
3. **配置开发环境**
4. **加载 Qwen3-0.6B 模型**

准备好开始了吗？

