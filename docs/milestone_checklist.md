# 里程碑检查清单

> 每个 Milestone 完成前必须满足的条件

## 通用检查项

每个阶段都需要完成以下内容：

### 📝 文档
- [ ] 学习笔记 (`docs/learn/XX_*.md`)
  - [ ] 技术原理讲解
  - [ ] 核心算法/数据结构
  - [ ] 实现要点
  - [ ] 与 vLLM 对比
  - [ ] 面试问题汇总 (至少 5 个)
  - [ ] 参考资料链接

- [ ] 开发日志 (`docs/dev/milestone_X.md`)
  - [ ] 功能清单
  - [ ] 实现细节
  - [ ] 遇到的问题和解决方案
  - [ ] 代码结构说明
  - [ ] 为下一阶段预留的接口

- [ ] README 更新
  - [ ] 更新进度表
  - [ ] 更新功能说明
  - [ ] 更新使用示例

### 🧪 测试
- [ ] 单元测试
  - [ ] 覆盖率 > 80%
  - [ ] 所有核心组件有测试
  - [ ] 边界条件测试

- [ ] 集成测试
  - [ ] 端到端推理验证
  - [ ] 与上一阶段的回归测试
  - [ ] 正确性验证 (与原模型对比)

- [ ] 性能测试 (有性能优化时必须)
  - [ ] Baseline 建立
  - [ ] 优化前后对比
  - [ ] 性能指标数据收集

### 💻 代码质量
- [ ] 代码注释清晰
- [ ] 符合 PEP 8 规范
- [ ] 类型标注完整
- [ ] 无 TODO 或 FIXME (应完成或文档化)

### 📊 可运行示例
- [ ] 提供命令行示例
- [ ] 提供 Python API 示例
- [ ] 示例代码可直接运行

---

## M0: 项目初始化

### 核心功能
- [ ] 项目目录结构创建
- [ ] 基础配置系统
  - [ ] ModelConfig
  - [ ] CacheConfig  
  - [ ] SchedulerConfig (预留)
- [ ] 模型加载器
  - [ ] 支持 HuggingFace 模型
  - [ ] Qwen3-0.6B 模型加载
  - [ ] 权重管理
- [ ] 基础数据结构
  - [ ] Request
  - [ ] Sequence
  - [ ] SamplingParams
  - [ ] Output 格式

### 测试
- [ ] 测试模型加载
- [ ] 测试配置解析
- [ ] 测试数据结构创建

### 文档
- [ ] 开发日志: `docs/dev/milestone_0.md`
- [ ] 项目结构说明
- [ ] 环境配置文档

### 验收标准
- [ ] 能成功加载 Qwen3-0.6B 模型
- [ ] 配置系统可用
- [ ] 基础数据结构完整

---

## M1: 基础离线推理

### 核心功能
- [ ] LLMEngine 基础实现
  - [ ] `generate()` 方法
  - [ ] 单请求处理流程
- [ ] Model forward pass
  - [ ] Qwen3 模型前向传播
  - [ ] KV Cache 维护
- [ ] Sampling 策略
  - [ ] Greedy sampling
  - [ ] Top-k sampling
  - [ ] Top-p (nucleus) sampling
  - [ ] Temperature scaling
- [ ] 简单 KV Cache
  - [ ] 连续内存分配
  - [ ] Cache 读写

### 测试
- [ ] 单元测试
  - [ ] 各采样策略测试
  - [ ] KV Cache 更新测试
  - [ ] Tokenization 测试
- [ ] 集成测试
  - [ ] 端到端推理
  - [ ] 输出正确性验证 (对比 HF)
- [ ] 性能测试
  - [ ] 单请求延迟 (TTFT, TPOT)
  - [ ] Token 生成速度
  - [ ] 显存占用

### 文档
- [ ] 学习笔记: `docs/learn/01_basic_inference.md`
  - [ ] KV Cache 原理
  - [ ] Transformer 推理流程
  - [ ] Sampling 策略详解
- [ ] 开发日志: `docs/dev/milestone_1.md`

### 示例
- [ ] `examples/basic_inference.py`
- [ ] CLI 工具: `python -m folovllm.run`

### 验收标准
- [ ] 能正确推理 Qwen3-0.6B
- [ ] 输出与 HuggingFace 一致 (相同种子)
- [ ] 支持多种 sampling 策略
- [ ] Baseline 性能数据建立

---

## M2: 连续批处理

### 核心功能
- [ ] Scheduler 实现
  - [ ] 请求队列 (waiting, running, finished)
  - [ ] 动态 batch 组装
  - [ ] 请求优先级处理
- [ ] 批处理引擎
  - [ ] 不同长度序列 padding
  - [ ] Attention mask 构建
  - [ ] Batch forward pass
- [ ] 请求生命周期管理
  - [ ] 状态转换 (WAITING → RUNNING → FINISHED)
  - [ ] 完成序列移除
  - [ ] 新序列添加
- [ ] 抢占机制 (基础版)
  - [ ] OOM 处理
  - [ ] 请求抢占

### 测试
- [ ] 单元测试
  - [ ] Scheduler 调度逻辑
  - [ ] Batch 组装
  - [ ] Padding 和 masking
- [ ] 集成测试
  - [ ] 多请求并发处理
  - [ ] 动态添加/移除
  - [ ] 正确性验证
- [ ] 性能测试
  - [ ] 不同 batch size 的吞吐量
  - [ ] GPU 利用率
  - [ ] 与 M1 对比 (应有 3-5x 提升)

### 文档
- [ ] 学习笔记: `docs/learn/02_continuous_batching.md`
  - [ ] Iteration-level scheduling
  - [ ] Dynamic batching 原理
  - [ ] 对比 static batching
- [ ] 开发日志: `docs/dev/milestone_2.md`

### 示例
- [ ] `examples/batch_inference.py`
- [ ] 并发请求测试脚本

### 验收标准
- [ ] 支持动态批处理
- [ ] 吞吐量相对 M1 提升 3-5x
- [ ] GPU 利用率 > 60%

---

## M3: Paged KV Cache

### 核心功能
- [ ] Block Pool Manager
  - [ ] 固定大小 block 分配
  - [ ] Free block 管理
  - [ ] Block 引用计数
- [ ] KV Cache Manager
  - [ ] 逻辑 block → 物理 block 映射
  - [ ] Block table 维护
  - [ ] Block 分配和回收
  - [ ] Copy-on-Write 基础
- [ ] PagedAttention 算子
  - [ ] 分页 KV 的 attention 计算
  - [ ] Block table 传递
  - [ ] 正确性验证

### 测试
- [ ] 单元测试
  - [ ] Block 分配/回收
  - [ ] Block table 更新
  - [ ] 引用计数
- [ ] 集成测试
  - [ ] 端到端验证
  - [ ] 与 M2 结果一致性
  - [ ] 碎片化测试
- [ ] 性能测试
  - [ ] 显存利用率 (应 >90%)
  - [ ] 吞吐量 (因更大 batch)
  - [ ] 内部碎片分析

### 文档
- [ ] 学习笔记: `docs/learn/03_paged_kv_cache.md`
  - [ ] PagedAttention 原理
  - [ ] Block 管理算法
  - [ ] 虚拟内存类比
- [ ] 开发日志: `docs/dev/milestone_3.md`

### 示例
- [ ] `examples/paged_kv_demo.py`
- [ ] 显存利用率可视化

### 验收标准
- [ ] 显存利用率 > 90%
- [ ] 支持更大 batch size (相对 M2)
- [ ] 输出正确性不变
- [ ] 零外部碎片

---

## M4: Flash Attention

### 核心功能
- [ ] Flash Attention 集成
  - [ ] Flash Attention 2 库安装
  - [ ] Prefill 阶段集成
  - [ ] Decode 阶段集成
- [ ] Attention Backend 抽象
  - [ ] Backend 接口定义
  - [ ] Naive backend (M1)
  - [ ] Paged backend (M3)
  - [ ] Flash backend (M4)
  - [ ] 自动 backend 选择
- [ ] 与 Paged KV 结合
  - [ ] Flash + Paged 的适配
  - [ ] Block table 传递

### 测试
- [ ] 单元测试
  - [ ] Flash Attention 计算正确性
  - [ ] Backend 切换
- [ ] 集成测试
  - [ ] 不同 backend 结果一致性
  - [ ] 长序列测试 (2K, 4K, 8K)
- [ ] 性能测试
  - [ ] Attention 计算速度 (应 2-4x 快)
  - [ ] 不同序列长度性能
  - [ ] 显存占用

### 文档
- [ ] 学习笔记: `docs/learn/04_flash_attention.md`
  - [ ] IO-aware 算法
  - [ ] Tiling 和 recomputation
  - [ ] 复杂度分析
- [ ] 开发日志: `docs/dev/milestone_4.md`

### 示例
- [ ] `examples/flash_attention_demo.py`
- [ ] 长上下文推理示例

### 验收标准
- [ ] Attention 计算 1.5-2x 快
- [ ] 支持 8K+ 上下文
- [ ] TTFT 降低 20-30%

---

## M5: Chunked Prefill

### 核心功能
- [ ] Chunked Prefill 实现
  - [ ] Prefill chunk size 配置
  - [ ] Sequence 分块逻辑
  - [ ] Chunk 状态管理
- [ ] 混合调度器
  - [ ] Prefill chunk 和 Decode 混合
  - [ ] Budget 管理 (token 数)
  - [ ] 优先级策略
- [ ] 两阶段处理
  - [ ] Prefill 阶段标记
  - [ ] Decode 阶段转换
  - [ ] KV Cache 正确维护

### 测试
- [ ] 单元测试
  - [ ] Chunk 分割
  - [ ] 混合调度逻辑
  - [ ] 状态转换
- [ ] 集成测试
  - [ ] 长 prefill + decode 混合
  - [ ] 正确性验证
- [ ] 性能测试
  - [ ] 不同 chunk size 的 TTFT
  - [ ] Decode 延迟改善
  - [ ] 总吞吐量影响

### 文档
- [ ] 学习笔记: `docs/learn/05_chunked_prefill.md`
  - [ ] Prefill-Decode 差异
  - [ ] Chunk size 选择
  - [ ] 权衡分析
- [ ] 开发日志: `docs/dev/milestone_5.md`

### 示例
- [ ] `examples/chunked_prefill_demo.py`
- [ ] Chunk size 调优脚本

### 验收标准
- [ ] Decode 请求 TTFT 显著降低
- [ ] 支持 prefill-decode 混合
- [ ] 总吞吐量基本不变

---

## M6: 前缀复用

### 核心功能
- [ ] Prefix Hash
  - [ ] Token 序列哈希
  - [ ] Hash 冲突处理
- [ ] Prefix 匹配
  - [ ] Trie 数据结构
  - [ ] 最长前缀匹配
  - [ ] 快速查找
- [ ] Block 共享
  - [ ] Copy-on-Write 完善
  - [ ] 引用计数管理
  - [ ] 自动前缀检测
- [ ] 缓存淘汰
  - [ ] LRU 策略
  - [ ] 引用计数考虑
  - [ ] 缓存大小限制

### 测试
- [ ] 单元测试
  - [ ] Hash 函数
  - [ ] Trie 匹配
  - [ ] COW 机制
  - [ ] LRU 淘汰
- [ ] 集成测试
  - [ ] 共享前缀场景
  - [ ] Few-shot 测试
  - [ ] 多轮对话测试
- [ ] 性能测试
  - [ ] 缓存命中率
  - [ ] TTFT 改善 (命中时)
  - [ ] 不同命中率的性能

### 文档
- [ ] 学习笔记: `docs/learn/06_prefix_caching.md`
  - [ ] Prefix matching 算法
  - [ ] COW 机制详解
  - [ ] 应用场景
- [ ] 开发日志: `docs/dev/milestone_6.md`

### 示例
- [ ] `examples/prefix_caching_demo.py`
- [ ] Few-shot 推理示例
- [ ] 多轮对话示例

### 验收标准
- [ ] 支持前缀自动检测
- [ ] 缓存命中时 TTFT 降低 3-10x
- [ ] 命中率统计准确

---

## M7: GPTQ 量化

### 核心功能
- [ ] GPTQ 权重加载
  - [ ] GPTQ 格式解析
  - [ ] 量化参数加载 (scale, zero-point)
  - [ ] 权重重组
- [ ] 量化算子
  - [ ] AutoGPTQ 集成
  - [ ] GPTQ kernel
  - [ ] Linear 层替换
- [ ] 端到端量化推理
  - [ ] 量化模型加载
  - [ ] 推理流程适配
  - [ ] 精度验证

### 测试
- [ ] 单元测试
  - [ ] 量化算子
  - [ ] 权重加载
  - [ ] 反量化正确性
- [ ] 集成测试
  - [ ] 量化模型推理
  - [ ] 与 FP16 对比
- [ ] 性能测试
  - [ ] 显存占用 (应 75% ↓)
  - [ ] 推理速度
  - [ ] 精度损失 (perplexity)

### 文档
- [ ] 学习笔记: `docs/learn/07_gptq_quantization.md`
  - [ ] GPTQ 算法原理
  - [ ] 量化流程
  - [ ] 精度-性能权衡
- [ ] 开发日志: `docs/dev/milestone_7.md`

### 示例
- [ ] `examples/gptq_inference.py`
- [ ] 精度对比脚本

### 验收标准
- [ ] 支持 GPTQ-4bit 推理
- [ ] 显存占用降低 ~75%
- [ ] Perplexity 下降 < 1%
- [ ] 推理速度提升 20%+

---

## 最终项目验收

### 完整性
- [ ] 所有 7 个 Milestone 完成
- [ ] 所有核心功能实现
- [ ] 文档完整 (学习笔记 + 开发日志)

### 质量
- [ ] 整体测试覆盖率 > 80%
- [ ] 所有测试通过
- [ ] 代码规范统一

### 性能
- [ ] 相对 M1 性能提升 > 10x (综合)
- [ ] 显存占用降低 > 80%
- [ ] GPU 利用率 > 75%

### 文档
- [ ] 完整的开发计划 ✅
- [ ] 技术路线图 ✅
- [ ] 面试准备指南 ✅
- [ ] 技术对比文档 ✅
- [ ] API 文档
- [ ] 用户手册

### 可运行示例
- [ ] 基础推理示例
- [ ] 批处理示例
- [ ] 长上下文示例
- [ ] 量化推理示例
- [ ] 性能测试脚本

---

## 开发流程

### 开始一个 Milestone
1. 创建功能分支: `git checkout -b milestone-X`
2. 阅读开发计划和上一阶段日志
3. 设计数据结构和接口
4. 编写测试用例 (TDD)
5. 实现功能

### 完成一个 Milestone
1. 运行所有测试，确保通过
2. 运行性能测试，收集数据
3. 编写学习笔记
4. 编写开发日志
5. 更新 README 和文档
6. 提交代码: `git commit -m "feat: Milestone X completed"`
7. **在本清单中勾选完成项**

### 代码审查要点
- [ ] 功能完整
- [ ] 测试充分
- [ ] 文档齐全
- [ ] 性能达标
- [ ] 代码清晰

---

## 工具和脚本

### 测试
```bash
# 运行所有测试
pytest tests/

# 运行单个 milestone 测试
pytest tests/unit/test_m1_*.py

# 覆盖率报告
pytest --cov=folovllm tests/
```

### 性能测试
```bash
# Benchmark
python tests/benchmark/run_benchmark.py --milestone m2

# 性能对比
python tests/benchmark/compare.py m1 m2
```

### 文档生成
```bash
# API 文档
pdoc --html folovllm -o docs/api/

# 检查文档完整性
python scripts/check_docs.py
```

---

**记住**: 每个 Milestone 都是独立可验证的，不要留下未完成的功能！

