# FoloVLLM 文档索引

欢迎来到 FoloVLLM 文档中心！

## 📚 文档导航

### 🚀 快速开始

| 文档                           | 描述                 | 适合人群 |
| ------------------------------ | -------------------- | -------- |
| [快速开始](getting_started.md) | 安装、配置、基础使用 | 新用户   |
| [快速参考](quick_reference.md) | API 速查、命令速查   | 开发者   |

### 📋 规划与设计

| 文档                                     | 描述                     | 用途         |
| ---------------------------------------- | ------------------------ | ------------ |
| [开发计划](development_plan.md)          | 完整的渐进式开发路线图   | 了解项目全貌 |
| [技术路线图](roadmap.md)                 | 各阶段技术演进和依赖关系 | 把握技术脉络 |
| [里程碑检查清单](milestone_checklist.md) | 每个阶段的验收标准       | 跟踪开发进度 |
| [项目总结](project_summary.md)           | 项目概览和核心价值       | 整体理解     |
| [项目结构](project_structure.md)         | 目录结构详细说明         | 理解架构     |

### 🎓 学习资料

| 文档                                | 描述                          | 重点内容         |
| ----------------------------------- | ----------------------------- | ---------------- |
| [学习笔记](learn/)                  | 各 Milestone 技术原理深度讲解 | 原理、算法、实现 |
| [技术对比](technical_comparison.md) | 性能指标和优化对比            | 性能分析         |

### 🔧 开发文档

| 文档                           | 描述                     | 适合场景 |
| ------------------------------ | ------------------------ | -------- |
| [开发日志](dev/)               | 各阶段实现细节和问题记录 | 开发参考 |
| [贡献指南](../CONTRIBUTING.md) | 开发规范和流程           | 参与贡献 |
| [API 文档](api/)               | 自动生成的 API 文档      | API 查询 |

---

## 📖 学习笔记清单

按 Milestone 组织的技术深度学习资料：

### M0: 项目初始化
- 📄 `dev/milestone_0.md` - 开发日志

### M1: 基础离线推理
- 📘 `learn/01_basic_inference.md` - KV Cache 原理、Sampling 策略
- 📄 `dev/milestone_1.md` - 开发日志

### M2: 连续批处理
- 📘 `learn/02_continuous_batching.md` - Dynamic Batching、调度策略
- 📄 `dev/milestone_2.md` - 开发日志

### M3: Paged KV Cache
- 📘 `learn/03_paged_kv_cache.md` - PagedAttention、Block 管理
- 📄 `dev/milestone_3.md` - 开发日志

### M4: Flash Attention
- 📘 `learn/04_flash_attention.md` - IO-aware 算法、Tiling
- 📄 `dev/milestone_4.md` - 开发日志

### M5: Chunked Prefill
- 📘 `learn/05_chunked_prefill.md` - 混合调度、Chunk size 选择
- 📄 `dev/milestone_5.md` - 开发日志

### M6: 前缀复用
- 📘 `learn/06_prefix_caching.md` - Trie 匹配、COW 机制
- 📄 `dev/milestone_6.md` - 开发日志

### M7: GPTQ 量化
- 📘 `learn/07_gptq_quantization.md` - GPTQ 算法、量化推理
- 📄 `dev/milestone_7.md` - 开发日志

---

## 🎯 文档使用指南

### 新手入门路径

1. **了解项目** → [项目总结](project_summary.md)
2. **快速上手** → [快速开始](getting_started.md)
3. **理解架构** → [技术路线图](roadmap.md)
4. **深入学习** → [学习笔记](learn/)

### 开发者路径

1. **开发规范** → [贡献指南](../CONTRIBUTING.md)
2. **开发计划** → [开发计划](development_plan.md)
3. **检查清单** → [里程碑检查清单](milestone_checklist.md)
4. **实现参考** → [开发日志](dev/)

### 面试准备路径

1. **项目亮点** → [项目总结](project_summary.md)
2. **技术深度** → [学习笔记](learn/)
3. **问答准备** → [面试准备](interview_guide.md)
4. **性能数据** → [技术对比](technical_comparison.md)

---

## 📊 文档统计

### 文档概览

| 类型     | 数量     | 预计字数 |
| -------- | -------- | -------- |
| 规划文档 | 5        | ~25K     |
| 学习笔记 | 7        | ~30K     |
| 开发日志 | 8        | ~20K     |
| 使用指南 | 3        | ~15K     |
| API 文档 | 自动生成 | -        |
| **总计** | **23+**  | **~90K** |

### 完成度追踪

- ✅ 规划文档: 5/5 (100%)
- ⏳ 学习笔记: 0/7 (0%) - 随开发进行
- ⏳ 开发日志: 0/8 (0%) - 随开发进行
- ✅ 使用指南: 3/3 (100%)

---

## 🔍 快速查找

### 按主题查找

#### 性能优化
- [技术对比](technical_comparison.md) - 性能指标对比
- [开发计划](development_plan.md#性能演进路径) - 优化路径
- [快速参考](quick_reference.md#性能优化速查) - 优化配置

#### 核心技术
- **KV Cache**: [学习笔记 M1](learn/01_basic_inference.md)
- **PagedAttention**: [学习笔记 M3](learn/03_paged_kv_cache.md)
- **Continuous Batching**: [学习笔记 M2](learn/02_continuous_batching.md)
- **Flash Attention**: [学习笔记 M4](learn/04_flash_attention.md)

#### 使用说明
- [快速开始](getting_started.md) - 安装和基础使用
- [快速参考](quick_reference.md) - API 和命令速查
- [API 文档](api/) - 详细 API 说明

#### 面试准备
- [面试准备](interview_guide.md) - 问答汇总
- [项目总结](project_summary.md) - 项目亮点
- 各 [学习笔记](learn/) - 技术深度

---

## 📝 文档更新规范

### 开发阶段

每完成一个 Milestone，需要更新：

1. **学习笔记** (`learn/XX_*.md`)
   - 技术原理
   - 实现要点
   - 面试问题

2. **开发日志** (`dev/milestone_X.md`)
   - 功能清单
   - 实现细节
   - 问题记录

3. **README**
   - 更新进度表
   - 更新功能列表

4. **本索引**
   - 更新完成度

### 文档规范

- **格式**: Markdown
- **编码**: UTF-8
- **行宽**: 100 字符
- **标题**: 使用 ATX 风格 (`#`)
- **代码块**: 指定语言

---

## 🌟 推荐阅读顺序

### 第一周: 基础理解

1. [项目总结](project_summary.md) - 1小时
2. [开发计划](development_plan.md) - 2小时
3. [技术路线图](roadmap.md) - 1小时
4. [快速开始](getting_started.md) - 1小时

### 第二-七周: 渐进开发

每周一个 Milestone:
1. 阅读对应开发计划
2. 实现功能
3. 编写学习笔记
4. 更新开发日志

### 第八周: 总结提升

1. [面试准备](interview_guide.md) - 准备答案
2. [技术对比](technical_comparison.md) - 整理数据
3. 完善所有文档

---

## 🔗 外部资源

### 论文
- [vLLM: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [GPTQ](https://arxiv.org/abs/2210.17323)

### 代码
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

### 博客
- vLLM 官方博客
- HuggingFace 博客

---

## 💬 反馈与建议

如果你发现文档有问题或有改进建议：

1. 提交 Issue: [GitHub Issues](../../issues)
2. 提交 PR: [贡献指南](../CONTRIBUTING.md)
3. 讨论: [GitHub Discussions](../../discussions)

---

**文档持续更新中... 📚**

最后更新: 2025-10-06

