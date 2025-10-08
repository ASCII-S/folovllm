# 什么是 Speculative Decoding（推测解码）

## 问题
Speculative Decoding 是什么？它是如何工作的？

## 回答

### 核心思想
Speculative Decoding（推测解码）是一种**加速大语言模型推理**的技术，在**不降低输出质量**的前提下显著提升生成速度。

### 工作原理

传统的自回归生成是**串行**的，一次只能生成一个 token：
```
输入 → [大模型] → token1 → [大模型] → token2 → [大模型] → token3
        慢         慢              慢              慢
```

Speculative Decoding 使用**小模型推测 + 大模型验证**的策略：

```
步骤1: 小模型快速推测（Draft）
输入 → [小模型] → token1' → [小模型] → token2' → [小模型] → token3'
        快         快               快               快

步骤2: 大模型并行验证（Verify）
输入 + [token1', token2', token3'] → [大模型并行推理]
                                     ↓
                      验证结果: ✓ token1, ✓ token2, ✗ token3
                                     ↓
                      接受前2个，重新生成 token3
```

### 关键特点

1. **无损加速**：最终输出的分布与直接用大模型生成**完全相同**
2. **并行验证**：大模型可以一次性验证多个 token（利用 KV cache）
3. **动态长度**：如果推测正确，一次可以接受多个 token；如果错误，回退重来

### 具体步骤

**步骤1 - 推测（Draft）**：
- 使用小而快的 draft model 生成 K 个 token（如 K=4）
- Draft model 可以是：量化模型、小规模模型、或简化的模型

**步骤2 - 验证（Verify）**：
- 将推测的 K 个 token **一次性**喂给大模型
- 大模型并行计算每个位置的概率分布
- 比较 draft 和 target 的概率分布

**步骤3 - 接受/拒绝（Accept/Reject）**：
- 从左到右检查每个 token
- 如果概率分布接近，**接受** token（跳过重新生成）
- 遇到第一个不匹配的 token，**拒绝**它及之后的所有 token
- 从拒绝位置重新开始

### 数学原理

使用 rejection sampling 保证输出分布无损：

对于每个推测的 token \( t \)：
- Draft model 概率：\( p_{\text{draft}}(t) \)
- Target model 概率：\( p_{\text{target}}(t) \)

**接受概率**：
\[
\alpha = \min\left(1, \frac{p_{\text{target}}(t)}{p_{\text{draft}}(t)}\right)
\]

- 如果 \( \alpha = 1 \)：target 认为这个 token 更好，直接接受
- 如果 \( \alpha < 1 \)：以概率 \( \alpha \) 接受，否则从调整后的分布重新采样

### 加速效果

**理论加速比**：
- 如果平均接受 \( \gamma \) 个 token（\( \gamma \in [1, K] \)）
- 每次迭代生成 \( \gamma \) 个 token，而不是 1 个
- 加速比 ≈ \( \gamma \times \frac{\text{大模型推理时间}}{\text{小模型推理时间} + \text{大模型验证时间}} \)

**实际效果**：
- Draft model 质量高：2-3x 加速
- Draft model 质量差：可能没有加速（推测总是错，退化为串行）

### 在 folovllm 中的应用

在 `Sequence.fork()` 的使用场景中：

```python
# 原始序列
original_seq = Sequence(seq_id="main", prompt_token_ids=[1, 2, 3])

# 1. Draft model 生成推测序列
draft_seq = original_seq.fork("draft")
draft_seq.add_token_id(100)  # 推测 token1
draft_seq.add_token_id(101)  # 推测 token2
draft_seq.add_token_id(102)  # 推测 token3

# 2. Target model 验证推测序列
verify_results = target_model.verify(draft_seq)
# 结果: [✓, ✓, ✗]

# 3. 接受验证通过的 token
original_seq.add_token_id(100)  # 接受
original_seq.add_token_id(101)  # 接受
# token 102 被拒绝，重新生成
```

**fork 的作用**：
- 创建独立的推测序列，不污染原序列
- 验证失败时，原序列保持不变
- 验证成功时，才合并 token 到原序列

### 优缺点

**优点**：
- ✅ 无损加速（输出分布不变）
- ✅ 对用户透明（无需修改 prompt）
- ✅ 适合高质量 draft model 的场景

**缺点**：
- ❌ 需要额外的 draft model
- ❌ Draft model 质量差时无效
- ❌ 实现复杂度较高

### 相关技术

- **Parallel Decoding**：直接并行生成多个候选，用模型打分选择
- **Medusa**：给大模型加多个解码头，直接预测多个 token
- **Lookahead Decoding**：使用 n-gram 匹配等启发式方法加速

## 总结

Speculative Decoding 是用**小模型推测、大模型验证**的方式，在保持输出质量的前提下加速 LLM 推理。核心是利用并行验证多个 token，而不是串行生成。

## 参考资源
- 论文：[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Google, 2023)
- 论文：[SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781)

## 相关代码
- `folovllm/request.py`: `Sequence.fork()` 方法用于创建推测序列
- `docs/interview/milestone_0.md`: Speculative Decoding 应用场景说明

