# 什么是 Beam Search（束搜索）

## 问题
Beam Search 是什么？它与 Greedy Sampling 有什么区别？

## 回答

### 核心概念

**Beam Search（束搜索）** 是一种**启发式搜索算法**，用于在大语言模型中找到**高质量的输出序列**。它在每一步保留多个候选路径（beam），而不是像贪心算法那样只保留一个最优解。

### 问题背景：贪心解码的局限

**Greedy Decoding（贪心解码）** 的问题：每步只选择当前最优的token，可能错过全局最优解。

```python
# 示例：翻译 "我爱你"
Step 1: "I"    (概率 0.6) ✓ 选择最优
        "You"  (概率 0.3)
        "Love" (概率 0.1)

Step 2: "I love" (概率 0.5) ✓ 选择最优
        "I am"   (概率 0.3)

Step 3: "I love it" (概率 0.4) ✓ 选择最优

最终: "I love it"
总概率 = 0.6 × 0.5 × 0.4 = 0.12
```

**但可能存在更好的路径**：
```python
如果在 Step 1 选择 "You"：
Step 1: "You"
Step 2: "You are"     (概率 0.9)
Step 3: "You are loved" (概率 0.8)

总概率 = 0.3 × 0.9 × 0.8 = 0.216 > 0.12 ✅ 更好！
```

**贪心的问题**：只看当前一步，错过了全局更优的路径。

## Beam Search 工作原理

### 基本思想

在每一步保留 **beam_size** 个最优的候选序列，而不是只保留 1 个。

### 详细流程

**设置**：`beam_size = 3`，翻译"我爱你"

**Step 1 - 初始扩展**：
```python
# 从 <BOS> 开始，生成第一个token
候选：
1. "I"    (log_prob = -0.5)
2. "You"  (log_prob = -1.2)
3. "Love" (log_prob = -2.3)

保留前3个 → beam = ["I", "You", "Love"]
```

**Step 2 - 扩展所有beam**：
```python
# 从每个beam扩展
"I" → {
    "I love"  (log_prob = -0.5 + -0.7 = -1.2)
    "I am"    (log_prob = -0.5 + -1.0 = -1.5)
    "I hate"  (log_prob = -0.5 + -3.0 = -3.5)
}

"You" → {
    "You are"   (log_prob = -1.2 + -0.1 = -1.3)
    "You love"  (log_prob = -1.2 + -0.5 = -1.7)
    "You can"   (log_prob = -1.2 + -1.8 = -3.0)
}

"Love" → {
    "Love is"   (log_prob = -2.3 + -0.5 = -2.8)
    "Love you"  (log_prob = -2.3 + -0.8 = -3.1)
    ...
}

# 总共 3 × vocab_size 个候选
# 选择全局最优的 3 个：
beam = [
    "I love"    (-1.2) ✓
    "You are"   (-1.3) ✓
    "I am"      (-1.5) ✓
]
```

**Step 3 - 继续扩展**：
```python
"I love" → "I love you"     (-1.2 + -0.3 = -1.5)
"You are" → "You are loved" (-1.3 + -0.2 = -1.5)
"I am" → "I am happy"       (-1.5 + -0.5 = -2.0)

最终保留 beam_size 个最优序列
```

**Step 4 - 终止条件**：
- 达到最大长度
- 所有beam都生成了 `<EOS>` token
- 或提前停止（early stopping）

### 伪代码实现

```python
def beam_search(model, prompt, beam_size=3, max_len=20):
    # 1. 初始化
    beams = [Sequence(prompt)]  # 初始只有一个序列
    
    for step in range(max_len):
        candidates = []
        
        # 2. 扩展每个beam
        for beam in beams:
            # 获取下一个token的概率分布
            logits = model.forward(beam.token_ids)
            log_probs = F.log_softmax(logits[-1], dim=-1)
            
            # 获取 top-k 个候选token
            top_k_probs, top_k_ids = torch.topk(log_probs, k=beam_size * 2)
            
            # 为每个候选token创建新序列
            for token_id, token_prob in zip(top_k_ids, top_k_probs):
                new_beam = beam.fork(f"{beam.id}-{token_id}")
                new_beam.add_token_id(token_id)
                new_beam.score = beam.score + token_prob  # 累积log概率
                candidates.append(new_beam)
        
        # 3. 选择全局最优的 beam_size 个
        candidates.sort(key=lambda x: x.score, reverse=True)
        beams = candidates[:beam_size]
        
        # 4. 检查终止条件
        if all(beam.is_finished() for beam in beams):
            break
    
    # 5. 返回最优序列
    return beams[0]
```

### 关键实现：Sequence.fork()

Beam Search 需要不断复制和扩展序列，这就是 `fork()` 的用途：

```python
# 当前beam
beam = Sequence(seq_id="seq-1", token_ids=[1, 2, 3])

# 扩展：为top-k个token各创建一个新序列
candidates = []
top_k_tokens = [100, 200, 300]  # 假设这是前3个最优token

for token in top_k_tokens:
    # fork出新序列
    new_beam = beam.fork(f"seq-1-{token}")
    new_beam.add_token_id(token)
    candidates.append(new_beam)

# 结果：
# candidates[0]: [1, 2, 3, 100]
# candidates[1]: [1, 2, 3, 200]
# candidates[2]: [1, 2, 3, 300]
```

**为什么需要fork？**
- 每个beam需要独立扩展，不能互相影响
- 深拷贝保证修改一个序列不会影响其他序列

## Beam Search vs Greedy Decoding

| 维度         | Greedy Decoding    | Beam Search (beam_size=k) |
| ------------ | ------------------ | ------------------------- |
| **搜索空间** | 贪心（局部最优）   | 保留k条路径（更全局）     |
| **计算量**   | O(1) 每步一次前向  | O(k) 每步k次前向          |
| **内存占用** | 低                 | k倍                       |
| **结果质量** | 可能次优           | 通常更好                  |
| **多样性**   | 无（确定性）       | 低（都是高概率路径）      |
| **使用场景** | 速度优先、对话生成 | 质量优先（翻译、摘要）    |

## Beam Search vs Parallel Sampling

| 维度              | Beam Search                | Parallel Sampling        |
| ----------------- | -------------------------- | ------------------------ |
| **目标**          | 找到最可能的输出           | 生成多个不同的输出       |
| **选择标准**      | 累积概率最高               | 随机采样（temperature）  |
| **多样性**        | 低（都是高概率路径）       | 高（随机性带来多样性）   |
| **候选来源**      | 搜索算法（保留top-k）      | 独立随机采样             |
| **典型beam_size** | 3-5                        | n=1-10                   |
| **使用场景**      | 机器翻译、摘要（质量优先） | 对话、创意写作（多样性） |

## 优缺点

### 优点
- ✅ 比贪心解码质量更高（考虑更多可能性）
- ✅ 在机器翻译、摘要等任务上效果显著
- ✅ 可控的计算成本（通过beam_size）
- ✅ 确定性输出（给定beam_size，结果固定）

### 缺点
- ❌ 计算成本是贪心的 k 倍（k = beam_size）
- ❌ 内存占用高（需要存储k个序列的KV Cache）
- ❌ **多样性低**：所有beam都在高概率区域，输出趋于"安全"
- ❌ **对话生成效果差**：容易生成通用、无聊的回复

## 为什么Beam Search对话生成效果不好？

```python
# 对话示例
User: "今天天气怎么样？"

# Beam Search倾向于生成：
Bot: "今天天气不错。"     (高概率但无聊)
Bot: "天气很好。"         (安全但缺乏信息)
Bot: "今天的天气很不错。" (重复、模板化)

# 而随机采样可能生成：
Bot: "今天阳光明媚，适合出去散步！" (更有信息量)
Bot: "看起来会下雨，记得带伞哦~"    (更个性化)
```

**原因**：
1. **Likelihood陷阱**：高概率 ≠ 有趣、有信息量
2. **缺乏随机性**：beam search总是选择高概率路径，缺少惊喜
3. **通用回复**："我不知道"、"是的"、"好的" 这类回复概率很高，但没有价值

**解决方案**：对话生成使用**随机采样** + **temperature/top-p**
```python
sampling_params = SamplingParams(
    temperature=0.8,  # 增加随机性
    top_p=0.9,        # nucleus sampling
    n=1,              # 不需要多个beam
)
```

## 实际应用场景

### 适合 Beam Search 的任务

**1. 机器翻译**：
```python
# 需要准确性和流畅性
Input: "我爱自然语言处理"
Beam Search: "I love natural language processing" ✅
```

**2. 文本摘要**：
```python
# 需要保留关键信息，减少冗余
Beam Search 能找到信息密度高的摘要
```

**3. 代码生成**（特定场景）：
```python
# 生成语法正确、逻辑清晰的代码
Beam Search 倾向于生成更"安全"的代码
```

### 不适合 Beam Search 的任务

**1. 对话生成**：
```python
# 需要多样性和创意
使用 temperature=0.7-0.9 的随机采样
```

**2. 创意写作**：
```python
# 需要意外性和想象力
使用 temperature=0.8-1.0，top_p=0.9
```

**3. 头脑风暴**：
```python
# 需要多个不同的想法
使用 n=5-10 的 parallel sampling
```

## 高级变体

### 1. Diverse Beam Search
```python
# 强制不同的beam探索不同的区域
# 增加多样性同时保持质量
```

### 2. Length Penalty
```python
# 调整得分公式，避免过短或过长的序列
score = log_prob / (length ** length_penalty)

# length_penalty > 1: 鼓励更长的序列
# length_penalty < 1: 鼓励更短的序列
```

### 3. Early Stopping
```python
# 当已有 beam_size 个完成的序列时停止
# 不需要等所有beam都完成
```

## FoloVLLM 中的实现状态

在 Milestone 0-1 中，**Beam Search 未实现**：

```python
# sampling_params.py
if self.use_beam_search:
    raise NotImplementedError(
        "Beam search is not supported in M0-M1. "
        "It will be implemented in future milestones."
    )
```

**但已预留接口**：
- `Sequence.fork()` 方法用于复制序列
- `cumulative_logprob` 字段用于累积概率
- `SamplingParams.use_beam_search` 参数已定义

**实现计划**：在未来的 Milestone 中实现完整的 Beam Search。

## 总结

**Beam Search** 是一种**启发式搜索算法**，通过在每一步保留多个候选路径来找到高质量的输出序列。

**核心特点**：
- 保留 `beam_size` 个最优候选
- 每步从所有候选中选择全局最优的 k 个
- 比贪心解码质量高，但计算成本也更高
- 适合机器翻译、摘要等需要准确性的任务
- 不适合需要创意和多样性的任务（如对话生成）

**关键公式**：
```
Beam Score = Σ log P(token_i | context)
           = log P(token_1) + log P(token_2) + ... + log P(token_n)
```

**实现要点**：
- 使用 `Sequence.fork()` 复制序列
- 每步扩展 beam_size × vocab_size 个候选
- 选择累积log概率最高的 beam_size 个

## 相关代码
- `folovllm/request.py`: `Sequence.fork()` 方法（为beam search预留）
- `folovllm/sampling_params.py`: `use_beam_search` 参数定义
- `docs/interview/milestone_0.md`: Beam Search 面试题
- `docs/learn/milestone_0.md`: Beam Search 详细说明

## 参考资料
- [Beam Search 原理](https://d2l.ai/chapter_recurrent-modern/beam-search.html)
- [The Curious Case of Neural Text Degeneration (Nucleus Sampling paper)](https://arxiv.org/abs/1904.09751)

