# 什么是 Parallel Sampling（并行采样）

## 问题
Parallel Sampling 是什么？它与 n 参数有什么关系？

## 回答

### 核心概念

**Parallel Sampling（并行采样）** 是指从**同一个输入**生成**多个不同的输出序列**的技术。这是大语言模型中常见的需求，用于提供多样化的候选答案。

### 使用场景

**示例1 - 创意写作**：
```python
prompt = "请写一个关于未来的故事开头"

# 生成 3 个不同的开头
sampling_params = SamplingParams(n=3, temperature=0.8)

# 可能得到：
outputs[0]: "2050年，人类终于实现了星际移民..."
outputs[1]: "在遥远的未来，AI已经成为人类最好的伙伴..."  
outputs[2]: "时间旅行成为现实的那一天，世界发生了巨变..."
```

**示例2 - 代码生成**：
```python
prompt = "写一个Python函数计算斐波那契数列"

# 生成 5 个不同实现
sampling_params = SamplingParams(n=5, temperature=0.6)

# 可能得到：递归版本、迭代版本、动态规划版本、生成器版本、矩阵快速幂版本
```

### 参数说明

#### n 参数
```python
SamplingParams(n=3)
```
- **含义**：最终返回的输出序列数量
- **默认值**：1
- **范围**：n ≥ 1

#### best_of 参数
```python
SamplingParams(n=3, best_of=5)
```
- **含义**：生成的候选序列总数（内部生成更多，然后筛选）
- **默认值**：如果不设置，`best_of = n`
- **约束**：`best_of ≥ n`

### n vs best_of 的区别

**仅使用 n**：
```python
SamplingParams(n=3)  # 等价于 n=3, best_of=3

流程：
1. 生成 3 个序列
2. 直接返回这 3 个序列
```

**使用 n 和 best_of**：
```python
SamplingParams(n=3, best_of=5)

流程：
1. 生成 5 个候选序列（并行采样）
2. 根据累积 log 概率排序
3. 返回最好的 3 个序列
```

### 工作原理

**步骤1 - 创建多个序列**：
```python
# 从一个 Request 派生多个 Sequence
sequences = [
    request.fork_sequence(f"seq-{i}") 
    for i in range(best_of)
]
```

**步骤2 - 并行生成**：
```python
# 所有序列共享 prompt 的 KV Cache
for step in range(max_tokens):
    for seq in sequences:
        # 每个序列独立采样，生成不同的 token
        next_token = sample(logits, temperature=0.8)
        seq.add_token_id(next_token)
```

**步骤3 - 选择最优**（如果 best_of > n）：
```python
# 按累积 log 概率排序
sequences.sort(key=lambda s: s.cumulative_logprob, reverse=True)

# 返回前 n 个
return sequences[:n]
```

### 关键实现：Sequence.fork()

并行采样依赖 `fork()` 方法创建独立的序列：

```python
# 原始序列
original_seq = Sequence(
    seq_id="main",
    prompt_token_ids=[1, 2, 3, 4, 5]
)

# 派生多个独立序列用于并行采样
sequences = [original_seq.fork(f"seq-{i}") for i in range(3)]

# 每个序列独立生成，互不影响
sequences[0].add_token_id(100)  # "太阳"
sequences[1].add_token_id(200)  # "月亮"
sequences[2].add_token_id(300)  # "星星"

# 原始序列不受影响
print(original_seq.output_token_ids)  # []
```

**为什么需要 fork？**
- 所有序列共享相同的 prompt（输入）
- 但输出 token 不同（独立采样）
- 需要**深拷贝**避免互相干扰

### 累积 log 概率

用于评估序列质量：

```python
# 每个 token 的 log 概率
seq.add_token_id(100, logprob=-0.5)   # log P(token_1) = -0.5
seq.add_token_id(200, logprob=-0.3)   # log P(token_2) = -0.3
seq.add_token_id(300, logprob=-0.8)   # log P(token_3) = -0.8

# 累积 log 概率（相加）
cumulative_logprob = -0.5 + (-0.3) + (-0.8) = -1.6

# 等价于实际概率（相乘）
actual_prob = exp(-0.5) × exp(-0.3) × exp(-0.8) = exp(-1.6)
```

**越高越好**：
- Cumulative logprob 越接近 0，概率越高
- Cumulative logprob 越负，概率越低

### 内存优化：共享 KV Cache

并行采样的关键优化：

```
Prompt: [1, 2, 3, 4, 5]
         ↓
     计算 KV Cache（只需一次）
         ↓
    共享给所有序列
    ↙     ↓      ↘
Seq1   Seq2    Seq3
[100]  [200]   [300]  ← 每个序列独立采样
```

**优势**：
- Prompt KV Cache 只计算一次
- 所有序列共享，节省显存和计算
- 只有输出部分的 KV Cache 是独立的

### 完整示例

```python
from folovllm import SamplingParams

# 场景1：生成 3 个不同的故事
params1 = SamplingParams(
    n=3,              # 返回 3 个序列
    temperature=0.8,  # 高温度 → 多样性
    top_p=0.9,
    max_tokens=100,
)

# 场景2：生成 5 个候选，返回最好的 2 个
params2 = SamplingParams(
    n=2,              # 最终返回 2 个
    best_of=5,        # 内部生成 5 个候选
    temperature=0.7,
    max_tokens=50,
)

# 场景3：确定性生成（无并行采样）
params3 = SamplingParams(
    n=1,              # 只生成 1 个
    temperature=0.0,  # 贪心解码（确定性）
    max_tokens=50,
)
```

### 与其他技术的区别

| 技术                     | 目的             | 多样性来源                  |
| ------------------------ | ---------------- | --------------------------- |
| **Parallel Sampling**    | 生成多个不同输出 | 随机采样（temperature）     |
| **Beam Search**          | 找到最可能的输出 | 搜索算法（保留 top-k 路径） |
| **Speculative Decoding** | 加速单个输出生成 | 用小模型推测 + 大模型验证   |

### 优缺点

**优点**：
- ✅ 提供多样化的输出选项
- ✅ 用户可以选择最满意的结果
- ✅ best_of 机制可以提高质量（过滤掉差的候选）
- ✅ 共享 prompt KV Cache，内存效率高

**缺点**：
- ❌ 计算成本高（生成 n 个序列的成本 ≈ n 倍）
- ❌ 如果 best_of > n，会浪费 (best_of - n) 个序列的计算
- ❌ 需要更多显存存储多个序列的 KV Cache

### 实际应用

**1. ChatGPT 式应用**：
```python
# 生成多个候选，让用户选择
params = SamplingParams(n=3, temperature=0.8)
```

**2. 代码生成工具**（如 GitHub Copilot）：
```python
# 生成多个代码实现供开发者选择
params = SamplingParams(n=5, temperature=0.6, best_of=10)
```

**3. 内容创作**：
```python
# 生成多个营销文案/广告语
params = SamplingParams(n=10, temperature=0.9)
```

## 总结

**Parallel Sampling** 是通过在同一个输入上**独立随机采样**来生成多个不同输出的技术。核心是：

1. 使用 `n` 参数控制返回数量
2. 使用 `best_of` 参数生成更多候选并筛选
3. 通过 `Sequence.fork()` 创建独立序列
4. 共享 prompt KV Cache 优化内存
5. 用累积 log 概率评估序列质量

## 相关代码
- `folovllm/sampling_params.py`: `SamplingParams` 类定义
- `folovllm/request.py`: `Sequence.fork()` 方法实现
- `docs/interview/milestone_0.md`: Parallel Sampling 面试题
- `docs/learn/milestone_0.md`: n vs best_of 详解

