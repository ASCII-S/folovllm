# Milestone 1 面试指南

> 本文档整理 M1 基础离线推理阶段可能遇到的面试问题及回答要点

---

## 📋 目录

1. [KV Cache 相关](#1-kv-cache-相关)
2. [Attention 机制相关](#2-attention-机制相关)
3. [位置编码相关](#3-位置编码相关)
4. [采样策略相关](#4-采样策略相关)
5. [模型架构相关](#5-模型架构相关)
6. [推理优化相关](#6-推理优化相关)
7. [系统设计相关](#7-系统设计相关)
8. [数值稳定性相关](#8-数值稳定性相关)

---

## 1. KV Cache 相关

### Q1.1: 什么是 KV Cache？为什么需要它？

**回答要点**：

**问题背景**：
- Transformer 生成第 N 个 token 时，需要计算它与前 N-1 个 token 的 attention
- 前 N-1 个 token 的 K 和 V 在每一步都重新计算是浪费的

**KV Cache 原理**：
- 缓存已计算的 K 和 V 矩阵
- 每次只计算新 token 的 K 和 V，追加到缓存

**数学表达**：
```
Without cache:
  Step 1: Q₁ @ [K₁]ᵀ
  Step 2: Q₂ @ [K₁, K₂]ᵀ  ← K₁ 重复计算
  Step 3: Q₃ @ [K₁, K₂, K₃]ᵀ  ← K₁, K₂ 重复计算

With cache:
  Step 1: Q₁ @ [K₁]ᵀ, cache=[K₁]
  Step 2: Q₂ @ [K₁, K₂]ᵀ, cache=[K₁, K₂]  ← K₁ 从 cache 读取
  Step 3: Q₃ @ [K₁, K₂, K₃]ᵀ, cache=[K₁, K₂, K₃]  ← K₁, K₂ 从 cache 读取
```

**性能提升**：
- 时间复杂度：$O(n^2)$ → $O(n)$（n 为序列长度）
- 实际加速：10-50x（取决于序列长度）

---

### Q1.2: KV Cache 的内存开销如何计算？

**回答要点**：

**单层 Cache 大小**：
```python
# 每层的 KV cache
memory_per_layer = 2 * batch_size * num_kv_heads * seq_len * head_dim * bytes_per_element

# 示例：Qwen3-0.6B (float16)
# - num_kv_heads = 2
# - head_dim = 64
# - seq_len = 2048
# - batch_size = 1
# - bytes_per_element = 2 (float16)

memory_per_layer = 2 * 1 * 2 * 2048 * 64 * 2 = 1,048,576 bytes = 1 MB
```

**全模型 Cache**：
```python
# Qwen3-0.6B 有 28 层
total_memory = 28 * 1 MB = 28 MB (per request, seq_len=2048)
```

**批处理场景**：
```python
# batch_size = 16, seq_len = 2048
total_memory = 28 * 16 MB = 448 MB
```

**追问：如何减少 KV Cache 内存？**

**回答**：
1. **GQA (Grouped Query Attention)**：减少 KV heads
   - MHA: 16 Q heads, 16 KV heads
   - GQA: 16 Q heads, 2 KV heads → 8x 内存减少
2. **MQA (Multi-Query Attention)**：所有 Q 共享一个 KV
   - 16 Q heads, 1 KV head → 16x 内存减少
3. **Paged Attention** (M3)：分页管理，避免连续内存
4. **Quantization**：int8 或 int4 KV cache

---

### Q1.3: Prefill 和 Decode 阶段的 KV Cache 如何使用？

**回答要点**：

**Prefill 阶段**（处理 prompt）：
```python
# 输入: "Hello, how are you?" → [token1, token2, ..., token6]
input_ids = [9906, 11, 1268, 527, 499, 30]  # shape: [1, 6]
positions = [0, 1, 2, 3, 4, 5]

# 一次性计算所有 token 的 Q, K, V
Q = [Q₀, Q₁, Q₂, Q₃, Q₄, Q₅]  # shape: [1, 16, 6, 64]
K = [K₀, K₁, K₂, K₃, K₄, K₅]  # shape: [1, 2, 6, 64]
V = [V₀, V₁, V₂, V₃, V₄, V₅]  # shape: [1, 2, 6, 64]

# Attention 计算（需要 causal mask）
attn = softmax(Q @ Kᵀ + mask) @ V

# 初始化 cache
kv_cache = (K, V)  # shape: [1, 2, 6, 64]
```

**Decode 阶段**（生成每个 token）：
```python
# Step 1: 生成第 7 个 token
input_ids = [358]  # shape: [1, 1]
positions = [6]

# 只计算新 token 的 Q, K, V
Q₆ = ...  # shape: [1, 16, 1, 64]
K₆ = ...  # shape: [1, 2, 64] (3D!)
V₆ = ...  # shape: [1, 2, 64]

# 追加到 cache
K_cached = [K₀, K₁, K₂, K₃, K₄, K₅, K₆]  # shape: [1, 2, 7, 64]
V_cached = [V₀, V₁, V₂, V₃, V₄, V₅, V₆]  # shape: [1, 2, 7, 64]

# Attention（不需要 mask，可以看所有历史）
attn = softmax(Q₆ @ K_cachedᵀ) @ V_cached

# Step 2: 生成第 8 个 token
Q₇ @ [K₀, ..., K₆, K₇]ᵀ @ [V₀, ..., V₆, V₇]
...
```

**关键区别**：

| 阶段    | 输入形状         | KV 形状                                | Mask             | 计算量   |
| ------- | ---------------- | -------------------------------------- | ---------------- | -------- |
| Prefill | [batch, seq_len] | [batch, heads, seq_len, dim] (4D)      | 需要 causal mask | $O(n^2)$ |
| Decode  | [batch, 1]       | [batch, heads, dim] (3D) → 追加后变 4D | 不需要           | $O(n)$   |

---

### Q1.4: 为什么 M1 使用连续内存 KV Cache，而 M3 要改用 Paged Attention？

**回答要点**：

**M1 方案（连续内存）**：
```python
# 每次追加都创建新 tensor
key_cache = torch.cat([key_cache, new_key], dim=2)

问题：
1. 需要连续内存块
2. torch.cat 会复制所有数据
3. 内存碎片化
4. 最大序列长度受限
```

**内存示例**：
```
Request 1: [K₀, K₁, K₂, K₃, ...] (2048 tokens)
Request 2: [K₀, K₁, K₂, ...] (512 tokens, 但预留 2048)
Request 3: [K₀, K₁, ...] (256 tokens, 但预留 2048)

浪费的内存 = (2048-512) + (2048-256) = 3328 tokens
```

**M3 方案（Paged Attention）**：
```python
# 分页管理，类似操作系统的虚拟内存
Page 0: [K₀, K₁, ..., K₁₅]  # 16 tokens per page
Page 1: [K₁₆, K₁₇, ..., K₃₁]
...

# 按需分配，无浪费
Request 1: [Page0, Page1, ..., Page127]  # 2048 tokens = 128 pages
Request 2: [Page128, ..., Page159]        # 512 tokens = 32 pages
Request 3: [Page160, ..., Page175]        # 256 tokens = 16 pages
```

**优势**：
- ✅ 内存利用率接近 100%
- ✅ 支持变长序列
- ✅ 便于多请求共享（shared prefix）
- ✅ 避免内存碎片

---

## 2. Attention 机制相关

### Q2.1: 为什么 Attention 要除以 $\sqrt{d_k}$？

**回答要点**：

**数学推导**：

假设 $Q$ 和 $K$ 的元素是独立同分布的随机变量，均值 0，方差 1：
- $Q, K \sim \mathcal{N}(0, 1)$

点积结果：
$$\text{score} = Q \cdot K = \sum_{i=1}^{d_k} Q_i K_i$$

根据中心极限定理：
- $\mathbb{E}[\text{score}] = 0$
- $\text{Var}[\text{score}] = d_k$（因为 $d_k$ 个独立变量相加）

**问题**：
- 当 $d_k$ 很大时（如 64、128），方差会很大
- Softmax 输入方差大 → 输出接近 one-hot → 梯度接近 0

**解决**：
$$\text{score}_{\text{scaled}} = \frac{Q \cdot K}{\sqrt{d_k}}$$

此时：
$$\text{Var}[\text{score}_{\text{scaled}}] = \frac{d_k}{d_k} = 1$$

**实验验证**：
```python
# 不 scale
d_k = 64
scores = torch.randn(1, 64) @ torch.randn(64, 100)
print(scores.var())  # ~64

# Scale
scores_scaled = scores / (64 ** 0.5)
print(scores_scaled.var())  # ~1
```

**追问：有没有其他 scale 方法？**

**回答**：
1. **可学习 scale**：$\text{score} = Q \cdot K / \alpha$，其中 $\alpha$ 是可学习参数
2. **固定 scale**：某些模型用固定值（如 8）
3. **QK Norm**：对 Q 和 K 分别做 LayerNorm（Qwen3 使用）

---

### Q2.2: 什么是 Grouped Query Attention (GQA)？为什么要用它？

**回答要点**：

**三种 Attention 模式**：

1. **MHA (Multi-Head Attention)**：
```python
num_heads = 16
num_kv_heads = 16  # 每个 head 独立的 K, V

Q: [batch, 16, seq, 64]
K: [batch, 16, seq, 64]  # 16 份独立的 K
V: [batch, 16, seq, 64]  # 16 份独立的 V
```

2. **MQA (Multi-Query Attention)**：
```python
num_heads = 16
num_kv_heads = 1  # 所有 head 共享 K, V

Q: [batch, 16, seq, 64]
K: [batch, 1, seq, 64]  # 只有 1 份 K
V: [batch, 1, seq, 64]  # 只有 1 份 V
```

3. **GQA (Grouped Query Attention)**：
```python
num_heads = 16
num_kv_heads = 2  # 8 个 Q heads 共享 1 个 KV head

Q: [batch, 16, seq, 64]
K: [batch, 2, seq, 64]  # 2 份 K
V: [batch, 2, seq, 64]  # 2 份 V

# Q heads 分组
Group 0: Q[0:8] 使用 K[0], V[0]
Group 1: Q[8:16] 使用 K[1], V[1]
```

**性能对比**：

| 模式 | KV Cache 内存 | 质量 | 推理速度 |
| ---- | ------------- | ---- | -------- |
| MHA  | 100%          | 最好 | 慢       |
| GQA  | 12.5% (2/16)  | 很好 | 快       |
| MQA  | 6.25% (1/16)  | 好   | 最快     |

**实现细节**：
```python
# GQA 时需要重复 KV heads
batch_size, num_heads, seq_len, head_dim = query.shape
_, num_kv_heads, _, _ = key.shape

if num_heads > num_kv_heads:
    # 重复 KV
    num_repeats = num_heads // num_kv_heads
    key = key.unsqueeze(2).expand(
        batch_size, num_kv_heads, num_repeats, seq_len, head_dim
    ).reshape(batch_size, num_heads, seq_len, head_dim)
    # value 同理
```

**为什么 GQA 有效**：
- Q 主要负责"查询"（多样性重要）
- K, V 主要负责"内容"（可以共享）
- 实验表明 GQA 质量接近 MHA，但内存和速度大幅提升

---

### Q2.3: Causal Mask 是什么？如何实现？

**回答要点**：

**定义**：
- Causal Mask 确保每个 token 只能看到**自己和之前**的 token
- 防止信息泄露（生成时不能看未来）

**实现**：
```python
def create_causal_mask(seq_len_q, seq_len_k):
    # 创建全 1 矩阵
    mask = torch.ones(seq_len_q, seq_len_k)
    
    # 保留上三角（不包括对角线）
    mask = torch.triu(mask, diagonal=seq_len_k - seq_len_q + 1)
    
    # 将 1 替换为 -inf（禁止 attend）
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    return mask
```

**示例**：
```python
# Prefill: seq_len_q = seq_len_k = 4
mask = create_causal_mask(4, 4)
"""
[[  0, -∞, -∞, -∞],
 [  0,   0, -∞, -∞],
 [  0,   0,   0, -∞],
 [  0,   0,   0,   0]]

解释：
- Token 0 只能看 Token 0
- Token 1 可以看 Token 0, 1
- Token 2 可以看 Token 0, 1, 2
- Token 3 可以看 Token 0, 1, 2, 3
"""

# Decode: seq_len_q = 1, seq_len_k = 5
mask = create_causal_mask(1, 5)
"""
[[0, 0, 0, 0, 0]]

解释：新 token 可以看所有历史 token
"""
```

**应用**：
```python
# Attention 计算
attn_weights = Q @ K.transpose(-2, -1)
attn_weights = attn_weights + mask  # 加 mask
attn_weights = F.softmax(attn_weights, dim=-1)
```

**为什么加 -inf**：
$$\text{softmax}(x_i + (-\infty)) = \frac{e^{x_i} \cdot e^{-\infty}}{\sum_j e^{x_j}} = \frac{e^{x_i} \cdot 0}{\sum_j e^{x_j}} = 0$$

**追问：Encoder-Decoder 模型的 mask 有什么不同？**

**回答**：
- **Encoder**: 双向 attention，不需要 causal mask
- **Decoder Self-Attention**: Causal mask（同上）
- **Decoder Cross-Attention**: 不需要 causal mask（可以看完整 encoder 输出）

---

## 3. 位置编码相关

### Q3.1: RoPE 是什么？为什么比传统位置编码更好？

**回答要点**：

**传统位置编码（Sinusoidal PE）**：
```python
# 在输入加位置编码
x = embedding(tokens) + positional_encoding(positions)
```

**问题**：
- 位置信息在深层网络中会被稀释
- 外推性差（训练 512，推理 2048 会失效）

**RoPE (Rotary Position Embedding)**：
```python
# 在 Attention 中旋转 Q 和 K
Q_rot = rotate(Q, position)
K_rot = rotate(K, position)
attn = softmax(Q_rot @ K_rotᵀ) @ V
```

**核心思想**：
- 将位置信息编码为**旋转**
- 在复平面上旋转向量

**数学原理**：

对于位置 $m$ 和 $n$：
$$\text{score}(m, n) = q_m^T k_n = (R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{m-n} k$$

其中 $R_\theta$ 是旋转矩阵：
$$R_\theta = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}$$

**结论**：Attention score 只依赖**相对位置** $m-n$！

**实现**：
```python
# 预计算旋转频率
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
# base = 10000, dim = 64
# inv_freq = [1.0, 0.1, 0.01, 0.001, ...]

# 计算 cos 和 sin
t = torch.arange(seq_len)  # [0, 1, 2, ...]
freqs = torch.outer(t, inv_freq)  # [seq_len, dim//2]
cos = freqs.cos()
sin = freqs.sin()

# 应用旋转
x1, x2 = x.chunk(2, dim=-1)
x_rot = torch.cat([
    x1 * cos - x2 * sin,
    x1 * sin + x2 * cos,
], dim=-1)
```

**优势**：
1. ✅ **相对位置**：自然编码相对位置信息
2. ✅ **外推性**：可以推理更长序列
3. ✅ **无参数**：不增加模型参数
4. ✅ **保持内积**：不破坏 Q·K 的语义

**追问：RoPE 如何支持长序列外推？**

**回答**：
```python
# 方法1：线性插值
rope_scaling = {"type": "linear", "factor": 2.0}
# 原本 max_len=2048，现在支持 4096

# 方法2：NTK-aware scaling
rope_scaling = {"type": "ntk", "factor": 2.0}
# 调整 base 值，更好的外推性

# 方法3：YaRN (Yet another RoPE extensioN)
rope_scaling = {"type": "yarn", "factor": 4.0}
# 混合多种策略
```

---

### Q3.2: 为什么 RoPE 只应用在 Q 和 K，不应用在 V？

**回答要点**：

**Attention 机制分解**：
$$\text{Attention}(Q, K, V) = \underbrace{\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)}_{\text{权重计算}} \underbrace{V}_{\text{内容聚合}}$$

**位置信息的作用**：
- **QK 计算**：决定"哪些位置重要"（需要位置信息）
- **V 计算**：聚合"内容"（不需要位置信息）

**数学验证**：

如果对 V 也应用旋转：
$$\text{output} = \text{softmax}(Q_{\text{rot}} K_{\text{rot}}^T) V_{\text{rot}}$$

问题：
- V 的旋转不参与权重计算
- 只是对输出做额外变换
- 不增加任何位置信息，反而引入干扰

**实验证明**：
- 只旋转 QK：性能最好
- 旋转 QKV：性能下降
- 只旋转 V：完全失效

**直觉理解**：
```
Q: "我想找什么位置的信息？" （需要位置）
K: "我在哪个位置？" （需要位置）
V: "我的内容是什么？" （只需要内容，不需要位置）
```

---

## 4. 采样策略相关

### Q4.1: Greedy、Top-k、Top-p、Temperature 有什么区别？如何选择？

**回答要点**：

**1. Greedy Sampling（贪心）**：
```python
next_token = torch.argmax(logits, dim=-1)
```
- **优点**：确定性，可复现
- **缺点**：重复、无聊、陷入循环
- **适用场景**：翻译、摘要等需要确定性的任务

**2. Temperature Scaling（温度）**：
```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```
- **temperature < 1.0**：分布更尖锐（更确定）
- **temperature = 1.0**：原始分布
- **temperature > 1.0**：分布更平滑（更随机）

**直觉**：
```
原始 logits: [5.0, 3.0, 1.0]
原始 probs:  [0.84, 0.16, 0.00]

T = 0.5 (冷): [0.98, 0.02, 0.00]  ← 更保守
T = 1.0:      [0.84, 0.16, 0.00]  ← 标准
T = 2.0 (热): [0.62, 0.32, 0.06]  ← 更多样
```

**3. Top-k Sampling**：
```python
# 只保留概率最高的 k 个 token
top_k_values, top_k_indices = torch.topk(logits, k)
filtered_logits = torch.full_like(logits, float('-inf'))
filtered_logits.scatter_(-1, top_k_indices, top_k_values)
probs = F.softmax(filtered_logits, dim=-1)
```
- **k=1**：等价于 Greedy
- **k=50**：常用值，平衡多样性和质量
- **问题**：固定 k 不够灵活（有时需要更多/更少选择）

**4. Top-p (Nucleus) Sampling**：
```python
# 动态选择，保留累积概率 >= p 的最小集合
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
mask = cumsum_probs > p
# 移除 mask 为 True 的 token
```
- **p=0.9**：常用值
- **优势**：动态调整候选集大小
- **例子**：
  ```
  probs = [0.6, 0.25, 0.1, 0.03, 0.02]
  p = 0.9
  累积: [0.6, 0.85, 0.95, ...]
  保留: [0.6, 0.25, 0.1]  # 3 个 token
  ```

**5. Min-p Sampling**：
```python
# 过滤概率 < min_p * max_prob 的 token
threshold = min_p * max(probs)
mask = probs < threshold
```
- **min_p=0.05**：常用值
- **优势**：过滤长尾，但保留相对重要的选择

**组合使用**（推荐）：
```python
# 1. Temperature: 调整分布
logits = logits / 0.7

# 2. Min-p: 粗过滤
logits = apply_min_p(logits, 0.05)

# 3. Top-k: 固定上限
logits = apply_top_k(logits, 50)

# 4. Top-p: 动态调整
logits = apply_top_p(logits, 0.95)

# 5. 采样
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**选择建议**：

| 任务     | 推荐配置                        |
| -------- | ------------------------------- |
| 创意写作 | T=0.8-1.0, top_p=0.95, top_k=50 |
| 对话     | T=0.7, top_p=0.9, top_k=40      |
| 翻译     | Greedy 或 T=0.3                 |
| 代码生成 | T=0.2-0.5, top_p=0.95           |
| 摘要     | T=0.3-0.5, top_p=0.9            |

---

### Q4.2: 为什么采样顺序是 Temperature → Min-p → Top-k → Top-p？

**回答要点**：

**顺序逻辑**：

1. **Temperature（全局调整）**：
   - 作用：调整整个分布的陡峭度
   - 原因：影响后续所有过滤步骤
   - 必须最先应用

2. **Min-p（粗过滤）**：
   - 作用：去除长尾（概率过低的 token）
   - 原因：快速减少候选集，提升后续效率
   - 相对宽松的过滤

3. **Top-k（固定过滤）**：
   - 作用：保留固定数量的候选
   - 原因：设置硬上限，防止候选过多
   - 固定数量，不依赖概率分布

4. **Top-p（精细过滤）**：
   - 作用：基于累积概率动态调整
   - 原因：在 top-k 基础上进一步精细化
   - 动态数量，适应分布特点

**示例**：
```python
原始 logits: [5.0, 4.5, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1, ...]

# Step 1: Temperature (T=0.7)
scaled: [7.14, 6.43, 5.71, 4.29, 2.86, 1.43, 0.71, 0.14, ...]
probs: [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01, 0.00, ...]

# Step 2: Min-p (0.05 * 0.45 = 0.0225)
过滤: probs < 0.0225
保留: [0.45, 0.25, 0.15, 0.08, 0.04, 0.02]  # 6 个

# Step 3: Top-k (k=5)
保留: [0.45, 0.25, 0.15, 0.08, 0.04]  # 5 个

# Step 4: Top-p (p=0.9)
累积: [0.45, 0.70, 0.85, 0.93, ...]
保留: [0.45, 0.25, 0.15, 0.08]  # 4 个（累积到 0.93 > 0.9）
```

**如果顺序错误会怎样？**

错误顺序：Top-p → Temperature
```python
# Step 1: Top-p (p=0.9) on 原始分布
保留: [0.6, 0.3, 0.1]

# Step 2: Temperature (T=0.5)
结果: [0.8, 0.15, 0.05]
# 问题：Temperature 失去作用（已经过滤了）
```

正确顺序：Temperature → Top-p
```python
# Step 1: Temperature (T=0.5)
分布: [0.8, 0.15, 0.03, 0.01, 0.01]

# Step 2: Top-p (p=0.9)
保留: [0.8, 0.15]
# 正确：基于调整后的分布过滤
```

---

### Q4.3: 如何实现可复现的随机采样？

**回答要点**：

**问题**：
- `torch.multinomial` 是随机的
- 相同输入，每次输出不同

**解决方案1：设置全局随机种子**
```python
torch.manual_seed(42)
next_token = torch.multinomial(probs, num_samples=1)
```
**问题**：影响所有随机操作

**解决方案2：使用 Generator（推荐）**
```python
generator = torch.Generator(device='cuda')
generator.manual_seed(42)
next_token = torch.multinomial(probs, num_samples=1, generator=generator)
```
**优势**：
- 独立的随机流
- 不影响其他随机操作
- 每个请求可以有独立的 seed

**实现细节**：
```python
class Sampler:
    def __init__(self):
        self._generator = None
    
    def sample(self, logits, sampling_params):
        # 设置随机种子
        if sampling_params.seed is not None:
            if self._generator is None:
                self._generator = torch.Generator(device=logits.device)
            self._generator.manual_seed(sampling_params.seed)
        
        # 采样
        if sampling_params.sampling_type == SamplingType.GREEDY:
            sampled_tokens = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            sampled_tokens = torch.multinomial(
                probs,
                num_samples=1,
                generator=self._generator,  # 使用独立 generator
            ).squeeze(-1)
        
        return sampled_tokens
```

**验证可复现性**：
```python
# 测试
engine = LLMEngine(...)
params1 = SamplingParams(seed=42, temperature=0.8)
params2 = SamplingParams(seed=42, temperature=0.8)

output1 = engine.generate("Hello", params1)
output2 = engine.generate("Hello", params2)

assert output1.outputs[0].text == output2.outputs[0].text  # 相同！
```

**注意事项**：
- Greedy 采样天然可复现（无需 seed）
- 不同 PyTorch 版本可能有差异
- GPU 和 CPU 可能产生不同结果（浮点精度）

---

## 5. 模型架构相关

### Q5.1: 为什么 Qwen3 使用 RMSNorm 而不是 LayerNorm？

**回答要点**：

**LayerNorm**：
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_i x_i$（均值）
- $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$（标准差）

**RMSNorm**：
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

其中：
- $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$（均方根）

**区别**：
1. **Re-centering（减均值）**：
   - LayerNorm: 有
   - RMSNorm: 无
2. **计算量**：
   - LayerNorm: 2 次遍历（计算均值、标准差）
   - RMSNorm: 1 次遍历（只计算 RMS）

**为什么 RMSNorm 更好**：

**理论分析**：
- 论文《Root Mean Square Layer Normalization》发现：
  - Re-centering 对性能提升有限
  - Scaling（除以标准差）才是关键
- RMSNorm 保留 scaling，去掉 re-centering

**性能提升**：
- 计算量：减少 ~10-15%
- 速度：加速 ~5-10%
- 精度：几乎无损

**实验结果**：
```python
# 在相同训练设置下
LayerNorm: PPL = 12.3, Time = 100s
RMSNorm:   PPL = 12.4, Time = 93s  # 略快，质量相当
```

**代码对比**：
```python
# LayerNorm
mean = x.mean(-1, keepdim=True)
var = ((x - mean) ** 2).mean(-1, keepdim=True)
x_norm = (x - mean) / sqrt(var + eps)  # 需要减均值

# RMSNorm
rms = sqrt(x.pow(2).mean(-1, keepdim=True))
x_norm = x / (rms + eps)  # 不需要减均值，更简单
```

**追问：为什么 RMSNorm 有 Fused Residual 版本？**

**回答**：
```python
# 传统方式（2 次内存访问）
residual = x + residual  # 第 1 次
x_norm = RMSNorm(residual)  # 第 2 次

# Fused 方式（1 次内存访问）
x_norm, new_residual = RMSNorm(x, residual)
# 内部一次性完成：new_residual = x + residual, x_norm = norm(new_residual)
```

**好处**：
- 减少内存带宽
- 减少 kernel launch
- 提升 5-10% 性能

---

### Q5.2: 为什么 Qwen3 的 MLP 使用 SiLU 而不是 ReLU？

**回答要点**：

**激活函数对比**：

1. **ReLU**：
$$\text{ReLU}(x) = \max(0, x)$$
- 优点：简单，计算快
- 缺点：非光滑，梯度消失（x<0）

2. **GELU**：
$$\text{GELU}(x) = x \cdot \Phi(x)$$
- 优点：光滑，性能好
- 缺点：计算复杂（涉及误差函数）

3. **SiLU (Swish)**：
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
- 优点：光滑，接近 GELU 性能，计算简单
- 缺点：比 ReLU 稍慢

**为什么选 SiLU**：

**性能**：
```python
# 实验结果（相同训练 setup）
ReLU: PPL = 15.2
GELU: PPL = 12.8
SiLU: PPL = 12.9  # 接近 GELU
```

**计算效率**：
```python
# 相对速度（ReLU = 1.0）
ReLU: 1.0
GELU: 0.85  # 慢 15%
SiLU: 0.95  # 慢 5%
```

**光滑性**：
```python
# 导数
ReLU':  {0, 1}  # 不连续
SiLU':  连续且光滑
```

**Gated MLP**：
```python
# Qwen3 使用 Gated MLP
gate = linear_gate(x)
up = linear_up(x)
output = silu(gate) * up  # SiLU 作为 gate 激活

# 为什么：
# - SiLU 输出 [0, +∞)，适合做 gate
# - 光滑性有助于梯度流动
```

**Fused 实现**：
```python
# 合并 gate 和 up 投影
gate_up = linear_gate_up(x)  # [hidden, 2*intermediate]
gate, up = gate_up.chunk(2, dim=-1)
output = F.silu(gate) * up  # 一次 kernel

# 好处：
# - 单个 linear layer 更高效
# - 减少内存访问
```

---

### Q5.3: 为什么要合并 QKV 投影为一个 Linear Layer？

**回答要点**：

**传统方式（3 个 Linear）**：
```python
Q = linear_q(x)  # [hidden, q_size]
K = linear_k(x)  # [hidden, kv_size]
V = linear_v(x)  # [hidden, kv_size]
```

**合并方式（1 个 Linear）**：
```python
QKV = linear_qkv(x)  # [hidden, q_size + 2*kv_size]
Q, K, V = QKV.split([q_size, kv_size, kv_size], dim=-1)
```

**优势**：

**1. 减少 Kernel Launch**：
```python
# 3 个 Linear
kernel_launch × 3  # 每次有启动开销

# 1 个 Linear
kernel_launch × 1  # 开销减少 2/3
```

**2. 内存访问优化**：
```python
# 3 个 Linear（3 次读取 x）
读 x → 计算 Q → 写 Q
读 x → 计算 K → 写 K
读 x → 计算 V → 写 V

# 1 个 Linear（1 次读取 x）
读 x → 计算 QKV → 写 QKV
```

**3. 计算效率**：
```python
# 单个大矩阵乘法更高效
[batch*seq, hidden] @ [hidden, q_size+2*kv_size]
# GPU 可以更好地利用 tensor cores
```

**性能提升**：
```python
# 实测（Qwen3-0.6B, batch=1, seq=1）
3 个 Linear: 0.064 ms/token
1 个 Linear: 0.058 ms/token  # 快 ~10%
```

**权重加载**：
```python
# HuggingFace checkpoint 通常是分离的
state_dict = {
    'q_proj.weight': ...,
    'k_proj.weight': ...,
    'v_proj.weight': ...,
}

# 需要合并
qkv_weight = torch.cat([
    state_dict['q_proj.weight'],
    state_dict['k_proj.weight'],
    state_dict['v_proj.weight'],
], dim=0)
```

**注意事项**：
- 并非所有模型都合并（如 LLaMA）
- M1 使用 HF 原生模型（未合并）
- M2 可以考虑合并优化

---

## 6. 推理优化相关

### Q6.1: Prefill 和 Decode 的性能瓶颈分别是什么？

**回答要点**：

**Prefill 阶段**（处理 prompt）：

**特点**：
- 输入：多个 token（如 100 个）
- 计算：$O(n^2)$ 的 attention
- 并行度高：所有 token 同时计算

**瓶颈**：
```
计算密集型 (Compute-bound)

原因：
- 大量矩阵乘法（Q@K^T, attention@V）
- GPU 计算单元利用率高
- 内存带宽压力相对较小
```

**优化方向**：
- Flash Attention（减少内存访问，提升计算效率）
- Tensor Parallelism（分布式计算）
- 混合精度（FP16/BF16）

**Decode 阶段**（生成 token）：

**特点**：
- 输入：1 个 token
- 计算：$O(n)$ 的 attention
- 并行度低：单个 token

**瓶颈**：
```
内存带宽密集型 (Memory-bound)

原因：
- 需要读取整个 KV cache
- 计算量小（单个 token）
- GPU 计算单元利用率低（<30%）
```

**数据**：
```python
# Qwen3-0.6B, seq_len=2048
Prefill: 
  - FLOPS: ~1.2 TFLOPs
  - Memory: ~100 MB
  - Time: 262 ms
  - GPU Util: ~90%

Decode:
  - FLOPS: ~0.6 GFLOPs (每 token)
  - Memory: ~50 MB (读取 KV cache)
  - Time: 64 ms (每 token)
  - GPU Util: ~25%
```

**优化方向**：
- Continuous Batching（增加并行度）
- Paged Attention（减少内存访问）
- Speculative Decoding（减少 decode 步数）

**对比图**：
```
Prefill:
GPU: [████████████████████] 90% (计算饱和)
MEM: [███░░░░░░░░░░░░░░░░] 15% (内存空闲)

Decode:
GPU: [█████░░░░░░░░░░░░░░] 25% (计算空闲)
MEM: [████████████████░░░] 80% (内存饱和)
```

---

### Q6.2: 什么是 Continuous Batching？为什么能提升吞吐量？

**回答要点**：

**传统 Batching（Static Batching）**：
```python
# 所有请求同时开始和结束
requests = [req1, req2, req3, req4]  # batch_size=4
while not all_finished:
    logits = model(requests)
    sample(logits)

# 问题：
req1: [=============================] (30 tokens)
req2: [===============] (15 tokens) ················  ← 等待
req3: [====================] (20 tokens) ··········  ← 等待
req4: [========================] (24 tokens) ······  ← 等待
                                  ↑
                            浪费的 GPU 时间
```

**Continuous Batching（Dynamic Batching）**：
```python
# 请求可以动态加入/离开 batch
active_requests = []

while True:
    # 添加新请求
    active_requests += new_requests()
    
    # 移除完成的请求
    active_requests = [r for r in active_requests if not r.finished]
    
    # 执行一步
    logits = model(active_requests)
    sample(logits)

# 效果：
Batch 1: [req1, req2, req3, req4]
Batch 2: [req1, req2, req3, req4]
Batch 3: [req1, req3, req4, req5]  ← req2 完成，req5 加入
Batch 4: [req1, req3, req4, req5, req6]
Batch 5: [req1, req4, req5, req6]  ← req3 完成
...
```

**性能提升**：

**吞吐量**：
```python
# Static Batching
吞吐量 = batch_size / max_completion_time
       = 4 / 30 tokens = 0.13 req/token

# Continuous Batching
吞吐量 = 总请求数 / 总时间
       ≈ 2-3x 提升
```

**GPU 利用率**：
```python
# Static Batching
平均 batch size = (4+4+4+3+2+1) / 6 = 3.0

# Continuous Batching
平均 batch size = (4+4+4+5+5+4+...) / N ≈ 4.5
                ↑ 持续保持高 batch size
```

**实现挑战**：

1. **KV Cache 管理**：
   - 每个请求有不同的 cache 大小
   - 需要动态分配内存

2. **Attention 计算**：
   - 不同请求的 seq_len 不同
   - 需要支持 variable-length attention

3. **调度策略**：
   - 如何选择下一个 batch？
   - FCFS、Priority、Fairness？

**M2 实现方式**：
```python
class Scheduler:
    def __init__(self):
        self.waiting = []
        self.running = []
    
    def schedule(self) -> List[Request]:
        # 添加等待的请求到 running
        while len(self.running) < MAX_BATCH_SIZE and self.waiting:
            req = self.waiting.pop(0)
            self.running.append(req)
        
        # 移除完成的请求
        self.running = [r for r in self.running if not r.is_finished()]
        
        return self.running
```

---

### Q6.3: 如何优化 Transformer 推理的内存使用？

**回答要点**：

**内存组成**：
```python
Total Memory = Model Weights + KV Cache + Activations + Temporary Buffers

# Qwen3-0.6B 示例（batch=1, seq=2048, fp16）
Model Weights: 1.2 GB
KV Cache:      28 MB (per request)
Activations:   50 MB (per layer)
Temporary:     100 MB
Total:         ~1.4 GB (单请求)
```

**优化策略**：

**1. 模型权重优化**：
```python
# 量化
FP16:  1.2 GB
INT8:  600 MB (-50%)
INT4:  300 MB (-75%)

# 稀疏化
剪枝: 减少 10-30% 权重

# LoRA（推理时合并）
只加载 base model + LoRA adapters
```

**2. KV Cache 优化**：

**GQA (Grouped Query Attention)**：
```python
# MHA: 16 Q heads, 16 KV heads
KV Cache = 28 MB

# GQA: 16 Q heads, 2 KV heads
KV Cache = 3.5 MB (-87.5%)
```

**Paged Attention (M3)**：
```python
# 传统：连续内存，预留最大长度
每请求 = max_seq_len * kv_size = 浪费多

# Paged：按需分配，利用率 100%
每请求 = actual_seq_len * kv_size
```

**KV Cache Quantization**：
```python
# FP16 → INT8
KV Cache 内存 -50%
精度损失 <1%
```

**3. Activation 优化**：

**Activation Checkpointing (Gradient Checkpointing)**：
```python
# 推理时不需要（只在训练时）
```

**Fused Kernels**：
```python
# 减少中间结果存储
# 例如：RMSNorm + Residual 一次性完成
```

**4. Batching 优化**：

**Continuous Batching**：
```python
# 动态管理，避免 padding
Static: [req1(2048), req2(512)] → 2*2048 = 4096 tokens
        浪费: 2048 - 512 = 1536 tokens

Continuous: [req1(2048), req2(512)] → 2048+512 = 2560 tokens
            节省: 1536 tokens (-38%)
```

**5. FlashAttention (M4)**：
```python
# 减少 attention 的内存占用
# Naive: O(n²) 内存
# Flash: O(n) 内存

# Qwen3-0.6B, seq=2048
Naive: 200 MB attention 中间结果
Flash: 20 MB (-90%)
```

**综合优化效果**：
```python
# 基础配置（FP16）
单请求内存: 1.4 GB
最大 batch: 16 (24GB GPU)

# 优化后（GQA + Paged + Flash + INT8 KV）
单请求内存: 0.3 GB
最大 batch: 70 (24GB GPU)
吞吐量: 4-5x 提升
```

---

## 7. 系统设计相关

### Q7.1: 为什么需要分层设计（Engine → Executor → Worker → ModelRunner）？

**回答要点**：

**架构层次**：
```
User API:       LLMEngine
                   ↓
Logic Layer:    Scheduler, Sampler, Processor
                   ↓
Execution:      GPUExecutor (接口层)
                   ↓
Device:         GPUWorker (设备管理)
                   ↓
Model:          ModelRunner (模型运行)
                   ↓
Hardware:       GPU/CUDA
```

**每层职责**：

**1. LLMEngine（业务逻辑）**：
```python
职责：
- 提供用户 API (generate, add_request)
- 协调各组件（scheduler, sampler, executor）
- 管理请求生命周期
- 构造输出

不关心：
- GPU 如何执行
- 模型如何加载
- 内存如何分配
```

**2. GPUExecutor（执行接口）**：
```python
职责：
- 统一的执行接口
- 隐藏分布式细节（单 GPU vs 多 GPU）
- 提供 get_next_token_logits() 等高层接口

不关心：
- 请求如何调度
- 采样如何进行
- 具体设备细节
```

**3. GPUWorker（设备管理）**：
```python
职责：
- 管理 GPU 设备
- 加载模型到设备
- 管理设备内存
- 创建 ModelRunner

不关心：
- 模型内部结构
- 前向传播细节
- 分布式通信
```

**4. ModelRunner（模型运行）**：
```python
职责：
- 准备模型输入
- 执行前向传播
- 管理 KV cache
- 返回 logits

不关心：
- 请求从哪来
- 结果如何处理
- 设备如何管理
```

**好处**：

**1. 关注点分离**：
```python
# Engine 关心业务逻辑
engine.generate(prompt, params)

# Executor 关心执行
executor.get_next_token_logits(tokens, pos)

# Worker 关心设备
worker = GPUWorker(model_config, device="cuda:0")

# Runner 关心模型
runner.execute_model(input_ids, positions)
```

**2. 便于扩展**：
```python
# M1: 单 GPU
executor = GPUExecutor(config, device="cuda")

# M6: 多 GPU Tensor Parallelism
executor = GPUExecutor(config, tensor_parallel_size=4)
# 内部创建 4 个 GPUWorker，用户无感知

# M7: Pipeline Parallelism
executor = PipelineExecutor(config, pipeline_stages=4)
```

**3. 便于测试**：
```python
# 单元测试：只测 ModelRunner
runner = ModelRunner(model, config, device)
logits = runner.execute_model(input_ids, positions)
assert logits.shape == expected_shape

# 集成测试：测 Engine
engine = LLMEngine(config)
output = engine.generate(prompt, params)
assert output.outputs[0].text != ""
```

**4. 便于维护**：
```python
# 修改模型运行逻辑：只改 ModelRunner
# 修改调度逻辑：只改 Scheduler
# 修改采样逻辑：只改 Sampler

# 不会相互影响
```

**对比 Monolithic 设计**：
```python
# 不分层（所有逻辑在一起）
class LLMEngine:
    def generate(self, prompt):
        # 设备管理
        model = load_model_to_gpu()
        # 模型运行
        logits = model(input_ids)
        # 采样
        next_token = sample(logits)
        # ...
        
# 问题：
# - 无法测试单个组件
# - 无法扩展（如添加多 GPU）
# - 代码难以理解
```

---

### Q7.2: HuggingFace 模型和自定义模型有什么区别？为什么 M1 选择 HF？

**回答要点**：

**自定义模型（原计划）**：
```python
# folovllm/model_executor/models/qwen.py
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(...)
    
    def forward(self, input_ids, positions, kv_caches):
        hidden = self.model(input_ids, positions, kv_caches)
        logits = self.lm_head(hidden)
        return logits

优点：
- ✅ 完全可控
- ✅ 可以优化（merged QKV, fused ops）
- ✅ 学习价值高

缺点：
- ❌ 权重加载复杂
- ❌ 架构必须完全匹配 HF
- ❌ 调试困难
- ❌ 维护成本高
```

**HuggingFace 模型（M1 实际）**：
```python
# 直接使用 HF
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.float16,
    device_map="cuda",
)

优点：
- ✅ 权重直接加载
- ✅ 稳定可靠
- ✅ 快速开发
- ✅ 与 HF 输出一致（便于验证）

缺点：
- ❌ 黑盒（难以深入优化）
- ❌ 学习价值较低
```

**为什么 M1 选择 HF**：

**1. 架构差异**：
```python
# 自定义模型
class Qwen3Attention:
    self.qkv_proj = nn.Linear(hidden, q_size + 2*kv_size)  # 合并

# HF 模型
class Qwen2Attention:
    self.q_proj = nn.Linear(hidden, q_size)
    self.k_proj = nn.Linear(hidden, kv_size)
    self.v_proj = nn.Linear(hidden, kv_size)
    self.q_norm = nn.LayerNorm(...)  # 额外的 norm
    self.k_norm = nn.LayerNorm(...)
```

**2. 权重映射复杂**：
```python
# 需要这样映射
hf_state_dict = torch.load("model.safetensors")
custom_state_dict = {}

# QKV 合并
q_weight = hf_state_dict['q_proj.weight']
k_weight = hf_state_dict['k_proj.weight']
v_weight = hf_state_dict['v_proj.weight']
custom_state_dict['qkv_proj.weight'] = torch.cat([q, k, v], dim=0)

# 处理 q_norm, k_norm（自定义模型没有）
# ... 复杂的逻辑

model.load_state_dict(custom_state_dict)  # 容易出错
```

**3. 稳定性优先**：
```python
# M1 目标：快速验证整体流程
# 自定义模型：调试权重加载可能花费大量时间
# HF 模型：开箱即用，专注核心逻辑
```

**权衡**：
```python
M1: 使用 HF（稳定性）
M2: 继续 HF（专注 scheduling）
M3: 考虑自定义（优化 KV cache）
M4: 自定义（Flash Attention）
M5: 自定义 + 优化（性能关键期）
```

**如何兼容 HF 模型**：
```python
# folovllm/model_loader.py
def _wrap_model_for_folovllm(self, model):
    # 添加 folovllm 需要的接口
    if not hasattr(model, 'compute_logits'):
        def compute_logits(hidden_states):
            return model.lm_head(hidden_states)
        
        import types
        model.compute_logits = types.MethodType(compute_logits, model)
    
    return model

# folovllm/worker/model_runner.py
def execute_model(self, token_ids, start_pos):
    # 检测 HF 模型
    if 'position_ids' in str(self.model.forward.__code__.co_varnames):
        # HF 模型
        outputs = self.model(
            input_ids=input_ids,
            position_ids=positions,
            past_key_values=self.past_key_values,  # HF cache
            use_cache=True,
        )
        logits = outputs.logits
        self.past_key_values = outputs.past_key_values
    else:
        # 自定义模型
        hidden = self.model(input_ids, positions, self.kv_caches)
        logits = self.model.compute_logits(hidden)
    
    return logits
```

---

## 8. 数值稳定性相关

### Q8.1: 为什么 Softmax 要在 FP32 下计算？

**回答要点**：

**问题背景**：
```python
# FP16 的数值范围
Max: 65504
Min: 6e-8
```

**Softmax 公式**：
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**FP16 的问题**：

**1. 上溢（Overflow）**：
```python
x = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float16)
exp_x = torch.exp(x)
# exp(30) ≈ 1e13 > 65504 → Overflow → inf

softmax = exp_x / exp_x.sum()
# inf / inf = nan
```

**2. 下溢（Underflow）**：
```python
x = torch.tensor([-100.0, -90.0, -80.0], dtype=torch.float16)
exp_x = torch.exp(x)
# exp(-100) ≈ 3e-44 < 6e-8 → Underflow → 0

softmax = exp_x / exp_x.sum()
# 0 / 0 = nan
```

**解决方案：FP32 + 数值稳定技巧**：
```python
def stable_softmax(x):
    # 1. 转 FP32
    x = x.float()
    
    # 2. 减去最大值（防止上溢）
    x_max = x.max(dim=-1, keepdim=True)
    x_shifted = x - x_max  # 最大值变为 0
    
    # 3. 计算 exp
    exp_x = torch.exp(x_shifted)  # 最大值的 exp 为 1，不会溢出
    
    # 4. 归一化
    softmax = exp_x / exp_x.sum(dim=-1, keepdim=True)
    
    return softmax
```

**数学证明**：
$$\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i}}{e^c} \cdot \frac{e^c}{\sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**实现**：
```python
# PyTorch 的 F.softmax 内置了数值稳定性
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

# 然后转回原 dtype
attn_weights = attn_weights.to(query.dtype)
```

**性能影响**：
```python
# FP16 softmax: 快但不稳定
# FP32 softmax: 慢 ~10% 但稳定

# 权衡：稳定性 > 性能
```

---

### Q8.2: RMSNorm 为什么要加 epsilon？如何选择 epsilon 值？

**回答要点**：

**RMSNorm 公式**：
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}$$

**为什么需要 epsilon**：

**问题：除零**：
```python
# 极端情况：x 全为 0
x = torch.zeros(128)
rms = torch.sqrt(x.pow(2).mean())  # rms = 0
x_norm = x / rms  # 0 / 0 = nan
```

**解决：加 epsilon**：
```python
rms = torch.sqrt(x.pow(2).mean() + eps)  # rms = sqrt(eps) ≠ 0
x_norm = x / rms  # 0 / sqrt(eps) = 0（正常）
```

**如何选择 epsilon**：

**常见值**：
```python
LayerNorm: eps = 1e-5
RMSNorm:   eps = 1e-6  # Qwen3 使用
           eps = 1e-8  # LLaMA 使用
```

**考虑因素**：

**1. 数值精度**：
```python
# FP32
最小正数 ≈ 1e-38
1e-6 是安全的

# FP16
最小正数 ≈ 6e-8
1e-6 是安全的
1e-8 可能有问题（接近极限）
```

**2. 对结果的影响**：
```python
# eps 太大
rms = sqrt(variance + 1e-3)
# 如果 variance 很小（如 1e-5），rms 主要由 eps 决定
# 导致归一化失效

# eps 太小
rms = sqrt(variance + 1e-10)
# 如果 variance 为 0，可能数值不稳定
```

**3. 实验验证**：
```python
# Qwen3 的选择
eps = 1e-6

原因：
- FP16/FP32 都安全
- 足够小，不影响归一化效果
- 足够大，避免数值问题
```

**实现**：
```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x_norm
```

**注意**：
```python
# torch.rsqrt = 1/sqrt
# 比 1/torch.sqrt 更快（单个 CUDA kernel）
```

---

### Q8.3: 为什么要在 FP16 和 FP32 之间转换？

**回答要点**：

**混合精度策略**：
```python
# 模型主体：FP16
model = model.half()  # 转 FP16

# 特定操作：FP32
x = x.float()  # 转 FP32
x = operation(x)
x = x.half()  # 转回 FP16
```

**为什么 FP16**：

**1. 内存**：
```python
FP32: 4 bytes/param
FP16: 2 bytes/param  # -50% 内存
```

**2. 速度**：
```python
# GPU Tensor Cores（专门为 FP16 设计）
FP16: ~2x 速度提升
```

**3. 带宽**：
```python
FP16: 传输数据减半
```

**为什么某些操作需要 FP32**：

**1. 累加操作**：
```python
# Softmax
sum = torch.sum(exp_x)  # 累加很多小数，FP16 精度不够

# LayerNorm / RMSNorm
mean = x.sum() / n  # 累加
variance = ((x - mean) ** 2).sum() / n  # 累加
```

**2. 数值范围大**：
```python
# Loss 计算
loss = -log(prob)  # log 可能产生很大/很小的值
```

**3. 梯度**：
```python
# 训练时
grad = compute_grad()  # FP16 梯度容易 underflow
grad = grad.float()  # 转 FP32
param = param - lr * grad  # FP32 更新
```

**实现模式**：

**Pattern 1: 局部 FP32**：
```python
def rmsnorm(x):
    input_dtype = x.dtype
    x = x.float()  # → FP32
    
    # 计算（FP32）
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    
    return x.to(input_dtype)  # → 回到原 dtype
```

**Pattern 2: Softmax**：
```python
attn_weights = Q @ K.T  # FP16
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # FP32
attn_weights = attn_weights.to(Q.dtype)  # → FP16
```

**性能权衡**：
```python
# 全 FP32
速度: 1.0x
内存: 1.0x
精度: 最高

# 全 FP16
速度: 2.0x
内存: 0.5x
精度: 可能不稳定

# 混合精度（推荐）
速度: 1.8x
内存: 0.5x
精度: 稳定
```

---

## 9. 总结：核心要点

### M1 最重要的 10 个概念

1. **KV Cache**：避免重复计算，是推理加速的核心
2. **Prefill vs Decode**：两阶段有不同的性能瓶颈
3. **Causal Mask**：确保 autoregressive 生成的正确性
4. **RoPE**：高效的相对位置编码
5. **GQA**：平衡内存和性能的 attention 变体
6. **Temperature/Top-k/Top-p**：控制生成质量和多样性
7. **RMSNorm**：比 LayerNorm 更高效的归一化
8. **Fused Operations**：减少内存访问，提升性能
9. **混合精度**：FP16 为主，FP32 for 稳定性
10. **分层架构**：便于扩展和维护

### M1 到 M2 的演进

```
M1: 单请求同步推理
  ↓
M2: 多请求 Continuous Batching
  - Scheduler（请求调度）
  - 异步接口
  - 动态 batching
  - 更高吞吐量
```

---

**文档完成！**

本文档涵盖了 M1 可能遇到的所有关键面试问题。

