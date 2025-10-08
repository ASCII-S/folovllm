# Milestone 1: 基础离线推理 - 学习笔记

本文档深入讲解 M1 阶段实现的核心技术原理。

---

## 📚 核心技术

### 1. KV Cache (键值缓存)

#### 原理

在 Transformer 解码过程中，每个新 token 的生成需要与之前所有 token 进行 attention 计算。如果每次都重新计算所有 token 的 K 和 V，会造成大量重复计算。

KV Cache 的核心思想：
- **Prefill 阶段**：处理完整 prompt，计算并缓存所有 token 的 K、V
- **Decode 阶段**：每次只计算新 token 的 K、V，追加到缓存中
- Attention 计算时，Q 是新 token 的查询，K/V 是缓存中的所有历史 token

#### 数学表示

```
传统方式（每步重算）:
  Step t: Attention(Q_0:t, K_0:t, V_0:t)
  时间复杂度: O(t^2)

KV Cache 方式:
  Step 0: K_cache = [K_0], V_cache = [V_0]
  Step t: K_cache.append(K_t), V_cache.append(V_t)
          Attention(Q_t, K_cache, V_cache)
  时间复杂度: O(t) per step
```

#### 实现要点

在 M1 中，我们使用**连续内存分配**：
- 每个 layer 维护一个 `(key_cache, value_cache)` tuple
- Shape: `[batch_size, num_kv_heads, seq_len, head_dim]`
- 每次 decode 通过 `torch.cat` 追加新 token

```python
# folovllm/attention/ops.py
if key_cache.numel() == 0:
    # 首次：初始化
    key_cache = key.unsqueeze(2)
else:
    # 后续：追加
    key = key.unsqueeze(2)
    key_cache = torch.cat([key_cache, key], dim=2)
```

#### 优劣分析

**优势**：
- ✅ 简单直观，易于实现和调试
- ✅ 适合单请求场景
- ✅ 避免重复计算，提升速度

**局限**（M3 将改进）：
- ❌ 连续内存分配效率低（concat 操作需要复制）
- ❌ 不同长度序列难以 batch
- ❌ 内存碎片化严重
- ❌ 无法动态调整序列（抢占、swap）

---

### 2. Transformer 推理流程

#### 两阶段推理

```
┌─────────────────────────────────────────────┐
│  Prefill Phase (预填充阶段)                   │
├─────────────────────────────────────────────┤
│  Input: 完整 prompt tokens                   │
│  Output: 第一个新 token + 初始化 KV Cache    │
│  特点:                                        │
│    - 一次性处理多个 token (并行)             │
│    - 需要 causal mask (下三角)               │
│    - 计算密集型 (matmul dominant)            │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Decode Phase (解码阶段)                      │
├─────────────────────────────────────────────┤
│  Input: 上一步生成的 token                    │
│  Output: 下一个 token + 更新 KV Cache        │
│  特点:                                        │
│    - 每次只处理 1 个 token                   │
│    - 不需要 mask（只 attend 历史）           │
│    - 内存带宽密集型 (IO bound)               │
│  循环直到:                                    │
│    - 遇到 EOS token                         │
│    - 达到 max_tokens                        │
│    - 满足 stop condition                    │
└─────────────────────────────────────────────┘
```

#### 详细流程

```python
# Prefill
tokens = tokenize(prompt)  # [1, 2, 3, 4]
hidden = model(tokens, positions=[0,1,2,3], kv_cache=None)
logits = lm_head(hidden)
next_token = sample(logits[-1])  # 只用最后一个位置的 logits

# Decode
for step in range(max_tokens):
    hidden = model([next_token], positions=[len(tokens)], kv_cache=kv_cache)
    logits = lm_head(hidden)
    next_token = sample(logits[0])
    if next_token == EOS: break
```

#### 性能指标

- **TTFT (Time To First Token)**: Prefill 时间
  - 主要影响因素：prompt 长度、模型大小
  - 优化方向：chunked prefill (M5)

- **TPOT (Time Per Output Token)**: 每个 decode token 的平均时间
  - 主要影响因素：模型大小、KV cache 大小
  - 优化方向：Flash Attention (M4), Paged Attention (M3)

- **Throughput**: tokens/second
  - 综合指标，受 batch size 影响大
  - 优化方向：Continuous Batching (M2)

---

### 3. Sampling 策略详解

#### Greedy Sampling (贪心采样)

```python
next_token = argmax(logits)
```

**特点**：
- ✅ 确定性，相同输入总是相同输出
- ✅ 速度快
- ❌ 容易产生重复文本
- ❌ 缺乏多样性

**使用场景**：翻译、摘要等需要确定性的任务

#### Temperature Scaling (温度缩放)

```python
logits = logits / temperature
probs = softmax(logits)
next_token = multinomial(probs)
```

**效果**：
- `temperature → 0`: 接近 greedy，确定性强
- `temperature = 1`: 原始分布
- `temperature > 1`: 更平滑，更随机

**原理**：温度调节概率分布的"尖锐度"

```
Original: [0.7, 0.2, 0.08, 0.02]
T=0.5:    [0.85, 0.12, 0.025, 0.005]  # 更尖锐
T=2.0:    [0.55, 0.28, 0.12, 0.05]    # 更平滑
```

#### Top-k Sampling

```python
top_k_values, top_k_indices = topk(logits, k)
# 将非 top-k 的 logits 设为 -inf
filtered_logits = full_like(logits, -inf)
filtered_logits[top_k_indices] = top_k_values
```

**特点**：
- 只考虑概率最高的 k 个 token
- 防止采样到低概率 token
- k 固定，不考虑概率分布形状

**问题**：有时 top-k 包含很多低质量 token，有时排除了合理选项

#### Top-p (Nucleus) Sampling

```python
sorted_probs, sorted_indices = sort(probs, descending=True)
cumsum_probs = cumsum(sorted_probs)
# 保留累积概率 < p 的 token
nucleus = cumsum_probs <= p
```

**特点**：
- 动态选择 token 数量
- 保留累积概率达到 p 的最小 token 集合
- 更符合概率分布的"自然形状"

**对比 Top-k**：
```
Distribution A: [0.5, 0.3, 0.1, 0.05, 0.05]
  Top-k (k=3): 选 3 个
  Top-p (p=0.9): 选 3 个

Distribution B: [0.9, 0.05, 0.03, 0.02]
  Top-k (k=3): 选 3 个 (包含 0.03, 0.02)
  Top-p (p=0.9): 只选 1 个 (0.9 已够)
```

#### Min-p Sampling

```python
max_prob = max(probs)
threshold = min_p * max_prob
mask = probs < threshold
```

**特点**：
- 相对于最大概率的阈值
- 避免"长尾"低质量 token
- 与 top-p 互补

#### 组合策略

实践中常组合使用：
```python
# 典型配置
temperature = 0.7      # 稍微随机
top_k = 50            # 粗过滤
top_p = 0.95          # 精细过滤
min_p = 0.05          # 去除长尾
```

执行顺序：`temperature → min_p → top_k → top_p → sample`

---

### 4. RoPE (Rotary Position Embedding)

#### 为什么需要位置编码？

Transformer 的 self-attention 是**置换不变**的（permutation invariant）：
```
Attention([Q1, Q2, Q3]) = Attention([Q3, Q1, Q2])
```

但语言是**位置敏感**的：
```
"dog bites man" ≠ "man bites dog"
```

所以需要注入位置信息。

#### RoPE 的核心思想

不是在输入加位置编码，而是**在 Q、K 向量上应用旋转变换**：

```python
# 对于位置 m 的 token
q_m = rotate(q, m * θ)
k_n = rotate(k, n * θ)

# Attention score
score = q_m^T k_n
      = rotate(q, m*θ)^T rotate(k, n*θ)
      = rotate(q^T k, (m-n)*θ)  # 只依赖相对位置!
```

#### 数学细节

旋转矩阵（2D 子空间）：
```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

对 d 维向量，分成 d/2 个 2D 子空间，每个独立旋转：
```
θ_i = base^(-2i/d),  i = 0, 1, ..., d/2-1
```

对位置 m：
```
RoPE(x, m) = [R(m*θ_0) x[0:2], R(m*θ_1) x[2:4], ...]
```

#### 优势

1. **相对位置编码**：自动编码相对位置关系
2. **长度外推**：训练长度 2k，推理时可扩展到 4k+
3. **旋转不变性**：保持向量长度不变
4. **高效计算**：只需元素级乘法

#### 实现

```python
# folovllm/model_executor/models/utils.py - RotaryEmbedding
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
freqs = torch.outer(positions, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos, sin = emb.cos(), emb.sin()

# 应用旋转
x1, x2 = x[..., :d//2], x[..., d//2:]
x_rotated = [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
```

---

### 5. RMSNorm

#### LayerNorm 的问题

传统 LayerNorm：
```python
mean = x.mean()
var = x.var()
x_norm = (x - mean) / sqrt(var + eps)
x_out = x_norm * weight + bias
```

**计算量**：需要计算均值和方差

#### RMSNorm 的简化

RMSNorm (Root Mean Square Normalization)：
```python
rms = sqrt(mean(x^2))
x_norm = x / (rms + eps)
x_out = x_norm * weight
```

**优势**：
- ✅ 不需要计算均值（省略 re-centering）
- ✅ 不需要 bias
- ✅ 计算更简单，速度更快
- ✅ 效果与 LayerNorm 相当

**理论**：re-centering 对性能提升有限，而 re-scaling 是关键

**实践**：LLaMA、Qwen3、GPT-NeoX 等现代 LLM 都使用 RMSNorm

---

### 6. GQA (Grouped Query Attention)

#### 演化路径

```
MHA (Multi-Head Attention)
  - num_heads 个独立的 Q, K, V heads
  - 例: 32 heads, 每个 head 都有独立 K, V
  - 问题: KV cache 大（32 份）

MQA (Multi-Query Attention)  
  - num_heads 个 Q heads, 但只有 1 个 K, V head
  - 所有 Q heads 共享同一个 K, V
  - 优势: KV cache 小 32x
  - 问题: 表达能力下降

GQA (Grouped Query Attention)
  - num_heads 个 Q heads, num_kv_heads 个 K, V heads
  - Q heads 分组共享 K, V heads
  - 平衡: 例如 32 个 Q, 8 个 KV (4 组)
```

#### Qwen3 的配置

```python
num_attention_heads = 16      # Q heads
num_key_value_heads = 2       # KV heads
# 每 8 个 Q heads 共享 1 个 KV head
```

#### 实现

```python
# folovllm/attention/ops.py - naive_attention
if num_heads > num_kv_heads:
    num_repeats = num_heads // num_kv_heads
    # 重复 KV heads 以匹配 Q heads
    key = key.unsqueeze(2).expand(...).reshape(...)
```

---

## 🎯 面试问题汇总

### Q1: KV Cache 如何节省计算？具体节省了多少？

**答**：
- **不使用 KV Cache**：每生成一个 token，都要重新计算从第一个 token 到当前的所有 K、V
  - 生成 N 个 token，总计算量：`O(N^2 * d)`
  
- **使用 KV Cache**：只计算新 token 的 K、V，复用历史
  - 生成 N 个 token，总计算量：`O(N * d)`
  
- **节省**：从 O(N^2) 降到 O(N)，对长序列效果显著
  - 例：生成 100 个 token，理论上快 100 倍（实际受其他因素影响）

### Q2: Prefill 和 Decode 阶段有什么区别？为什么 Decode 是 IO bound？

**答**：
- **Prefill**：
  - 批量处理多个 token（seq_len > 1）
  - 需要大量矩阵乘法（Q@K^T，attn@V）
  - **计算密集型**（compute bound）：GPU 核心利用率高
  
- **Decode**：
  - 每次只处理 1 个 token（seq_len = 1）
  - 矩阵乘法退化为向量-矩阵乘法
  - **内存带宽密集型**（memory bound）：需要读取整个 KV cache
  - 瓶颈是显存带宽，而非计算能力
  
- **优化方向**：
  - Prefill：更大 batch，更优 GEMM kernel
  - Decode：减少内存访问（Flash Attention），增加 batch（Continuous Batching）

### Q3: Top-k 和 Top-p 采样有什么区别？各自适用场景？

**答**：
- **Top-k**：固定选择概率最高的 k 个 token
  - 优点：简单，可预测
  - 缺点：不考虑概率分布形状
  - 适用：需要一定随机性但不要太离谱的场景
  
- **Top-p**：动态选择，累积概率达到 p 为止
  - 优点：自适应，更符合分布特性
  - 缺点：token 数量不固定
  - 适用：需要高质量随机性的创作任务
  
- **实践**：通常组合使用，先 top-k 粗过滤，再 top-p 精细选择

### Q4: RoPE 相比绝对位置编码有什么优势？

**答**：
1. **相对位置编码**：Attention score 只依赖相对位置差 (m-n)，更符合语言特性
2. **长度外推**：训练时 2k 长度，推理时可扩展到更长（通过调整 scaling_factor）
3. **旋转不变性**：保持向量长度和方向关系
4. **高效**：不需要额外加法，只在 Q、K 上做旋转

绝对位置编码（如 BERT）：训练时最大长度 512，推理时无法超过 512。

### Q5: M1 的 KV Cache 实现有什么局限？M3 将如何改进？

**答**：
- **M1 局限**：
  - 使用连续内存（`torch.cat`），每次追加需要复制整个 cache
  - 不同长度序列无法高效 batch
  - 无法动态调整（抢占、swap）
  - 内存碎片化
  
- **M3 改进 (Paged Attention)**：
  - 将 KV cache 分成固定大小的 blocks（类似虚拟内存分页）
  - 使用 block table 管理，支持非连续存储
  - 支持序列间共享 blocks（前缀缓存）
  - 支持动态扩展、抢占、swap
  
- **效果**：显存利用率提升，支持更大 batch size

### Q6: 为什么现代 LLM 都用 RMSNorm 而不是 LayerNorm？

**答**：
1. **计算效率**：RMSNorm 不需要计算均值，只需要 RMS（均方根），减少计算量
2. **无 bias**：不需要 bias 参数，减少参数量
3. **效果相当**：大量实验表明，去除 re-centering（减均值）对性能影响很小
4. **训练稳定性**：RMSNorm 在大规模训练中表现稳定
5. **实践验证**：LLaMA、Qwen、GPT-NeoX 等顶级模型都采用

**底层原理**：LayerNorm 的 re-centering 主要是为了数值稳定性，但在现代硬件和训练技巧下，re-scaling 才是关键。

### Q7: 如何理解 Grouped Query Attention (GQA)？

**答**：
- **MHA**：每个 Q head 有独立的 K、V head
  - 表达能力强，但 KV cache 大
  
- **MQA**：所有 Q heads 共享 1 个 K、V head
  - KV cache 小，但表达能力受限
  
- **GQA**：Q heads 分组，每组共享 K、V head
  - 平衡表达能力和显存占用
  - 例：Qwen3-0.6B 用 16 个 Q heads，2 个 KV heads（8:1 分组）
  
**实现**：在 attention 计算时，重复 KV heads 以匹配 Q heads 数量

**优势**：在不显著降低性能的前提下，减少 KV cache 8x（对于 8:1 分组）

---

## 📖 参考资料

### 论文
1. **Attention Is All You Need** (2017) - Transformer 原理
2. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (2021) - RoPE
3. **Root Mean Square Layer Normalization** (2019) - RMSNorm
4. **GQA: Training Generalized Multi-Query Transformer Models** (2023) - GQA
5. **Fast Transformer Decoding: One Write-Head is All You Need** (2019) - MQA

### 博客
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)
- [KV Cache Explained](https://kipp.ly/blog/transformer-inference-arithmetic/)

### 代码
- vLLM v1: https://github.com/vllm-project/vllm
- HuggingFace Transformers: https://github.com/huggingface/transformers
- Flash Attention: https://github.com/Dao-AILab/flash-attention

---

## 🎓 总结

M1 实现了完整的离线推理流程，掌握以下核心概念：

1. ✅ **KV Cache**：理解预填充和解码阶段，掌握缓存机制
2. ✅ **Attention**：朴素实现，理解 Q、K、V 的计算流程
3. ✅ **Sampling**：多种采样策略，理解各自优劣和组合使用
4. ✅ **RoPE**：旋转位置编码，理解如何注入位置信息
5. ✅ **RMSNorm**：高效归一化，理解与 LayerNorm 的区别
6. ✅ **GQA**：分组查询注意力，平衡性能与显存

这些是 LLM 推理的基础，后续 milestone 将在此基础上优化性能和功能。

