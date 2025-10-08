# Grouped Query Attention (GQA) - Qwen3 中的 Q、K、V 头数不同

## 问题：为什么 Qwen3 的头数不一样？

在 Qwen3-0.6B 中：
- **Q (Query) 头数**：16 个
- **K (Key) 头数**：8 个  
- **V (Value) 头数**：8 个

这不是错误，而是一种优化技术：**Grouped Query Attention (GQA，分组查询注意力)**。

## Attention 的三种架构

### 1. Multi-Head Attention (MHA) - 传统方式

**所有头数相同**：
```python
num_q_heads = 16
num_k_heads = 16  # 与 Q 相同
num_v_heads = 16  # 与 Q 相同
```

**结构**：
```
Q heads: [Q₀] [Q₁] [Q₂] ... [Q₁₅]  (16个)
         ↓     ↓     ↓         ↓
K heads: [K₀] [K₁] [K₂] ... [K₁₅]  (16个)
         ↓     ↓     ↓         ↓
V heads: [V₀] [V₁] [V₂] ... [V₁₅]  (16个)
```

**特点**：
- 每个 Q head 对应一个专属的 K、V head
- KV cache 大小：16 × seq_len × head_dim

### 2. Multi-Query Attention (MQA) - 极端优化

**只有 1 个 K、V head**：
```python
num_q_heads = 16
num_k_heads = 1   # 只有1个
num_v_heads = 1   # 只有1个
```

**结构**：
```
Q heads: [Q₀] [Q₁] [Q₂] ... [Q₁₅]  (16个)
         ↓     ↓     ↓         ↓
K head:        [K₀]                 (1个，共享)
               ↓
V head:        [V₀]                 (1个，共享)
```

**特点**：
- 所有 Q heads 共享同一个 K、V head
- KV cache 大小：1 × seq_len × head_dim
- **问题**：表达能力下降，性能有损失

### 3. Grouped Query Attention (GQA) - Qwen3 使用的折中方案

**K、V 头数少于 Q 头数**：
```python
num_q_heads = 16
num_k_heads = 8   # Q heads 的 1/2
num_v_heads = 8   # Q heads 的 1/2
```

**结构**：
```
Q heads: [Q₀ Q₁] [Q₂ Q₃] [Q₄ Q₅] [Q₆ Q₇] [Q₈ Q₉] [Q₁₀ Q₁₁] [Q₁₂ Q₁₃] [Q₁₄ Q₁₅]
         └ 组1 ┘ └ 组2 ┘ └ 组3 ┘ └ 组4 ┘ └ 组5 ┘ └─ 组6 ─┘ └─ 组7 ─┘ └─ 组8 ─┘
            ↓       ↓       ↓       ↓       ↓        ↓        ↓        ↓
K heads:  [K₀]    [K₁]    [K₂]    [K₃]    [K₄]     [K₅]     [K₆]     [K₇]     (8个)
           ↓       ↓       ↓       ↓       ↓        ↓        ↓        ↓
V heads:  [V₀]    [V₁]    [V₂]    [V₃]    [V₄]     [V₅]     [V₆]     [V₇]     (8个)
```

**分组**：
- 第1组：Q₀-Q₁ 共享 K₀、V₀
- 第2组：Q₂-Q₃ 共享 K₁、V₁
- ...
- 第8组：Q₁₄-Q₁₅ 共享 K₇、V₇
- 每组有 2 个 Q heads（16 / 8 = 2）

**特点**：
- KV cache 大小：8 × seq_len × head_dim
- 在性能和效率间取得平衡

## 三种方式对比

| 特性              | MHA        | GQA (Qwen3) | MQA       |
| ----------------- | ---------- | ----------- | --------- |
| **Q heads**       | 16         | 16          | 16        |
| **K heads**       | 16         | 8           | 1         |
| **V heads**       | 16         | 8           | 1         |
| **KV cache 大小** | 16 × S × D | 8 × S × D   | 1 × S × D |
| **相对大小**      | 100%       | 50%         | 6.25%     |
| **表达能力**      | ★★★★★      | ★★★★☆       | ★★★☆☆     |
| **推理速度**      | ★★★☆☆      | ★★★★☆       | ★★★★★     |

## 在代码中的实现

### 配置（Qwen3-0.6B）

```python
# transformers Qwen2Config
config.hidden_size = 1024
config.num_attention_heads = 16      # Q heads
config.num_key_value_heads = 8       # K, V heads
config.head_dim = 128
```

### Attention 层初始化

**文件**：`folovllm/model_executor/layers/attention.py`

```python
class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,           # 16 (Q heads)
        num_kv_heads: int,        # 8  (K, V heads)
        head_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads          # 16
        self.num_kv_heads = num_kv_heads    # 8
        self.head_dim = head_dim            # 128
        
        # 验证：Q heads 必须能被 KV heads 整除
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        # 计算每组的 Q heads 数量
        self.num_repeats = num_heads // num_kv_heads  # 16 // 8 = 2
        
        # QKV 投影维度
        self.q_size = num_heads * head_dim      # 16 × 128 = 2048
        self.kv_size = num_kv_heads * head_dim  # 8 × 128 = 1024
        
        # 合并的 QKV 投影
        self.qkv_proj = nn.Linear(
            hidden_size,                    # 1024
            self.q_size + 2 * self.kv_size, # 2048 + 2×1024 = 4096
            bias=False,
        )
```

### Forward 中的拆分

```python
def forward(self, hidden_states):
    # hidden_states: [batch, seq_len, 1024]
    
    # QKV 投影
    qkv = self.qkv_proj(hidden_states)
    # qkv: [batch, seq_len, 4096]
    
    # 拆分成 Q, K, V
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    # q: [batch, seq_len, 2048]  - 16 heads × 128
    # k: [batch, seq_len, 1024]  - 8 heads × 128
    # v: [batch, seq_len, 1024]  - 8 heads × 128
    
    # Reshape 为多头形式
    q = q.view(batch, seq_len, self.num_heads, self.head_dim)
    # q: [batch, seq_len, 16, 128]
    
    k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
    # k: [batch, seq_len, 8, 128]
    
    v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
    # v: [batch, seq_len, 8, 128]
    
    # Transpose 为 [batch, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)  # [batch, 16, seq_len, 128]
    k = k.transpose(1, 2)  # [batch, 8, seq_len, 128]
    v = v.transpose(1, 2)  # [batch, 8, seq_len, 128]
```

### KV Heads 的重复（在 Attention 计算中）

**文件**：`folovllm/attention/ops.py` (第78-87行)

```python
def naive_attention(query, key, value, scale, attn_mask=None):
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    _, num_kv_heads, seq_len_k, _ = key.shape
    
    # 如果 Q heads 数量 > KV heads 数量，需要重复 KV
    if num_heads > num_kv_heads:
        # 计算重复次数
        num_repeats = num_heads // num_kv_heads  # 16 // 8 = 2
        
        # 重复 key: [batch, 8, seq_len, 128] -> [batch, 16, seq_len, 128]
        key = key.unsqueeze(2).expand(
            batch_size, num_kv_heads, num_repeats, seq_len_k, head_dim
        ).reshape(batch_size, num_heads, seq_len_k, head_dim)
        
        # 重复 value: [batch, 8, seq_len, 128] -> [batch, 16, seq_len, 128]
        value = value.unsqueeze(2).expand(
            batch_size, num_kv_heads, num_repeats, seq_len_k, head_dim
        ).reshape(batch_size, num_heads, seq_len_k, head_dim)
    
    # 现在 Q、K、V 的 heads 数量相同，可以做 attention
    # query: [batch, 16, seq_len_q, 128]
    # key:   [batch, 16, seq_len_k, 128]  (重复后)
    # value: [batch, 16, seq_len_k, 128]  (重复后)
    
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale
    # ...
```

### 重复过程可视化

```python
# 原始 K: [batch, 8, seq_len, 128]
K₀ = [batch, seq_len, 128]  # 第1个 K head
K₁ = [batch, seq_len, 128]  # 第2个 K head
...
K₇ = [batch, seq_len, 128]  # 第8个 K head

# unsqueeze(2): [batch, 8, 1, seq_len, 128]
[[K₀],
 [K₁],
 [K₂],
 [K₃],
 [K₄],
 [K₅],
 [K₆],
 [K₇]]

# expand: [batch, 8, 2, seq_len, 128]
[[K₀, K₀],  # K₀ 重复 2 次
 [K₁, K₁],  # K₁ 重复 2 次
 [K₂, K₂],
 [K₃, K₃],
 [K₄, K₄],
 [K₅, K₅],
 [K₆, K₆],
 [K₇, K₇]]

# reshape: [batch, 16, seq_len, 128]
[K₀, K₀, K₁, K₁, K₂, K₂, K₃, K₃, K₄, K₄, K₅, K₅, K₆, K₆, K₇, K₇]
 └组1┘  └组2┘  └组3┘  └组4┘  └组5┘  └组6┘  └组7┘  └组8┘
```

## 为什么这样设计？

### 1. 减少 KV Cache 大小

**关键瓶颈**：在推理阶段，KV cache 是显存的主要消耗。

```python
# MHA (传统方式)
kv_cache_size = 2 × num_layers × num_heads × seq_len × head_dim
              = 2 × 28 × 16 × 2048 × 128
              = 234,881,024 元素
              ≈ 940 MB (float32)

# GQA (Qwen3)
kv_cache_size = 2 × num_layers × num_kv_heads × seq_len × head_dim
              = 2 × 28 × 8 × 2048 × 128
              = 117,440,512 元素
              ≈ 470 MB (float32)

# 节省: (940 - 470) / 940 = 50% 的 KV cache 显存
```

### 2. 加速推理

**Decode 阶段**（生成单个 token）是**内存带宽受限**的：

```python
# 每个 decode step 需要读取的 KV cache
# MHA: 16 × seq_len × 128 × 2 (K和V) = 很大
# GQA: 8 × seq_len × 128 × 2 (K和V) = 小一半

# 内存带宽节省 = 16/8 = 2倍
# 实际加速: 1.2-1.5x (因为还有其他计算)
```

### 3. 保持表达能力

**比 MQA 更好**：
- MQA (1 个 KV head)：所有 Q heads 共享，信息瓶颈
- GQA (8 个 KV heads)：8 组，每组有独立的 K、V
- 实验表明：GQA 性能接近 MHA，远优于 MQA

### 4. 适合长上下文

**长序列推理**时，KV cache 是主要瓶颈：

```python
# 序列长度 = 32K tokens
# MHA KV cache: 28 × 16 × 32768 × 128 = 15 GB
# GQA KV cache: 28 × 8 × 32768 × 128 = 7.5 GB

# GQA 可以支持 2 倍长的序列！
```

## 实际效果

### Qwen3-0.6B 的配置选择

```python
num_attention_heads = 16
num_key_value_heads = 8

# 分组比例 = 16 / 8 = 2
# 每个 KV head 服务 2 个 Q heads
```

**为什么选 8 个 KV heads？**
- 太少 (1个)：性能下降明显
- 太多 (16个)：KV cache 太大
- 8个：平衡点，性能损失 < 2%，显存节省 50%

### 与其他模型对比

| 模型           | Q heads | KV heads | 比例  | 架构    |
| -------------- | ------- | -------- | ----- | ------- |
| GPT-3          | 96      | 96       | 1:1   | MHA     |
| LLaMA-7B       | 32      | 32       | 1:1   | MHA     |
| **LLaMA-2-7B** | 32      | 32       | 1:1   | MHA     |
| **LLaMA-3-8B** | 32      | 8        | 4:1   | **GQA** |
| **Qwen3-0.6B** | 16      | 8        | 2:1   | **GQA** |
| **Mistral-7B** | 32      | 8        | 4:1   | **GQA** |
| PaLM           | 128     | 1        | 128:1 | MQA     |

**趋势**：新模型普遍采用 GQA，平衡性能和效率。

## 权衡考虑

### 优点

✅ **显存节省**：KV cache 减少 50%  
✅ **推理加速**：内存带宽消耗减少  
✅ **支持长序列**：可以处理更长的上下文  
✅ **性能保持**：相比 MHA 损失 < 2%

### 缺点

❌ **实现复杂**：需要处理 heads 重复  
❌ **轻微性能损失**：不如 MHA（但可接受）

## 总结

**Qwen3 中 Q、K、V 头数不同**：
- **Q heads**: 16 个（每个 token 需要独立的查询）
- **K, V heads**: 8 个（共享，节省显存）
- **架构**: Grouped Query Attention (GQA)

**分组方式**：
```
16 个 Q heads 分成 8 组
每组 2 个 Q heads 共享 1 个 K head 和 1 个 V head
```

**实现关键**：
1. QKV 投影维度不同：`2048 + 1024 + 1024 = 4096`
2. Reshape 时头数不同：Q 是 16，K/V 是 8
3. Attention 计算前重复 K/V：8 个 → 16 个

**收益**：
- KV cache 减少 **50%**
- 推理加速 **1.2-1.5x**（decode 阶段）
- 性能损失 < **2%**

**这是现代 LLM 的标准优化技术**，在效率和性能间取得完美平衡！

