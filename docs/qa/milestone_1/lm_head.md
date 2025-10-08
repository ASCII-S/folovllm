# LM Head (Language Model Head) - 语言模型头

## 基本概念

**LM Head**（Language Model Head）是语言模型的**输出层**，负责将模型的隐藏状态转换为**词汇表上的概率分布**。

简单来说：**LM Head 预测下一个 token 是什么**。

## 在模型中的位置

```
输入 token IDs
    ↓
Embedding 层 (token → vector)
    ↓
Transformer 层 (N 层)
    ↓
Hidden States [batch, seq_len, hidden_size]
    ↓
LM Head (Linear 层)  ← 这里！
    ↓
Logits [batch, seq_len, vocab_size]
    ↓
Softmax
    ↓
概率分布 (每个 token 的概率)
```

## 数学定义

```python
# hidden_states: [batch, seq_len, hidden_size]
# LM head 是一个 Linear 层
logits = W_lm @ hidden_states + b_lm
# logits: [batch, seq_len, vocab_size]

# 每个位置的 logits 表示词汇表中每个 token 的得分
# 通过 softmax 转换为概率
probs = softmax(logits)
```

**作用**：将 `hidden_size` 维的向量映射到 `vocab_size` 维的 logits。

## 在 Qwen3 中的实现

### 代码

**文件**：`folovllm/model_executor/models/qwen.py` (第240-295行)

```python
class Qwen3ForCausalLM(nn.Module):
    """Qwen3 for causal language modeling."""
    
    def __init__(self, config: Qwen2Config):
        super().__init__()
        
        # Base model (Transformer)
        self.model = Qwen3Model(config)
        
        # LM head (输出层)
        self.lm_head = nn.Linear(
            config.hidden_size,    # 896 (输入维度)
            config.vocab_size,     # 151,936 (输出维度)
            bias=False,            # 不使用 bias
        )
        
        # 权重共享（可选）
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self, input_ids, positions, kv_caches=None):
        # 1. 通过 Transformer 得到 hidden states
        hidden_states = self.model(input_ids, positions, kv_caches)
        # hidden_states: [batch, seq_len, 896]
        
        return hidden_states
    
    def compute_logits(self, hidden_states):
        # 2. LM head 计算 logits
        logits = self.lm_head(hidden_states)
        # logits: [batch, seq_len, 151936]
        
        return logits
```

### 参数量

```python
# Qwen3-0.6B 的 LM head
hidden_size = 896
vocab_size = 151,936

# 参数量
params = hidden_size × vocab_size
       = 896 × 151,936
       = 136,134,656
       ≈ 136M 参数

# 占整个模型的比例
total_params ≈ 600M
lm_head_ratio = 136M / 600M ≈ 22.7%
```

**LM head 占了模型参数的 1/5 以上！**

## 具体例子

### 前向传播流程

```python
# 输入：一个句子 "The cat is"
input_ids = [The=1, cat=2, is=3]  # token IDs

# Step 1: Embedding
embeddings = embed_tokens(input_ids)
# [3, 896] - 3 个 tokens，每个 896 维

# Step 2: Transformer layers
hidden_states = transformer_layers(embeddings)
# [3, 896] - 经过 28 层后，仍然是 896 维

# Step 3: LM head
logits = lm_head(hidden_states)
# [3, 151936] - 每个位置对整个词汇表的得分

# 解释：
logits[0] = 对 "The" 之后下一个 token 的预测
logits[1] = 对 "cat" 之后下一个 token 的预测
logits[2] = 对 "is" 之后下一个 token 的预测
```

### 预测下一个 token

```python
# 获取最后一个位置的 logits
last_logits = logits[-1]  # [151936] - 对 "is" 后面的预测

# 找到得分最高的 token
next_token_id = torch.argmax(last_logits)
# 假设 = 4 (对应 "sleeping")

# 完整句子：The cat is sleeping
```

### Softmax 转概率

```python
# logits: [-2.3, 5.1, -0.8, 8.2, ...]  (151936 个值)

# Softmax
probs = F.softmax(last_logits, dim=-1)
# [0.0001, 0.0023, 0.0005, 0.9800, ...]

# 解释：
# token 0 ("The") 的概率：0.01%
# token 1 ("cat") 的概率：0.23%
# token 2 ("is") 的概率：0.05%
# token 3 ("sleeping") 的概率：98.00%  ← 最可能
# ...
```

## 权重共享（Weight Tying）

### 概念

**权重共享**：LM head 的权重与 Embedding 层的权重**共享**（相同的参数）。

```python
# 不共享（默认）
embed_weight: [vocab_size, hidden_size] = [151936, 896]
lm_head_weight: [vocab_size, hidden_size] = [151936, 896]
# 总参数：2 × 136M = 272M

# 共享（tie_word_embeddings=True）
shared_weight: [vocab_size, hidden_size] = [151936, 896]
embed_tokens.weight = shared_weight
lm_head.weight = shared_weight  # 指向同一个 tensor
# 总参数：136M （节省 136M！）
```

### 代码实现

```python
if config.tie_word_embeddings:
    # 让 lm_head 的 weight 指向 embed_tokens 的 weight
    self.lm_head.weight = self.model.embed_tokens.weight
    # 现在它们共享内存，更新一个会影响另一个
```

### 为什么可以共享？

**数学对称性**：

```python
# Embedding: token_id → hidden_vector
embedding = embed_weight[token_id]  # 查表

# LM head: hidden_vector → token_scores
logits = hidden_states @ lm_head_weight.T  # 矩阵乘法

# 如果共享权重
logits = hidden_states @ embed_weight.T
# 相当于计算 hidden_states 与每个 token embedding 的相似度
```

**直观理解**：
- Embedding：将 token 映射到语义空间
- LM head：计算隐藏状态与各个 token 语义的相似度
- 使用相同的语义空间是合理的！

### 优缺点

**优点**：
- ✅ 节省参数（~20% 的参数量）
- ✅ 节省显存
- ✅ 某些情况下性能更好（共享语义空间）

**缺点**：
- ❌ 限制了 embedding 和 lm_head 的独立性
- ❌ 可能略微影响性能（取决于任务）

**Qwen3 的选择**：
```python
config.tie_word_embeddings = False  # 不共享（性能优先）
```

## LM Head 的变体

### 1. 基础 Linear（Qwen3 使用）

```python
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

**特点**：
- 简单直接
- 参数量大
- 性能好

### 2. Adaptive Softmax

```python
# 对高频词和低频词使用不同的策略
# 节省计算，但实现复杂
```

**用途**：大词汇表（> 100万）时使用

### 3. Factorized LM Head

```python
# 分解为两个小矩阵
self.lm_head_1 = nn.Linear(hidden_size, intermediate_size)
self.lm_head_2 = nn.Linear(intermediate_size, vocab_size)
```

**用途**：极致压缩参数

## 训练和推理的区别

### 训练时

```python
# 对每个位置都计算 logits
logits = self.lm_head(hidden_states)
# [batch, seq_len, vocab_size]

# 计算 loss（对比真实标签）
loss = cross_entropy(logits[:, :-1], labels[:, 1:])
# 预测下一个 token，所以错开一位
```

### 推理时（生成）

```python
# 只需要最后一个位置的 logits
last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
logits = self.lm_head(last_hidden)     # [batch, vocab_size]

# 采样下一个 token
next_token = sample(logits, temperature, top_k, top_p)
```

**优化**：推理时只计算最后一个位置，节省计算。

## 性能影响

### 计算量

```python
# LM head 的 FLOPs
batch_size = 1
seq_len = 1  # decode 阶段
hidden_size = 896
vocab_size = 151,936

FLOPs = 2 × batch × seq_len × hidden_size × vocab_size
      = 2 × 1 × 1 × 896 × 151,936
      = 272,268,288
      ≈ 272M FLOPs

# 对比：一个 Attention 层 ~ 100M FLOPs
# LM head 是最大的单个操作！
```

### 内存占用

```python
# Logits 的内存
batch = 1
seq_len = 100
vocab_size = 151,936

logits_memory = batch × seq_len × vocab_size × 4 bytes (float32)
              = 1 × 100 × 151,936 × 4
              = 60,774,400 bytes
              ≈ 58 MB

# 长序列时，logits 占用很多内存！
```

### 优化技巧

**1. 只计算需要的位置**
```python
# ❌ 全部计算
logits = lm_head(hidden_states)  # [batch, seq_len, vocab]

# ✅ 只计算最后一个
logits = lm_head(hidden_states[:, -1, :])  # [batch, vocab]
```

**2. 延迟计算**
```python
# 先返回 hidden_states，需要时再计算 logits
hidden_states = model.forward(...)
# 只在采样时计算 logits
logits = model.compute_logits(hidden_states[:, -1, :])
```

**3. Vocabulary 截断**
```python
# 只计算 top-k 个 token 的 logits
# 用于某些特殊场景（如代码补全）
```

## 在 FoloVLLM 中的使用

### 完整流程

```python
# 文件：folovllm/engine/llm_engine.py

# 1. Forward 得到 hidden states
hidden_states = self.model.forward(input_ids, positions, kv_caches)

# 2. 只取最后一个位置
last_hidden = hidden_states[:, -1, :]

# 3. 计算 logits
logits = self.model.compute_logits(last_hidden)

# 4. 采样
next_token, log_prob = self.sampler.sample(
    logits,
    sampling_params,
)
```

### 为什么分离 forward 和 compute_logits？

**好处**：
1. **灵活性**：可以只计算需要的 logits
2. **性能**：decode 时只算最后一个位置
3. **未来扩展**：为 speculative decoding 预留接口

```python
# 未来：Speculative decoding
draft_hidden = draft_model.forward(...)
verify_hidden = main_model.forward(...)

# 一次性计算多个位置的 logits
logits = model.compute_logits(verify_hidden)
```

## 调试技巧

### 检查 logits 的合理性

```python
# 1. 检查形状
assert logits.shape == (batch_size, vocab_size)

# 2. 检查数值范围（logits 通常在 -10 到 10 之间）
print(f"Logits min: {logits.min()}, max: {logits.max()}")

# 3. 检查 top-k token
top_k_values, top_k_indices = torch.topk(logits, k=10)
print("Top 10 tokens:", tokenizer.convert_ids_to_tokens(top_k_indices[0]))

# 4. 检查概率分布
probs = F.softmax(logits, dim=-1)
print(f"Prob sum: {probs.sum()}")  # 应该 ≈ 1.0
```

### 常见问题

**问题1：logits 全是 NaN**
```python
# 原因：hidden_states 有 NaN
# 检查：是否有梯度爆炸、数值溢出
```

**问题2：logits 分布很平**
```python
# 原因：模型未训练好，或 temperature 太高
# 检查：是否加载了正确的权重
```

**问题3：总是预测同一个 token**
```python
# 原因：某个 token 的 logit 远大于其他
# 检查：是否有 bias 项错误、权重异常
```

## 总结

**LM Head 是什么**：
- 语言模型的**输出层**
- 一个 **Linear 层**：`hidden_size → vocab_size`
- 将隐藏状态转换为词汇表上的 **logits**

**在 Qwen3 中**：
```python
self.lm_head = nn.Linear(896, 151936, bias=False)
# 参数量：136M (占模型 22.7%)
```

**关键概念**：
- **Logits**：每个 token 的原始得分
- **Softmax**：转换为概率分布
- **权重共享**：可选的参数优化

**作用**：
```
Hidden States → LM Head → Logits → Softmax → 概率 → 采样 → 下一个 token
```

**优化要点**：
- 推理时只计算最后一个位置
- 分离 forward 和 compute_logits
- 注意 logits 的内存占用

**调试**：
- 检查 logits 形状和数值范围
- 查看 top-k token 是否合理
- 确认概率和为 1

LM Head 虽然简单（就是一个 Linear 层），但它是语言模型的**关键输出组件**，决定了模型的预测能力！

