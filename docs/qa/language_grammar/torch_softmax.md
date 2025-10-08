# PyTorch F.softmax 函数

## 调用的包

```python
import torch.nn.functional as F

F.softmax(...)  # 调用 torch.nn.functional.softmax
```

**来源**：PyTorch 的 `torch.nn.functional` 模块

## 计算原理

Softmax 将任意实数向量转换为**概率分布**（所有值在 0-1 之间，和为 1）。

**数学公式**：
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**实际计算**（数值稳定版本）：
```
softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```
减去最大值防止 `exp()` 溢出。

## 示例

```python
import torch.nn.functional as F

x = torch.tensor([1.0, 2.0, 3.0])

# 计算 softmax
result = F.softmax(x, dim=0)
# 结果: [0.0900, 0.2447, 0.6652]
# 和为 1.0

# 计算过程：
# exp(1) = 2.718, exp(2) = 7.389, exp(3) = 20.086
# sum = 30.193
# [2.718/30.193, 7.389/30.193, 20.086/30.193] = [0.09, 0.24, 0.67]
```

## 在代码第99行的应用

```python
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

**参数**：
- `attn_weights`: attention scores，形状 `[batch, num_heads, seq_len_q, seq_len_k]`
- `dim=-1`: 在最后一维（seq_len_k）上做 softmax
- `dtype=torch.float32`: 用 float32 计算（更精确）
- `.to(query.dtype)`: 转回原来的数据类型（可能是 float16/bfloat16）

**作用**：
- 将 attention scores 转换为 attention 权重（概率分布）
- 每行的权重和为 1.0
- 决定每个 query 关注哪些 key（权重越大越关注）

**示例**：
```python
# 假设某个 query 对3个 key 的 scores
scores = torch.tensor([2.0, 5.0, 1.0])

# softmax 后
weights = F.softmax(scores, dim=0)
# 结果: [0.042, 0.843, 0.115]
# 第2个 key 的权重最高（0.843），说明这个 query 最关注它
```

## 为什么用 float32 计算

```python
dtype=torch.float32
```

**原因**：
1. **数值稳定性**：softmax 涉及 `exp()` 运算，float16 容易溢出/下溢
2. **精度**：概率分布的和必须严格为 1.0，float16 精度不够
3. **性能权衡**：只在 softmax 时用 float32，其他计算仍用 float16

**典型流程**：
```python
# 输入: bfloat16
attn_weights = ...  # bfloat16

# Softmax: 转 float32 计算
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
# 结果: float32

# 转回原类型
attn_weights = attn_weights.to(query.dtype)  # 转回 bfloat16
```

## 在 Attention 中的完整流程

```python
# Step 1: 计算 scores
attn_weights = query @ key.T  # [B, H, Q, K]
attn_weights = attn_weights * scale  # 缩放

# Step 2: 应用 mask（causal attention）
attn_weights = attn_weights + attn_mask  # 被 mask 的位置设为 -inf

# Step 3: Softmax（重点）
attn_weights = F.softmax(attn_weights, dim=-1)
# 每行变成概率分布，-inf 位置的概率为 0

# Step 4: 加权求和
output = attn_weights @ value  # [B, H, Q, D]
```

## Softmax 的特性

1. **输出范围**：`[0, 1]`
2. **和为 1**：`sum(softmax(x)) = 1.0`
3. **单调性**：输入越大，输出越大
4. **温度缩放**：`softmax(x/T)`，T 越大分布越平滑

**示例**：
```python
x = torch.tensor([1.0, 2.0, 3.0])

# 标准 softmax
F.softmax(x, dim=0)
# [0.09, 0.24, 0.67]  # 差异明显

# 高温度（T=2）
F.softmax(x/2.0, dim=0)
# [0.16, 0.26, 0.58]  # 更平滑

# 低温度（T=0.5）
F.softmax(x/0.5, dim=0)
# [0.02, 0.12, 0.86]  # 更尖锐
```

