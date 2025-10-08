# PyTorch torch.triu 函数

## 功能

`torch.triu` 提取矩阵的**上三角部分**（triangle upper），将下三角部分设为 0。

**triu = TRIangle Upper**

## 语法

```python
torch.triu(input, diagonal=0)
```

**参数**：
- `input`: 输入矩阵
- `diagonal`: 对角线偏移
  - `diagonal=0`: 主对角线（默认）
  - `diagonal=1`: 主对角线上方一条
  - `diagonal=-1`: 主对角线下方一条

## 基础示例

### 默认（主对角线）

```python
import torch

x = torch.ones(4, 4)
# [[1, 1, 1, 1],
#  [1, 1, 1, 1],
#  [1, 1, 1, 1],
#  [1, 1, 1, 1]]

torch.triu(x)
# [[1, 1, 1, 1],    ← 保留
#  [0, 1, 1, 1],    ← 保留对角线及右上
#  [0, 0, 1, 1],    ← 保留对角线及右上
#  [0, 0, 0, 1]]    ← 保留对角线
```

### diagonal=1（对角线上方）

```python
torch.triu(x, diagonal=1)
# [[0, 1, 1, 1],    ← 对角线也被清零
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]
```

### diagonal=-1（对角线下方）

```python
torch.triu(x, diagonal=-1)
# [[1, 1, 1, 1],
#  [1, 1, 1, 1],    ← 对角线下方也保留
#  [0, 1, 1, 1],
#  [0, 0, 1, 1]]
```

## 在代码第132行的应用

### Causal Mask 创建

```python
def create_causal_mask(seq_len_q, seq_len_k, ...):
    mask = torch.ones(seq_len_q, seq_len_k)
    mask = torch.triu(mask, diagonal=seq_len_k - seq_len_q + 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

**作用**：创建因果注意力掩码，防止"看到未来"。

### 示例1: Prefill 阶段（seq_len_q = seq_len_k = 4）

```python
seq_len_q = 4
seq_len_k = 4
diagonal = 4 - 4 + 1 = 1

# Step 1: 全1矩阵
mask = torch.ones(4, 4)
# [[1, 1, 1, 1],
#  [1, 1, 1, 1],
#  [1, 1, 1, 1],
#  [1, 1, 1, 1]]

# Step 2: triu(diagonal=1) - 保留右上三角（不含对角线）
mask = torch.triu(mask, diagonal=1)
# [[0, 1, 1, 1],
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]

# Step 3: 将1替换为-inf（被mask的位置）
mask = mask.masked_fill(mask == 1, float('-inf'))
# [[  0, -inf, -inf, -inf],
#  [  0,    0, -inf, -inf],
#  [  0,    0,    0, -inf],
#  [  0,    0,    0,    0]]
```

**含义**：
- 第0个 token 只能看到自己（位置0）
- 第1个 token 能看到位置0-1
- 第2个 token 能看到位置0-2
- 第3个 token 能看到位置0-3

这就是**因果注意力**（causal attention）：每个位置只能看到自己和之前的位置。

### 示例2: Decode 阶段（seq_len_q = 1, seq_len_k = 5）

```python
seq_len_q = 1  # 当前生成的token
seq_len_k = 5  # 已有5个token（包括当前）
diagonal = 5 - 1 + 1 = 5

# Step 1: 全1矩阵 [1, 5]
mask = torch.ones(1, 5)
# [[1, 1, 1, 1, 1]]

# Step 2: triu(diagonal=5) - diagonal>=5，全部保留为0
mask = torch.triu(mask, diagonal=5)
# [[0, 0, 0, 0, 0]]

# Step 3: 没有1，无需mask
# [[0, 0, 0, 0, 0]]
```

**含义**：Decode 时当前 token 可以看到所有历史 token（全0 = 不mask）。

## 可视化理解

### Causal Mask 的作用

```
Query Token:  0    1    2    3
             ┌────────────────┐
Key Token 0  │ ✓   ✗   ✗   ✗ │  ← Token 0 只能被自己看到
         1  │ ✓   ✓   ✗   ✗ │  ← Token 1 能被 0,1 看到
         2  │ ✓   ✓   ✓   ✗ │  ← Token 2 能被 0,1,2 看到
         3  │ ✓   ✓   ✓   ✓ │  ← Token 3 能被所有人看到
             └────────────────┘

✓ = 0 (允许注意)
✗ = -inf (禁止注意，softmax后为0)
```

对应的 mask 矩阵：
```
[[  0, -∞, -∞, -∞],
 [  0,  0, -∞, -∞],
 [  0,  0,  0, -∞],
 [  0,  0,  0,  0]]
```

## diagonal 参数的计算

在代码中：
```python
diagonal = seq_len_k - seq_len_q + 1
```

**为什么这样算？**

### 情况1: Prefill（seq_len_q = seq_len_k = N）
```
diagonal = N - N + 1 = 1
```
- 使用 `triu(diagonal=1)` 保留严格的右上三角
- 主对角线为 0（可以看到自己）
- 右上为 1（不能看到未来）→ 替换为 -inf

### 情况2: Decode（seq_len_q = 1, seq_len_k = N）
```
diagonal = N - 1 + 1 = N
```
- 使用 `triu(diagonal=N)`，由于 diagonal >= N，全部为 0
- 全0 = 可以看到所有历史 token

## torch.tril（下三角）

与 `triu` 相反，`tril` 保留**下三角**部分：

```python
x = torch.ones(4, 4)

torch.tril(x)
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

# 可以用 tril 创建 causal mask 的另一种方式
mask = torch.tril(torch.ones(4, 4))
mask = mask.masked_fill(mask == 0, float('-inf'))
mask = mask.masked_fill(mask == 1, 0)
# 效果与 triu 方式相同
```

## 常见使用场景

1. **Causal Attention Mask**（如本例）
   - GPT 系列模型
   - 自回归生成

2. **下三角矩阵运算**
   - Cholesky 分解
   - 三角矩阵求解

3. **遮蔽未来信息**
   - 时间序列预测
   - 强化学习

## 性能提示

`triu` 和 `tril` 是**视图操作**（返回新 tensor，但可能共享内存）：

```python
x = torch.randn(1000, 1000)

# 高效：直接使用视图
mask = torch.triu(x)

# 如果需要独立副本
mask = torch.triu(x).clone()
```

## 完整示例：Attention 中的应用

```python
def attention_with_causal_mask(query, key, value):
    # query, key, value: [batch, num_heads, seq_len, head_dim]
    
    seq_len = query.size(2)
    
    # 计算 attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    # Shape: [batch, num_heads, seq_len, seq_len]
    
    # 创建 causal mask
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=query.device),
        diagonal=1  # 右上三角为1
    )
    mask = mask.masked_fill(mask == 1, float('-inf'))
    # Shape: [seq_len, seq_len]
    
    # 应用 mask
    scores = scores + mask
    
    # Softmax（-inf位置概率为0）
    attn_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attn_weights, value)
    
    return output
```

## 总结

- `torch.triu`: 提取上三角部分，下三角清零
- `diagonal` 参数控制对角线位置
- 在 causal mask 中用于防止"看到未来"
- Prefill 时创建三角 mask，Decode 时全部可见

