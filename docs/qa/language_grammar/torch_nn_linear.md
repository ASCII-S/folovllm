# PyTorch nn.Linear 层

## 功能

`nn.Linear` 是 PyTorch 的**全连接层（Fully Connected Layer）**，也叫**线性层（Linear Layer）**或**密集层（Dense Layer）**。

**作用**：对输入进行**线性变换（仿射变换）**。

## 数学原理

```
y = xW^T + b
```

其中：
- `x`: 输入，shape `[..., in_features]`
- `W`: 权重矩阵，shape `[out_features, in_features]`
- `b`: 偏置向量，shape `[out_features]`（可选）
- `y`: 输出，shape `[..., out_features]`

## 语法

```python
import torch.nn as nn

layer = nn.Linear(in_features, out_features, bias=True)
```

**参数**：
- `in_features`: 输入特征维度
- `out_features`: 输出特征维度
- `bias`: 是否使用偏置（默认 `True`）

## 基础示例

```python
import torch
import torch.nn as nn

# 创建 Linear 层：输入维度 10，输出维度 5
linear = nn.Linear(10, 5)

# 输入数据
x = torch.randn(3, 10)  # [batch_size=3, in_features=10]

# 前向传播
y = linear(x)  # [batch_size=3, out_features=5]

print(x.shape)  # torch.Size([3, 10])
print(y.shape)  # torch.Size([3, 5])
```

### 内部计算过程

```python
# nn.Linear 内部做的事情
x = torch.randn(3, 10)
linear = nn.Linear(10, 5)

# 等价于：
y = x @ linear.weight.T + linear.bias
# 或者：
y = torch.matmul(x, linear.weight.T) + linear.bias
```

### 查看参数

```python
linear = nn.Linear(10, 5)

# 权重矩阵
print(linear.weight.shape)  # torch.Size([5, 10])

# 偏置向量
print(linear.bias.shape)    # torch.Size([5])

# 参数总数
num_params = 10 * 5 + 5 = 55
```

## 在 FoloVLLM 中的应用

### 位置：`folovllm/model_executor/layers/attention.py`

### 应用1：QKV 投影（第61-65行）

```python
self.qkv_proj = nn.Linear(
    hidden_size,                      # 输入：hidden_size (例如 896)
    self.q_size + 2 * self.kv_size,  # 输出：Q + K + V 的总维度
    bias=bias,
)
```

**示例参数**（Qwen3-0.6B）：
- `hidden_size = 896`
- `num_heads = 14`，`head_dim = 64` → `q_size = 14 * 64 = 896`
- `num_kv_heads = 2`，`head_dim = 64` → `kv_size = 2 * 64 = 128`
- 输出维度 = `896 + 2 * 128 = 1152`

**计算流程**（第107行）：
```python
# hidden_states: [batch, seq_len, 896]
qkv = self.qkv_proj(hidden_states)
# qkv: [batch, seq_len, 1152]

# 拆分成 Q, K, V
q, k, v = qkv.split([896, 128, 128], dim=-1)
# q: [batch, seq_len, 896]
# k: [batch, seq_len, 128]
# v: [batch, seq_len, 128]
```

**为什么合并 QKV？**
- 性能优化：1次矩阵乘法比 3次快
- 减少 kernel launch 次数（GPU 优化）

### 应用2：输出投影（第68-72行）

```python
self.o_proj = nn.Linear(
    self.q_size,      # 输入：896 (14 heads * 64)
    hidden_size,      # 输出：896
    bias=bias,
)
```

**计算流程**（第147行）：
```python
# attn_output: [batch, seq_len, num_heads, head_dim]
#             = [batch, seq_len, 14, 64]

# Reshape 为 [batch, seq_len, 896]
attn_output = attn_output.reshape(batch_size, seq_len, self.q_size)

# 输出投影
output = self.o_proj(attn_output)
# output: [batch, seq_len, 896]
```

## 支持多维输入

Linear 层会保留前面的所有维度，只变换**最后一维**：

```python
linear = nn.Linear(10, 5)

# 2D 输入
x = torch.randn(3, 10)
y = linear(x)  # [3, 5]

# 3D 输入
x = torch.randn(2, 3, 10)
y = linear(x)  # [2, 3, 5]

# 4D 输入
x = torch.randn(2, 3, 4, 10)
y = linear(x)  # [2, 3, 4, 5]
```

## 参数初始化

PyTorch 默认使用 **Kaiming Uniform** 初始化：

```python
linear = nn.Linear(10, 5)

# 权重初始化：均匀分布 U(-sqrt(k), sqrt(k))
# 其中 k = 1 / in_features
bound = 1 / math.sqrt(10)
# weight ~ U(-0.316, 0.316)

# 偏置初始化：同样的均匀分布
# bias ~ U(-0.316, 0.316)
```

### 自定义初始化

```python
linear = nn.Linear(10, 5)

# 零初始化
nn.init.zeros_(linear.weight)
nn.init.zeros_(linear.bias)

# Xavier 初始化
nn.init.xavier_uniform_(linear.weight)

# 正态分布初始化
nn.init.normal_(linear.weight, mean=0, std=0.02)
```

## 无偏置的 Linear

```python
linear = nn.Linear(10, 5, bias=False)

print(linear.bias)  # None

# 计算
y = x @ linear.weight.T  # 没有 + bias 项
```

**使用场景**：
- LayerNorm 之后通常不需要 bias
- 某些模型架构（如 Qwen3）不使用 bias

## 与卷积层的对比

| 特性     | nn.Linear | nn.Conv2d               |
| -------- | --------- | ----------------------- |
| 操作     | 全连接    | 局部连接                |
| 权重共享 | 无        | 有（卷积核）            |
| 参数量   | in × out  | kernel_size² × in × out |
| 适用     | 特征变换  | 图像、序列              |

## 内存和计算量

### 参数量

```python
linear = nn.Linear(in_features, out_features, bias=True)

# 参数量
num_params = in_features * out_features + out_features
           = out_features * (in_features + 1)
```

**示例**：
```python
# Qwen3 的 QKV 投影
in_features = 896
out_features = 1152
num_params = 896 * 1152 + 1152 = 1,033,344 ≈ 1M 参数
```

### 计算量（FLOPs）

对于输入 `[batch, seq_len, in_features]`：

```python
# 每个输出元素需要 in_features 次乘法和加法
FLOPs = 2 * batch * seq_len * in_features * out_features
```

**示例**：
```python
# batch=1, seq_len=100, in=896, out=1152
FLOPs = 2 * 1 * 100 * 896 * 1152 ≈ 206M FLOPs
```

## 在 Transformer 中的角色

Transformer 的 Attention 层通常有 **4 个 Linear 层**：

```python
# 1. Q 投影
Q = linear_q(x)

# 2. K 投影
K = linear_k(x)

# 3. V 投影
V = linear_v(x)

# 4. 输出投影
output = linear_o(attention_output)
```

在 FoloVLLM 中，为了性能优化，将 Q、K、V 合并为一个 Linear：

```python
# 优化：1 个 Linear 代替 3 个
qkv = self.qkv_proj(x)  # 一次矩阵乘法
q, k, v = qkv.split(...)  # 拆分（几乎零开销）
```

## 与 MLP 的关系

**MLP（多层感知机）** = 多个 Linear 层 + 激活函数

```python
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x):
        x = self.fc1(x)         # Linear
        x = self.activation(x)  # 非线性
        x = self.fc2(x)         # Linear
        return x
```

在 Qwen3 中的 MLP（`folovllm/model_executor/models/qwen.py`）：

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x):
        gate_up = self.gate_up_proj(x)  # [B, S, 2*I]
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up           # Gated activation
        x = self.down_proj(x)           # [B, S, H]
        return x
```

## 性能优化技巧

### 1. 合并多个 Linear

```python
# ❌ 慢：3 次矩阵乘法
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)

# ✅ 快：1 次矩阵乘法
qkv = self.qkv_proj(x)
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
```

### 2. 使用混合精度

```python
# 使用 bfloat16 或 float16
linear = nn.Linear(1024, 1024).to(torch.bfloat16)
x = x.to(torch.bfloat16)
y = linear(x)  # 更快，内存更少
```

### 3. 预分配输出 tensor（避免）

PyTorch 已经优化，通常不需要手动预分配。

## 常见错误

### 错误1：维度不匹配

```python
linear = nn.Linear(10, 5)
x = torch.randn(3, 8)  # ❌ 最后一维是 8，不是 10

y = linear(x)  # RuntimeError: size mismatch
```

### 错误2：忘记转置权重

```python
# ❌ 错误
y = x @ linear.weight  # shape 不匹配

# ✅ 正确
y = x @ linear.weight.T  # 或使用 F.linear(x, linear.weight, linear.bias)
```

## 总结

**nn.Linear 的本质**：
- 数学：`y = xW^T + b`
- 作用：将输入从 `in_features` 维映射到 `out_features` 维
- 应用：特征变换、降维/升维、分类器头

**在 FoloVLLM Attention 中**：
- QKV 投影：`[hidden_size] → [q_size + 2*kv_size]`
- 输出投影：`[q_size] → [hidden_size]`
- 优化：合并 QKV 为一个 Linear 层

**关键要点**：
- 保留前面所有维度，只变换最后一维
- 参数量 = `in × out + out`（含 bias）
- Transformer 中最常用的组件之一

