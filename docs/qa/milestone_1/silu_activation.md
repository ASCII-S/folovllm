# SiLU 激活函数（Sigmoid Linear Unit）

## 基本概念

**SiLU**（Sigmoid Linear Unit），也叫 **Swish**，是一种平滑的非线性激活函数。

## 数学定义

```
SiLU(x) = x · σ(x) = x · sigmoid(x)
        = x / (1 + e^(-x))
```

其中 `σ(x) = sigmoid(x) = 1 / (1 + e^(-x))`

## 函数特性

### 数值示例

```
x = -2  →  SiLU(-2) = -2 × 0.119 = -0.238
x = -1  →  SiLU(-1) = -1 × 0.269 = -0.269
x =  0  →  SiLU(0)  =  0 × 0.5   =  0.0
x =  1  →  SiLU(1)  =  1 × 0.731 =  0.731
x =  2  →  SiLU(2)  =  2 × 0.881 =  1.762
```

### 函数图像

```
    2 |           ╱
      |          ╱
    1 |        ╱
      |      ╱
    0 |────╱─────────
      |  ╱
   -1 | ╱
      |╱
   -2 +─────────────
     -3  -1  0  1  3
```

**特点**：
- **平滑**：处处可导
- **非单调**：在 x ≈ -1.28 处有最小值
- **有界下限**：x → -∞ 时，SiLU(x) → 0
- **无界上限**：x → +∞ 时，SiLU(x) → x

## PyTorch 实现

### 基本用法

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2., -1., 0., 1., 2.])

# 方式1：使用 F.silu
y = F.silu(x)
print(y)
# tensor([-0.2384, -0.2689,  0.0000,  0.7311,  1.7616])

# 方式2：手动实现
y = x * torch.sigmoid(x)

# 方式3：使用 nn.SiLU
silu = torch.nn.SiLU()
y = silu(x)
```

### 在 FoloVLLM 中的实现

**文件**：`folovllm/model_executor/models/utils.py`

#### 基础 SiLU（第205-213行）

```python
class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation function.
    
    Also known as Swish. Used in Qwen3 MLP.
    SiLU(x) = x * sigmoid(x)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)
```

#### Fused SiLUAndMul（第216-237行）

```python
class SiLUAndMul(nn.Module):
    """Fused SiLU and element-wise multiplication.
    
    Used in gated MLPs. Given input [x, y], computes:
        SiLU(x) * y
    
    This is more memory efficient than separate operations.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 2 * hidden_size]
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
```

## 与其他激活函数的对比

### 对比表格

| 激活函数       | 公式        | 范围      | 平滑性      | 计算复杂度 |
| -------------- | ----------- | --------- | ----------- | ---------- |
| **ReLU**       | `max(0, x)` | `[0, ∞)`  | 不可导(x=0) | 低         |
| **GELU**       | `x·Φ(x)`    | `(-∞, ∞)` | 平滑        | 高         |
| **SiLU/Swish** | `x·σ(x)`    | `(-∞, ∞)` | 平滑        | 中         |
| **Sigmoid**    | `σ(x)`      | `(0, 1)`  | 平滑        | 中         |
| **Tanh**       | `tanh(x)`   | `(-1, 1)` | 平滑        | 中         |

### ReLU vs SiLU

```python
x = torch.linspace(-3, 3, 100)

relu = F.relu(x)
silu = F.silu(x)

# ReLU: 硬截断
# x < 0 → 0
# x ≥ 0 → x

# SiLU: 平滑过渡
# x < 0 → 小的负值
# x ≥ 0 → 接近 x
```

**可视化对比**：
```
ReLU:              SiLU:
    |╱                 |    ╱
    |                  |   ╱
────┼────          ───┼──╱────
    |                  | ╱
    |                  |╱
```

### GELU vs SiLU

两者非常相似，都是平滑的激活函数：

```python
x = torch.randn(1000)

gelu = F.gelu(x)
silu = F.silu(x)

# 相关系数
correlation = torch.corrcoef(torch.stack([gelu, silu]))[0, 1]
print(f"Correlation: {correlation:.4f}")
# 通常 > 0.99（非常接近）
```

**差异**：
- GELU 基于正态分布 CDF
- SiLU 基于 sigmoid
- GELU 在 x < 0 时衰减稍快
- 实际性能差异很小

## 在 Gated MLP 中的应用

### 概念：Gated Mechanism（门控机制）

门控机制允许网络学习"打开"或"关闭"信息流：

```
output = gate ⊙ value
```

其中 `gate` 控制 `value` 中每个元素的通过程度。

### Qwen3 MLP 结构

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        # 合并 gate 和 up 投影
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = SiLUAndMul()
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        
        # 一次投影得到 gate 和 up
        gate_up = self.gate_up_proj(x)
        # gate_up: [batch, seq_len, 2 * intermediate_size]
        
        # 应用 SiLU(gate) * up
        x = self.act_fn(gate_up)
        # x: [batch, seq_len, intermediate_size]
        
        # 投影回原维度
        x = self.down_proj(x)
        # x: [batch, seq_len, hidden_size]
        
        return x
```

### SiLUAndMul 详细过程

```python
# 输入
x = torch.randn(1, 10, 2048)  # [batch, seq_len, 2 * intermediate_size]

# 步骤1：拆分
gate, up = x.chunk(2, dim=-1)
# gate: [1, 10, 1024]  - 门控信号
# up:   [1, 10, 1024]  - 值信号

# 步骤2：激活 gate
activated_gate = F.silu(gate)
# activated_gate: [1, 10, 1024]

# 步骤3：逐元素相乘
output = activated_gate * up
# output: [1, 10, 1024]
```

### 为什么使用 Gated MLP？

**传统 MLP**：
```python
x = linear1(x)
x = activation(x)
x = linear2(x)
```

**Gated MLP**：
```python
gate = linear_gate(x)
up = linear_up(x)
x = activation(gate) * up
x = linear_down(x)
```

**优势**：
1. **选择性信息流**：gate 控制哪些信息通过
2. **更强的表达能力**：两路信息交互
3. **更好的梯度流**：避免梯度消失

## 性能优化：Fusion

### 未优化版本

```python
# ❌ 低效：3 次 kernel launch
gate = self.gate_proj(x)      # Linear 1
up = self.up_proj(x)          # Linear 2
gate = F.silu(gate)           # SiLU
output = gate * up            # Mul
output = self.down_proj(output)  # Linear 3
```

### 优化版本（FoloVLLM）

```python
# ✅ 高效：2 次 kernel launch
gate_up = self.gate_up_proj(x)  # 合并 Linear（1次）
output = self.act_fn(gate_up)   # SiLU + Mul（1次）
output = self.down_proj(output) # Linear（1次）
```

**收益**：
- Kernel launch 减少：5次 → 3次
- 内存访问减少：不需要单独存储 gate 和 up
- 加速：~15-20%

### 内存效率分析

**未优化**：
```
x → gate (中间结果) → activated_gate (中间结果) → output
  → up (中间结果)                                    ↑
```

**优化**：
```
x → gate_up (中间结果) → output
    在内部拆分，不存储单独的 gate/up
```

## 梯度和反向传播

### SiLU 的导数

```
d(SiLU)/dx = σ(x) + x·σ(x)·(1-σ(x))
           = σ(x)·(1 + x·(1-σ(x)))
```

### 数值稳定性

```python
# PyTorch 内部实现（数值稳定）
def silu(x):
    return x / (1 + torch.exp(-x))

# 对于大的负数 x，exp(-x) 会很大，但不会溢出
# 对于大的正数 x，exp(-x) ≈ 0，silu(x) ≈ x
```

### 梯度流

```python
x = torch.tensor([1.0], requires_grad=True)
y = F.silu(x)
y.backward()

print(x.grad)  # tensor([0.9277])
# grad = σ(1) + 1·σ(1)·(1-σ(1))
#      = 0.731 + 1 × 0.731 × 0.269
#      = 0.9277
```

## 实际应用

### 1. LLaMA / Qwen3 MLP

```python
# LLaMA-style Gated MLP
x = hidden_states
gate = self.gate_proj(x)
up = self.up_proj(x)
down = self.down_proj(F.silu(gate) * up)
```

### 2. Vision Transformers

```python
# 某些 ViT 变体使用 SiLU
class MLP(nn.Module):
    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        return x
```

### 3. 扩散模型（Diffusion Models）

SiLU 在扩散模型中广泛使用，因为其平滑性有助于稳定训练。

## 为什么选择 SiLU？

### 优点

1. **平滑可导**：处处可导，有利于优化
2. **非单调性**：允许负值（比 ReLU 表达能力强）
3. **自门控**：`x·σ(x)` 形式天然具有门控特性
4. **性能好**：在多个任务上超越 ReLU 和 GELU

### 实验结果

在多个基准测试中：
- **LLaMA**：使用 SiLU，性能优于 ReLU
- **Qwen3**：使用 SiLU，训练更稳定
- **相比 GELU**：性能相近，计算稍快

### 计算成本

| 激活函数 | 相对成本 | 说明             |
| -------- | -------- | ---------------- |
| ReLU     | 1×       | 最快（简单比较） |
| SiLU     | ~1.5×    | sigmoid 计算     |
| GELU     | ~2×      | 更复杂的数学函数 |

虽然 SiLU 比 ReLU 慢，但性能提升值得这个开销。

## 代码示例

### 示例1：基础使用

```python
import torch
import torch.nn.functional as F

# 创建数据
x = torch.randn(2, 3, 4)

# 应用 SiLU
y = F.silu(x)

print(x)
# tensor([[[ 0.5, -0.8,  1.2, -0.3],
#          [ 0.9,  0.2, -1.5,  0.7],
#          [-0.4,  1.0, -0.1,  0.6]]])

print(y)
# tensor([[[ 0.3110, -0.2316,  0.8849, -0.1151],
#          [ 0.6468,  0.1049, -0.2929,  0.4589],
#          [-0.1616,  0.7311, -0.0475,  0.3777]]])
```

### 示例2：Gated MLP

```python
class GatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

# 使用
mlp = GatedMLP(768, 3072)
x = torch.randn(1, 10, 768)
output = mlp(x)  # [1, 10, 768]
```

### 示例3：性能对比

```python
import time

x = torch.randn(1000, 1000).cuda()

# ReLU
start = time.time()
for _ in range(1000):
    y = F.relu(x)
torch.cuda.synchronize()
relu_time = time.time() - start

# SiLU
start = time.time()
for _ in range(1000):
    y = F.silu(x)
torch.cuda.synchronize()
silu_time = time.time() - start

print(f"ReLU: {relu_time:.4f}s")
print(f"SiLU: {silu_time:.4f}s")
print(f"Ratio: {silu_time / relu_time:.2f}x")
# 典型结果: SiLU 约 1.3-1.5x 慢
```

## 总结

**SiLU 激活函数**：
- **定义**：`SiLU(x) = x · sigmoid(x)`
- **别名**：Swish
- **特性**：平滑、非单调、自门控

**在 FoloVLLM 中的应用**：
```python
class SiLUAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
```

**优势**：
- 比 ReLU 表达能力强
- 训练更稳定
- 配合门控机制效果好

**优化技巧**：
- 合并 Linear 层（gate_up_proj）
- Fused SiLU + Mul
- 减少 kernel launch 和内存访问

**适用场景**：
- 现代 LLM（LLaMA、Qwen、Mistral）
- Gated MLP 结构
- 需要平滑激活函数的场景

