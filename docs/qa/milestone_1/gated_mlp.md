# Gated MLP（门控多层感知机）

## 基本概念

**Gated MLP** 是一种改进的多层感知机（MLP）结构，使用**门控机制（Gating Mechanism）** 来选择性地控制信息流。

## 传统 MLP vs Gated MLP

### 传统 MLP

```python
class TraditionalMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.ReLU()  # 或 GELU
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        x = self.fc1(x)              # 升维
        x = self.activation(x)       # 激活
        x = self.fc2(x)              # 降维
        return x
```

**流程**：
```
输入 → Linear(升维) → 激活函数 → Linear(降维) → 输出
```

### Gated MLP（门控MLP）

```python
class GatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.SiLU()  # 或其他激活函数
    
    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        gate = self.gate_proj(x)     # 门控路径
        up = self.up_proj(x)         # 值路径
        x = self.activation(gate) * up  # 门控 * 值
        x = self.down_proj(x)        # 降维
        return x
```

**流程**：
```
         ┌─→ Linear(gate) → 激活 ─┐
输入 ────┤                        ├─→ 逐元素相乘 → Linear(降维) → 输出
         └─→ Linear(up) ──────────┘
```

## 核心思想：门控机制

### 什么是门控？

**门控（Gating）** 是一种控制信息流的机制，通过一个"门"来决定有多少信息可以通过。

```python
output = gate ⊙ value
```

- `gate`：控制信号（0-1之间或其他范围）
- `value`：数据信号
- `⊙`：逐元素相乘

### 直观理解

想象一个水龙头：
- **value**：水流的强度
- **gate**：水龙头的开关程度
- **output**：实际流出的水量

```
value: [1.0, 2.0, 3.0, 4.0]
gate:  [0.0, 0.5, 1.0, 0.3]  ← 激活后的门控值
output:[0.0, 1.0, 3.0, 1.2]  ← gate * value
```

## 在 FoloVLLM 中的实现：Qwen3MLP

### 完整代码

**文件**：`folovllm/model_executor/models/qwen.py` (第63-108行)

```python
class Qwen3MLP(nn.Module):
    """Qwen3 MLP with SwiGLU activation (gated MLP).
    
    This uses a gated mechanism where the output is computed as:
        SiLU(gate_proj(x)) * up_proj(x)
    
    For efficiency, gate_proj and up_proj are merged into a single linear layer.
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size          # 896
        self.intermediate_size = config.intermediate_size  # 4864
        
        # 合并 gate 和 up 投影（性能优化）
        self.gate_up_proj = nn.Linear(
            self.hidden_size,              # 输入：896
            2 * self.intermediate_size,    # 输出：2 × 4864 = 9728
            bias=False,
        )
        
        # Down projection
        self.down_proj = nn.Linear(
            self.intermediate_size,  # 输入：4864
            self.hidden_size,        # 输出：896
            bias=False,
        )
        
        # SwiGLU: SiLU(gate) * up
        self.act_fn = SiLUAndMul()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch, seq_len, 896]
        
        # 一次性投影得到 gate 和 up
        gate_up = self.gate_up_proj(hidden_states)
        # gate_up: [batch, seq_len, 9728]
        
        # 拆分并应用 SiLU(gate) * up
        hidden_states = self.act_fn(gate_up)
        # hidden_states: [batch, seq_len, 4864]
        
        # 降维回原始维度
        hidden_states = self.down_proj(hidden_states)
        # hidden_states: [batch, seq_len, 896]
        
        return hidden_states
```

### 详细流程图

```
输入: [batch, seq_len, 896]
    ↓
gate_up_proj (Linear)
    ↓
[batch, seq_len, 9728]
    ↓
split into two halves
    ↓                    ↓
gate: [B, S, 4864]    up: [B, S, 4864]
    ↓                    ↓
SiLU(gate)               |
    ↓                    |
激活后的gate             |
    └────────┬───────────┘
             ↓
    逐元素相乘 (*)
             ↓
    [batch, seq_len, 4864]
             ↓
    down_proj (Linear)
             ↓
输出: [batch, seq_len, 896]
```

### 数值示例

```python
# 假设简化的例子
hidden_states = torch.randn(1, 1, 4)  # [B, S, H]

# 步骤1：gate_up 投影
gate_up = self.gate_up_proj(hidden_states)
# gate_up shape: [1, 1, 16]  (4 -> 2*8 = 16)

# 步骤2：拆分
gate, up = gate_up.chunk(2, dim=-1)
# gate: [1, 1, 8]  - 门控信号
# up:   [1, 1, 8]  - 值信号

# 假设具体数值
gate = [[ [2.0, -1.0, 0.5, 1.5, -0.5, 3.0, -2.0, 1.0] ]]
up   = [[ [1.0,  2.0, 3.0, 4.0,  5.0, 6.0,  7.0, 8.0] ]]

# 步骤3：激活 gate
activated_gate = F.silu(gate)
# ≈ [[ [1.76, -0.27, 0.31, 1.11, -0.19, 2.86, -0.24, 0.73] ]]

# 步骤4：逐元素相乘
result = activated_gate * up
# ≈ [[ [1.76, -0.54, 0.93, 4.44, -0.95, 17.16, -1.68, 5.84] ]]

# 步骤5：降维
output = self.down_proj(result)
# output: [1, 1, 4]
```

## 为什么使用 Gated MLP？

### 1. 更强的表达能力

**传统 MLP**：
```
output = activation(Wx + b)
```
- 每个神经元只有一个控制维度

**Gated MLP**：
```
output = activation(W_gate × x) ⊙ (W_up × x)
```
- 两个独立的变换路径
- gate 可以学习"关闭"某些维度

### 2. 选择性信息流

```python
# 示例：gate 学会过滤不重要的信息
gate_activated = [0.1, 0.9, 0.05, 0.95]  # 激活后的门控
up_values      = [5.0, 3.0, 8.0,  2.0]   # 值

output = gate_activated * up_values
       = [0.5, 2.7, 0.4,  1.9]
#        ↑低    ↑高   ↑低    ↑高
# gate 选择性地允许某些信息通过
```

### 3. 更好的梯度流

**问题**：传统 MLP 在激活函数饱和时梯度消失

```python
# 传统 MLP
x → Linear → ReLU → Linear
         ↑ 如果输入全是负数，梯度为0

# Gated MLP
x → gate路径 → SiLU(gate) × up
  → up路径 ──────────────┘
# up 路径提供直接的梯度路径
```

### 4. 实验效果更好

多项研究表明，Gated MLP 在多种任务上优于传统 MLP：
- **LLaMA**：使用 Gated MLP，性能显著提升
- **Qwen**：采用 Gated MLP，训练更稳定
- **GLU 系列**：多种变体（GLU, SwiGLU, GeGLU）都优于标准 MLP

## GLU 变体家族

Gated MLP 有多种变体，统称为 **GLU（Gated Linear Unit）** 家族：

| 变体       | 激活函数   | 公式              | 使用模型        |
| ---------- | ---------- | ----------------- | --------------- |
| **GLU**    | Sigmoid    | `σ(gate) * up`    | 原始论文        |
| **ReGLU**  | ReLU       | `ReLU(gate) * up` | -               |
| **GELU**   | GELU       | `GELU(gate) * up` | BERT变体        |
| **SwiGLU** | SiLU/Swish | `SiLU(gate) * up` | **LLaMA, Qwen** |

### SwiGLU（Qwen3 使用）

```python
SwiGLU(x) = SiLU(W_gate × x) ⊙ (W_up × x)
          = (W_gate×x / (1 + e^(-W_gate×x))) ⊙ (W_up × x)
```

**特点**：
- 平滑可导（SiLU 的优势）
- 自门控（SiLU 的特性）
- 性能优异

## 性能优化：合并投影

### 未优化版本

```python
class GatedMLP_Slow(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x):
        gate = self.gate_proj(x)     # Linear 1
        up = self.up_proj(x)         # Linear 2
        x = F.silu(gate) * up        # SiLU + Mul
        x = self.down_proj(x)        # Linear 3
        return x
```

**问题**：
- 3 次矩阵乘法
- 3 次 kernel launch
- 需要分别存储 gate 和 up

### 优化版本（FoloVLLM）

```python
class GatedMLP_Fast(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        # 合并 gate 和 up 投影
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = SiLUAndMul()
    
    def forward(self, x):
        gate_up = self.gate_up_proj(x)  # 1 次 Linear（合并）
        x = self.act_fn(gate_up)        # SiLU + Mul（fused）
        x = self.down_proj(x)           # Linear
        return x
```

**优势**：
- 2 次矩阵乘法（减少 33%）
- 2 次 kernel launch（减少 33%）
- 更好的内存局部性

### 性能提升

```python
import time
import torch

hidden_size = 896
intermediate_size = 4864
batch_size = 1
seq_len = 128

x = torch.randn(batch_size, seq_len, hidden_size).cuda()

# 未优化版本
model_slow = GatedMLP_Slow(hidden_size, intermediate_size).cuda()
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y = model_slow(x)
torch.cuda.synchronize()
time_slow = time.time() - start

# 优化版本
model_fast = GatedMLP_Fast(hidden_size, intermediate_size).cuda()
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y = model_fast(x)
torch.cuda.synchronize()
time_fast = time.time() - start

print(f"Slow: {time_slow:.4f}s")
print(f"Fast: {time_fast:.4f}s")
print(f"Speedup: {time_slow / time_fast:.2f}x")
# 典型结果: 1.2-1.3x 加速
```

## 参数量分析

### 传统 MLP

```python
# 参数量
fc1_params = hidden_size * intermediate_size
fc2_params = intermediate_size * hidden_size
total = 2 * hidden_size * intermediate_size
```

**示例**（Qwen3-0.6B）：
```
hidden_size = 896
intermediate_size = 4864
total = 2 × 896 × 4864 = 8,714,752 ≈ 8.7M 参数
```

### Gated MLP

```python
# 参数量
gate_proj_params = hidden_size * intermediate_size
up_proj_params = hidden_size * intermediate_size
down_proj_params = intermediate_size * hidden_size
total = 3 * hidden_size * intermediate_size
```

**示例**（Qwen3-0.6B）：
```
total = 3 × 896 × 4864 = 13,072,128 ≈ 13.1M 参数
```

**结论**：Gated MLP 参数量是传统 MLP 的 **1.5 倍**。

但性能提升超过参数增加，所以参数效率（performance per parameter）更高。

## 在 Transformer 中的位置

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Attention block
        x = x + self.attention(self.ln1(x))
        
        # MLP block（这里！）
        x = x + self.mlp(self.ln2(x))
        
        return x
```

每个 Transformer 层都有一个 MLP，通常参数量占整个模型的 **60-70%**。

## 总结

**Gated MLP** 是什么：
- 使用门控机制的改进 MLP
- 两条路径：gate（控制）+ up（值）
- 输出 = `activation(gate) * up`

**在 Qwen3 中的实现**：
```python
gate_up = self.gate_up_proj(x)      # 合并投影
output = SiLU(gate) * up            # SwiGLU
output = self.down_proj(output)     # 降维
```

**优势**：
- ✅ 更强的表达能力（两条独立路径）
- ✅ 选择性信息流（门控机制）
- ✅ 更好的梯度流（直接路径）
- ✅ 实验效果更好（多种模型验证）

**代价**：
- ❌ 参数量增加 50%
- ❌ 计算量略增（但可优化）

**优化技巧**：
- 合并 gate_proj 和 up_proj
- Fused SiLU + Mul
- 减少 kernel launch

**使用场景**：
- 现代 LLM（LLaMA, Qwen, Mistral, etc.）
- 需要强表达能力的模型
- 性能优先的场景

