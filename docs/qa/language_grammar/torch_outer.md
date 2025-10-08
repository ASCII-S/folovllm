# PyTorch torch.outer 函数

## 功能

`torch.outer` 计算两个 1D 向量的**外积（outer product）**，也叫**张量积（tensor product）**。

**外积**：将两个向量组合成一个矩阵，矩阵中每个元素是两个向量对应元素的乘积。

## 数学定义

对于向量 `a = [a₀, a₁, ..., aₘ]` 和 `b = [b₀, b₁, ..., bₙ]`：

```
outer(a, b)[i, j] = a[i] × b[j]
```

结果是一个 `(m+1) × (n+1)` 的矩阵。

## 语法

```python
torch.outer(input, vec2) → Tensor
```

**参数**：
- `input`: 1D tensor，长度 m
- `vec2`: 1D tensor，长度 n

**返回**：2D tensor，形状 `[m, n]`

## 基础示例

### 示例1：简单外积

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5])

result = torch.outer(a, b)
print(result)
# tensor([[ 4,  5],
#         [ 8, 10],
#         [12, 15]])
```

**计算过程**：
```
result[0, 0] = 1 × 4 = 4
result[0, 1] = 1 × 5 = 5
result[1, 0] = 2 × 4 = 8
result[1, 1] = 2 × 5 = 10
result[2, 0] = 3 × 4 = 12
result[2, 1] = 3 × 5 = 15
```

### 示例2：矩阵形式

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30, 40])

result = torch.outer(a, b)
print(result)
# tensor([[10, 20, 30, 40],
#         [20, 40, 60, 80],
#         [30, 60, 90, 120]])
```

**可视化**：
```
    [10, 20, 30, 40]
1 × [10, 20, 30, 40] = [10, 20, 30, 40]
2 × [10, 20, 30, 40] = [20, 40, 60, 80]
3 × [10, 20, 30, 40] = [30, 60, 90, 120]
```

### 示例3：浮点数

```python
a = torch.tensor([0.5, 1.0, 1.5])
b = torch.tensor([2.0, 3.0])

result = torch.outer(a, b)
print(result)
# tensor([[1.0, 1.5],
#         [2.0, 3.0],
#         [3.0, 4.5]])
```

## 在第114行的应用：RoPE 频率计算

### 代码分析

```python
# 第106-110行：生成位置索引
t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
t = t / self.scaling_factor
# t: [0, 1, 2, 3, ..., seq_len-1]

# 第114行：计算频率矩阵
freqs = torch.outer(t, self.inv_freq)
```

### 具体示例

假设 `seq_len = 4`，`dim = 64`（则 `inv_freq` 有 32 个元素）：

```python
# t: 位置索引
t = torch.tensor([0., 1., 2., 3.])  # [4]

# inv_freq: 频率向量（简化显示前4个）
inv_freq = torch.tensor([1.0, 0.68, 0.46, 0.32, ...])  # [32]

# 外积
freqs = torch.outer(t, inv_freq)
# Shape: [4, 32]
```

**结果**：
```
freqs[i, j] = position[i] × frequency[j]

freqs = [[0×1.0,  0×0.68,  0×0.46,  0×0.32,  ...],  ← 位置0
         [1×1.0,  1×0.68,  1×0.46,  1×0.32,  ...],  ← 位置1
         [2×1.0,  2×0.68,  2×0.46,  2×0.32,  ...],  ← 位置2
         [3×1.0,  3×0.68,  3×0.46,  3×0.32,  ...]]  ← 位置3
      
      = [[0.00,  0.00,  0.00,  0.00,  ...],
         [1.00,  0.68,  0.46,  0.32,  ...],
         [2.00,  1.36,  0.92,  0.64,  ...],
         [3.00,  2.04,  1.38,  0.96,  ...]]
```

### 数学含义

**RoPE 的核心思想**：为每个位置、每个维度生成不同的旋转角度。

```
angle[position, dim] = position × frequency[dim]
```

- **行（position）**：不同的序列位置（0, 1, 2, ...）
- **列（frequency）**：不同维度的基础频率
- **元素**：该位置在该维度的旋转角度

### 后续计算

```python
# 第117行：复制一遍（用于 sin 和 cos）
emb = torch.cat((freqs, freqs), dim=-1)
# Shape: [4, 64]

# 第119-120行：计算 cos 和 sin
cos_cached = emb.cos()  # [4, 64]
sin_cached = emb.sin()  # [4, 64]
```

## 等价实现

### 方式1：使用广播

```python
# outer 的等价实现
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5])

# 使用 outer
result1 = torch.outer(a, b)

# 使用广播
result2 = a.unsqueeze(1) * b.unsqueeze(0)
# a: [3, 1]
# b: [1, 2]
# 广播后相乘得到 [3, 2]

print(torch.equal(result1, result2))  # True
```

### 方式2：使用矩阵乘法

```python
# 使用矩阵乘法
result3 = a.unsqueeze(1) @ b.unsqueeze(0)

print(torch.equal(result1, result3))  # True
```

### 方式3：使用 einsum

```python
# 使用 einsum（最简洁）
result4 = torch.einsum('i,j->ij', a, b)

print(torch.equal(result1, result4))  # True
```

## 与其他操作的对比

### outer vs matmul

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5.])

# outer: [3] × [2] -> [3, 2]
torch.outer(a, b)
# tensor([[4., 5.],
#         [8., 10.],
#         [12., 15.]])

# matmul: 不同的操作
# [3] @ [2] 会报错（内积需要维度匹配）
# torch.matmul(a, b)  # ❌ 错误
```

### outer vs dot (内积)

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

# 内积（dot product）：1×4 + 2×5 + 3×6 = 32
torch.dot(a, b)  # tensor(32.)
# 返回标量

# 外积（outer product）
torch.outer(a, b)
# tensor([[ 4.,  5.,  6.],
#         [ 8., 10., 12.],
#         [12., 15., 18.]])
# 返回矩阵
```

| 操作             | 输入         | 输出     | 含义   |
| ---------------- | ------------ | -------- | ------ |
| **内积** `dot`   | `[n]`, `[n]` | 标量     | Σ aᵢbᵢ |
| **外积** `outer` | `[m]`, `[n]` | `[m, n]` | aᵢbⱼ   |

### outer vs cross (叉积)

```python
# 叉积（仅限 3D 向量）
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

torch.cross(a, b)
# tensor([-3.,  6., -3.])
# 返回垂直于 a 和 b 的向量（3D）

# 外积
torch.outer(a, b)
# 返回 [3, 3] 矩阵
```

## 常见使用场景

### 场景1：生成网格

```python
# 生成坐标网格
x = torch.arange(5)  # [0, 1, 2, 3, 4]
y = torch.arange(3)  # [0, 1, 2]

# 使用 outer 生成网格的一种方式
grid_x = torch.outer(torch.ones(3), x)
grid_y = torch.outer(y, torch.ones(5))

print(grid_x)
# [[0, 1, 2, 3, 4],
#  [0, 1, 2, 3, 4],
#  [0, 1, 2, 3, 4]]

print(grid_y)
# [[0, 0, 0, 0, 0],
#  [1, 1, 1, 1, 1],
#  [2, 2, 2, 2, 2]]
```

### 场景2：批量计算距离

```python
# 计算点集之间的距离矩阵
points1 = torch.tensor([1., 2., 3.])
points2 = torch.tensor([0., 5.])

# 差值矩阵
diff = torch.outer(points1, torch.ones_like(points2)) - torch.outer(torch.ones_like(points1), points2)
# [[1-0, 1-5],
#  [2-0, 2-5],
#  [3-0, 3-5]]
```

### 场景3：协方差矩阵

```python
# 简化的协方差计算
x = torch.tensor([1., 2., 3.])
x_centered = x - x.mean()

cov = torch.outer(x_centered, x_centered) / (len(x) - 1)
```

### 场景4：注意力权重可视化

```python
# 生成注意力模式
query_weights = torch.tensor([0.5, 1.0, 0.8])
key_weights = torch.tensor([1.0, 0.6, 0.9, 0.7])

attention_pattern = torch.outer(query_weights, key_weights)
# [3, 4] - 3 个 query 对 4 个 key 的基础权重
```

## 性能考虑

### 时间复杂度

对于向量 `a[m]` 和 `b[n]`：
- **时间复杂度**：O(m × n)
- **空间复杂度**：O(m × n)

### 性能对比

```python
import time

a = torch.randn(1000).cuda()
b = torch.randn(1000).cuda()

# 方法1：outer
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    result = torch.outer(a, b)
torch.cuda.synchronize()
print(f"outer: {(time.time() - start) * 1000:.2f} ms")

# 方法2：广播
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    result = a.unsqueeze(1) * b.unsqueeze(0)
torch.cuda.synchronize()
print(f"broadcast: {(time.time() - start) * 1000:.2f} ms")

# 典型结果：两者性能相近
```

## RoPE 中的完整流程

### 第114行在整个 RoPE 中的位置

```python
class RotaryEmbedding:
    def __init__(self, dim, max_pos, base, scaling):
        # 步骤1：预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq: [dim/2] = [32]
        
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        # 步骤2：生成位置索引
        t = torch.arange(seq_len, device=device)
        # t: [seq_len] 例如 [128]
        
        # 步骤3：外积 - 计算所有位置×频率组合
        freqs = torch.outer(t, self.inv_freq)
        # freqs: [128, 32] - 128个位置，32个频率
        
        # 步骤4：扩展维度
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb: [128, 64]
        
        # 步骤5：计算 cos 和 sin
        self._cos_cached = emb.cos()  # [128, 64]
        self._sin_cached = emb.sin()  # [128, 64]
```

### 可视化矩阵

```
位置 × 频率 = 角度矩阵

    freq₀  freq₁  freq₂  ...  freq₃₁
pos₀ [  0      0      0   ...    0   ]  ← 位置0的所有角度
pos₁ [  1.0    0.68   0.46 ...  0.0002]  ← 位置1的所有角度
pos₂ [  2.0    1.36   0.92 ...  0.0004]  ← 位置2的所有角度
...
pos₁₂₇[127.0  86.4   58.4 ...  0.025 ]  ← 位置127的所有角度
```

## 注意事项

### 1. 输入必须是 1D

```python
a = torch.randn(3, 4)  # 2D
b = torch.randn(5)

torch.outer(a, b)  # ❌ RuntimeError: outer: expected 1D tensors
```

### 2. 结果总是 2D

```python
a = torch.tensor([1.])  # 长度1
b = torch.tensor([2.])

result = torch.outer(a, b)
print(result.shape)  # torch.Size([1, 1])
print(result)  # tensor([[2.]])
```

### 3. 梯度传播

```python
a = torch.tensor([1., 2., 3.], requires_grad=True)
b = torch.tensor([4., 5.], requires_grad=True)

result = torch.outer(a, b)
loss = result.sum()
loss.backward()

print(a.grad)  # tensor([9., 9., 9.])  - sum(b) 的广播
print(b.grad)  # tensor([6., 6.])      - sum(a) 的广播
```

## 总结

**torch.outer() 的作用**：
- 计算两个 1D 向量的外积
- `outer(a, b)[i, j] = a[i] × b[j]`
- 返回 2D 矩阵 `[len(a), len(b)]`

**在第114行的应用**：
```python
freqs = torch.outer(t, self.inv_freq)
# t: 位置索引 [seq_len]
# inv_freq: 频率向量 [dim/2]
# freqs: 位置×频率矩阵 [seq_len, dim/2]
```

**数学含义**：
- 为每个位置、每个维度计算旋转角度
- `freqs[pos, dim] = position × frequency[dim]`
- 用于 RoPE 的 sin/cos 计算

**等价操作**：
```python
torch.outer(a, b)
# 等价于
a.unsqueeze(1) * b.unsqueeze(0)
# 或
torch.einsum('i,j->ij', a, b)
```

**关键特性**：
- 输入：两个 1D tensor
- 输出：2D tensor
- 时间复杂度：O(m × n)

