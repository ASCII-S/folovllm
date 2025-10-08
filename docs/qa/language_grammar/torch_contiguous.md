# PyTorch contiguous() 函数

## 核心概念

### 什么是 Contiguous？

**Contiguous（连续的）** 指 tensor 的元素在内存中是**按照行优先顺序（row-major order）连续存储**的。

```python
# Contiguous tensor（连续）
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# 内存布局：[1, 2, 3, 4, 5, 6]（行优先）

# Non-contiguous tensor（不连续）
y = x.transpose(0, 1)  # [[1, 4],
                       #  [2, 5],
                       #  [3, 6]]
# 内存布局仍然是：[1, 2, 3, 4, 5, 6]
# 但逻辑顺序是：[1, 4, 2, 5, 3, 6]
# 内存不连续！
```

### 为什么会出现 Non-contiguous？

某些操作**只改变 tensor 的视图**（view），不实际移动数据：

- `transpose()` - 转置
- `permute()` - 任意维度重排
- `narrow()` - 窄化
- `expand()` - 扩展

这些操作返回的 tensor 可能是 non-contiguous 的。

## 基础示例

### 示例1：transpose 导致 non-contiguous

```python
import torch

x = torch.randn(2, 3)
print(x.is_contiguous())  # True

# Transpose
y = x.transpose(0, 1)
print(y.is_contiguous())  # False（不连续！）

# 使其连续
y_cont = y.contiguous()
print(y_cont.is_contiguous())  # True
```

### 示例2：查看内存布局

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("原始 tensor:")
print(f"Shape: {x.shape}")
print(f"Stride: {x.stride()}")  # (3, 1) - 行跨度3，列跨度1
print(f"Contiguous: {x.is_contiguous()}")  # True

y = x.transpose(0, 1)
print("\n转置后:")
print(f"Shape: {y.shape}")
print(f"Stride: {y.stride()}")  # (1, 3) - 行跨度1，列跨度3
print(f"Contiguous: {y.is_contiguous()}")  # False

y_cont = y.contiguous()
print("\ncontiguous() 后:")
print(f"Stride: {y_cont.stride()}")  # (2, 1) - 重新排列
print(f"Contiguous: {y_cont.is_contiguous()}")  # True
```

### Stride（步幅）解释

**Stride** 表示在每个维度上前进一步需要跳过多少元素。

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# Shape: [2, 3]
# Stride: (3, 1)

# 访问 x[i, j] 的内存位置：
# memory_offset = i * stride[0] + j * stride[1]
#               = i * 3 + j * 1

# x[0, 0] -> 0*3 + 0*1 = 0 -> 内存位置0 -> 值1
# x[0, 1] -> 0*3 + 1*1 = 1 -> 内存位置1 -> 值2
# x[1, 0] -> 1*3 + 0*1 = 3 -> 内存位置3 -> 值4
```

**Transpose 后**：
```python
y = x.transpose(0, 1)
# Shape: [3, 2]
# Stride: (1, 3) - 只改变了 stride，没有移动数据！

# y[0, 0] -> 0*1 + 0*3 = 0 -> 值1
# y[0, 1] -> 0*1 + 1*3 = 3 -> 值4
# y[1, 0] -> 1*1 + 0*3 = 1 -> 值2
```

## 在第152行的应用

### 代码分析

```python
# 第152行
output = output.transpose(1, 2).contiguous()
```

**步骤分解**：

**1. Attention 输出**：
```python
# output 形状: [batch_size, num_heads, seq_len, head_dim]
#              例如: [1, 14, 10, 64]
```

**2. Transpose（交换维度1和2）**：
```python
output = output.transpose(1, 2)
# 形状: [batch_size, seq_len, num_heads, head_dim]
#       [1, 10, 14, 64]
# 此时 output 是 non-contiguous！
```

**3. Contiguous**：
```python
output = output.contiguous()
# 形状不变: [1, 10, 14, 64]
# 但内存重新排列，变成连续的
```

**4. View（第153行）**：
```python
output = output.view(batch_size, seq_len, self.q_size)
# 形状: [1, 10, 896]  (14 * 64 = 896)
```

### 为什么需要 contiguous()？

**关键原因**：`view()` **要求 tensor 必须是 contiguous 的**！

```python
# ❌ 错误示例
x = torch.randn(2, 3, 4)
y = x.transpose(0, 1)  # Non-contiguous
z = y.view(-1)  # RuntimeError: view size is not compatible with input tensor's size and stride

# ✅ 正确示例
x = torch.randn(2, 3, 4)
y = x.transpose(0, 1).contiguous()  # 变连续
z = y.view(-1)  # OK
```

### 替代方案：reshape

```python
# reshape 会自动调用 contiguous()
output = output.transpose(1, 2).reshape(batch_size, seq_len, self.q_size)

# 等价于
output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_size)
```

但显式调用 `contiguous()` 更清晰，表明意图。

## 哪些操作需要 Contiguous？

### 必须 Contiguous 的操作

1. **`view()`**：改变形状
   ```python
   x.transpose(0, 1).view(-1)  # ❌ 错误
   x.transpose(0, 1).contiguous().view(-1)  # ✅ 正确
   ```

2. **某些 CUDA 操作**：
   ```python
   # 一些自定义 CUDA kernel 要求连续输入
   ```

3. **保存模型**：
   ```python
   torch.save(model.state_dict(), 'model.pt')
   # 建议参数都是 contiguous 的
   ```

### 自动处理 Contiguous 的操作

1. **`reshape()`**：自动调用 contiguous
   ```python
   x.transpose(0, 1).reshape(-1)  # OK，内部会调用 contiguous
   ```

2. **大部分数学运算**：
   ```python
   x.transpose(0, 1) + y  # OK
   torch.matmul(x.transpose(0, 1), y)  # OK
   ```

## 性能影响

### contiguous() 的开销

```python
import time
import torch

x = torch.randn(1000, 1000).cuda()
y = x.transpose(0, 1)

# 测量 contiguous() 的时间
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    z = y.contiguous()
torch.cuda.synchronize()
print(f"Time: {(time.time() - start) * 1000:.2f} ms")
# 典型结果: ~5-10 ms（需要复制数据）
```

**开销**：需要**复制整个 tensor 到新的内存位置**，时间复杂度 O(n)。

### 何时避免 contiguous()

如果后续操作不需要连续性，可以避免：

```python
# ❌ 不必要的 contiguous
x = x.transpose(0, 1).contiguous()
y = x + 1  # 加法不需要 contiguous

# ✅ 只在必要时调用
x = x.transpose(0, 1)
y = x + 1
z = x.contiguous().view(-1)  # view 需要 contiguous
```

## 内存布局可视化

### Contiguous Tensor

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

内存：[1][2][3][4][5][6]
访问：x[0,0]=1, x[0,1]=2, x[0,2]=3, x[1,0]=4, ...
     ↑连续访问
```

### Non-contiguous Tensor

```python
y = x.transpose(0, 1)  # [[1, 4],
                       #  [2, 5],
                       #  [3, 6]]

内存仍然：[1][2][3][4][5][6]
访问：y[0,0]=1, y[0,1]=4, y[1,0]=2, y[1,1]=5, ...
               ↑        ↑跳跃访问（stride=3）
```

### Contiguous 后

```python
z = y.contiguous()

新内存：[1][4][2][5][3][6]
访问：z[0,0]=1, z[0,1]=4, z[1,0]=2, z[1,1]=5, ...
     ↑连续访问
```

## 检查和调试

### 检查是否 Contiguous

```python
x = torch.randn(2, 3)
print(x.is_contiguous())  # True

y = x.transpose(0, 1)
print(y.is_contiguous())  # False

z = y.contiguous()
print(z.is_contiguous())  # True
```

### 查看 Stride

```python
x = torch.randn(2, 3, 4)
print(x.stride())  # (12, 4, 1) - contiguous

y = x.transpose(0, 2)
print(y.stride())  # (1, 4, 12) - non-contiguous
```

**Contiguous 的 Stride 特征**：最后一维的 stride 是 1，前面的依次递增。

### 常见错误和修复

#### 错误1：view 要求 contiguous

```python
# ❌ 错误
x = torch.randn(2, 3, 4)
y = x.transpose(0, 1).view(-1)
# RuntimeError: view size is not compatible with input tensor's size and stride

# ✅ 修复1：使用 contiguous
y = x.transpose(0, 1).contiguous().view(-1)

# ✅ 修复2：使用 reshape
y = x.transpose(0, 1).reshape(-1)
```

#### 错误2：忘记 contiguous 导致性能下降

```python
# ❌ 低效：non-contiguous tensor 的访问慢
x = x.transpose(0, 1)
for _ in range(1000):
    y = some_operation(x)  # 每次访问都跳跃

# ✅ 高效：先变连续
x = x.transpose(0, 1).contiguous()
for _ in range(1000):
    y = some_operation(x)  # 连续访问
```

## 在 Transformer 中的典型用法

### Attention 输出重排

```python
# Multi-head attention 输出
# [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]

output = attn_output.transpose(1, 2).contiguous()
output = output.view(batch, seq_len, hidden_size)
```

### MLP 计算

```python
# 通常不需要 transpose，所以不需要 contiguous
x = self.fc1(x)
x = F.gelu(x)
x = self.fc2(x)
```

### Embedding

```python
# Embedding 输出通常是 contiguous 的
embeddings = self.embedding(input_ids)  # Contiguous
```

## 高级：自定义 CUDA Kernel

某些自定义 CUDA kernel 要求输入是 contiguous 的：

```python
# 假设有个自定义 kernel
def my_cuda_kernel(x):
    assert x.is_contiguous(), "Input must be contiguous"
    # CUDA kernel 假设内存连续
    ...

# 使用前确保 contiguous
x = x.transpose(0, 1).contiguous()
output = my_cuda_kernel(x)
```

## 总结

**contiguous() 的作用**：
- 将 tensor 在内存中重新排列，使其连续存储
- 返回一个新的 tensor（如果原本就连续，则返回自身）

**何时需要**：
1. 使用 `view()` 之前
2. 某些 CUDA 操作要求
3. 提高访问性能（避免跳跃访问）

**在第152行的应用**：
```python
output = output.transpose(1, 2).contiguous()
output = output.view(batch_size, seq_len, self.q_size)
```
- `transpose` 使 tensor 变为 non-contiguous
- `contiguous()` 重新排列内存
- `view()` 改变形状（要求 contiguous）

**性能考虑**：
- `contiguous()` 有开销（需复制数据）
- 只在必要时调用
- 可用 `reshape()` 自动处理

**最佳实践**：
- 使用 `is_contiguous()` 检查
- 必要时显式调用 `contiguous()`
- 优先使用 `reshape()` 而非 `view()` + `contiguous()`

