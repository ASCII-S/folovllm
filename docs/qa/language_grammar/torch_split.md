# PyTorch split 函数

## 功能

`split` 用于**将 tensor 沿指定维度拆分**成多个子 tensor。

## 语法

### 方式1：按固定大小拆分

```python
torch.split(tensor, split_size, dim=0)
# 或
tensor.split(split_size, dim=0)
```

**参数**：
- `split_size`: 每块的大小（整数）
- `dim`: 拆分的维度

**返回**：tuple of tensors

### 方式2：按指定大小列表拆分

```python
torch.split(tensor, split_size_list, dim=0)
# 或
tensor.split([size1, size2, size3], dim=0)
```

**参数**：
- `split_size_list`: 每块的大小列表
- `dim`: 拆分的维度

## 基础示例

### 方式1：均等拆分

```python
import torch

x = torch.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 按大小 3 拆分
result = x.split(3)
# 结果: (tensor([0, 1, 2]),
#        tensor([3, 4, 5]),
#        tensor([6, 7, 8]),
#        tensor([9]))         # 最后一块可能不足 3 个
```

### 方式2：指定大小拆分

```python
x = torch.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 按 [4, 3, 3] 拆分
a, b, c = x.split([4, 3, 3])

print(a)  # tensor([0, 1, 2, 3])
print(b)  # tensor([4, 5, 6])
print(c)  # tensor([7, 8, 9])
```

### 多维 tensor 拆分

```python
x = torch.randn(2, 6, 4)  # [2, 6, 4]

# 在第1维拆分，每块大小 2
result = x.split(2, dim=1)
# 结果: 3 个 tensor，每个形状 [2, 2, 4]

# 在第2维拆分，按 [1, 3] 大小
a, b = x.split([1, 3], dim=2)
# a: [2, 6, 1]
# b: [2, 6, 3]
```

## 在代码第108行的应用

### QKV 拆分

```python
# 第107行：QKV 投影
qkv = self.qkv_proj(hidden_states)
# qkv: [batch, seq_len, q_size + 2*kv_size]
# 例如: [1, 10, 1152] (896 + 128 + 128)

# 第108行：拆分成 Q, K, V
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
# q: [1, 10, 896]
# k: [1, 10, 128]
# v: [1, 10, 128]
```

### 具体示例（Qwen3-0.6B）

```python
# 配置
num_heads = 14
num_kv_heads = 2
head_dim = 64

q_size = num_heads * head_dim = 14 * 64 = 896
kv_size = num_kv_heads * head_dim = 2 * 64 = 128

# QKV 投影输出
qkv = torch.randn(1, 10, 1152)  # [batch=1, seq_len=10, 896+128+128]

# 拆分
q, k, v = qkv.split([896, 128, 128], dim=-1)

# 结果
q.shape  # torch.Size([1, 10, 896])  - Query: 14 heads
k.shape  # torch.Size([1, 10, 128])  - Key:   2 heads (GQA)
v.shape  # torch.Size([1, 10, 128])  - Value: 2 heads (GQA)
```

### 为什么要拆分？

**步骤1**：合并投影（性能优化）
```python
# ❌ 慢：3 次矩阵乘法
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)

# ✅ 快：1 次矩阵乘法
qkv = self.qkv_proj(x)  # 一次性计算 QKV
```

**步骤2**：拆分（几乎零开销）
```python
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
# split 只是创建视图，不复制数据
```

**收益**：
- 减少 kernel launch（GPU 优化）
- 更好的内存局部性
- 整体加速 10-20%

## 与其他函数的对比

### split vs chunk

```python
x = torch.arange(10)

# split: 指定每块的大小
x.split(3)  # (3个, 3个, 3个, 1个) - 最后一块可能不同

# chunk: 指定块的数量
x.chunk(3)  # (4个, 3个, 3个) - 尽量均分
```

| 函数    | 指定       | 结果           |
| ------- | ---------- | -------------- |
| `split` | 每块的大小 | 块数不固定     |
| `chunk` | 块的数量   | 每块大小不固定 |

### split vs unbind

```python
x = torch.randn(3, 4)

# split: 返回 tuple，每个元素保留该维度
parts = x.split(1, dim=0)  # 3 个 tensor，每个形状 [1, 4]

# unbind: 返回 tuple，移除该维度
parts = x.unbind(dim=0)    # 3 个 tensor，每个形状 [4]
```

### split vs tensor indexing

```python
x = torch.arange(10)

# split
a, b, c = x.split([4, 3, 3])

# 等价于 indexing
a = x[0:4]
b = x[4:7]
c = x[7:10]
```

**split 的优势**：更简洁，自动计算边界。

## 内存和性能

### split 是视图操作

```python
x = torch.randn(10)
a, b = x.split([6, 4])

# a 和 b 共享 x 的内存
a[0] = 999
print(x[0])  # 999（x 也被修改）

# 如果需要独立副本
a = a.clone()
```

### 性能对比

```python
import time

x = torch.randn(1000, 1000, 3000).cuda()

# 方式1: split（推荐）
start = time.time()
a, b, c = x.split([1000, 1000, 1000], dim=2)
print(f"split: {time.time() - start:.6f}s")  # ~0.000001s（极快）

# 方式2: indexing
start = time.time()
a = x[:, :, 0:1000]
b = x[:, :, 1000:2000]
c = x[:, :, 2000:3000]
print(f"indexing: {time.time() - start:.6f}s")  # 类似

# 方式3: 多次 Linear（慢）
start = time.time()
a = linear_a(x)
b = linear_b(x)
c = linear_c(x)
print(f"3 linears: {time.time() - start:.6f}s")  # ~0.01s（慢100倍）
```

## 常见使用场景

### 1. QKV 拆分（如本例）

```python
qkv = self.qkv_proj(x)
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
```

### 2. Gate + Up 拆分（Gated MLP）

```python
# Qwen3MLP
gate_up = self.gate_up_proj(x)  # [B, S, 2*I]
gate, up = gate_up.split([intermediate_size, intermediate_size], dim=-1)
# 或者使用 chunk
gate, up = gate_up.chunk(2, dim=-1)
```

### 3. 批次拆分

```python
# 将大 batch 拆成小 batch
batch = torch.randn(128, 1024)
mini_batches = batch.split(32, dim=0)  # 4 个 [32, 1024]

for mini_batch in mini_batches:
    output = model(mini_batch)
```

### 4. 多头注意力

```python
# 拆分成多个 head
# x: [batch, seq_len, num_heads * head_dim]
heads = x.split(head_dim, dim=-1)
# 每个 head: [batch, seq_len, head_dim]
```

### 5. RGB 通道拆分

```python
# 图像: [batch, 3, height, width]
r, g, b = image.split(1, dim=1)
# r, g, b: 每个 [batch, 1, height, width]
```

## 错误示例

### 错误1: 大小不匹配

```python
x = torch.arange(10)
a, b = x.split([6, 6], dim=0)  # ❌ 6+6=12 > 10

# RuntimeError: split_with_sizes expects split_sizes to sum exactly to 10
```

### 错误2: 维度越界

```python
x = torch.randn(3, 4)
x.split(2, dim=2)  # ❌ 只有 2 维（0, 1），没有维度 2

# IndexError: dimension out of range
```

### 错误3: 解包数量不匹配

```python
x = torch.arange(10)
a, b = x.split(3)  # ❌ split 返回 4 个，但只有 2 个变量

# ValueError: too many values to unpack (expected 2)
```

## 高级用法

### 动态拆分

```python
# 根据配置动态决定拆分大小
if use_gqa:  # Grouped Query Attention
    sizes = [q_size, kv_size, kv_size]
else:  # Multi-Head Attention
    sizes = [hidden_size, hidden_size, hidden_size]

q, k, v = qkv.split(sizes, dim=-1)
```

### 与 cat 配合（逆操作）

```python
# split 的逆操作是 cat
x = torch.randn(10)
a, b, c = x.split([4, 3, 3])

# 重新拼接
x_reconstructed = torch.cat([a, b, c], dim=0)
assert torch.equal(x, x_reconstructed)
```

### 批量处理不同长度

```python
# 处理变长序列
sequences = torch.randn(5, 100)  # 5 个序列，最大长度 100
lengths = [80, 60, 90, 70, 100]

# 按实际长度拆分
valid_parts = []
for i, seq in enumerate(sequences):
    valid = seq[:lengths[i]]
    valid_parts.append(valid)
```

## 在 Transformer 中的典型应用

### Multi-Query Attention (MQA)

```python
# Q: num_heads 个, K/V: 1 个
q_size = num_heads * head_dim
kv_size = 1 * head_dim

qkv = self.qkv_proj(x)
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
```

### Grouped Query Attention (GQA)

```python
# Q: num_heads 个, K/V: num_kv_heads 个
q_size = num_heads * head_dim  # 14 * 64 = 896
kv_size = num_kv_heads * head_dim  # 2 * 64 = 128

qkv = self.qkv_proj(x)
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
```

### Multi-Head Attention (MHA)

```python
# Q, K, V: 都是 num_heads 个
hidden_size = num_heads * head_dim

qkv = self.qkv_proj(x)
q, k, v = qkv.split([hidden_size, hidden_size, hidden_size], dim=-1)
# 或者
q, k, v = qkv.chunk(3, dim=-1)
```

## 总结

**split 的作用**：
- 将 tensor 沿指定维度拆分成多个部分
- 两种方式：固定大小 vs 指定大小列表
- 返回 tuple of tensors

**在第108行的应用**：
- 将合并的 QKV 投影结果拆分成 Q、K、V
- 配合合并投影优化性能（1次矩阵乘法 + 拆分）
- GQA 架构：Q 维度 > K/V 维度

**关键特性**：
- 视图操作，几乎零开销
- 自动处理边界计算
- 与 cat 互为逆操作

**最佳实践**：
- 需要均分时用 `chunk`
- 需要不同大小时用 `split([size1, size2, ...])`
- 需要独立副本时调用 `.clone()`

