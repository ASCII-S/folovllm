# PyTorch torch.cat 函数

## 功能
`torch.cat` 用于**在指定维度上拼接（concatenate）多个 tensor**。

## 语法
```python
torch.cat(tensors, dim=0)
```

**参数**：
- `tensors`: tensor 列表或元组，要拼接的 tensor
- `dim`: 整数，在哪个维度上拼接（默认 dim=0）

**要求**：
- 所有 tensor 的维度数量必须相同
- 除了拼接维度外，其他维度的大小必须一致

## 基础示例

### 1维拼接
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

torch.cat([a, b], dim=0)
# 结果: tensor([1, 2, 3, 4, 5, 6])
```

### 2维拼接
```python
a = torch.randn(2, 3)  # [2, 3]
b = torch.randn(2, 3)  # [2, 3]

# 在第0维（行）拼接
torch.cat([a, b], dim=0)  # [4, 3]

# 在第1维（列）拼接
torch.cat([a, b], dim=1)  # [2, 6]
```

### 多个 tensor 拼接
```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(2, 3)

torch.cat([a, b, c], dim=0)  # [6, 3]
```

### 不同大小的拼接（仅拼接维度可以不同）
```python
a = torch.randn(2, 3)  # [2, 3]
b = torch.randn(2, 5)  # [2, 5]

torch.cat([a, b], dim=1)  # [2, 8]  ✅ 正确
# torch.cat([a, b], dim=0)  # ❌ 错误！第1维大小不匹配
```

## 在 FoloVLLM Milestone 1 中的应用

### KV Cache 的序列拼接（`folovllm/attention/ops.py` L45-46）

```python
def reshape_and_cache_kv(...):
    """每次生成新 token 时，追加到 KV cache"""
    
    if key_cache.numel() == 0:
        # 第一个 token: 初始化
        key_cache = key.unsqueeze(2)
        # Shape: [batch, num_kv_heads, 1, head_dim]
    else:
        # 后续 token: 追加到已有 cache
        key = key.unsqueeze(2)  # [batch, num_kv_heads, 1, head_dim]
        
        # 在 dim=2 (序列维度) 上拼接
        key_cache = torch.cat([key_cache, key], dim=2)
        #   旧: [batch, num_kv_heads, past_seq_len, head_dim]
        #   新: [batch, num_kv_heads, 1, head_dim]
        #  结果: [batch, num_kv_heads, past_seq_len+1, head_dim]
```

**实际效果**：
- 第1个 token: `[1, 2, 1, 64]`
- 第2个 token: `[1, 2, 1, 64]` + `[1, 2, 1, 64]` → `[1, 2, 2, 64]`
- 第3个 token: `[1, 2, 2, 64]` + `[1, 2, 1, 64]` → `[1, 2, 3, 64]`
- ...
- 第N个 token: `[1, 2, N-1, 64]` + `[1, 2, 1, 64]` → `[1, 2, N, 64]`

### 可视化示例

假设 `head_dim=3`，`num_kv_heads=1`，`batch=1`：

```python
# Step 1: 第一个 token
key_cache = [[[[0.1, 0.2, 0.3]]]]  # [1, 1, 1, 3]

# Step 2: 追加第二个 token
new_key = [[[[0.4, 0.5, 0.6]]]]    # [1, 1, 1, 3]
key_cache = torch.cat([key_cache, new_key], dim=2)
# 结果: [[[[0.1, 0.2, 0.3],
#          [0.4, 0.5, 0.6]]]]       # [1, 1, 2, 3]

# Step 3: 追加第三个 token
new_key = [[[[0.7, 0.8, 0.9]]]]    # [1, 1, 1, 3]
key_cache = torch.cat([key_cache, new_key], dim=2)
# 结果: [[[[0.1, 0.2, 0.3],
#          [0.4, 0.5, 0.6],
#          [0.7, 0.8, 0.9]]]]       # [1, 1, 3, 3]
```

## 与其他拼接函数的对比

### `torch.cat` vs `torch.stack`

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# cat: 在现有维度上拼接，不增加维度
torch.cat([a, b], dim=0)    # [4, 3] (2维)

# stack: 创建新维度并拼接
torch.stack([a, b], dim=0)  # [2, 2, 3] (3维)
```

**区别**：
- `cat`: 在已有维度拼接，维度数不变
- `stack`: 创建新维度，维度数 +1

### `torch.cat` vs `+` (拼接 vs 相加)

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# cat: 拼接
torch.cat([a, b])  # [1, 2, 3, 4, 5, 6]

# +: 逐元素相加
a + b              # [5, 7, 9]
```

## 性能注意事项

### 1. 避免在循环中频繁拼接

**❌ 低效写法**（每次都重新分配内存）：
```python
result = torch.tensor([])
for i in range(1000):
    new_data = torch.randn(1, 100)
    result = torch.cat([result, new_data], dim=0)  # 每次都复制整个 result
```

**✅ 高效写法**（预分配或使用列表）：
```python
# 方法1: 预分配内存
result = torch.zeros(1000, 100)
for i in range(1000):
    result[i] = torch.randn(100)

# 方法2: 先收集再拼接
data_list = []
for i in range(1000):
    data_list.append(torch.randn(1, 100))
result = torch.cat(data_list, dim=0)  # 只拼接一次
```

### 2. 在 M1 中的权衡

在 `reshape_and_cache_kv` 中，我们每个 decode step 都要 `cat` 一次：

```python
# M1: 简单但低效（每次都复制整个 cache）
key_cache = torch.cat([key_cache, key], dim=2)
```

**M3 改进**：使用 **Paged Attention**，预分配固定大小的 cache 块：
```python
# M3: 高效（直接写入预分配的内存）
key_cache[:, :, slot_mapping, :] = key
```

这是为什么 M3 会有显著的性能提升。

## 错误示例和修复

### 错误1: 维度不匹配
```python
a = torch.randn(2, 3)
b = torch.randn(2, 5)

torch.cat([a, b], dim=0)  # ❌ RuntimeError: sizes must match except in dimension 0
```

**修复**：在第1维拼接
```python
torch.cat([a, b], dim=1)  # ✅ [2, 8]
```

### 错误2: tensor 维度数量不同
```python
a = torch.randn(2, 3)     # 2维
b = torch.randn(2, 3, 4)  # 3维

torch.cat([a, b], dim=0)  # ❌ RuntimeError: all input tensors must have the same number of dimensions
```

**修复**：先用 `unsqueeze` 调整维度
```python
a = a.unsqueeze(-1)       # [2, 3, 1]
torch.cat([a, b], dim=2)  # ✅ [2, 3, 5]
```

## 常见使用场景

1. **序列拼接**：追加新 token（如本例中的 KV cache）
2. **批次合并**：合并多个小 batch 为大 batch
3. **特征融合**：在 channel 维度拼接不同特征
4. **数据增强**：拼接原始数据和增强数据

## 总结

- `torch.cat` 在指定维度拼接 tensor，维度数不变
- 除拼接维度外，其他维度大小必须一致
- 在循环中频繁拼接效率低，优先考虑预分配或批量拼接
- 在 M1 的 KV cache 中，每次生成新 token 就在序列维度追加一次

