# PyTorch unsqueeze 函数

## 功能
`unsqueeze` 用于在指定位置插入一个大小为 1 的新维度。

## 语法
```python
tensor.unsqueeze(dim)  # dim: 要插入新维度的位置（从0开始）
```

## 示例

### 基础用法
```python
x = torch.randn(2, 3)      # shape: [2, 3]

x.unsqueeze(0)             # shape: [1, 2, 3]  在最前面插入
x.unsqueeze(1)             # shape: [2, 1, 3]  在中间插入
x.unsqueeze(2)             # shape: [2, 3, 1]  在最后面插入
x.unsqueeze(-1)            # shape: [2, 3, 1]  负索引，从后往前
```

### 在 FoloVLLM 中的应用（Milestone 1）

**位置**: `folovllm/attention/ops.py`

**代码**:
```python
# key 形状: [batch_size, num_kv_heads, head_dim]
key_cache = key.unsqueeze(2)
# key_cache 形状: [batch_size, num_kv_heads, 1, head_dim]
```

**用途**: 在 KV Cache 管理中，为序列维度添加维度以便拼接
- 第一个 token: 初始化 cache 为 `[batch, heads, 1, dim]`
- 后续 token: 将新 token `unsqueeze` 后与旧 cache 在 dim=2 拼接
  ```python
  key_cache = torch.cat([key_cache, key.unsqueeze(2)], dim=2)
  # 旧: [batch, heads, seq_len, dim]
  # 新: [batch, heads, 1, dim]
  # 结果: [batch, heads, seq_len+1, dim]
  ```

## 对应的反操作
`squeeze(dim)` 用于移除大小为 1 的维度：
```python
x = torch.randn(1, 2, 1, 3)    # [1, 2, 1, 3]
x.squeeze(0)                    # [2, 1, 3]  移除第0维
x.squeeze(2)                    # [1, 2, 3]  移除第2维
x.squeeze()                     # [2, 3]     移除所有大小为1的维度
```

## 常见使用场景
1. **广播运算**: 添加维度以匹配其他 tensor 的形状
2. **batch 维度**: 给单个样本添加 batch 维度
3. **序列拼接**: 如 KV Cache 中的 token 追加
4. **维度对齐**: 使不同 tensor 的维度数量一致

