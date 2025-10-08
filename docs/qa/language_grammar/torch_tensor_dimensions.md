# PyTorch Tensor 的维度

## 基本概念

PyTorch tensor 可以是 **0 到任意多维** 的多维数组，理论上没有上限，但实际应用中最常见的是 0-5 维。

## 如何查看 Tensor 的维度

```python
x = torch.randn(2, 3, 4)

# 方法1: 查看形状（最常用）
x.shape         # torch.Size([2, 3, 4])
x.size()        # torch.Size([2, 3, 4])

# 方法2: 查看维度数量
x.dim()         # 3 (表示是3维tensor)
x.ndim          # 3 (与 dim() 相同)

# 方法3: 查看总元素数量
x.numel()       # 24 (2*3*4 = 24个元素)
```

## 常见维度及其含义

### 0-5 维 Tensor 示例

| 维度数  | 形状示例          | 别名          | 常见用途                           |
| ------- | ----------------- | ------------- | ---------------------------------- |
| **0维** | `[]`              | Scalar (标量) | loss 值、学习率                    |
| **1维** | `[10]`            | Vector (向量) | 权重、bias、token IDs              |
| **2维** | `[3, 4]`          | Matrix (矩阵) | Linear 层权重、batch tokens        |
| **3维** | `[2, 3, 4]`       | 3D Tensor     | 时间序列批次、单个 token 的 KV     |
| **4维** | `[2, 3, 4, 5]`    | 4D Tensor     | 图像批次 (NCHW)、Attention weights |
| **5维** | `[2, 3, 4, 5, 6]` | 5D Tensor     | 视频批次 (NCTHW)                   |

### 在 FoloVLLM Milestone 1 中的应用

| 维度数  | 形状示例                                | 在代码中的具体应用                     | 文件位置        |
| ------- | --------------------------------------- | -------------------------------------- | --------------- |
| **1维** | `[seq_len]`                             | `position_ids`, `input_ids` (单个序列) | `llm_engine.py` |
| **2维** | `[batch, seq_len]`                      | 批量的 token IDs (M2会用到)            | `processor.py`  |
| **3维** | `[batch, num_heads, head_dim]`          | decode 阶段的 key/value                | `ops.py` L20-21 |
| **4维** | `[batch, num_heads, seq_len, head_dim]` | prefill 阶段的 key/value, KV cache     | `ops.py` L36-37 |
| **4维** | `[1, 1, seq_len_q, seq_len_k]`          | Causal attention mask                  | `ops.py` L107   |

## 在 `folovllm/attention/ops.py` 中的实例

### reshape_and_cache_kv 函数

```python
def reshape_and_cache_kv(
    key: torch.Tensor,        # 输入: 3维 [batch, num_kv_heads, head_dim]
    value: torch.Tensor,      # 输入: 3维 [batch, num_kv_heads, head_dim]
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    slot_mapping: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入维度: 3维 (decode 阶段单个 token)
    输出维度: 4维 (完整序列的 cache)
    """
    key_cache, value_cache = kv_cache
    
    if key_cache.numel() == 0:
        # 3维 -> 4维: [B, H, D] -> [B, H, 1, D]
        key_cache = key.unsqueeze(2)
        value_cache = value.unsqueeze(2)
    else:
        # 追加: [B, H, S, D] + [B, H, 1, D] -> [B, H, S+1, D]
        key = key.unsqueeze(2)
        key_cache = torch.cat([key_cache, key], dim=2)
    
    return key_cache, value_cache  # 返回: 4维
```

### naive_attention 函数

```python
def naive_attention(
    query: torch.Tensor,   # 4维: [batch, num_heads, seq_len_q, head_dim]
    key: torch.Tensor,     # 4维: [batch, num_kv_heads, seq_len_k, head_dim]
    value: torch.Tensor,   # 4维: [batch, num_kv_heads, seq_len_k, head_dim]
    attn_mask: Optional[torch.Tensor] = None,  # 4维: [1, 1, seq_len_q, seq_len_k]
) -> torch.Tensor:
    """
    所有输入都是 4维
    输出也是 4维: [batch, num_heads, seq_len_q, head_dim]
    """
    # 计算 attention scores: 4维 @ 4维 -> 4维
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    # Shape: [batch, num_heads, seq_len_q, seq_len_k]
    
    return output  # 4维: [batch, num_heads, seq_len_q, head_dim]
```

## 维度操作总结

### 增加维度
- `unsqueeze(dim)`: 在指定位置插入大小为 1 的维度
  ```python
  x = torch.randn(2, 3)        # 2维
  x.unsqueeze(0)                # 3维: [1, 2, 3]
  x.unsqueeze(1)                # 3维: [2, 1, 3]
  ```

### 减少维度
- `squeeze(dim)`: 移除大小为 1 的维度
  ```python
  x = torch.randn(1, 2, 1, 3)  # 4维
  x.squeeze(0)                  # 3维: [2, 1, 3]
  x.squeeze()                   # 2维: [2, 3] (移除所有大小为1的维度)
  ```

### 重组维度
- `reshape()` / `view()`: 改变形状（总元素数不变）
  ```python
  x = torch.randn(2, 3, 4)     # 2*3*4 = 24个元素
  x.reshape(6, 4)               # [6, 4]
  x.reshape(2, 12)              # [2, 12]
  ```

- `transpose()` / `permute()`: 交换维度顺序
  ```python
  x = torch.randn(2, 3, 4)
  x.transpose(0, 1)             # [3, 2, 4]
  x.permute(2, 0, 1)            # [4, 2, 3]
  ```

## 检查维度的最佳实践

在处理复杂的 tensor 操作时，建议添加断言检查：

```python
def my_function(x: torch.Tensor):
    # 检查维度数量
    assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
    
    # 检查具体形状
    batch, num_heads, seq_len, head_dim = x.shape
    assert head_dim == 64, f"Expected head_dim=64, got {head_dim}"
    
    # 或者直接检查整个形状
    expected_shape = (2, 16, 128, 64)
    assert x.shape == expected_shape, f"Shape mismatch: {x.shape} vs {expected_shape}"
```

## 调试技巧

```python
# 在开发时打印维度信息
def debug_tensor(name: str, tensor: torch.Tensor):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dim: {tensor.dim()}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print()

# 使用示例
debug_tensor("key", key)
# 输出:
# key:
#   Shape: torch.Size([1, 2, 128])
#   Dim: 3
#   Dtype: torch.float32
#   Device: cuda:0
```

