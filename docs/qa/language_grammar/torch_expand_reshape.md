# PyTorch expand 和 reshape

## expand - 扩展维度（不复制数据）

**功能**：在指定维度上重复 tensor，**共享内存，不实际复制数据**。

```python
x = torch.randn(2, 1, 3)
x.expand(2, 4, 3)  # [2, 1, 3] -> [2, 4, 3]
# 将第1维从1扩展到4，数据重复4次但不占用额外内存
```

**限制**：只能扩展大小为 1 的维度。

## reshape - 改变形状

**功能**：改变 tensor 形状，**总元素数必须不变**。

```python
x = torch.randn(2, 3, 4)  # 24个元素
x.reshape(6, 4)           # [6, 4]
x.reshape(2, 12)          # [2, 12]
x.reshape(24)             # [24]
```

## 在代码第82-84行的应用

```python
# GQA: 2个KV heads 需要重复成 16个 query heads
key = key.unsqueeze(2).expand(
    batch_size, num_kv_heads, num_repeats, seq_len_k, head_dim
).reshape(batch_size, num_heads, seq_len_k, head_dim)
```

**步骤**：
1. `unsqueeze(2)`: `[B, 2, S, D]` → `[B, 2, 1, S, D]`
2. `expand()`: `[B, 2, 1, S, D]` → `[B, 2, 8, S, D]` (重复8次)
3. `reshape()`: `[B, 2, 8, S, D]` → `[B, 16, S, D]` (2×8=16)

**效果**：每个 KV head 重复 8 次，匹配 16 个 query heads。

## 核心区别

| 函数      | 作用              | 是否复制数据         | 限制                   |
| --------- | ----------------- | -------------------- | ---------------------- |
| `expand`  | 扩展大小为1的维度 | ❌ 不复制（共享内存） | 只能扩展 size=1 的维度 |
| `reshape` | 改变形状          | 视情况而定           | 总元素数不变           |

