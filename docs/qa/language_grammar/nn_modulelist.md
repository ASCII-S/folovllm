# nn.ModuleList 是什么

## 简介

`nn.ModuleList` 是 PyTorch 中的一个容器类，用于存储多个 `nn.Module` 子模块。

## 核心特点

1. **自动注册参数**: 将模块放入 `ModuleList` 后，这些模块的参数会自动注册到父模块中
2. **支持迭代**: 可以像普通 Python list 一样遍历
3. **正确的设备管理**: 调用 `.cuda()` 或 `.to(device)` 时，所有子模块会一起移动

## 与普通 Python List 的区别

```python
# ❌ 错误：参数不会被注册
self.layers = [nn.Linear(10, 10) for _ in range(5)]

# ✅ 正确：参数会被自动注册
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
```

如果使用普通 list，优化器无法找到这些层的参数，导致训练失败。

## 在 Qwen 模型中的使用

```python
self.layers = nn.ModuleList([
    Qwen3DecoderLayer(config, layer_idx)
    for layer_idx in range(config.num_hidden_layers)
])
```

这里创建了多个 Decoder 层（例如 28 层），每一层都是一个独立的 `Qwen3DecoderLayer` 模块。

## 使用方式

```python
# 遍历所有层
for layer in self.layers:
    hidden_states = layer(hidden_states, ...)

# 索引访问
first_layer = self.layers[0]
last_layer = self.layers[-1]

# 获取数量
num_layers = len(self.layers)
```

## 总结

`nn.ModuleList` = **能被 PyTorch 正确识别和管理的模块列表**，是构建深度神经网络时存储多个层的标准方式。

