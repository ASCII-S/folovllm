# 为什么调用 layer() 时不需要传入权重参数？

## 问题

在调用每一层时：
```python
hidden_states, residual = layer(positions, hidden_states, residual, kv_cache)
```

明明需要使用权重来计算 QKV，为什么没有传入权重参数？

## 答案

**权重已经存储在每个 layer 实例内部了！**

## 详细解释

### 1. 权重在初始化时创建并存储

看 `Qwen3DecoderLayer` 的 `__init__` 方法：

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        # 每个 layer 实例都有自己的 attention 模块
        self.self_attn = Qwen3Attention(config)  # ← 权重在这里！
        # 每个 layer 实例都有自己的 MLP 模块
        self.mlp = Qwen3MLP(config)              # ← 权重在这里！
        ...
```

### 2. Attention 模块包含 QKV 权重

再看 `Qwen3Attention` → `Attention`：

```python
class Attention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # QKV 投影权重矩阵
        self.qkv_proj = nn.Linear(
            hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=bias,
        )  # ← QKV 权重在这里！
        
        # 输出投影权重矩阵
        self.o_proj = nn.Linear(
            self.q_size,
            hidden_size,
            bias=bias,
        )  # ← O 权重在这里！
```

`nn.Linear` 内部包含权重张量 `weight` 和可选的 `bias`。

### 3. MLP 模块包含 FFN 权重

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        # Gate 和 Up 投影权重
        self.gate_up_proj = nn.Linear(...)  # ← 权重在这里！
        # Down 投影权重
        self.down_proj = nn.Linear(...)     # ← 权重在这里！
```

### 4. 调用 forward 时自动使用内部权重

当你调用：
```python
hidden_states, residual = layer(positions, hidden_states, residual, kv_cache)
```

实际执行流程：
```python
# 等价于调用 layer.forward(...)
def forward(self, positions, hidden_states, residual, kv_cache):
    # 使用 self.self_attn（内部有权重）
    hidden_states = self.self_attn(positions, hidden_states, kv_cache)
    
    # 使用 self.mlp（内部有权重）
    hidden_states = self.mlp(hidden_states)
    
    return hidden_states, residual
```

然后在 `self.self_attn` 内部：
```python
def forward(self, positions, hidden_states, kv_cache):
    # 使用 self.qkv_proj.weight 和 self.qkv_proj.bias 来计算 QKV
    qkv = self.attn.qkv_proj(hidden_states)  # 等价于 hidden_states @ W_qkv + b_qkv
    # ... 后续计算
```

## 关键理解

### 面向对象的封装
- **每个 layer 实例是独立的对象**，拥有自己的权重参数
- 第 0 层的权重 ≠ 第 1 层的权重 ≠ ... ≠ 第 27 层的权重
- 权重作为**成员变量**存储在实例中，不需要每次传递

### 对比传统函数式编程

如果用函数式编程，需要这样写：
```python
# ❌ 函数式风格（不推荐）
def decoder_layer_forward(hidden_states, W_qkv, W_o, W_gate_up, W_down, ...):
    qkv = hidden_states @ W_qkv
    # ... 需要手动传递所有权重
```

而 PyTorch 的面向对象风格：
```python
# ✅ 面向对象风格（推荐）
layer = Qwen3DecoderLayer(config, layer_idx)  # 权重在初始化时创建
output = layer(hidden_states)                 # 自动使用内部权重
```

## 权重存储层级

```
Qwen3Model
├── self.layers[0]: Qwen3DecoderLayer
│   ├── self.self_attn: Qwen3Attention
│   │   └── self.attn: Attention
│   │       ├── self.qkv_proj: nn.Linear (包含 W_qkv 权重)
│   │       └── self.o_proj: nn.Linear (包含 W_o 权重)
│   └── self.mlp: Qwen3MLP
│       ├── self.gate_up_proj: nn.Linear (包含权重)
│       └── self.down_proj: nn.Linear (包含权重)
├── self.layers[1]: Qwen3DecoderLayer
│   ├── self.self_attn: ... (不同的权重！)
│   └── self.mlp: ...       (不同的权重！)
...
└── self.layers[27]: Qwen3DecoderLayer
    ├── self.self_attn: ... (不同的权重！)
    └── self.mlp: ...       (不同的权重！)
```

## 总结

- **权重存储位置**：在每个模块实例的 `nn.Linear`、`nn.Embedding` 等子模块中
- **为什么不传参**：面向对象封装，权重作为对象的属性自动使用
- **好处**：代码简洁、参数管理自动化、易于加载预训练权重

