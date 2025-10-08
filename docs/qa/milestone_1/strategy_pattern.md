# 策略模式（Strategy Pattern）

## 概念

策略模式是一种**行为型设计模式**，它定义了一系列算法，将每个算法封装起来，并使它们可以**互相替换**。策略模式让算法的变化独立于使用算法的客户端。

**核心思想**：
- 定义一个**抽象接口**（策略接口）
- 实现多个**具体策略**（算法实现）
- 客户端通过接口使用策略，可以**动态切换**不同的实现

## 基本结构

```python
from abc import ABC, abstractmethod

# 1. 策略接口（抽象基类）
class Strategy(ABC):
    @abstractmethod
    def execute(self, data):
        """执行策略的抽象方法"""
        pass

# 2. 具体策略A
class ConcreteStrategyA(Strategy):
    def execute(self, data):
        return f"Strategy A processing: {data}"

# 3. 具体策略B
class ConcreteStrategyB(Strategy):
    def execute(self, data):
        return f"Strategy B processing: {data}"

# 4. 上下文（使用策略的类）
class Context:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: Strategy):
        """动态切换策略"""
        self.strategy = strategy
    
    def do_work(self, data):
        """使用当前策略执行任务"""
        return self.strategy.execute(data)
```

## 使用示例

```python
# 创建不同的策略
strategy_a = ConcreteStrategyA()
strategy_b = ConcreteStrategyB()

# 创建上下文，使用策略A
context = Context(strategy_a)
print(context.do_work("data"))  # Strategy A processing: data

# 动态切换到策略B
context.set_strategy(strategy_b)
print(context.do_work("data"))  # Strategy B processing: data
```

## 在 FoloVLLM 中的应用：Attention Backend

### 抽象策略接口

文件：`folovllm/attention/backends/abstract.py`

```python
from abc import ABC, abstractmethod

class AttentionBackend(ABC):
    """Attention 实现的抽象接口（策略接口）"""
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播的抽象方法"""
        raise NotImplementedError
    
    @abstractmethod
    def get_name(self) -> str:
        """获取后端名称"""
        raise NotImplementedError
```

### 具体策略实现

#### 策略1：TorchNaiveBackend（M1）

文件：`folovllm/attention/backends/torch_naive.py`

```python
class TorchNaiveBackend(AttentionBackend):
    """朴素的 PyTorch 实现"""
    
    def forward(self, query, key, value, kv_cache, scale, attn_mask=None):
        # 纯 PyTorch 实现
        # 简单、易读，但性能较低
        ...
        output = naive_attention(query, key, value, scale, attn_mask)
        return output, updated_kv_cache
    
    def get_name(self) -> str:
        return "torch_naive"
```

#### 策略2：PagedAttentionBackend（M3 将实现）

```python
class PagedAttentionBackend(AttentionBackend):
    """Paged Attention 实现"""
    
    def forward(self, query, key, value, kv_cache, scale, attn_mask=None):
        # 使用 PagedAttention
        # 高效的 KV cache 管理
        ...
        output = paged_attention(query, key, value, ...)
        return output, updated_kv_cache
    
    def get_name(self) -> str:
        return "paged_attention"
```

#### 策略3：FlashAttentionBackend（M4 将实现）

```python
class FlashAttentionBackend(AttentionBackend):
    """Flash Attention 2 实现"""
    
    def forward(self, query, key, value, kv_cache, scale, attn_mask=None):
        # 使用 Flash Attention 2
        # IO 感知，速度最快
        ...
        output = flash_attn_func(query, key, value)
        return output, updated_kv_cache
    
    def get_name(self) -> str:
        return "flash_attention_2"
```

### 客户端代码（Attention 层）

文件：`folovllm/model_executor/layers/attention.py`

```python
class Attention(nn.Module):
    """通用 Attention 层（上下文/客户端）"""
    
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.backend = backend  # 注入策略
        ...
    
    def forward(self, hidden_states, kv_cache, ...):
        # 投影得到 Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 应用 RoPE
        ...
        
        # 使用策略执行 attention（核心）
        attn_output, updated_cache = self.backend.forward(
            query, key, value, kv_cache, self.scale, attn_mask
        )
        
        # 输出投影
        output = self.o_proj(attn_output)
        return output, updated_cache
```

### 无缝切换策略

```python
# M1: 使用朴素实现
backend = TorchNaiveBackend()
attention_layer = Attention(config, backend)

# M3: 切换到 Paged Attention（无需修改 Attention 类）
backend = PagedAttentionBackend()
attention_layer = Attention(config, backend)

# M4: 切换到 Flash Attention（仍然无需修改 Attention 类）
backend = FlashAttentionBackend()
attention_layer = Attention(config, backend)
```

## 策略模式的优点

### 1. 开闭原则（Open-Closed Principle）

**对扩展开放，对修改关闭**

```python
# ✅ 添加新策略：只需新建类，不修改现有代码
class NewOptimizedBackend(AttentionBackend):
    def forward(self, ...):
        # 新的优化实现
        pass
```

### 2. 单一职责原则

每个策略类只负责一种算法实现：
- `TorchNaiveBackend`：只负责朴素实现
- `PagedAttentionBackend`：只负责 Paged Attention
- `FlashAttentionBackend`：只负责 Flash Attention

### 3. 依赖倒置原则

高层模块（`Attention` 层）依赖抽象（`AttentionBackend`），不依赖具体实现。

```python
# Attention 依赖抽象接口，不依赖具体实现
class Attention(nn.Module):
    def __init__(self, backend: AttentionBackend):  # 依赖抽象
        self.backend = backend
```

### 4. 易于测试

可以方便地 mock 或替换策略：

```python
# 测试时使用简单的 mock backend
class MockBackend(AttentionBackend):
    def forward(self, ...):
        return torch.zeros(...), (None, None)

# 单元测试
attention = Attention(config, MockBackend())
```

### 5. 运行时切换

可以根据配置、硬件条件等动态选择策略：

```python
def create_attention_backend(config):
    if config.use_flash_attn and is_flash_attn_available():
        return FlashAttentionBackend()
    elif config.use_paged_attn:
        return PagedAttentionBackend()
    else:
        return TorchNaiveBackend()

backend = create_attention_backend(config)
```

## 经典应用场景

### 1. 排序算法选择

```python
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class QuickSort(SortStrategy):
    def sort(self, data):
        # 快排实现
        pass

class MergeSort(SortStrategy):
    def sort(self, data):
        # 归并排序实现
        pass

# 根据数据特征选择策略
if len(data) < 10:
    strategy = InsertionSort()
else:
    strategy = QuickSort()
```

### 2. 支付方式

```python
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paying {amount} by credit card")

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paying {amount} by PayPal")

class WeChatPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paying {amount} by WeChat Pay")
```

### 3. 压缩算法

```python
class CompressionStrategy(ABC):
    @abstractmethod
    def compress(self, data):
        pass

class ZipCompression(CompressionStrategy):
    def compress(self, data):
        # ZIP 压缩
        pass

class GzipCompression(CompressionStrategy):
    def compress(self, data):
        # Gzip 压缩
        pass
```

## 与其他模式的区别

### 策略模式 vs 工厂模式

| 特性     | 策略模式         | 工厂模式     |
| -------- | ---------------- | ------------ |
| 目的     | 封装算法，可互换 | 封装对象创建 |
| 关注点   | 算法的行为       | 对象的创建   |
| 切换时机 | 运行时切换       | 创建时决定   |

**可以组合使用**：
```python
# 工厂模式创建策略
class BackendFactory:
    @staticmethod
    def create(backend_type: str) -> AttentionBackend:
        if backend_type == "naive":
            return TorchNaiveBackend()
        elif backend_type == "paged":
            return PagedAttentionBackend()
        elif backend_type == "flash":
            return FlashAttentionBackend()

# 使用
backend = BackendFactory.create("flash")
attention = Attention(config, backend)
```

### 策略模式 vs 模板方法模式

| 特性   | 策略模式             | 模板方法模式         |
| ------ | -------------------- | -------------------- |
| 结构   | 组合（has-a）        | 继承（is-a）         |
| 粒度   | 整个算法可替换       | 算法步骤可重写       |
| 灵活性 | 更灵活（运行时切换） | 较固定（编译时确定） |

## FoloVLLM 中的演进路线

```
M1 (当前):
    AttentionBackend
        ↓
    TorchNaiveBackend
    - 纯 PyTorch 实现
    - 性能基线

M3:
    AttentionBackend
        ↓
    TorchNaiveBackend, PagedAttentionBackend
    - 添加 Paged Attention
    - 高效 KV cache 管理
    - 无需修改 Attention 层

M4:
    AttentionBackend
        ↓
    TorchNaiveBackend, PagedAttentionBackend, FlashAttentionBackend
    - 添加 Flash Attention 2
    - IO 优化，最快性能
    - 仍然无需修改 Attention 层
```

## 总结

**策略模式的核心**：
1. 定义抽象接口
2. 实现多个具体策略
3. 客户端通过接口使用，可随时切换

**在 FoloVLLM 中的价值**：
- 支持多种 Attention 实现（Naive, Paged, Flash）
- 轻松添加新后端（如 xFormers）
- 根据硬件/配置自动选择最优策略
- 便于测试和性能对比

**本质**：将"做什么"（Attention）与"怎么做"（具体实现）分离，提高代码的灵活性和可维护性。

