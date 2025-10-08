# 为什么设置随机种子可以保证结果复现

## 问题
代码中的 `set_random_seed()` 函数设置了多个随机种子，为什么这样可以保证结果复现？

## 回答

### 核心原理
深度学习中的"随机"操作实际上是**伪随机**的。伪随机数生成器（PRNG）从一个初始状态（种子）开始，通过确定性算法生成看似随机的数字序列。相同的种子 → 相同的初始状态 → 相同的随机数序列。

### 代码解析

```python
def set_random_seed(seed: int) -> None:
    random.seed(seed)              # 1. Python标准库
    np.random.seed(seed)           # 2. NumPy
    torch.manual_seed(seed)        # 3. PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 4. PyTorch GPU
```

设置4个不同的随机种子是因为不同库有独立的随机数生成器：

1. **`random.seed()`**: Python标准库的 `random` 模块
2. **`np.random.seed()`**: NumPy的随机操作（数组采样、shuffle等）
3. **`torch.manual_seed()`**: PyTorch在CPU上的随机操作
4. **`torch.cuda.manual_seed_all()`**: PyTorch在所有GPU上的随机操作

### LLM推理中的随机性

在大语言模型推理中，随机性主要来自**采样策略**：

- **temperature采样**: 调整logits分布的锐度
- **top-k采样**: 从概率最高的k个token中随机选择
- **top-p (nucleus)采样**: 从累积概率达到p的token集合中随机选择

**示例**：
```
输入: "今天天气"
logits: [很:0.4, 真:0.3, 不错:0.2, ...]

temperature=0.8, top_k=3
→ 随机从[很, 真, 不错]中采样
→ 设置种子后，每次都会选择相同的token
```

### 复现条件

要保证完全相同的结果，需要满足：

1. ✅ **相同的随机种子**
2. ✅ **相同的输入**（prompt）
3. ✅ **相同的模型权重**
4. ✅ **相同的采样参数**（temperature、top_k、top_p等）
5. ✅ **相同的硬件/软件环境**（某些CUDA操作可能有非确定性）

### 注意事项

- **贪心解码**（temperature=0或始终选概率最高的token）不需要随机种子，本身就是确定性的
- 某些GPU操作（如某些卷积算法）可能有非确定性，即使设置了种子也无法100%复现
- 分布式训练/推理时，需要为每个进程设置不同但确定的种子

## 相关代码
- `folovllm/utils/common.py`: `set_random_seed()` 函数实现

