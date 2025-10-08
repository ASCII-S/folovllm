# HuggingFace PreTrainedModel 详解

## 什么是 PreTrainedModel

`PreTrainedModel` 是 HuggingFace Transformers 库中所有模型的基类。它封装了预训练模型的加载、保存和推理功能。

```python
from transformers import PreTrainedModel, AutoModelForCausalLM

# 加载预训练模型（自动返回 PreTrainedModel 的子类）
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
# model 是 PreTrainedModel 的一个实例
```

## 在 model_runner.py 中的使用

### 代码位置（145-151行）
```python
outputs = self.model(
    input_ids=input_ids,           # 输入的 token IDs
    position_ids=positions,        # 位置编码
    past_key_values=self.past_key_values,  # KV 缓存
    use_cache=True,                # 启用缓存
    return_dict=True,              # 返回字典格式
)
```

## 参数详解

### 1. input_ids (必需)
**类型:** `torch.LongTensor`  
**形状:** `[batch_size, sequence_length]`  
**含义:** 输入文本转换成的 token ID 序列

```python
# 示例
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # shape: [1, 5]
# 表示：batch_size=1, sequence_length=5
```

**在代码中:**
- 来自 `self.prepare_inputs(token_ids, start_pos)` 的返回值
- 表示当前要处理的 token 序列

### 2. position_ids (可选)
**类型:** `torch.LongTensor`  
**形状:** `[batch_size, sequence_length]`  
**含义:** 每个 token 在序列中的位置索引，用于位置编码（RoPE）

```python
# 示例
# 如果是第一次输入（prefill阶段）
position_ids = torch.tensor([[0, 1, 2, 3, 4]])  # shape: [1, 5]

# 如果是生成阶段（decode阶段），假设已经生成了10个token，现在生成第11个
position_ids = torch.tensor([[10]])  # shape: [1, 1]
```

**为什么需要 position_ids？**
- RoPE（Rotary Position Embedding）需要知道每个token的绝对位置
- 在增量生成时，需要正确的位置信息来计算注意力

### 3. past_key_values (可选，但对推理很重要)
**类型:** `Tuple[Tuple[torch.FloatTensor]]`  
**含义:** KV Cache（键值缓存），存储之前计算过的 key 和 value

**结构:**
```python
past_key_values = (
    (layer_0_key, layer_0_value),  # 第0层的缓存
    (layer_1_key, layer_1_value),  # 第1层的缓存
    ...
    (layer_n_key, layer_n_value),  # 第n层的缓存
)

# 每个 tensor 的形状: [batch_size, num_heads, past_seq_len, head_dim]
```

**作用:**
- **避免重复计算**: 在自回归生成时，之前的token的K和V不需要重新计算
- **加速推理**: 只需计算新token的K和V，然后与缓存拼接

**工作流程:**
```python
# 第一次调用（prefill）: 输入 "你好"
outputs = model(
    input_ids=tensor([[1, 2]]),      # "你" "好"
    past_key_values=None,            # 没有缓存
)
# 输出包含 past_key_values，保存了 "你好" 的K和V

# 第二次调用（decode）: 生成 "世"
outputs = model(
    input_ids=tensor([[3]]),         # "世"
    past_key_values=outputs.past_key_values,  # 传入之前的缓存
)
# 只计算 "世" 的K和V，然后与缓存中的 "你好" 拼接

# 第三次调用: 生成 "界"
outputs = model(
    input_ids=tensor([[4]]),         # "界"
    past_key_values=outputs.past_key_values,  # 包含 "你好世" 的缓存
)
# 以此类推...
```

### 4. use_cache (可选)
**类型:** `bool`  
**默认:** `True`  
**含义:** 是否返回并使用 KV cache

```python
use_cache=True   # 返回 past_key_values，用于下一次调用
use_cache=False  # 不返回 past_key_values，训练时使用
```

**何时使用:**
- **推理/生成**: `True` - 加速生成
- **训练**: `False` - 训练不需要缓存

### 5. return_dict (可选)
**类型:** `bool`  
**默认:** `True`  
**含义:** 返回格式

```python
# return_dict=True（推荐）
outputs = model(..., return_dict=True)
logits = outputs.logits              # 通过属性访问
past_kv = outputs.past_key_values    # 清晰明了

# return_dict=False
outputs = model(..., return_dict=False)
logits = outputs[0]                  # 通过索引访问
past_kv = outputs[1]                 # 容易出错
```

## 返回值详解

当 `return_dict=True` 时，返回 `CausalLMOutputWithPast` 对象：

```python
class CausalLMOutputWithPast:
    logits: torch.FloatTensor          # 模型输出的 logits
    past_key_values: Tuple[...]        # 更新后的 KV cache
    hidden_states: Optional[...]       # 隐藏状态（如果需要）
    attentions: Optional[...]          # 注意力权重（如果需要）
```

**在代码中的使用（152-154行）:**
```python
logits = outputs.logits                    # 获取 logits
self.past_key_values = outputs.past_key_values  # 保存缓存供下次使用
```

### logits 的含义
**形状:** `[batch_size, sequence_length, vocab_size]`

```python
# 示例
logits.shape = [1, 1, 151936]
# batch_size=1: 一个请求
# sequence_length=1: 当前生成1个token（decode阶段）
# vocab_size=151936: Qwen模型的词表大小

# logits[0, 0, :] 是一个长度为 151936 的向量
# 每个值表示对应 token 的 "分数"
# 通过 softmax 转换为概率，然后采样得到下一个 token
```

## 完整的推理流程示例

```python
# 初始化
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
past_key_values = None

# Prefill 阶段：处理输入 prompt "你好"
input_ids = torch.tensor([[1, 2]])           # "你" "好"
position_ids = torch.tensor([[0, 1]])        # 位置 0, 1

outputs = model(
    input_ids=input_ids,
    position_ids=position_ids,
    past_key_values=None,          # 首次调用，无缓存
    use_cache=True,
    return_dict=True,
)

logits = outputs.logits              # [1, 2, 151936]
past_key_values = outputs.past_key_values  # 保存缓存

# 取最后一个位置的 logits 进行采样
next_token = torch.argmax(logits[0, -1, :])  # 假设得到 token_id=3 "世"

# Decode 阶段：生成第一个新 token "世"
input_ids = torch.tensor([[3]])              # 只输入新 token
position_ids = torch.tensor([[2]])           # 位置=2（接在"你好"后面）

outputs = model(
    input_ids=input_ids,
    position_ids=position_ids,
    past_key_values=past_key_values,  # 传入缓存（包含"你好"）
    use_cache=True,
    return_dict=True,
)

logits = outputs.logits              # [1, 1, 151936] 只有新token的logits
past_key_values = outputs.past_key_values  # 更新缓存（包含"你好世"）

# 继续生成...
next_token = torch.argmax(logits[0, -1, :])  # 假设得到 token_id=4 "界"
# ... 重复上述过程
```

## KV Cache 的性能优势

### 不使用 KV Cache
```python
# 生成每个新 token 都要重新计算所有历史 token
# 时间复杂度: O(n²)，n 是生成的 token 数

输入: "你"      -> 计算 1 个 token 的 K,V
输入: "你好"    -> 计算 2 个 token 的 K,V（重复计算"你"）
输入: "你好世"  -> 计算 3 个 token 的 K,V（重复计算"你好"）
输入: "你好世界" -> 计算 4 个 token 的 K,V（重复计算"你好世"）
```

### 使用 KV Cache
```python
# 每次只计算新 token 的 K,V
# 时间复杂度: O(n)

输入: "你"      -> 计算 1 个 token，缓存
输入: "好"      -> 计算 1 个 token，与缓存拼接
输入: "世"      -> 计算 1 个 token，与缓存拼接
输入: "界"      -> 计算 1 个 token，与缓存拼接
```

**加速比例:** 生成100个token时，使用KV Cache可以加速约 50倍！

## 总结

1. **PreTrainedModel** 是 HuggingFace 模型的基类，通过 `__call__` 方法执行推理
2. **input_ids**: 当前要处理的 token 序列
3. **position_ids**: token 的位置索引，用于 RoPE
4. **past_key_values**: KV 缓存，避免重复计算，大幅加速推理
5. **use_cache**: 控制是否返回/使用缓存
6. **return_dict**: 控制返回格式，建议使用 `True`
7. **推理流程**: Prefill（处理prompt）→ Decode（逐个生成token，使用KV cache）

## 在 folovllm 中的位置

- **model_runner.py L145-151**: HuggingFace 模型的调用
- **model_runner.py L159-163**: 自定义模型的调用（参数略有不同）
- **KV cache 管理**: `self.past_key_values` 保存并传递给下一次调用

