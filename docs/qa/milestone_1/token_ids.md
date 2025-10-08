# Token IDs - 什么是 IDs？

## IDs 的含义

**ID** = **Identifier**（标识符），是一个**唯一的数字编号**。

**Token ID** = 给每个 token（词或子词）分配的**唯一整数编号**。

## 为什么要用 ID？

神经网络**只能处理数字**，不能直接处理文本。所以需要把文本转换成数字。

```python
# ❌ 神经网络无法直接处理文本
text = "Hello world"
model(text)  # 不行！

# ✅ 需要先转换为数字 ID
token_ids = [15496, 1917]  # Hello=15496, world=1917
model(token_ids)  # 可以！
```

## 完整流程

### 1. 文本 → Token IDs（编码）

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# 输入文本
text = "Hello, how are you?"

# Tokenization（分词 + 转 ID）
token_ids = tokenizer.encode(text)
print(token_ids)
# [15496, 11, 1128, 525, 498, 30]
```

**步骤**：
1. **分词**：`"Hello, how are you?"` → `["Hello", ",", "how", "are", "you", "?"]`
2. **转 ID**：查词汇表，每个 token 对应一个 ID

### 2. Token IDs → 文本（解码）

```python
# Token IDs
token_ids = [15496, 11, 1128, 525, 498, 30]

# 解码回文本
text = tokenizer.decode(token_ids)
print(text)
# "Hello, how are you?"
```

## Token ID 的本质

### 词汇表（Vocabulary）

Token ID 就是词汇表中的**索引**：

```python
# 简化的词汇表示例
vocab = {
    0: "<pad>",      # 特殊 token：填充
    1: "<unk>",      # 特殊 token：未知词
    2: "<bos>",      # 特殊 token：句子开始
    3: "<eos>",      # 特殊 token：句子结束
    4: "the",
    5: "a",
    6: "hello",
    7: "world",
    8: "cat",
    9: "dog",
    # ... 151,936 个 tokens（Qwen3 的词汇表大小）
}

# Token → ID
"hello" → 6
"world" → 7

# ID → Token
6 → "hello"
7 → "world"
```

### Qwen3 的词汇表

```python
# Qwen3-0.6B
vocab_size = 151,936  # 151,936 个不同的 tokens

# ID 范围
min_id = 0
max_id = 151,935

# 每个 ID 对应一个 token（可能是字、词、子词）
```

## 具体例子

### 中文文本

```python
text = "你好，世界！"

# Tokenization
token_ids = tokenizer.encode(text)
print(token_ids)
# [108, 109, 3837, 111, 112]

# 对应关系
# 108 → "你"
# 109 → "好"
# 3837 → "，"
# 111 → "世"
# 112 → "界"
# (实际 ID 可能不同，这只是示例)
```

### 英文文本

```python
text = "The cat is sleeping."

token_ids = tokenizer.encode(text)
# [791, 8758, 374, 21811, 13]

# 对应关系
# 791 → "The"
# 8758 → " cat"  (注意空格)
# 374 → " is"
# 21811 → " sleeping"
# 13 → "."
```

### 子词切分（Subword）

现代 tokenizer 会将不常见的词切分成子词：

```python
text = "unhappiness"

# 可能的切分
tokens = ["un", "happiness"]  # 或 ["un", "happy", "ness"]
token_ids = [1234, 5678]

# 而不是一个完整的词 "unhappiness" → 一个 ID
```

**好处**：
- 处理未见过的词（OOV, Out-of-Vocabulary）
- 词汇表更小
- 更灵活

## 在模型中的使用

### Embedding 层

Token ID 用于查找 embedding：

```python
# Token IDs
input_ids = torch.tensor([15496, 1917])  # [Hello, world]

# Embedding 层
embedding = nn.Embedding(vocab_size, hidden_size)
# vocab_size = 151,936
# hidden_size = 896

# 查表得到向量
vectors = embedding(input_ids)
# vectors: [2, 896]
# 每个 ID 对应一个 896 维的向量
```

**过程**：
```python
# ID 15496 → embedding[15496] → [0.12, -0.34, 0.56, ..., 0.78]  (896维)
# ID 1917  → embedding[1917]  → [0.23, 0.45, -0.12, ..., 0.34]  (896维)
```

### LM Head

模型输出 logits，每个位置对应词汇表中每个 ID 的得分：

```python
# 模型输出
logits = model(input_ids)
# logits: [batch, seq_len, vocab_size]
#       = [1, 2, 151936]

# 对于每个位置，有 151,936 个得分
# 得分最高的 ID 就是预测的下一个 token
next_token_id = torch.argmax(logits[0, -1, :])
# 例如 next_token_id = 13 (对应 "!")

# 预测：Hello world!
```

## 特殊 Token IDs

大部分 tokenizer 有一些特殊的 token：

```python
# Qwen tokenizer 的特殊 tokens
tokenizer.bos_token_id    # Beginning of Sentence
tokenizer.eos_token_id    # End of Sentence  
tokenizer.pad_token_id    # Padding
tokenizer.unk_token_id    # Unknown

# 例如
print(tokenizer.eos_token_id)  # 151643
print(tokenizer.decode([151643]))  # "<|endoftext|>"
```

**用途**：
- **BOS/EOS**：标记句子开始/结束
- **PAD**：填充到固定长度（批处理时）
- **UNK**：表示未知词

## ID 的范围和类型

### 数据类型

```python
# Token IDs 通常是整数
import torch

input_ids = torch.tensor([15496, 1917])
print(input_ids.dtype)  # torch.int64

# 或者 int32（节省内存）
input_ids = torch.tensor([15496, 1917], dtype=torch.int32)
```

### 范围

```python
# 有效范围
0 <= token_id < vocab_size

# Qwen3
0 <= token_id < 151,936

# 如果超出范围
invalid_id = 200000  # 超过 vocab_size
# embedding[invalid_id]  # 会报错！
```

## 可视化示例

### 编码过程

```
文本: "I love AI"
  ↓ Tokenization
Tokens: ["I", " love", " AI"]
  ↓ 查词汇表
Token IDs: [40, 3021, 15592]
  ↓ Embedding
Vectors: [[0.1, 0.2, ...],   # 40 → 896维向量
          [0.3, -0.1, ...],  # 3021 → 896维向量
          [-0.2, 0.5, ...]]  # 15592 → 896维向量
  ↓ 模型处理
...
```

### 解码过程

```
模型输出: Logits [1, 3, 151936]
  ↓ Argmax
Predicted IDs: [40, 3021, 15592]
  ↓ 查词汇表（反向）
Tokens: ["I", " love", " AI"]
  ↓ 拼接
文本: "I love AI"
```

## 常见问题

### Q1: 为什么不直接用文本？

**A**: 神经网络只能处理数字，文本需要转换。

### Q2: 同一个词的 ID 总是相同吗？

**A**: 对于同一个 tokenizer，是的。但不同的 tokenizer 可能分配不同的 ID。

```python
# Qwen tokenizer
qwen_tokenizer.encode("hello")  # [15496]

# GPT-2 tokenizer
gpt2_tokenizer.encode("hello")  # [31373]
# 不同的 ID！
```

### Q3: 中文和英文的 ID 有区别吗？

**A**: 没有本质区别，都是 0 到 vocab_size-1 的整数。只是对应不同的字符/词。

### Q4: 为什么 ID 不是连续使用的？

**A**: ID 的分配由 tokenizer 的训练过程决定，高频词通常有较小的 ID。

```python
# 高频词（小ID）
"the" → 4
"a" → 5
"is" → 8

# 低频词（大ID）
"antidisestablishmentarianism" → 145678
```

## 在 FoloVLLM 中的使用

### 输入处理

```python
# 文件：folovllm/engine/processor.py

class InputProcessor:
    def process_request(self, prompt: str, ...):
        # 1. 文本 → Token IDs
        input_ids = self.tokenizer.encode(prompt)
        # input_ids: [15496, 1917, 13, ...]
        
        # 2. 转为 tensor
        input_ids = torch.tensor(input_ids)
        
        return Request(input_ids=input_ids, ...)
```

### 输出解码

```python
# 文件：folovllm/engine/llm_engine.py

def _build_output(self, request, output_tokens):
    # 1. Token IDs → 文本
    output_text = self.tokenizer.decode(output_tokens)
    
    return RequestOutput(
        request_id=request.request_id,
        prompt=request.prompt,
        output_text=output_text,
        output_tokens=output_tokens,  # 保留 IDs
    )
```

### 生成循环

```python
# 每次生成一个 token ID
while not finished:
    # 模型预测
    logits = model(input_ids)
    
    # 采样得到下一个 token ID
    next_token_id = sample(logits)  # 例如 15496
    
    # 添加到序列
    output_ids.append(next_token_id)
    
    # 用新 token 继续生成
    input_ids = torch.tensor([next_token_id])
```

## 与其他概念的关系

### Token vs Token ID

```python
Token: "hello"       # 文本片段
Token ID: 15496      # 数字标识

# 映射关系
"hello" ←→ 15496
```

### Vocabulary vs Token IDs

```python
Vocabulary: 所有可能的 tokens 的集合
Token IDs: Vocabulary 中每个 token 的索引

# 词汇表大小 = 最大 token ID + 1
vocab_size = 151,936
max_token_id = 151,935
```

### Embedding vs Token ID

```python
Token ID: 整数索引（输入）
Embedding: 稠密向量（Token ID 的表示）

# 关系
embedding_vector = embedding_table[token_id]
```

## 总结

**IDs 的含义**：
- **ID** = Identifier（标识符）
- **Token ID** = 给每个 token 分配的唯一整数编号

**为什么需要**：
- 神经网络只能处理数字
- 需要将文本转换为数字

**本质**：
- Token ID 是词汇表的索引
- 范围：`0` 到 `vocab_size - 1`

**在 Qwen3 中**：
- 词汇表大小：151,936
- ID 范围：0 - 151,935
- 每个 ID 对应一个 token（字、词、子词）

**完整流程**：
```
文本 → Tokenization → Token IDs → Embedding → 模型 → Logits → Token IDs → 文本
```

**关键理解**：
- Token ID 只是一个**数字标签**
- 用于**索引** embedding 表
- 便于计算机处理文本

Token IDs 是 NLP 模型的基础，连接了文本世界和数字世界！
