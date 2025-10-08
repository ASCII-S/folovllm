# 为什么分类任务关注最后一个真实token和使用右padding

## 问题
1. 为什么分类任务要关注最后一个真实token？
2. 为什么分类任务中要使用右padding（Right Padding）？

## 回答

### 核心原因

分类任务和生成任务的**目标不同**，导致它们需要不同的padding策略：

- **生成任务**：需要知道在**哪里生成下一个token** → 使用左padding（Left Padding）
- **分类任务**：需要提取**整个句子的语义** → 使用右padding（Right Padding）

## 一、为什么分类任务关注最后一个真实token？

### 1. 自回归模型的特点

像 GPT、Qwen 这样的**自回归（Causal）模型**使用**单向注意力**：

```
Token 1 → 只能看到 Token 1
Token 2 → 可以看到 Token 1, Token 2  
Token 3 → 可以看到 Token 1, Token 2, Token 3
Token N → 可以看到 Token 1, ..., Token N  ← 看到全部信息
```

**结论**：最后一个token的表示包含了**前面所有token的信息**。

### 2. 分类任务的需求

分类任务需要理解**整个句子**的语义：

```python
# 情感分类示例
输入: "这部电影真的很棒，我非常喜欢！"
任务: 判断情感（正面/负面）

需要：
- 理解 "很棒"（正面词）
- 理解 "非常喜欢"（正面词）
- 综合整个句子的语义
- 输出：正面情感
```

**为什么用最后一个token？**

```
Token 1: 这     → 只看到 "这"
Token 2: 部     → 看到 "这部"
Token 3: 电     → 看到 "这部电"
...
Token N: ！    → 看到完整句子 "这部电影真的很棒，我非常喜欢！"
           ↑
        这里包含了完整的句子信息
```

### 3. 代码实现

```python
# 分类模型结构
class ClassificationModel(nn.Module):
    def forward(self, input_ids, attention_mask):
        # 1. 获取模型输出
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # 2. 找到每个序列的最后一个真实token
        sequence_lengths = attention_mask.sum(dim=1) - 1  # 最后一个真实token的位置
        
        # 3. 提取最后一个token的表示
        batch_size = input_ids.shape[0]
        last_hidden = hidden_states[
            torch.arange(batch_size), 
            sequence_lengths
        ]  # [batch, hidden_dim]
        
        # 4. 分类
        logits = self.classifier(last_hidden)  # [batch, num_classes]
        return logits
```

## 二、为什么分类任务使用右padding？

### 1. 右padding的优势

**右padding让最后一个真实token的位置更加一致**：

```python
# 使用右padding（Right Padding）
Seq 1: [CLS] Hello world [SEP] [PAD] [PAD]
                           ↑ 位置4
Seq 2: [CLS] Hello world ! [SEP] [PAD]
                            ↑ 位置5
Seq 3: [CLS] Hello [SEP] [PAD] [PAD] [PAD]
                      ↑ 位置3

# 最后一个真实token位置：[4, 5, 3]
# 虽然不完全对齐，但都在"靠前"的位置
```

### 2. 与左padding对比

**左padding会让最后一个真实token的位置差异更大**：

```python
# 使用左padding（Left Padding）
Seq 1: [PAD] [PAD] [CLS] Hello world [SEP]
                                       ↑ 位置6
Seq 2: [PAD] [CLS] Hello world ! [SEP]
                                  ↑ 位置6
Seq 3: [PAD] [PAD] [PAD] [CLS] Hello [SEP]
                                       ↑ 位置6

# 咦？位置都是6？
```

**注意**：上面的对齐是**误导性的**！实际问题在于：

```python
# 真正的问题：左padding在批处理时的索引
# 批处理需要固定长度，假设 max_len=6

# 原始序列长度不同：
Seq 1: [CLS] Hello world [SEP]        # 长度4
Seq 2: [CLS] Hello world ! [SEP]      # 长度5
Seq 3: [CLS] Hello [SEP]              # 长度3

# 左padding到长度6：
Seq 1: [PAD] [PAD] [CLS] Hello world [SEP]  # 最后真实token在位置5
Seq 2: [PAD] [CLS] Hello world ! [SEP]      # 最后真实token在位置5
Seq 3: [PAD] [PAD] [PAD] [CLS] Hello [SEP]  # 最后真实token在位置5
         ↑
     padding数量不同，导致"最后真实token"在tensor中的绝对位置相同，
     但相对位置（从序列开始计算）不同
```

### 3. 为什么右padding更自然？

**人类阅读习惯**：从左到右，句子结尾自然在右边

```
自然语言：Hello world [结束]
         ↓
右padding：Hello world [SEP] [PAD] [PAD]
                       ↑ 结尾在右边
```

**序列长度统计**：右padding让真实序列长度的计算更直观

```python
# 右padding
tokens = [101, 202, 303, 0, 0]  # 0 是 PAD
real_length = (tokens != 0).sum()  # = 3 ✅

# 左padding
tokens = [0, 0, 101, 202, 303]
real_length = len(tokens) - (tokens == 0).sum()  # 需要减法 ⚠️
```

### 4. 实际代码示例

```python
# BERT式分类（使用 [CLS] token）
class BERTClassification:
    def forward(self, input_ids):
        # BERT是双向的，直接用[CLS]（第一个token）做分类
        outputs = self.bert(input_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        logits = self.classifier(cls_embedding)
        return logits

# GPT式分类（使用最后一个真实token）
class GPTClassification:
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids, attention_mask=attention_mask)
        
        # 找到最后一个真实token
        # 右padding: [1,1,1,1,0,0] → 最后真实token在位置3
        seq_lengths = attention_mask.sum(dim=1) - 1
        
        # 提取最后token的表示
        last_token_embed = outputs.last_hidden_state[
            torch.arange(batch_size),
            seq_lengths
        ]
        
        logits = self.classifier(last_token_embed)
        return logits
```

## 三、生成任务为什么用左padding？

对比理解：

### 生成任务的需求

```python
# 生成下一个token
Prompt: "今天天气"
任务: 生成 "很好"

需要：在prompt**末尾**生成新token
```

### 批处理中的对齐问题

**左padding（生成任务推荐）**：

```python
Seq 1: [PAD] [PAD] 今天 天气
Seq 2: [PAD] 今天 天气 很好
                      ↑ 生成位置对齐

# 生成时：
Seq 1: [PAD] [PAD] 今天 天气 → 很好  # 在位置4生成
Seq 2: [PAD] 今天 天气 很好 → ！    # 在位置4生成
                            ↑ 位置对齐！
```

**右padding（生成任务不推荐）**：

```python
Seq 1: 今天 天气 [PAD] [PAD]
Seq 2: 今天 天气 很好 [PAD]
              ↑    ↑ 生成位置不对齐

# 生成时：
Seq 1: 今天 天气 [PAD] [PAD] → 很好？ # 应该在位置2生成
Seq 2: 今天 天气 很好 [PAD] → ！？   # 应该在位置3生成
# 每个序列的生成位置不同，批处理困难！
```

## 四、总结对比表

| 维度            | 分类任务（Right Padding） | 生成任务（Left Padding）   |
| --------------- | ------------------------- | -------------------------- |
| **关注点**      | 整个句子的语义表示        | 下一个token的生成位置      |
| **关键位置**    | 最后一个真实token         | 序列末尾（生成新token）    |
| **Padding方向** | 右padding（句子→PAD）     | 左padding（PAD→句子）      |
| **位置对齐**    | 真实token在左边，自然阅读 | 生成位置在右边，批处理对齐 |
| **典型模型**    | BERT（[CLS]）、GPT分类    | GPT、Qwen等生成模型        |
| **代码实现**    | `hidden[:, seq_len-1, :]` | 直接在末尾append新token    |

## 五、实际示例对比

### 分类任务（情感分析）

```python
# 输入
texts = [
    "这部电影很棒！",
    "非常好看",
]

# 右padding（推荐）
# Tokenizer设置
tokenizer.padding_side = "right"

# 结果：
# [CLS] 这 部 电 影 很 棒 ！ [SEP] [PAD] [PAD]  ← 最后真实token: [SEP]
# [CLS] 非 常 好 看 [SEP] [PAD] [PAD] [PAD] [PAD]  ← 最后真实token: [SEP]

# 提取特征：找到[SEP]的位置 → 提取embedding → 分类
```

### 生成任务（文本续写）

```python
# 输入
prompts = [
    "今天天气",
    "今天天气很好",
]

# 左padding（推荐）
# Tokenizer设置
tokenizer.padding_side = "left"

# 结果：
# [PAD] [PAD] 今天 天气                ← 在位置3生成新token
# [PAD] 今天 天气 很好                ← 在位置3生成新token

# 生成：在对齐的位置3批量生成新token
```

## 六、特殊情况

### BERT等双向模型

```python
# BERT使用特殊的[CLS] token做分类
# [CLS]在句首，通过双向注意力看到整个句子

Input: [CLS] 这 部 电 影 很 棒 [SEP] [PAD] [PAD]
        ↑
    直接用这个token的embedding做分类

# 因为BERT是双向的，[CLS]可以attend到所有token
# 所以不需要"最后一个token"
```

### GPT等单向模型做分类

```python
# GPT只能用最后一个真实token
Input: [BOS] 这 部 电 影 很 棒 [EOS] [PAD] [PAD]
                               ↑
                        用这个token的embedding

# 因为GPT是单向的，只有最后一个token看到了全部信息
```

## 相关代码

- `folovllm/model_loader.py`: tokenizer加载时设置 `padding_side="left"` 用于生成
- `docs/interview/milestone_0.md`: padding策略面试题
- `docs/learn/milestone_0.md`: padding详细说明

## 参考资料

- [Hugging Face Padding & Truncation](https://huggingface.co/docs/transformers/pad_truncation)
- [GPT Text Classification](https://github.com/openai/gpt-3/blob/master/classification.md)
