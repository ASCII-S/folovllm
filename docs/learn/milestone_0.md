# Milestone 0 学习笔记：项目基础架构

> 本文档介绍 M0 阶段涉及的核心技术原理和设计思想

---

## 📚 目录

1. [配置系统设计](#1-配置系统设计)
2. [采样参数原理](#2-采样参数原理)
3. [请求和序列抽象](#3-请求和序列抽象)
4. [模型加载机制](#4-模型加载机制)
5. [数据流设计](#5-数据流设计)
6. [关键设计模式](#6-关键设计模式)

---

## 1. 配置系统设计

### 1.1 为什么需要配置系统？

在大语言模型推理系统中，配置系统是核心基础设施：

1. **模型配置**：不同模型有不同的参数（dtype、max_length 等）
2. **资源管理**：需要控制 GPU 显存使用、CPU swap 空间等
3. **调度策略**：批处理大小、序列数量等影响性能
4. **可扩展性**：为未来功能预留配置接口

### 1.2 配置系统层次结构

```
EngineConfig (引擎配置)
    ├── ModelConfig (模型配置)
    │   ├── model: 模型路径
    │   ├── dtype: 数据类型
    │   ├── tokenizer: 分词器路径
    │   └── max_model_len: 最大序列长度
    │
    ├── CacheConfig (缓存配置)
    │   ├── block_size: KV Cache 块大小
    │   ├── gpu_memory_utilization: GPU 显存利用率
    │   └── enable_prefix_caching: 前缀缓存开关
    │
    └── SchedulerConfig (调度配置)
        ├── max_num_batched_tokens: 最大批处理 token 数
        ├── max_num_seqs: 最大序列数
        └── enable_chunked_prefill: 分块预填充开关
```

### 1.3 关键设计决策

#### a) 使用 dataclass

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model: str
    dtype: str = "auto"
    # ...
```

**优点**：
- 自动生成 `__init__`、`__repr__` 等方法
- 类型提示清晰
- 支持默认值
- 可以使用 `__post_init__` 进行验证

#### b) 类型约束

```python
from typing import Literal

ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
```

**优点**：
- 编译时类型检查
- IDE 自动补全
- 避免无效值

#### c) 参数验证

```python
def __post_init__(self):
    if self.block_size <= 0:
        raise ValueError(f"block_size must be positive")
```

**优点**：
- 提前发现配置错误
- 清晰的错误信息
- 避免运行时错误

### 1.4 与 vLLM 的对齐

FoloVLLM 的配置系统完全参考 vLLM v1 设计：

| vLLM              | FoloVLLM          | 说明                     |
| ----------------- | ----------------- | ------------------------ |
| `ModelConfig`     | `ModelConfig`     | 模型配置，参数基本一致   |
| `CacheConfig`     | `CacheConfig`     | 缓存配置，简化了部分参数 |
| `SchedulerConfig` | `SchedulerConfig` | 调度配置，预留了扩展接口 |

---

## 2. 采样参数原理

### 2.1 什么是采样？

采样是从模型输出的概率分布中选择下一个 token 的过程：

```
模型输出 logits: [vocab_size]
    ↓ softmax
概率分布 probs: [vocab_size]
    ↓ 采样策略
下一个 token: int
```

### 2.2 采样策略详解

#### a) Greedy Sampling (贪心采样)

```python
next_token = argmax(probs)
```

**原理**：总是选择概率最高的 token

**特点**：
- 确定性输出（相同输入总是得到相同输出）
- 可能导致重复和单调的文本
- 适合需要确定性的任务（如翻译）

**实现**：`temperature = 0.0`

#### b) Temperature Scaling (温度缩放)

```python
logits_scaled = logits / temperature
probs = softmax(logits_scaled)
```

**原理**：调整概率分布的"陡峭"程度

**效果**：
- `temperature < 1.0`: 分布更陡峭，高概率 token 更容易被选中（更确定）
- `temperature = 1.0`: 不改变分布（原始概率）
- `temperature > 1.0`: 分布更平缓，低概率 token 也有机会（更随机）

**可视化**：
```
temperature = 0.5      temperature = 1.0      temperature = 2.0
    ▁▁█▁▁                  ▂▅█▅▂                  ▄▆█▆▄
   更确定                   平衡                   更随机
```

#### c) Top-k Sampling

```python
top_k_probs, top_k_indices = torch.topk(probs, k)
top_k_probs = top_k_probs / top_k_probs.sum()
next_token = sample(top_k_indices, top_k_probs)
```

**原理**：只从概率最高的 k 个 token 中采样

**优点**：
- 过滤掉低概率的"噪音" token
- 保持输出质量
- 增加多样性

**典型值**：`k = 50`

#### d) Top-p (Nucleus) Sampling

```python
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
mask = cumsum_probs <= p
nucleus_probs = sorted_probs[mask]
```

**原理**：选择累积概率达到 p 的最小 token 集合

**优点**：
- 动态调整候选集大小
- 在概率分布平缓时包含更多候选
- 在概率分布陡峭时减少候选

**典型值**：`p = 0.9` 或 `p = 0.95`

**示例**：
```
Token概率: [0.4, 0.3, 0.15, 0.1, 0.05]
p = 0.9:   [✓   ✓   ✓    ✗   ✗  ]  累积到 0.85 < 0.9
```

#### e) Min-p Sampling

```python
threshold = p * max(probs)
mask = probs >= threshold
```

**原理**：过滤掉概率低于 `p * max_prob` 的 token

**优点**：
- 相对阈值，适应不同的概率分布
- 避免选择"不太可能"的 token

### 2.3 采样策略组合

实际使用中，通常组合多种策略：

```python
SamplingParams(
    temperature=0.8,  # 增加随机性
    top_p=0.9,        # Nucleus sampling
    top_k=50,         # 过滤低概率 token
)
```

**执行顺序**：
1. Temperature scaling
2. Top-k filtering
3. Top-p filtering
4. Min-p filtering
5. Random sampling

### 2.4 停止条件

#### a) Stop Strings

```python
stop = ["</s>", "\n\n", "Human:"]
```

生成的文本包含任一停止字符串时停止。

#### b) Stop Token IDs

```python
stop_token_ids = [2, 50256]  # EOS tokens
```

生成的 token ID 在停止列表中时停止。

#### c) Max Tokens

```python
max_tokens = 100
```

生成指定数量的 token 后停止。

---

## 3. 请求和序列抽象

### 3.1 为什么需要序列抽象？

在 LLM 推理中，需要管理复杂的状态：

1. **多序列生成**：一个请求可能生成多个候选序列（n > 1）
2. **状态追踪**：每个序列有自己的生成状态
3. **资源管理**：KV Cache 需要按序列分配
4. **调度决策**：调度器需要知道哪些序列在运行

### 3.2 三层抽象结构

```
Request (请求)
    ├── 包含多个 Sequence
    ├── 共享 prompt 和 sampling_params
    └── 管理请求级别的状态

Sequence (序列)
    ├── 有独立的 seq_id
    ├── 包含一个 SequenceData
    ├── 有自己的状态 (WAITING/RUNNING/FINISHED)
    └── 管理序列级别的资源 (KV Cache blocks)

SequenceData (序列数据)
    ├── prompt_token_ids (输入)
    ├── output_token_ids (输出)
    └── 提供 token 操作接口
```

### 3.3 状态机设计

#### 请求状态机

```
WAITING → RUNNING → FINISHED_*
   ↓         ↓
   └─ SWAPPED ─┘
```

**状态说明**：
- `WAITING`: 在等待队列中
- `RUNNING`: 正在处理
- `SWAPPED`: 被换出到 CPU（内存不足时）
- `FINISHED_STOPPED`: 遇到停止条件
- `FINISHED_LENGTH_CAPPED`: 达到最大长度
- `FINISHED_ABORTED`: 被用户中止

#### 序列状态机

与请求状态机相同，但增加了：
- `FINISHED_IGNORED`: 在 best_of > n 时被忽略的序列

### 3.4 Sequence Fork 机制

```python
def fork(self, new_seq_id: str) -> "Sequence":
    """Fork 一个新序列（用于 beam search 或 parallel sampling）"""
    new_data = SequenceData(
        prompt_token_ids=self.data.prompt_token_ids.copy(),
        output_token_ids=self.data.output_token_ids.copy(),
    )
    return Sequence(
        seq_id=new_seq_id,
        request_id=self.request_id,
        data=new_data,
        sampling_params=self.sampling_params,
    )
```

**用途**：
- **Beam Search**：每次扩展时 fork 多个候选
- **Parallel Sampling**：生成 n > 1 个独立序列
- **Speculative Decoding**：验证推测的 token

**关键**：深拷贝数据，避免共享状态

### 3.5 n vs best_of

```python
SamplingParams(n=3, best_of=5)
```

- `best_of=5`: 生成 5 个候选序列
- `n=3`: 最终返回 3 个最好的序列

**流程**：
1. 创建 5 个 Sequence
2. 并行生成（共享 prompt KV Cache）
3. 根据累积 log 概率排序
4. 返回前 3 个

---

## 4. 模型加载机制

### 4.1 HuggingFace 模型加载

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
```

#### 关键参数

**torch_dtype**：
- `torch.float16`: 半精度，显存占用减半
- `torch.bfloat16`: Brain Float16，数值范围更大
- `torch.float32`: 全精度，最准确但最占显存

**trust_remote_code**：
- 允许执行模型仓库中的自定义代码
- Qwen 等模型需要此选项

**low_cpu_mem_usage**：
- 使用 accelerate 库的优化加载
- 减少 CPU 内存峰值

### 4.2 Tokenizer 配置

```python
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    use_fast=True,
    padding_side="left",
)
```

#### Padding Side

**Left Padding (推荐用于生成)**：
```
Sequence 1: [PAD][PAD]token1 token2 token3
Sequence 2: [PAD]token1 token2 token3 token4
                              ↑
                      生成从这里开始
```

**Right Padding (用于分类)**：
```
Sequence 1: token1 token2 token3[PAD][PAD]
Sequence 2: token1 token2 token3 token4[PAD]
                  ↑
         分类器看最后一个真实 token
```

### 4.3 Dtype 选择策略

```python
def _get_dtype(self) -> torch.dtype:
    if self.model_config.torch_dtype is not None:
        return self.model_config.torch_dtype
    
    # 默认：GPU 用 FP16，CPU 用 FP32
    if torch.cuda.is_available():
        return torch.float16
    else:
        return torch.float32
```

**决策因素**：
1. **精度需求**：科学计算用 FP32，推理用 FP16
2. **显存限制**：FP16 减半显存
3. **硬件支持**：A100/H100 支持 BF16
4. **模型训练 dtype**：最好与训练时一致

### 4.4 Max Model Length 推断

```python
if self.model_config.max_model_len is None:
    if hasattr(hf_config, "max_position_embeddings"):
        self.model_config.max_model_len = hf_config.max_position_embeddings
    else:
        self.model_config.max_model_len = 2048  # 默认值
```

**来源**：
1. 用户显式指定（最高优先级）
2. 模型配置文件中的 `max_position_embeddings`
3. 默认值（2048）

---

## 5. 数据流设计

### 5.1 端到端数据流

```
用户输入文本
    ↓ Tokenizer.encode()
prompt_token_ids: List[int]
    ↓ 创建 Request
Request + Sequence
    ↓ 调度器调度 (M2)
Batch of Sequences
    ↓ 模型前向传播 (M1)
Logits: [batch_size, vocab_size]
    ↓ 采样 (M1)
next_tokens: [batch_size]
    ↓ 添加到 Sequence
output_token_ids: List[int]
    ↓ Tokenizer.decode()
输出文本
```

### 5.2 配置传递

```
EngineConfig
    ↓ 分发
ModelConfig → ModelLoader → Model
CacheConfig → KVCacheManager (M3)
SchedulerConfig → Scheduler (M2)
```

### 5.3 状态更新流

```
Sequence.status = WAITING
    ↓ Scheduler.schedule()
Sequence.status = RUNNING
    ↓ Worker.execute()
Sequence.add_token_id(new_token)
    ↓ 检查停止条件
Sequence.status = FINISHED_*
```

---

## 6. 关键设计模式

### 6.1 Builder Pattern (配置构建)

```python
# 分步构建配置
model_config = ModelConfig(model="Qwen/Qwen3-0.6B")
cache_config = CacheConfig(block_size=16)
scheduler_config = SchedulerConfig(max_num_seqs=256)

# 组装成引擎配置
engine_config = EngineConfig(
    model_config=model_config,
    cache_config=cache_config,
    scheduler_config=scheduler_config,
)
```

### 6.2 Strategy Pattern (采样策略)

```python
class SamplingParams:
    @property
    def sampling_type(self) -> SamplingType:
        if self.temperature == 0.0:
            return SamplingType.GREEDY
        else:
            return SamplingType.RANDOM
```

不同的参数组合对应不同的采样策略，运行时动态选择。

### 6.3 Factory Pattern (模型加载)

```python
def get_model_and_tokenizer(config, device):
    loader = ModelLoader(config)
    return loader.load_model_and_tokenizer(device)
```

封装复杂的模型加载逻辑。

### 6.4 State Pattern (序列状态)

```python
class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "finished_stopped"
    
    def is_finished(self) -> bool:
        return self in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            # ...
        ]
```

封装状态转换逻辑。

---

## 7. 性能考虑

### 7.1 内存优化

1. **Dataclass vs Dict**：
   - Dataclass 有类型检查和更好的性能
   - Dict 更灵活但容易出错

2. **Deep Copy vs Shallow Copy**：
   - Sequence fork 使用深拷贝避免共享状态
   - 但会增加内存开销

3. **List vs Numpy Array**：
   - token_ids 使用 List 方便动态增长
   - 批处理时转换为 Tensor

### 7.2 类型提示的价值

```python
def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
    """类型提示帮助 IDE 和 mypy 检查"""
```

**优点**：
- 编译时发现类型错误
- IDE 自动补全
- 代码更易读

---

## 8. 与后续 Milestone 的连接

### M0 为后续阶段预留的接口

1. **Sequence.block_ids**: 用于 M3 PagedAttention
2. **CacheConfig.enable_prefix_caching**: 用于 M6 前缀缓存
3. **SchedulerConfig.enable_chunked_prefill**: 用于 M5 分块预填充
4. **SamplingParams.logprobs**: 用于 M1+ 日志概率返回

### 扩展点

- 新的配置类可以轻松添加
- 新的采样策略只需扩展 SamplingParams
- 新的状态可以加入枚举类
- 新的模型只需实现加载逻辑

---

## 9. 总结

### 核心技术点

1. **配置系统**：分层设计，类型安全，参数验证
2. **采样策略**：Temperature、Top-k、Top-p、停止条件
3. **序列抽象**：三层结构，状态机，fork 机制
4. **模型加载**：HuggingFace 集成，dtype 选择，自动配置

### 设计原则

1. **对齐 vLLM**：学习成熟框架的设计
2. **渐进式**：预留扩展接口
3. **类型安全**：使用类型提示和验证
4. **模块化**：清晰的职责分离

### 学习建议

1. 理解每个类的职责和关系
2. 对比 vLLM 源码理解设计思路
3. 动手实验不同的配置组合
4. 阅读后续 milestone 了解演进

---

## 参考资料

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原理
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Top-p Sampling

### 代码
- [vLLM Official Repo](https://github.com/vllm-project/vllm) - 参考实现
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 模型库

### 博客
- [How to generate text: using different decoding methods](https://huggingface.co/blog/how-to-generate)
- [Nucleus Sampling explained](https://towardsdatascience.com/the-curious-case-of-neural-text-degeneration-374f79c5c9a4)

