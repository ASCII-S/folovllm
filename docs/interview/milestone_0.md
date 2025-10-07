# Milestone 0 面试指南

> 本文档整理 M0 阶段可能遇到的面试问题及回答要点

---

## 📋 目录

1. [配置系统相关](#1-配置系统相关)
2. [采样策略相关](#2-采样策略相关)
3. [数据结构设计](#3-数据结构设计)
4. [模型加载相关](#4-模型加载相关)
5. [系统设计相关](#5-系统设计相关)
6. [性能优化相关](#6-性能优化相关)

---

## 1. 配置系统相关

### Q1.1: 为什么要设计分层的配置系统？

**回答要点**：

1. **职责分离**：
   - ModelConfig 管理模型相关配置
   - CacheConfig 管理缓存相关配置
   - SchedulerConfig 管理调度相关配置
   - 每个配置类职责单一，易于维护

2. **灵活组合**：
   - 可以独立修改某一层配置而不影响其他层
   - 便于测试（可以 mock 特定配置）

3. **扩展性**：
   - 新增配置类不影响现有代码
   - 符合开闭原则（对扩展开放，对修改封闭）

4. **可读性**：
   - 配置层次清晰，容易理解
   - 与 vLLM 等成熟框架对齐

**追问：如何保证配置之间的一致性？**

**回答**：
```python
class EngineConfig:
    def __post_init__(self):
        # 在顶层配置中同步子配置
        if self.scheduler_config.max_model_len is None:
            self.scheduler_config.max_model_len = self.model_config.max_model_len
```

---

### Q1.2: 为什么使用 dataclass 而不是普通类或字典？

**回答要点**：

**vs 普通类**：
```python
# dataclass 自动生成
@dataclass
class ModelConfig:
    model: str
    dtype: str = "auto"

# 等价于普通类的大量代码
class ModelConfig:
    def __init__(self, model: str, dtype: str = "auto"):
        self.model = model
        self.dtype = dtype
    
    def __repr__(self): ...
    def __eq__(self): ...
```

**vs 字典**：
- ✅ **类型安全**：IDE 可以检查类型
- ✅ **自动补全**：IDE 知道有哪些字段
- ✅ **验证**：`__post_init__` 可以验证参数
- ✅ **性能**：比字典访问更快（属性访问）

**示例**：
```python
# dataclass - IDE 会报错
config = ModelConfig(model=123)  # ❌ 类型错误

# dict - 运行时才发现错误
config = {"model": 123}  # ✅ 没问题，但后续会出错
```

---

### Q1.3: 配置验证为什么放在 `__post_init__` 而不是 `__init__`？

**回答要点**：

dataclass 的生成顺序：
1. 自动生成的 `__init__` 设置所有字段
2. 调用用户定义的 `__post_init__`

**好处**：
- 所有字段都已初始化，可以访问任何字段
- 可以进行跨字段验证
- 不需要手写 `__init__`

**示例**：
```python
@dataclass
class SamplingParams:
    n: int = 1
    best_of: Optional[int] = None
    
    def __post_init__(self):
        # 此时 self.n 和 self.best_of 都已设置
        if self.best_of is None:
            self.best_of = self.n
        
        # 跨字段验证
        if self.best_of < self.n:
            raise ValueError(f"best_of ({self.best_of}) must be >= n ({self.n})")
```

---

## 2. 采样策略相关

### Q2.1: 解释 Temperature、Top-k、Top-p 的区别和使用场景

**回答要点**：

**Temperature**：
- **原理**：调整概率分布的陡峭程度
- **公式**：`logits_scaled = logits / temperature`
- **效果**：
  - `< 1.0`: 更确定（分布更陡）
  - `= 1.0`: 原始分布
  - `> 1.0`: 更随机（分布更平）
- **使用**：控制输出的创造性

**Top-k**：
- **原理**：只从概率最高的 k 个 token 中采样
- **优点**：过滤低概率噪音
- **缺点**：固定 k 值，不适应不同的分布
- **使用**：一般设置 k=50

**Top-p (Nucleus)**：
- **原理**：选择累积概率达到 p 的最小 token 集
- **优点**：动态调整候选集大小
- **效果**：
  - 分布平缓时：包含更多候选
  - 分布陡峭时：只包含高概率 token
- **使用**：p=0.9 或 0.95

**组合使用**：
```python
SamplingParams(
    temperature=0.8,  # 增加随机性
    top_k=50,         # 过滤噪音
    top_p=0.95        # 动态候选集
)
# 执行顺序：temperature → top_k → top_p → sample
```

**追问：为什么要组合使用？**

**回答**：
- Temperature 调整整体随机性
- Top-k 去除明显的坏候选
- Top-p 在剩余候选中动态选择
- 三者互补，效果最佳

---

### Q2.2: Greedy Sampling 和 Beam Search 的区别？

**回答要点**：

**Greedy Sampling**：
```python
# 每步选择最优
step 1: token_a (prob=0.6) ✓
step 2: token_b (prob=0.5) ✓
step 3: token_c (prob=0.4) ✓
总分数: 0.6 × 0.5 × 0.4 = 0.12
```

**Beam Search** (beam_size=3):
```python
# 维护 3 个候选序列
step 1: [token_a, token_b, token_c]  # 前3个最优
step 2: 
  token_a → [token_x, token_y, token_z]
  token_b → [token_p, token_q, token_r]
  token_c → [token_m, token_n, token_o]
  # 从 9 个候选中选前 3 个最优
step 3: ...
```

**区别**：

| 维度     | Greedy           | Beam Search        |
| -------- | ---------------- | ------------------ |
| 搜索空间 | 贪心（局部最优） | 广度优先（更全局） |
| 计算量   | O(1)             | O(beam_size)       |
| 结果质量 | 可能次优         | 通常更好           |
| 多样性   | 低               | 中等               |
| 使用场景 | 速度优先         | 质量优先（翻译）   |

**追问：为什么 Beam Search 在对话生成中效果不好？**

**回答**：
- Beam Search 倾向于生成安全、通用的回复
- 缺乏多样性和创造性
- 对话需要随机采样增加趣味性

---

### Q2.3: 如何实现 n > 1 的并行采样？

**回答要点**：

**方法一：独立采样**（FoloVLLM 当前实现）
```python
# 创建 n 个独立序列
for i in range(n):
    seq = Sequence(...)
    sequences.append(seq)

# 每个序列独立采样
for seq in sequences:
    next_token = sample(logits, temperature, top_p, top_k)
    seq.add_token_id(next_token)
```

**方法二：Best-of-N**（更高级）
```python
# 生成 best_of 个候选
n=3, best_of=5

# 1. 生成 5 个序列
sequences = [Sequence(...) for _ in range(5)]

# 2. 并行生成
for seq in sequences:
    generate(seq)

# 3. 按累积 log 概率排序
sequences.sort(key=lambda s: s.cumulative_logprob, reverse=True)

# 4. 返回前 3 个
return sequences[:3]
```

**优化**：
- 共享 prompt 的 KV Cache
- 只在 decode 阶段独立计算
- 减少重复计算

**追问：如何共享 prompt 的 KV Cache？**

**回答**（预告 M3）：
```python
# 所有序列共享 prompt 的 KV Cache blocks
for seq in sequences:
    seq.block_ids[:prompt_blocks] = shared_blocks  # 共享
    seq.block_ids[prompt_blocks:] = allocate_new_blocks()  # 独立
```

---

## 3. 数据结构设计

### Q3.1: 为什么需要 Request、Sequence、SequenceData 三层抽象？

**回答要点**：

**Request**（请求级别）：
- 管理一个推理请求的所有序列
- 包含共享信息：prompt、sampling_params
- 提供请求级别的操作：is_finished()、get_seqs()

**Sequence**（序列级别）：
- 一个独立的生成序列
- 管理序列状态：WAITING/RUNNING/FINISHED
- 管理序列资源：KV Cache blocks（M3）
- 提供序列级别的操作：add_token_id()、fork()

**SequenceData**（数据级别）：
- 纯数据容器：prompt_token_ids、output_token_ids
- 不包含状态和逻辑
- 便于序列化和传输

**类比**：
```
Request   = 订单
Sequence  = 订单项
SequenceData = 商品信息
```

**好处**：
- 职责清晰，易于维护
- 便于扩展（如添加状态、资源管理）
- 符合单一职责原则

---

### Q3.2: Sequence 的 fork() 方法有什么用？

**回答要点**：

**用途**：

1. **Beam Search**：
```python
# 扩展候选
beam = [seq1, seq2, seq3]
new_beam = []
for seq in beam:
    for token in top_k_tokens:
        new_seq = seq.fork(f"{seq.seq_id}-{token}")
        new_seq.add_token_id(token)
        new_beam.append(new_seq)
# 保留最优的 beam_size 个
```

2. **Speculative Decoding**：
```python
# Draft model 生成推测序列
draft_seq = seq.fork("draft")
# 验证推测，不修改原序列
```

3. **Parallel Sampling**：
```python
# 从一个序列派生多个独立序列
sequences = [seq.fork(f"seq-{i}") for i in range(n)]
```

**关键**：深拷贝
```python
def fork(self, new_seq_id: str) -> "Sequence":
    new_data = SequenceData(
        prompt_token_ids=self.data.prompt_token_ids.copy(),  # 深拷贝
        output_token_ids=self.data.output_token_ids.copy(),
    )
    # 修改 fork 的序列不影响原序列
```

**追问：为什么要深拷贝？**

**回答**：
```python
# 如果浅拷贝
seq1 = Sequence(...)
seq2 = seq1.fork("seq2")
seq2.output_token_ids.append(100)
# ❌ seq1 也会被修改！

# 深拷贝避免共享状态
seq2.data.output_token_ids = seq1.data.output_token_ids.copy()
seq2.output_token_ids.append(100)
# ✅ seq1 不受影响
```

---

### Q3.3: 序列状态机是如何设计的？

**回答要点**：

**状态转换图**：
```
    WAITING
       ↓ schedule()
    RUNNING
    ↙     ↓     ↘
SWAPPED  step()  FINISHED_*
    ↓         ↗
    └─ resume()
```

**状态说明**：
- `WAITING`: 在等待队列，等待被调度
- `RUNNING`: 正在生成 token
- `SWAPPED`: 被换出到 CPU（显存不足时）
- `FINISHED_STOPPED`: 遇到停止条件（EOS、stop string）
- `FINISHED_LENGTH_CAPPED`: 达到 max_tokens
- `FINISHED_ABORTED`: 用户取消

**状态检查**：
```python
class SequenceStatus(Enum):
    def is_finished(self) -> bool:
        return self in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
        ]
```

**使用**：
```python
# 调度器根据状态决策
if seq.status == SequenceStatus.WAITING:
    schedule_seq(seq)
elif seq.status == SequenceStatus.RUNNING:
    if seq.is_finished():
        remove_from_running(seq)
```

---

## 4. 模型加载相关

### Q4.1: 不同 dtype 的区别和选择？

**回答要点**：

**数据类型对比**：

| dtype    | 位数 | 范围 | 精度 | 显存 | 速度 | 使用场景       |
| -------- | ---- | ---- | ---- | ---- | ---- | -------------- |
| float32  | 32   | 大   | 高   | 基准 | 慢   | 训练、科学计算 |
| float16  | 16   | 小   | 中   | 50%  | 快   | 推理（通用）   |
| bfloat16 | 16   | 大   | 低   | 50%  | 快   | 推理（新硬件） |

**FP16 vs BF16**：
```
FP32:  1 bit (符号) + 8 bits (指数) + 23 bits (尾数)
FP16:  1 bit (符号) + 5 bits (指数) + 10 bits (尾数)
BF16:  1 bit (符号) + 8 bits (指数) + 7 bits (尾数)
```

- **FP16**：精度高，但范围小，容易溢出
- **BF16**：范围大（与 FP32 相同），但精度低

**选择策略**：
```python
def choose_dtype(model_name, hardware):
    if "训练" in task:
        return torch.float32
    
    if "A100" in hardware or "H100" in hardware:
        return torch.bfloat16  # 新硬件支持
    
    if "V100" in hardware:
        return torch.float16  # 旧硬件
    
    if "CPU" in hardware:
        return torch.float32  # CPU 不支持 FP16
```

**追问：为什么模型推理可以用 FP16？**

**回答**：
- 推理不需要梯度，数值稳定性要求低
- Transformer 对精度不敏感
- 实验表明 FP16 推理精度损失 < 1%

---

### Q4.2: 为什么 tokenizer 的 padding_side 要设置为 left？

**回答要点**：

**生成任务的特点**：
- 需要知道每个序列的"结尾"在哪里
- 从结尾开始生成新 token

**Left Padding**（推荐）：
```python
# Batch
Seq 1: [PAD][PAD] Hello world
Seq 2: [PAD] Hello world !
         ↓
# 生成时，attention mask 保证 PAD 不参与计算
# 新 token 添加在右侧（已对齐）
Seq 1: [PAD][PAD] Hello world <new>
Seq 2: [PAD] Hello world ! <new>
                            ↑ 位置对齐
```

**Right Padding**（不推荐生成）：
```python
# Batch
Seq 1: Hello world [PAD][PAD]
Seq 2: Hello world ! [PAD]
         ↓
# 生成时，每个序列的"结尾"位置不同
Seq 1: Hello world [PAD][PAD]  # 在位置 2
Seq 2: Hello world ! [PAD]      # 在位置 3
# 需要额外处理来找到正确的生成位置
```

**实现**：
```python
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side="left",  # ← 关键
)
```

**追问：分类任务为什么用 Right Padding？**

**回答**：
```python
# 分类任务关注最后一个真实 token
Seq 1: [CLS] Hello world [SEP] [PAD]
                           ↑ 这里做分类
Seq 2: [CLS] Hello world ! [SEP]
                            ↑ 这里做分类
# Right padding 让分类位置更接近
```

---

### Q4.3: 如何处理模型和 tokenizer 不匹配的情况？

**回答要点**：

**问题场景**：
- 使用了不同的 tokenizer
- Tokenizer 缺少特殊 token
- Vocab size 不匹配

**处理策略**：

**1. 自动设置 pad_token**：
```python
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token  # 复用 EOS
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # 新建
        # 需要 resize 模型的 embedding
        model.resize_token_embeddings(len(tokenizer))
```

**2. 检查 vocab size**：
```python
model_vocab_size = model.config.vocab_size
tokenizer_vocab_size = len(tokenizer)

if model_vocab_size != tokenizer_vocab_size:
    logger.warning(f"Vocab size mismatch: {model_vocab_size} vs {tokenizer_vocab_size}")
    # 调整模型
    model.resize_token_embeddings(tokenizer_vocab_size)
```

**3. 验证特殊 token**：
```python
required_tokens = ["bos_token", "eos_token", "pad_token"]
for token_name in required_tokens:
    if getattr(tokenizer, token_name) is None:
        logger.warning(f"Missing {token_name}")
```

---

## 5. 系统设计相关

### Q5.1: 如果要支持多 GPU，配置系统需要怎么改？

**回答要点**：

**当前单 GPU 设计**：
```python
@dataclass
class ModelConfig:
    model: str
    dtype: str = "auto"
    # 没有 GPU 相关配置
```

**多 GPU 扩展**：
```python
@dataclass
class ParallelConfig:
    """并行配置"""
    tensor_parallel_size: int = 1  # 张量并行（模型切分）
    pipeline_parallel_size: int = 1  # 流水线并行（层切分）
    data_parallel_size: int = 1  # 数据并行（batch 切分）
    
    def get_world_size(self) -> int:
        return (self.tensor_parallel_size * 
                self.pipeline_parallel_size * 
                self.data_parallel_size)

@dataclass
class EngineConfig:
    model_config: ModelConfig
    parallel_config: ParallelConfig  # 新增
    # ...
```

**使用**：
```python
# 2-GPU 张量并行
config = EngineConfig(
    model_config=ModelConfig(...),
    parallel_config=ParallelConfig(tensor_parallel_size=2)
)
```

**追问：这三种并行的区别？**

**回答**：
- **Tensor Parallel**: 把每层参数切分到多个 GPU（需要通信）
- **Pipeline Parallel**: 把不同层放到不同 GPU（层间通信）
- **Data Parallel**: 每个 GPU 有完整模型，处理不同 batch

---

### Q5.2: 如何设计才能方便地添加新的配置项？

**回答要点**：

**原则**：开闭原则（对扩展开放，对修改封闭）

**1. 使用可选参数**：
```python
@dataclass
class CacheConfig:
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    # 新增配置（向后兼容）
    enable_prefix_caching: bool = False  # 默认值保持兼容
    prefix_cache_size: Optional[int] = None  # 可选
```

**2. 使用 Union 类型**：
```python
from typing import Union

@dataclass
class ModelConfig:
    # 支持多种输入类型
    dtype: Union[str, torch.dtype] = "auto"
    
    def __post_init__(self):
        # 统一转换
        if isinstance(self.dtype, str):
            self.torch_dtype = parse_dtype(self.dtype)
```

**3. 使用配置字典**：
```python
@dataclass
class EngineConfig:
    # 预留扩展字段
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        return self.extra_config.get(key, default)
```

**4. 版本化配置**：
```python
@dataclass
class EngineConfig:
    config_version: str = "1.0"
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        version = config_dict.get("config_version", "1.0")
        if version == "1.0":
            return cls(**config_dict)
        elif version == "2.0":
            # 处理版本迁移
            return cls._migrate_v1_to_v2(config_dict)
```

---

## 6. 性能优化相关

### Q6.1: 如何减少配置验证的开销？

**回答要点**：

**问题**：每次创建配置都要验证，可能影响性能

**优化策略**：

**1. 懒验证**：
```python
@dataclass
class CacheConfig:
    block_size: int = 16
    _validated: bool = field(default=False, init=False)
    
    def validate(self):
        """延迟到实际使用时验证"""
        if self._validated:
            return
        
        if self.block_size <= 0:
            raise ValueError(...)
        
        self._validated = True
```

**2. 缓存验证结果**：
```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    _sampling_type: Optional[SamplingType] = field(default=None, init=False)
    
    @property
    def sampling_type(self) -> SamplingType:
        """缓存计算结果"""
        if self._sampling_type is None:
            self._sampling_type = self._compute_sampling_type()
        return self._sampling_type
```

**3. 批量验证**：
```python
# 不要这样
for config in configs:
    config.validate()  # 每个都验证

# 应该这样
EngineConfig.validate_batch(configs)  # 批量验证，共享检查
```

**追问：这样做的权衡是什么？**

**回答**：
- **优点**：性能更好，特别是频繁创建配置时
- **缺点**：错误发现延迟，可能在运行时才发现配置错误
- **建议**：关键配置（如安全相关）立即验证，性能相关配置延迟验证

---

### Q6.2: 为什么 token_ids 使用 List 而不是 Tensor？

**回答要点**：

**动态增长**：
```python
# List - 动态添加很方便
output_token_ids = []
for _ in range(max_tokens):
    token = generate_next_token()
    output_token_ids.append(token)  # O(1) 均摊

# Tensor - 需要预分配或重新分配
output_tokens = torch.zeros(max_tokens, dtype=torch.long)
for i in range(actual_len):
    output_tokens[i] = generate_next_token()
# 或者
output_tokens = torch.cat([output_tokens, new_token.unsqueeze(0)])  # 每次都复制
```

**内存效率**：
```python
# List: 只存储实际生成的 token
[1, 2, 3]  # 3 个整数

# Tensor: 预分配最大长度
tensor([1, 2, 3, 0, 0, 0, ...])  # 浪费内存
```

**何时转换为 Tensor**：
```python
# 批处理时才转换
def prepare_batch(sequences):
    # 转换为 padded tensor
    token_ids_list = [seq.get_token_ids() for seq in sequences]
    return pad_sequence(token_ids_list)  # → Tensor
```

**追问：List 的缺点是什么？**

**回答**：
- 不能直接用于模型计算（需要转换）
- 单个元素访问比 Tensor 慢
- 但在动态增长场景下，优点大于缺点

---

### Q6.3: 如何优化大量 Sequence 对象的创建？

**回答要点**：

**问题**：创建大量序列对象开销大

**优化方法**：

**1. 对象池**：
```python
class SequencePool:
    def __init__(self, pool_size=1000):
        self.pool = [Sequence(...) for _ in range(pool_size)]
        self.free_list = list(range(pool_size))
    
    def acquire(self) -> Sequence:
        if not self.free_list:
            # 池满，创建新对象
            return Sequence(...)
        idx = self.free_list.pop()
        seq = self.pool[idx]
        seq.reset()  # 重置状态
        return seq
    
    def release(self, seq: Sequence):
        # 归还到池中
        self.free_list.append(seq.pool_index)
```

**2. 延迟初始化**：
```python
@dataclass
class Sequence:
    # 只在需要时创建
    _cached_token_ids: Optional[List[int]] = None
    
    def get_token_ids(self) -> List[int]:
        if self._cached_token_ids is None:
            self._cached_token_ids = (
                self.data.prompt_token_ids + 
                self.data.output_token_ids
            )
        return self._cached_token_ids
```

**3. 批量创建**：
```python
def create_sequences_batch(n: int, config: SamplingParams):
    # 一次性分配内存
    sequences = []
    base_data = SequenceData(...)
    
    for i in range(n):
        # 浅拷贝 + 深拷贝需要独立的部分
        seq = Sequence(
            seq_id=f"seq-{i}",
            data=base_data.copy(),  # 只拷贝必要的
            sampling_params=config,  # 共享不变的部分
        )
        sequences.append(seq)
    
    return sequences
```

---

## 总结：面试准备建议

### 重点掌握

1. **配置系统**：
   - 为什么分层？
   - 为什么用 dataclass？
   - 如何扩展？

2. **采样策略**：
   - Temperature、Top-k、Top-p 原理
   - 组合使用的理由
   - 不同任务的选择

3. **数据结构**：
   - 三层抽象的理由
   - 状态机设计
   - Fork 机制

4. **系统设计**：
   - 如何扩展到多 GPU
   - 配置管理最佳实践
   - 性能优化权衡

### 深入学习

- 阅读 vLLM 源码对比实现差异
- 实验不同配置的性能影响
- 思考未来功能的扩展方式

### 面试技巧

1. **结构化回答**：先总后分，列举要点
2. **举例说明**：用代码示例解释概念
3. **对比分析**：说明不同方案的优缺点
4. **追问准备**：预测面试官可能的追问
5. **连接后续**：提及与后续 milestone 的关系

---

**需要更深入的讨论？** 查看 [学习笔记](../learn/milestone_0.md) 了解技术原理。

