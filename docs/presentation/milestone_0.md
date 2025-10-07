# Milestone 0 口述展示文档

> 本文档以类/函数为单位，详细讲解 M0 的实现过程，适合向小白讲解

---

## 📋 文档结构

本文档按照开发顺序讲解：

1. [项目初始化与环境脚本](#1-项目初始化与环境脚本)
2. [配置系统实现](#2-配置系统实现)
3. [采样参数实现](#3-采样参数实现)
4. [请求和序列实现](#4-请求和序列实现)
5. [输出格式实现](#5-输出格式实现)
6. [模型加载器实现](#6-模型加载器实现)
7. [工具函数实现](#7-工具函数实现)
8. [包导出和测试](#8-包导出和测试)

---

## 1. 项目初始化与环境脚本

### 1.1 为什么需要环境脚本？

**场景思考**：
```
开发者 A: "怎么安装依赖？"
开发者 B: "Python 版本不对，怎么办？"
开发者 C: "CUDA 不可用，如何检查？"
```

**解决方案**：创建自动化环境设置脚本

### 1.2 setup_env.sh 实现（Linux/macOS）

#### 第一步：定义辅助函数

**为什么**：让脚本输出更友好，便于用户理解进度

```bash
# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# 打印辅助函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}
```

**讲解**：
- 使用 ANSI 转义码实现彩色输出
- 统一的消息格式，易于阅读
- 不同级别用不同颜色（信息=蓝，成功=绿，错误=红）

#### 第二步：检查 Python 版本

**为什么**：项目需要 Python 3.10+，需要确保环境符合要求

```bash
check_python_version() {
    print_header "检查 Python 版本"
    
    # 尝试不同的 Python 命令
    for cmd in python3.10 python3.11 python3.12 python3 python; do
        if command -v $cmd &> /dev/null; then
            PYTHON_CMD=$cmd
            PYTHON_VER=$($cmd --version 2>&1 | awk '{print $2}')
            print_info "找到 Python: $cmd (版本 $PYTHON_VER)"
            
            # 检查版本是否 >= 3.10
            MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
            MINOR=$(echo $PYTHON_VER | cut -d. -f2)
            
            if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
                print_success "Python 版本符合要求 (>= 3.10)"
                return 0
            fi
        fi
    done
    
    print_error "未找到 Python 3.10 或更高版本"
    exit 1
}
```

**讲解**：
1. **循环尝试**：不同系统 Python 命令不同（python3.10、python3、python）
2. **版本解析**：用 `awk` 和 `cut` 解析版本号
3. **数值比较**：检查主版本号 >= 3 且次版本号 >= 10
4. **友好提示**：找不到时给出明确的错误信息

#### 第三步：创建虚拟环境

**为什么**：隔离项目依赖，避免污染系统 Python

```bash
create_venv() {
    print_header "创建虚拟环境"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "虚拟环境已存在: $VENV_DIR"
        read -p "是否删除并重新创建？(y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "删除旧的虚拟环境..."
            rm -rf $VENV_DIR
        else
            print_info "跳过虚拟环境创建"
            return 0
        fi
    fi
    
    print_info "创建虚拟环境: $VENV_DIR"
    $PYTHON_CMD -m venv $VENV_DIR
    print_success "虚拟环境创建成功"
}
```

**讲解**：
1. **检查存在**：避免覆盖已有环境
2. **用户确认**：让用户决定是否重建
3. **使用找到的 Python**：用 `$PYTHON_CMD` 确保版本正确

#### 第四步：安装依赖

**为什么**：自动化安装过程，处理可选依赖

```bash
install_dependencies() {
    print_header "安装依赖"
    
    print_info "安装基础依赖..."
    pip install -r requirements.txt -q
    print_success "基础依赖安装成功"
    
    # 询问是否安装可选依赖
    echo ""
    print_info "可选依赖："
    echo "  1) Flash Attention 2 (需要 CUDA，编译时间较长)"
    echo "  2) AutoGPTQ (用于 M7 量化支持)"
    echo "  3) 跳过可选依赖"
    read -p "请选择 (1/2/3, 默认=3): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            print_info "安装 Flash Attention 2..."
            print_warning "这可能需要 10-20 分钟..."
            pip install flash-attn --no-build-isolation
            print_success "Flash Attention 2 安装成功"
            ;;
        2)
            print_info "安装 AutoGPTQ..."
            pip install auto-gptq optimum
            print_success "AutoGPTQ 安装成功"
            ;;
        *)
            print_info "跳过可选依赖"
            ;;
    esac
}
```

**讲解**：
1. **静默安装**：`-q` 参数减少输出，避免信息过载
2. **交互选择**：让用户决定安装哪些可选依赖
3. **清晰提示**：说明安装时间和用途

#### 第五步：验证安装

**为什么**：确保所有组件正确安装，提前发现问题

```bash
verify_installation() {
    print_header "验证安装"
    
    print_info "检查项目导入..."
    python -c "import folovllm; print('✓ folovllm 导入成功')"
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
    python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
    
    # 检查 CUDA
    echo ""
    print_info "检查 CUDA 可用性..."
    python -c "import torch; print('✓ CUDA 可用' if torch.cuda.is_available() else '✗ CUDA 不可用')"
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
        python -c "import torch; print(f'  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
    fi
    
    print_success "安装验证通过"
}
```

**讲解**：
1. **导入测试**：用 `python -c` 快速测试导入
2. **CUDA 检查**：检测 GPU 可用性和显存
3. **条件执行**：只在 CUDA 可用时显示 GPU 信息

### 1.3 activate.sh 实现

**为什么**：提供快捷的虚拟环境激活方式

```bash
#!/bin/bash
# 快速激活虚拟环境
source venv/bin/activate
echo "✓ FoloVLLM 虚拟环境已激活"
echo "Python: $(python --version)"
echo "位置: $(which python)"
```

**讲解**：
- 简单的激活脚本，避免输入长命令
- 显示 Python 版本和路径，便于确认

### 1.4 .gitignore 实现

**为什么**：防止不必要的文件进入版本控制

```gitignore
# Python
__pycache__/
*.py[cod]
venv/
*.egg-info/

# IDE
.vscode/
.cursor/
.idea/

# 模型文件
*.bin
*.safetensors
*.pth

# 日志和临时文件
*.log
logs/
tmp/
```

**讲解**：
1. **Python 相关**：忽略编译文件、虚拟环境
2. **IDE 配置**：不同开发者可能用不同 IDE
3. **大文件**：模型文件不应进入 git
4. **临时文件**：日志和缓存文件

---

## 2. 配置系统实现

### 2.1 整体设计思路

**目标**：创建一个分层、类型安全、易扩展的配置系统

**层次结构**：
```
EngineConfig (顶层)
    ├── ModelConfig (模型配置)
    ├── CacheConfig (缓存配置)
    └── SchedulerConfig (调度配置)
```

### 2.2 ModelConfig 实现

#### 第一步：定义基本结构

**思考**：模型配置需要哪些信息？

```python
from dataclasses import dataclass
from typing import Literal, Optional

ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]

@dataclass
class ModelConfig:
    """模型配置"""
    model: str  # 必需：模型路径
    tokenizer: Optional[str] = None  # 可选：tokenizer 路径
    tokenizer_mode: str = "auto"  # 默认：fast tokenizer
    trust_remote_code: bool = False  # 默认：不信任远程代码
    dtype: ModelDType = "auto"  # 默认：自动选择
    max_model_len: Optional[int] = None  # 可选：从模型推断
    seed: int = 0  # 默认：固定种子
```

**讲解**：
- **必需参数**：`model` 是唯一必需的
- **合理默认值**：大部分参数有合理的默认值
- **类型约束**：`Literal` 限制 dtype 的取值
- **可选参数**：用 `Optional` 标记

#### 第二步：添加后处理逻辑

**思考**：创建后需要做什么？

```python
@dataclass
class ModelConfig:
    # ... 字段定义 ...
    
    def __post_init__(self):
        """创建后的初始化逻辑"""
        # 1. tokenizer 默认使用 model 路径
        if self.tokenizer is None:
            self.tokenizer = self.model
        
        # 2. 将字符串 dtype 转换为 torch.dtype
        if isinstance(self.dtype, str):
            dtype_map = {
                "auto": None,
                "half": torch.float16,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float": torch.float32,
                "float32": torch.float32,
            }
            self.torch_dtype = dtype_map.get(self.dtype)
        else:
            self.torch_dtype = self.dtype
```

**讲解**：
1. **自动补全**：tokenizer 未指定时使用 model 路径
2. **类型转换**：字符串 → torch.dtype，方便后续使用
3. **保存结果**：转换结果保存到 `torch_dtype`

**为什么这样设计**：
- 用户可以写 `dtype="float16"`（易读）
- 内部使用 `torch.float16`（类型正确）

### 2.3 CacheConfig 实现

#### 第一步：定义配置项

**思考**：KV Cache 需要哪些配置？

```python
@dataclass
class CacheConfig:
    """KV Cache 配置"""
    block_size: int = 16  # 每个 block 的 token 数
    gpu_memory_utilization: float = 0.9  # GPU 显存利用率
    swap_space: float = 4.0  # CPU swap 空间 (GiB)
    enable_prefix_caching: bool = False  # 前缀缓存（M6）
```

**讲解**：
- **block_size**：PagedAttention 的基本单位（M3 使用）
- **gpu_memory_utilization**：控制显存使用，避免 OOM
- **swap_space**：显存不足时交换到 CPU
- **enable_prefix_caching**：预留给 M6

#### 第二步：添加验证逻辑

**思考**：哪些配置值是不合法的？

```python
@dataclass
class CacheConfig:
    # ... 字段定义 ...
    
    def __post_init__(self):
        """验证配置合法性"""
        # 验证 block_size
        if self.block_size <= 0:
            raise ValueError(
                f"block_size must be positive, got {self.block_size}"
            )
        
        # 验证 gpu_memory_utilization
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )
```

**讲解**：
1. **提前验证**：在创建时就发现错误，而不是运行时
2. **清晰的错误信息**：告诉用户哪里错了
3. **范围检查**：确保参数在合理范围内

**为什么这样做**：
- 早发现早修复
- 避免运行到一半才出错
- 错误信息清晰，易于调试

### 2.4 SchedulerConfig 实现

#### 预留设计

**思考**：调度器需要哪些配置？

```python
@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_num_batched_tokens: Optional[int] = None  # 最大批处理 token 数
    max_num_seqs: int = 256  # 最大序列数
    max_model_len: Optional[int] = None  # 最大序列长度
    enable_chunked_prefill: bool = False  # 分块预填充（M5）
    
    def __post_init__(self):
        """后处理"""
        # 预留：后续会从 engine config 同步 max_model_len
        pass
```

**讲解**：
- **预留接口**：虽然 M0 不用，但预留给 M2
- **可选参数**：大部分都是可选的
- **空的后处理**：明确表示有这个阶段，但暂无逻辑

### 2.5 EngineConfig 实现

#### 组装配置

**思考**：如何组合三个配置类？

```python
from dataclasses import dataclass, field

@dataclass
class EngineConfig:
    """引擎统一配置"""
    model_config: ModelConfig  # 必需
    cache_config: CacheConfig = field(default_factory=CacheConfig)  # 可选，有默认值
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)  # 可选
    
    def __post_init__(self):
        """配置同步"""
        # 将 max_model_len 同步到 scheduler_config
        if self.scheduler_config.max_model_len is None:
            self.scheduler_config.max_model_len = self.model_config.max_model_len
```

**讲解**：
1. **必需的 model_config**：必须明确指定模型
2. **可选的子配置**：用 `field(default_factory=...)` 提供默认值
3. **配置同步**：避免配置不一致

**为什么用 default_factory**：
```python
# ❌ 错误：所有实例共享同一个对象
cache_config: CacheConfig = CacheConfig()

# ✅ 正确：每个实例有自己的对象
cache_config: CacheConfig = field(default_factory=CacheConfig)
```

---

## 3. 采样参数实现

### 3.1 设计思路

**目标**：支持多种采样策略，参数验证，易于扩展

### 3.2 SamplingParams 实现

#### 第一步：定义采样类型

**思考**：有哪些采样类型？

```python
from enum import IntEnum

class SamplingType(IntEnum):
    """采样类型"""
    GREEDY = 0  # 贪心：总是选最大概率
    RANDOM = 1  # 随机：按概率分布采样
```

**讲解**：
- **IntEnum**：既有名字又有整数值
- **两种基本类型**：确定性（贪心）和随机性

#### 第二步：定义参数结构

**思考**：采样需要哪些参数？

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SamplingParams:
    """采样参数"""
    
    # 输出数量
    n: int = 1  # 返回几个序列
    best_of: Optional[int] = None  # 生成几个候选
    
    # 采样策略
    temperature: float = 1.0  # 温度
    top_p: float = 1.0  # Nucleus sampling
    top_k: int = -1  # Top-k sampling (-1 表示禁用)
    min_p: float = 0.0  # 最小概率阈值
    
    # 停止条件
    stop: Optional[List[str]] = None  # 停止字符串
    stop_token_ids: Optional[List[int]] = None  # 停止 token ID
    max_tokens: Optional[int] = 16  # 最大生成长度
    min_tokens: int = 0  # 最小生成长度
    
    # 其他
    seed: Optional[int] = None  # 随机种子
    skip_special_tokens: bool = True  # 跳过特殊 token
```

**讲解**：
- **分组**：按功能分组（输出、采样、停止、其他）
- **默认值**：每个参数都有合理的默认值
- **可选参数**：用 `Optional` 标记

#### 第三步：参数验证

**思考**：哪些参数组合是不合法的？

```python
@dataclass
class SamplingParams:
    # ... 字段定义 ...
    
    def __post_init__(self):
        """参数验证"""
        
        # 1. best_of 默认等于 n
        if self.best_of is None:
            self.best_of = self.n
        
        # 2. 验证 n
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")
        
        # 3. 验证 best_of >= n
        if self.best_of < self.n:
            raise ValueError(
                f"best_of ({self.best_of}) must be >= n ({self.n})"
            )
        
        # 4. 验证 temperature
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}"
            )
        
        # 5. 验证 top_p
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        
        # 6. 验证 top_k
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disabled) or >= 1, got {self.top_k}"
            )
        
        # 7. 检查未实现的功能
        if self.use_beam_search:
            raise NotImplementedError(
                "Beam search is not supported in M0-M1"
            )
        
        # 8. 初始化列表
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []
```

**讲解**：
1. **自动设置**：best_of 默认等于 n
2. **范围检查**：确保每个参数在合理范围
3. **逻辑检查**：best_of 必须 >= n（否则无法选出 n 个）
4. **功能检查**：明确标记未实现的功能
5. **默认值**：空列表而不是 None

**为什么这样验证**：
- 提前发现配置错误
- 清晰的错误信息
- 避免运行时意外行为

#### 第四步：添加辅助方法

**思考**：如何方便地获取采样类型？

```python
@dataclass
class SamplingParams:
    # ... 字段和验证 ...
    
    @property
    def sampling_type(self) -> SamplingType:
        """根据参数判断采样类型"""
        if self.temperature == 0.0:
            return SamplingType.GREEDY
        else:
            return SamplingType.RANDOM
```

**讲解**：
- **@property**：像访问属性一样调用方法
- **判断逻辑**：temperature=0 表示贪心，否则随机
- **使用**：`params.sampling_type` 而不是 `params.sampling_type()`

---

## 4. 请求和序列实现

### 4.1 设计思路

**目标**：三层抽象管理生成状态

```
Request (一个推理请求)
  ├── Sequence 1 (候选序列 1)
  │     └── SequenceData (token IDs)
  ├── Sequence 2 (候选序列 2)
  └── ...
```

### 4.2 SequenceData 实现

#### 纯数据容器

**思考**：序列需要保存哪些数据？

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class SequenceData:
    """序列数据（纯数据，无状态）"""
    prompt_token_ids: List[int]  # 输入 token
    output_token_ids: List[int] = field(default_factory=list)  # 输出 token
```

**讲解**：
- **两个列表**：输入和输出分开存储
- **default_factory**：每个实例有自己的列表

#### 添加操作方法

**思考**：需要哪些常用操作？

```python
@dataclass
class SequenceData:
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    
    def get_len(self) -> int:
        """总长度"""
        return len(self.prompt_token_ids) + len(self.output_token_ids)
    
    def get_prompt_len(self) -> int:
        """输入长度"""
        return len(self.prompt_token_ids)
    
    def get_output_len(self) -> int:
        """输出长度"""
        return len(self.output_token_ids)
    
    def get_token_ids(self) -> List[int]:
        """所有 token"""
        return self.prompt_token_ids + self.output_token_ids
    
    def get_last_token_id(self) -> int:
        """最后一个 token"""
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]
    
    def add_token_id(self, token_id: int):
        """添加新 token"""
        self.output_token_ids.append(token_id)
```

**讲解**：
- **命名规范**：get_ 前缀表示查询操作
- **边界处理**：`get_last_token_id` 考虑了 output 为空的情况
- **修改操作**：只有 `add_token_id` 会修改数据

### 4.3 Sequence 实现

#### 第一步：定义状态枚举

**思考**：序列有哪些状态？

```python
from enum import Enum

class SequenceStatus(Enum):
    """序列状态"""
    WAITING = "waiting"  # 等待调度
    RUNNING = "running"  # 正在生成
    SWAPPED = "swapped"  # 被换出到 CPU
    FINISHED_STOPPED = "finished_stopped"  # 遇到停止条件
    FINISHED_LENGTH_CAPPED = "finished_length_capped"  # 达到最大长度
    FINISHED_ABORTED = "finished_aborted"  # 被用户中止
    FINISHED_IGNORED = "finished_ignored"  # 被忽略（best_of > n）
    
    def is_finished(self) -> bool:
        """是否已完成"""
        return self in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]
```

**讲解**：
- **清晰的命名**：状态名反映了序列所处的阶段
- **辅助方法**：`is_finished()` 方便判断
- **三类状态**：运行中、等待中、完成

#### 第二步：定义 Sequence 类

**思考**：序列需要哪些信息？

```python
from dataclasses import dataclass, field

@dataclass
class Sequence:
    """一个生成序列"""
    seq_id: str  # 唯一标识
    request_id: str  # 所属请求
    data: SequenceData  # 数据
    sampling_params: SamplingParams  # 采样参数
    status: SequenceStatus = SequenceStatus.WAITING  # 状态
    
    # M3: KV Cache blocks (预留)
    block_ids: List[int] = field(default_factory=list)
```

**讲解**：
- **标识信息**：seq_id（自己）和 request_id（所属）
- **数据和参数**：分离关注点
- **状态**：默认为 WAITING
- **预留字段**：block_ids 用于 M3

#### 第三步：添加代理方法

**思考**：如何方便地访问 SequenceData 的方法？

```python
@dataclass
class Sequence:
    # ... 字段定义 ...
    
    def get_len(self) -> int:
        """代理到 data.get_len()"""
        return self.data.get_len()
    
    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()
    
    def get_output_len(self) -> int:
        return self.data.get_output_len()
    
    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()
    
    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()
    
    def add_token_id(self, token_id: int):
        return self.data.add_token_id(token_id)
```

**讲解**：
- **代理模式**：Sequence 代理到 SequenceData
- **便利性**：`seq.get_len()` 而不是 `seq.data.get_len()`

#### 第四步：实现 fork 方法

**思考**：如何复制一个序列？

```python
@dataclass
class Sequence:
    # ... 其他方法 ...
    
    def fork(self, new_seq_id: str) -> "Sequence":
        """复制序列（用于 beam search）"""
        # 深拷贝数据
        new_data = SequenceData(
            prompt_token_ids=self.data.prompt_token_ids.copy(),
            output_token_ids=self.data.output_token_ids.copy(),
        )
        
        # 创建新序列
        return Sequence(
            seq_id=new_seq_id,
            request_id=self.request_id,  # 相同的 request
            data=new_data,
            sampling_params=self.sampling_params,  # 共享
            status=self.status,
            block_ids=self.block_ids.copy(),  # 深拷贝
        )
```

**讲解**：
1. **深拷贝数据**：避免共享状态
2. **保留 request_id**：fork 的序列属于同一个请求
3. **共享 sampling_params**：不可变的可以共享
4. **拷贝 block_ids**：列表需要深拷贝

**为什么需要 fork**：
- Beam search：每次扩展都 fork 多个候选
- Speculative decoding：验证推测序列

### 4.4 Request 实现

#### 第一步：定义请求状态

**思考**：请求有哪些状态？（与 Sequence 类似）

```python
class RequestStatus(Enum):
    """请求状态"""
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"
```

#### 第二步：定义 Request 类

**思考**：请求需要管理什么？

```python
import time
from typing import Dict

@dataclass
class Request:
    """一个推理请求"""
    request_id: str  # 唯一标识
    prompt: str  # 原始文本
    prompt_token_ids: List[int]  # 编码后的 token
    sampling_params: SamplingParams  # 采样参数
    arrival_time: float = field(default_factory=time.time)  # 到达时间
    
    # 该请求的所有序列
    sequences: Dict[str, Sequence] = field(default_factory=dict)
    
    # 状态
    status: RequestStatus = RequestStatus.WAITING
```

**讲解**：
- **原始信息**：保存原始 prompt 用于输出
- **时间戳**：记录到达时间，用于统计延迟
- **序列字典**：`seq_id -> Sequence` 的映射
- **状态**：请求级别的状态

#### 第三步：初始化序列

**思考**：什么时候创建序列？

```python
@dataclass
class Request:
    # ... 字段定义 ...
    
    def __post_init__(self):
        """创建序列"""
        if not self.sequences:
            # 根据 best_of 创建多个序列
            for i in range(self.sampling_params.best_of):
                seq_id = f"{self.request_id}-{i}"
                
                # 创建 SequenceData
                seq_data = SequenceData(
                    prompt_token_ids=self.prompt_token_ids.copy()
                )
                
                # 创建 Sequence
                seq = Sequence(
                    seq_id=seq_id,
                    request_id=self.request_id,
                    data=seq_data,
                    sampling_params=self.sampling_params,
                )
                
                self.sequences[seq_id] = seq
```

**讲解**：
1. **检查是否已创建**：避免重复创建
2. **best_of 个序列**：生成多个候选
3. **唯一的 seq_id**：`request_id-索引`
4. **深拷贝 prompt**：每个序列独立

#### 第四步：添加辅助方法

**思考**：如何方便地操作序列？

```python
@dataclass
class Request:
    # ... 其他方法 ...
    
    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        """获取序列（可按状态过滤）"""
        if status is None:
            return list(self.sequences.values())
        return [seq for seq in self.sequences.values() if seq.status == status]
    
    def get_num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        """获取序列数量"""
        return len(self.get_seqs(status))
    
    def is_finished(self) -> bool:
        """是否所有序列都完成"""
        return all(seq.is_finished() for seq in self.sequences.values())
```

**讲解**：
- **过滤功能**：可以只获取特定状态的序列
- **计数方法**：快速获取数量
- **完成判断**：所有序列都完成才算完成

---

## 5. 输出格式实现

### 5.1 设计思路

**目标**：定义清晰的输出格式

```
RequestOutput (请求级别)
  └── CompletionOutput[] (每个序列一个)
```

### 5.2 CompletionOutput 实现

**思考**：一个完成的序列包含什么信息？

```python
@dataclass
class CompletionOutput:
    """单个完成序列的输出"""
    index: int  # 序列索引（0 到 n-1）
    text: str  # 生成的文本
    token_ids: List[int]  # 生成的 token ID
    cumulative_logprob: Optional[float] = None  # 累积对数概率
    logprobs: Optional[List[float]] = None  # 每个 token 的概率
    finish_reason: Optional[str] = None  # 完成原因
    
    def finished(self) -> bool:
        """是否完成"""
        return self.finish_reason is not None
```

**讲解**：
- **基本信息**：文本和 token
- **质量指标**：logprob 用于排序（best_of）
- **完成原因**：'stop'（停止条件）或 'length'（达到最大长度）
- **预留字段**：logprobs 在 M1+ 实现

### 5.3 RequestOutput 实现

**思考**：请求输出需要包含什么？

```python
@dataclass
class RequestOutput:
    """请求输出"""
    request_id: str  # 请求标识
    prompt: str  # 原始输入
    prompt_token_ids: List[int]  # 输入 token
    outputs: List[CompletionOutput]  # 所有完成的序列
    finished: bool  # 是否完成
    
    # 预留：性能指标
    metrics: Optional[dict] = None
    
    def __repr__(self) -> str:
        """友好的字符串表示"""
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt[:50]}..., "
            f"outputs={len(self.outputs)}, finished={self.finished})"
        )
```

**讲解**：
- **输入信息**：保存原始输入用于上下文
- **输出列表**：可能有多个（n > 1）
- **完成标志**：是否所有序列都完成
- **友好显示**：截断长文本避免刷屏

---

## 6. 模型加载器实现

### 6.1 设计思路

**目标**：封装 HuggingFace 模型加载逻辑，自动处理配置

### 6.2 ModelLoader 类实现

#### 第一步：初始化

**思考**：加载器需要什么信息？

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    """HuggingFace 模型加载器"""
    
    def __init__(self, model_config: ModelConfig):
        """初始化
        
        Args:
            model_config: 模型配置
        """
        self.model_config = model_config
```

**讲解**：
- 简单的初始化，只保存配置
- 实际加载在调用 load_model 时进行

#### 第二步：实现 load_model 方法

**思考**：加载模型需要哪些步骤？

```python
class ModelLoader:
    # ... __init__ ...
    
    def load_model(self, device: str = "cuda") -> PreTrainedModel:
        """加载模型
        
        Args:
            device: 设备（'cuda' 或 'cpu'）
            
        Returns:
            加载的模型
        """
        model_path = self.model_config.model
        
        # 1. 确定 dtype
        dtype = self._get_dtype()
        
        # 2. 加载配置
        hf_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        
        # 3. 推断 max_model_len
        if self.model_config.max_model_len is None:
            if hasattr(hf_config, "max_position_embeddings"):
                self.model_config.max_model_len = hf_config.max_position_embeddings
            else:
                self.model_config.max_model_len = 2048  # 默认值
        
        # 4. 打印信息
        print(f"Loading model from {model_path}...")
        print(f"  - dtype: {dtype}")
        print(f"  - max_model_len: {self.model_config.max_model_len}")
        print(f"  - device: {device}")
        
        # 5. 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=hf_config,
            torch_dtype=dtype,
            trust_remote_code=self.model_config.trust_remote_code,
            low_cpu_mem_usage=True,  # 减少 CPU 内存峰值
        )
        
        # 6. 移动到设备
        model = model.to(device)
        model.eval()  # 设置为评估模式
        
        # 7. 打印统计信息
        print(f"Model loaded successfully!")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Number of parameters: {self._count_parameters(model):,}")
        
        return model
```

**讲解**：
1. **分步骤**：每个步骤功能单一，易于理解
2. **自动推断**：max_model_len 从模型配置获取
3. **友好输出**：打印加载进度
4. **优化选项**：`low_cpu_mem_usage=True` 减少内存占用
5. **评估模式**：`eval()` 关闭 dropout 等训练特性

#### 第三步：实现 load_tokenizer 方法

**思考**：加载 tokenizer 需要注意什么？

```python
class ModelLoader:
    # ... 其他方法 ...
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """加载 tokenizer
        
        Returns:
            加载的 tokenizer
        """
        tokenizer_path = self.model_config.tokenizer or self.model_config.model
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        
        # 确定是否使用 fast tokenizer
        use_fast = self.model_config.tokenizer_mode == "auto"
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=use_fast,
            trust_remote_code=self.model_config.trust_remote_code,
            padding_side="left",  # 生成任务用 left padding
        )
        
        # 设置 pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        print(f"Tokenizer loaded successfully!")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        print(f"  - Special tokens: pad={tokenizer.pad_token}, eos={tokenizer.eos_token}")
        
        return tokenizer
```

**讲解**：
1. **路径处理**：tokenizer_path 默认使用 model_path
2. **fast tokenizer**：默认使用 fast 版本（更快）
3. **padding_side**：left padding 适合生成任务
4. **pad_token 处理**：自动设置缺失的 pad_token

#### 第四步：辅助方法

**思考**：需要哪些辅助功能？

```python
class ModelLoader:
    # ... 其他方法 ...
    
    def _get_dtype(self) -> torch.dtype:
        """获取 torch dtype"""
        if self.model_config.torch_dtype is not None:
            return self.model_config.torch_dtype
        
        # 默认：GPU 用 FP16，CPU 用 FP32
        if torch.cuda.is_available():
            return torch.float16
        else:
            return torch.float32
    
    @staticmethod
    def _count_parameters(model: PreTrainedModel) -> int:
        """统计参数数量"""
        return sum(p.numel() for p in model.parameters())
```

**讲解**：
- **dtype 选择**：根据硬件自动选择
- **参数统计**：`numel()` 获取元素数量

#### 第五步：便捷方法

**思考**：如何一次性加载模型和 tokenizer？

```python
class ModelLoader:
    # ... 其他方法 ...
    
    def load_model_and_tokenizer(
        self, device: str = "cuda"
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """同时加载模型和 tokenizer
        
        Args:
            device: 设备
            
        Returns:
            (model, tokenizer) 元组
        """
        model = self.load_model(device)
        tokenizer = self.load_tokenizer()
        return model, tokenizer

# 便捷函数
def get_model_and_tokenizer(
    model_config: ModelConfig, device: str = "cuda"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """便捷的加载函数
    
    Args:
        model_config: 模型配置
        device: 设备
        
    Returns:
        (model, tokenizer) 元组
    """
    loader = ModelLoader(model_config)
    return loader.load_model_and_tokenizer(device)
```

**讲解**：
- **一次性加载**：避免分两次调用
- **便捷函数**：更简洁的 API

---

## 7. 工具函数实现

### 7.1 common.py 工具函数

#### 随机种子管理

**思考**：如何确保结果可复现？

```python
import random
import numpy as np
import torch

def set_random_seed(seed: int) -> None:
    """设置全局随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU
```

**讲解**：
- **多个库**：Python、NumPy、PyTorch 都需要设置
- **CUDA**：GPU 也有独立的随机数生成器

#### 请求 ID 生成

**思考**：如何生成唯一的请求 ID？

```python
import uuid

def generate_request_id() -> str:
    """生成唯一的请求 ID
    
    Returns:
        UUID 字符串
    """
    return str(uuid.uuid4())
```

**讲解**：
- **UUID**：全局唯一标识符
- **uuid4**：基于随机数，冲突概率极低

#### GPU 显存监控

**思考**：如何监控 GPU 显存使用？

```python
def get_gpu_memory_info(device: int = 0) -> dict:
    """获取 GPU 显存信息
    
    Args:
        device: GPU 设备索引
        
    Returns:
        显存信息字典
    """
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    free = total - allocated
    
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
    }
```

**讲解**：
- **检查 CUDA**：先检查是否可用
- **三个指标**：已分配、已预留、总显存
- **单位转换**：字节 → GB

#### 设备管理

**思考**：如何自动选择设备？

```python
def get_device(device: Optional[str] = None) -> torch.device:
    """获取 torch device
    
    Args:
        device: 设备字符串，None 表示自动选择
        
    Returns:
        torch.device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
```

**讲解**：
- **自动选择**：有 CUDA 用 CUDA，否则用 CPU
- **显式指定**：也可以手动指定设备

---

## 8. 包导出和测试

### 8.1 __init__.py 实现

**思考**：如何让用户方便地导入？

```python
"""
FoloVLLM: A Lightweight LLM Inference Framework
"""

__version__ = "0.1.0"
__author__ = "FoloVLLM Contributors"

# 导出核心类
from folovllm.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
)
from folovllm.sampling_params import SamplingParams, SamplingType
from folovllm.request import (
    Request,
    RequestStatus,
    Sequence,
    SequenceData,
    SequenceStatus,
)
from folovllm.outputs import CompletionOutput, RequestOutput
from folovllm.model_loader import ModelLoader, get_model_and_tokenizer
from folovllm.utils.common import (
    generate_request_id,
    get_device,
    get_gpu_memory_info,
    is_cuda_available,
    set_random_seed,
)

__all__ = [
    "__version__",
    # 配置
    "ModelConfig",
    "CacheConfig",
    "SchedulerConfig",
    "EngineConfig",
    # 采样
    "SamplingParams",
    "SamplingType",
    # 请求和序列
    "Request",
    "RequestStatus",
    "Sequence",
    "SequenceData",
    "SequenceStatus",
    # 输出
    "RequestOutput",
    "CompletionOutput",
    # 模型加载
    "ModelLoader",
    "get_model_and_tokenizer",
    # 工具
    "generate_request_id",
    "set_random_seed",
    "get_device",
    "is_cuda_available",
    "get_gpu_memory_info",
]
```

**讲解**：
- **统一导出**：用户可以 `from folovllm import ModelConfig`
- **__all__**：明确哪些是公共 API
- **分组**：按功能分组，易于查找

### 8.2 测试实现思路

#### 单元测试结构

**思考**：如何组织测试？

```python
# tests/unit/test_m0_config.py
class TestModelConfig:
    """测试 ModelConfig"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        config = ModelConfig(model="test")
        assert config.model == "test"
        assert config.tokenizer == "test"  # 自动设置
    
    def test_dtype_conversion(self):
        """测试 dtype 转换"""
        config = ModelConfig(model="test", dtype="float16")
        assert config.torch_dtype == torch.float16
```

**讲解**：
- **类组织**：每个类一个测试类
- **命名规范**：test_ 前缀
- **清晰的文档字符串**：说明测试内容

#### 集成测试思路

**思考**：如何测试端到端流程？

```python
# tests/integration/test_m0_model_loading.py
class TestModelLoadingGPU:
    """测试 GPU 模型加载"""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_load_model_and_tokenizer(self):
        """测试加载模型和 tokenizer"""
        config = ModelConfig(
            model="Qwen/Qwen2.5-0.6B",
            dtype="float16",
        )
        
        try:
            model, tokenizer = get_model_and_tokenizer(config)
            
            # 测试 tokenization
            tokens = tokenizer.encode("Hello")
            assert isinstance(tokens, list)
            
            # 测试 decoding
            text = tokenizer.decode(tokens)
            assert isinstance(text, str)
            
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")
```

**讲解**：
- **条件跳过**：没有 CUDA 时跳过
- **异常处理**：模型未下载时跳过
- **实际验证**：测试完整流程

---

## 9. 总结：开发流程回顾

### 9.1 开发顺序

1. **环境脚本** → 让其他开发者能快速开始
2. **配置系统** → 定义项目的"规则"
3. **数据结构** → 定义核心抽象（Request/Sequence）
4. **工具函数** → 提供基础能力
5. **模型加载** → 连接 HuggingFace
6. **测试** → 验证正确性

### 9.2 设计原则

1. **分层设计**：配置、数据、逻辑分离
2. **类型安全**：使用类型提示和验证
3. **预留接口**：为后续 milestone 预留
4. **对齐 vLLM**：学习成熟框架

### 9.3 关键技巧

1. **dataclass**：减少样板代码
2. **__post_init__**：初始化后处理和验证
3. **Enum**：类型安全的常量
4. **代理模式**：简化接口

### 9.4 测试策略

1. **单元测试**：每个类、每个方法
2. **参数验证测试**：边界条件
3. **集成测试**：端到端流程
4. **异常处理**：优雅地处理错误

---

## 10. 下一步：Milestone 1

### 10.1 M1 需要实现什么？

基于 M0 的基础，M1 将实现：

1. **LLMEngine**：推理引擎主类
   - 使用 ModelConfig 加载模型
   - 使用 SamplingParams 配置采样
   - 管理 Request 生命周期

2. **Token 生成**：
   - 实现 Greedy、Top-k、Top-p 采样
   - 使用 Sequence 管理生成状态

3. **简单 KV Cache**：
   - 为每个 Sequence 分配连续内存

4. **完整推理循环**：
   - Tokenization → Forward → Sampling → Detokenization

### 10.2 M0 如何支持 M1？

- **Request/Sequence**：管理生成状态
- **SamplingParams**：控制采样行为
- **ModelLoader**：加载模型和 tokenizer
- **工具函数**：设置随机种子、管理设备

### 10.3 学习建议

1. **动手实验**：修改配置，观察行为
2. **阅读测试**：测试代码是最好的文档
3. **对比 vLLM**：理解设计思路
4. **提前预习 M1**：了解如何使用 M0 的接口

---

**恭喜完成 Milestone 0！** 🎉

你已经学习了：
- ✅ 项目环境设置自动化
- ✅ 分层配置系统设计
- ✅ 采样策略原理和实现
- ✅ 三层序列抽象（Request/Sequence/SequenceData）
- ✅ HuggingFace 模型加载
- ✅ 测试驱动开发

**准备好开始 Milestone 1 了吗？** 🚀

