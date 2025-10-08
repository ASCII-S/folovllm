# 如何使用 utils/common.py 中的工具函数

## 问题
`utils/common.py` 中实现的工具函数如何在代码中使用？

## 回答

### 方法1：从 folovllm.utils 导入（推荐）

```python
from folovllm.utils import (
    is_cuda_available,
    get_gpu_memory_info,
    get_device,
    set_random_seed,
    generate_request_id,
)

# 使用示例
print(f"CUDA 可用: {is_cuda_available()}")
set_random_seed(42)
device = get_device()
request_id = generate_request_id()
```

**原理**：
- 函数在 `folovllm/utils/common.py` 中实现
- `folovllm/utils/__init__.py` 将它们导出，使得可以通过 `folovllm.utils` 访问

### 方法2：直接从 common 模块导入

```python
from folovllm.utils.common import (
    is_cuda_available,
    get_gpu_memory_info,
    set_random_seed,
)
```

### 方法3：导入整个 utils 模块

```python
import folovllm.utils as utils

# 使用时需要加前缀
if utils.is_cuda_available():
    utils.set_random_seed(42)
```

## 各函数使用示例

### 1. `is_cuda_available()` - 检查CUDA是否可用

```python
from folovllm.utils import is_cuda_available

if is_cuda_available():
    print("可以使用 GPU 加速")
    device = "cuda"
else:
    print("使用 CPU")
    device = "cpu"
```

### 2. `get_gpu_memory_info()` - 获取GPU显存信息

```python
from folovllm.utils import get_gpu_memory_info, is_cuda_available

if is_cuda_available():
    mem_info = get_gpu_memory_info()
    print(f"总显存: {mem_info['total_gb']:.2f} GB")
    print(f"已使用: {mem_info['used_gb']:.2f} GB")
    print(f"可用: {mem_info['free_gb']:.2f} GB")
```

**返回值**：字典，包含 `total_gb`、`used_gb`、`free_gb` 等键

### 3. `get_device()` - 获取推荐的设备

```python
from folovllm.utils import get_device

device = get_device()  # 自动选择 "cuda" 或 "cpu"
print(f"使用设备: {device}")
```

### 4. `set_random_seed(seed)` - 设置随机种子

```python
from folovllm.utils import set_random_seed

# 设置种子以保证结果可复现
set_random_seed(42)
```

### 5. `generate_request_id()` - 生成唯一请求ID

```python
from folovllm.utils import generate_request_id

request_id = generate_request_id()
print(f"请求ID: {request_id}")  # 例如: "req_abc123def456"
```

### 6. `move_to_device(data, device)` - 移动数据到指定设备

```python
from folovllm.utils import move_to_device
import torch

tensor = torch.randn(3, 3)
tensor_gpu = move_to_device(tensor, "cuda")  # 移动到GPU
```

### 7. `print_gpu_memory_info()` - 打印GPU显存信息

```python
from folovllm.utils import print_gpu_memory_info

print_gpu_memory_info()
# 输出:
# GPU Memory Info:
#   Total: 24.00 GB
#   Used: 10.50 GB
#   Free: 13.50 GB
```

## 完整使用示例（从 m0_basic_usage.py）

```python
from folovllm.utils import (
    is_cuda_available,
    get_gpu_memory_info,
    get_device,
    set_random_seed,
    generate_request_id,
)

# 1. 设置随机种子
set_random_seed(42)

# 2. 检查设备
print(f"CUDA 可用: {is_cuda_available()}")
device = get_device()
print(f"使用设备: {device}")

# 3. 获取显存信息
if is_cuda_available():
    mem_info = get_gpu_memory_info()
    print(f"GPU 显存: {mem_info['total_gb']:.2f} GB (总)")
    print(f"         {mem_info['free_gb']:.2f} GB (可用)")

# 4. 生成请求ID
request_id = generate_request_id()
print(f"请求ID: {request_id}")
```

## 注意事项

1. **推荐使用方法1**：从 `folovllm.utils` 导入，代码更简洁
2. **GPU函数**：`get_gpu_memory_info()` 和 `print_gpu_memory_info()` 在没有GPU时会抛出异常，使用前先检查 `is_cuda_available()`
3. **类型提示**：这些函数都有完整的类型注解，IDE会提供自动补全和类型检查

## 相关文件
- `folovllm/utils/common.py` - 函数实现
- `folovllm/utils/__init__.py` - 导出定义
- `examples/m0_basic_usage.py` - 使用示例

