# PyTorch Tensor.to() 函数

## 功能

`to()` 是 PyTorch tensor 的多功能转换方法，可以：
1. **转换数据类型**（dtype）
2. **转移设备**（CPU ↔ GPU）
3. **同时转换类型和设备**

## 语法

```python
tensor.to(dtype)              # 转换数据类型
tensor.to(device)             # 转移设备
tensor.to(device, dtype)      # 同时转换
tensor.to(other_tensor)       # 匹配另一个 tensor 的 dtype 和 device
```

## 1. 转换数据类型（dtype）

### 基本用法

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])  # 默认 float32
print(x.dtype)  # torch.float32

# 转换为 float64
y = x.to(torch.float64)
print(y.dtype)  # torch.float64

# 转换为 int32
z = x.to(torch.int32)
print(z)  # tensor([1, 2, 3], dtype=torch.int32)

# 转换为 float16
w = x.to(torch.float16)
print(w.dtype)  # torch.float16
```

### 常用数据类型

| 数据类型   | PyTorch          | 别名           | 字节数 | 范围               |
| ---------- | ---------------- | -------------- | ------ | ------------------ |
| **浮点型** |
| 半精度     | `torch.float16`  | `torch.half`   | 2      | ±65504             |
| 脑浮点     | `torch.bfloat16` | -              | 2      | 更大范围，更低精度 |
| 单精度     | `torch.float32`  | `torch.float`  | 4      | ±3.4e38            |
| 双精度     | `torch.float64`  | `torch.double` | 8      | ±1.8e308           |
| **整数型** |
| 8位        | `torch.int8`     | -              | 1      | -128~127           |
| 16位       | `torch.int16`    | `torch.short`  | 2      | -32768~32767       |
| 32位       | `torch.int32`    | `torch.int`    | 4      | -2e9~2e9           |
| 64位       | `torch.int64`    | `torch.long`   | 8      | -9e18~9e18         |
| **布尔型** |
| 布尔       | `torch.bool`     | -              | 1      | True/False         |

### 在第47行的应用：转换为 float32

```python
# 第47行
hidden_states = new_residual.to(torch.float32)
```

**原因**：RMSNorm 计算需要更高精度
- 输入可能是 `bfloat16` 或 `float16`（省内存）
- 计算时转为 `float32`（保证精度）
- 输出时转回原类型（省内存）

**完整流程**：
```python
# 第41行：保存原始类型
input_dtype = hidden_states.dtype  # 例如 torch.bfloat16

# 第47行：转为 float32 计算
hidden_states = new_residual.to(torch.float32)

# 第54-55行：RMSNorm 计算（float32）
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

# 第58行：转回原始类型
hidden_states = self.weight * hidden_states.to(input_dtype)
```

## 2. 转移设备（device）

### CPU ↔ GPU

```python
# 创建 CPU tensor
x = torch.randn(3, 4)
print(x.device)  # cpu

# 转移到 GPU
x_gpu = x.to('cuda')
print(x_gpu.device)  # cuda:0

# 转移到指定 GPU
x_gpu1 = x.to('cuda:1')
print(x_gpu1.device)  # cuda:1

# 转回 CPU
x_cpu = x_gpu.to('cpu')
print(x_cpu.device)  # cpu
```

### 使用 torch.device 对象

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

### 常见设备字符串

```python
'cpu'           # CPU
'cuda'          # 默认 GPU (cuda:0)
'cuda:0'        # 第一块 GPU
'cuda:1'        # 第二块 GPU
'mps'           # Apple Silicon (M1/M2)
```

## 3. 同时转换类型和设备

```python
# CPU float32 tensor
x = torch.randn(3, 4)

# 转到 GPU 并变成 float16
x = x.to('cuda', torch.float16)
print(x.device)  # cuda:0
print(x.dtype)   # torch.float16

# 或者分开写（效果相同）
x = x.to('cuda').to(torch.float16)
```

## 4. 匹配另一个 tensor

```python
x = torch.randn(3, 4)  # CPU, float32
y = torch.randn(2, 3).cuda().half()  # GPU, float16

# 让 x 匹配 y 的 dtype 和 device
x = x.to(y)
print(x.device)  # cuda:0
print(x.dtype)   # torch.float16
```

## 5. 原地操作 vs 非原地操作

### 非原地（默认）

```python
x = torch.randn(3, 4)
y = x.to(torch.float16)  # 返回新 tensor

print(x.dtype)  # torch.float32 (不变)
print(y.dtype)  # torch.float16
```

### 原地操作（不推荐，易出错）

```python
# to() 没有原地版本，只能重新赋值
x = torch.randn(3, 4)
x = x.to(torch.float16)  # 必须重新赋值
```

## 常见使用场景

### 场景1：混合精度训练

```python
# 模型参数用 float32，计算用 float16
class Model(nn.Module):
    def forward(self, x):
        x = x.to(torch.float16)  # 输入转 fp16
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.to(torch.float32)  # 输出转回 fp32
        return x
```

### 场景2：设备无关的代码

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型和数据都转到同一设备
model = model.to(device)
data = data.to(device)

output = model(data)
```

### 场景3：保存和加载模型

```python
# 保存在 GPU 上的模型
torch.save(model.state_dict(), 'model.pt')

# 加载到 CPU
model = Model()
model.load_state_dict(torch.load('model.pt', map_location='cpu'))

# 或者加载后转移
model.load_state_dict(torch.load('model.pt'))
model = model.to('cpu')
```

### 场景4：推理时的类型转换

```python
# 模型用 float32 训练，推理用 float16 加速
model = model.to(torch.float16)
input_data = input_data.to(torch.float16)

with torch.no_grad():
    output = model(input_data)
```

## 性能考虑

### 1. 类型转换的开销

```python
import time

x = torch.randn(1000, 1000).cuda()

# 测量转换时间
start = time.time()
for _ in range(1000):
    y = x.to(torch.float16)
torch.cuda.synchronize()
print(f"Time: {(time.time() - start) * 1000:.2f} ms")
# 典型结果: ~10-20 ms
```

**开销**：需要复制并转换数据，O(n) 时间复杂度。

### 2. 设备转移的开销

```python
x = torch.randn(1000, 1000)

# CPU -> GPU 转移（慢）
start = time.time()
x_gpu = x.to('cuda')
torch.cuda.synchronize()
print(f"CPU->GPU: {(time.time() - start) * 1000:.2f} ms")
# 典型结果: ~5-10 ms

# GPU -> CPU 转移（更慢）
start = time.time()
x_cpu = x_gpu.to('cpu')
torch.cuda.synchronize()
print(f"GPU->CPU: {(time.time() - start) * 1000:.2f} ms")
# 典型结果: ~10-20 ms
```

**建议**：尽量减少 CPU-GPU 数据传输。

### 3. 避免不必要的转换

```python
# ❌ 低效：重复转换
for i in range(1000):
    x = x.to('cuda')  # 如果已经在 GPU 上，这是浪费
    y = model(x)

# ✅ 高效：只转换一次
x = x.to('cuda')
for i in range(1000):
    y = model(x)
```

### 4. to() 的智能行为

```python
# 如果已经是目标类型/设备，to() 会返回原 tensor（不复制）
x = torch.randn(3, 4).cuda()
y = x.to('cuda')  # 不复制，y 和 x 是同一个 tensor

print(x is y)  # True（同一个对象）
```

## 与其他转换方法的对比

### to() vs cuda()

```python
# 两者等价
x.to('cuda')
x.cuda()

# cuda() 只能转到 GPU
x.cuda()       # 转到默认 GPU
x.cuda(1)      # 转到 GPU 1

# to() 更通用
x.to('cuda')
x.to('cpu')
x.to('mps')
```

### to() vs cpu()

```python
# 两者等价
x.to('cpu')
x.cpu()
```

### to() vs type()

```python
# 两者等价（类型转换）
x.to(torch.float16)
x.type(torch.float16)

# 但 to() 更常用和推荐
```

## 在 FoloVLLM 中的应用

### RMSNorm 中的混合精度

```python
class RMSNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        input_dtype = hidden_states.dtype  # bfloat16
        
        # 转为 float32 计算（高精度）
        hidden_states = hidden_states.to(torch.float32)
        
        # RMSNorm 计算
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        
        # 转回原类型（省内存）
        hidden_states = hidden_states.to(input_dtype)
        
        return hidden_states, residual
```

**原因**：
- `bfloat16` 省内存，但精度低
- `float32` 保证归一化计算的精度
- 输出转回 `bfloat16` 节省后续计算的内存

### 模型加载时的设备转移

```python
# 加载模型到指定设备
def load_model(model_path, device):
    model = Qwen3ForCausalLM.from_pretrained(model_path)
    model = model.to(device)  # 转到 GPU/CPU
    return model

# 使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('Qwen/Qwen3-0.6B', device)
```

## 错误和调试

### 错误1：设备不匹配

```python
x = torch.randn(3, 4)  # CPU
y = torch.randn(3, 4).cuda()  # GPU

z = x + y  # ❌ RuntimeError: Expected all tensors to be on the same device

# 修复
z = x.to('cuda') + y  # ✅
```

### 错误2：类型不兼容

```python
x = torch.randn(3, 4)  # float32
y = torch.randint(0, 10, (3, 4))  # int64

z = x + y  # ⚠️ 可能有警告，结果类型可能不是你想要的

# 修复：显式转换
z = x + y.to(torch.float32)  # ✅
```

### 调试技巧

```python
def debug_tensor(name, tensor):
    print(f"{name}:")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  shape: {tensor.shape}")
    print()

x = torch.randn(3, 4).cuda().half()
debug_tensor("x", x)
# x:
#   dtype: torch.float16
#   device: cuda:0
#   shape: torch.Size([3, 4])
```

## 最佳实践

### 1. 保持一致性

```python
# ✅ 好：所有操作在同一设备、同一类型
device = torch.device('cuda')
dtype = torch.float16

x = torch.randn(3, 4).to(device, dtype)
y = torch.randn(3, 4).to(device, dtype)
z = x + y
```

### 2. 早期转换

```python
# ✅ 在函数开始时就转换
def process(data, device):
    data = data.to(device)  # 只转换一次
    # ... 后续操作
    return result
```

### 3. 使用上下文一致性

```python
# 确保模型和数据在同一设备
def forward(model, data):
    device = next(model.parameters()).device
    data = data.to(device)
    return model(data)
```

### 4. 混合精度的标准模式

```python
# 输入 -> float32 -> 计算 -> 原类型
def precise_operation(x):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    # 高精度计算
    result = some_operation(x)
    return result.to(input_dtype)
```

## 总结

**to() 函数的作用**：
- 转换数据类型（float32, float16, int, etc.）
- 转移设备（CPU ↔ GPU）
- 同时转换类型和设备

**在第47行的应用**：
```python
hidden_states = new_residual.to(torch.float32)
```
- 将输入转为 float32 以保证 RMSNorm 计算精度
- 避免 float16/bfloat16 的数值不稳定

**性能考虑**：
- 转换有开销（复制数据）
- 如果已是目标类型/设备，不会复制
- 尽量减少 CPU-GPU 传输

**最佳实践**：
- 早期转换，避免重复转换
- 保持设备和类型一致性
- 混合精度：计算用 float32，存储用 float16

