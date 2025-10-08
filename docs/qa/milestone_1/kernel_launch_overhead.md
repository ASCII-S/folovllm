# Kernel Launch Overhead（内核启动开销）

## 基本概念

### 什么是 Kernel？

在 GPU 编程中，**kernel（内核）** 是指在 GPU 上执行的一个函数/程序。

```cuda
// CUDA kernel 示例
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

在 PyTorch 中，每个操作（如 `matmul`、`add`、`softmax`）都会启动一个或多个 GPU kernel。

### 什么是 Kernel Launch？

**Kernel launch（内核启动）** 是指 CPU 向 GPU 发送指令，让 GPU 执行某个 kernel 的过程。

```python
# PyTorch 中的操作
result = torch.matmul(a, b)  # ← 这会启动一个 matmul kernel

# 等价的流程：
# 1. CPU 准备参数（a, b 的指针，维度等）
# 2. CPU 发送启动命令给 GPU
# 3. GPU 执行 matmul kernel
# 4. GPU 将结果写回内存
```

### 什么是 Overhead？

**Overhead（开销）** 是指启动 kernel 本身所需的时间，包括：

1. **参数准备**：打包 kernel 参数
2. **指令传输**：CPU → GPU 通信
3. **线程调度**：GPU 分配线程块
4. **上下文切换**：GPU 切换不同的 kernel

这些操作需要时间，通常在 **5-20 微秒**。

## 为什么 Overhead 是问题？

### 问题场景

当 kernel 执行时间很短时，overhead 占比很大：

```python
# 小矩阵乘法（执行时间 ~10 微秒）
a = torch.randn(10, 10).cuda()
b = torch.randn(10, 10).cuda()
c = torch.matmul(a, b)  # kernel 执行: 10μs, overhead: 10μs
                        # 总时间: 20μs，50% 是 overhead！

# 大矩阵乘法（执行时间 ~1000 微秒）
a = torch.randn(1000, 1000).cuda()
b = torch.randn(1000, 1000).cuda()
c = torch.matmul(a, b)  # kernel 执行: 1000μs, overhead: 10μs
                        # 总时间: 1010μs，1% 是 overhead
```

### 多次启动的累积影响

```python
# ❌ 糟糕：启动 3 次 kernel
for i in range(3):
    result = torch.matmul(a[i], b[i])  # 3 × 10μs overhead = 30μs

# ✅ 好：启动 1 次 kernel（批处理）
result = torch.matmul(a, b)  # 1 × 10μs overhead = 10μs
```

## 在 FoloVLLM 中的实际应用

### 问题：分离的 QKV 投影

```python
# ❌ 低效：3 次 kernel launch
class Attention(nn.Module):
    def __init__(self, hidden_size):
        self.q_proj = nn.Linear(hidden_size, hidden_size)  # Linear 1
        self.k_proj = nn.Linear(hidden_size, hidden_size)  # Linear 2
        self.v_proj = nn.Linear(hidden_size, hidden_size)  # Linear 3
    
    def forward(self, x):
        q = self.q_proj(x)  # Kernel launch 1 (overhead: ~10μs)
        k = self.k_proj(x)  # Kernel launch 2 (overhead: ~10μs)
        v = self.v_proj(x)  # Kernel launch 3 (overhead: ~10μs)
        # 总 overhead: ~30μs
        return q, k, v
```

**分析**：
- 每个 `nn.Linear` 调用启动一个 GEMM (矩阵乘法) kernel
- 3 次 Linear → 3 次 kernel launch
- 累积 overhead: 30μs

### 解决方案：合并 QKV 投影

```python
# ✅ 高效：1 次 kernel launch
class Attention(nn.Module):
    def __init__(self, hidden_size):
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)  # 合并
    
    def forward(self, x):
        qkv = self.qkv_proj(x)  # Kernel launch 1 (overhead: ~10μs)
        q, k, v = qkv.split([hidden_size, hidden_size, hidden_size], dim=-1)
        # split 是视图操作，几乎零开销
        # 总 overhead: ~10μs
        return q, k, v
```

**收益**：
- Overhead 减少：30μs → 10μs（**减少 67%**）
- 总时间减少：~15-20%（取决于矩阵大小）

### 实际测量

```python
import torch
import time

hidden_size = 896
batch_size = 1
seq_len = 128

x = torch.randn(batch_size, seq_len, hidden_size).cuda()

# 方式1：分离的投影
q_proj = torch.nn.Linear(hidden_size, hidden_size).cuda()
k_proj = torch.nn.Linear(hidden_size, hidden_size).cuda()
v_proj = torch.nn.Linear(hidden_size, hidden_size).cuda()

torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)
torch.cuda.synchronize()
time_separate = time.time() - start
print(f"Separate: {time_separate:.4f}s")

# 方式2：合并的投影
qkv_proj = torch.nn.Linear(hidden_size, 3 * hidden_size).cuda()

torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    qkv = qkv_proj(x)
    q, k, v = qkv.split([hidden_size, hidden_size, hidden_size], dim=-1)
torch.cuda.synchronize()
time_fused = time.time() - start
print(f"Fused: {time_fused:.4f}s")

print(f"Speedup: {time_separate / time_fused:.2f}x")
```

**典型结果**：
```
Separate: 0.0850s  (3000 次 kernel launch)
Fused:    0.0720s  (1000 次 kernel launch)
Speedup:  1.18x     (18% 加速)
```

## Kernel Launch 的详细流程

### CPU 侧（Host）

```python
# 当你调用：
result = torch.matmul(a, b)

# PyTorch 内部流程：
1. 检查输入 tensor 的设备、类型、形状
2. 选择合适的 kernel（cublas、自定义 CUDA kernel 等）
3. 准备 kernel 参数（指针、维度、配置）
4. 将参数复制到 GPU 内存
5. 调用 CUDA driver API 启动 kernel
6. 返回（异步执行，不等待完成）
```

### GPU 侧（Device）

```
1. 接收 kernel 启动命令
2. 分配 SM（Streaming Multiprocessor）
3. 创建线程块（thread blocks）
4. 分配共享内存、寄存器
5. 执行 kernel 代码
6. 写回结果到全局内存
```

### 时间开销分解

| 阶段            | 时间        | 占比     |
| --------------- | ----------- | -------- |
| 参数准备        | 1-3 μs      | ~20%     |
| CPU→GPU 通信    | 2-5 μs      | ~30%     |
| 线程调度        | 3-8 μs      | ~40%     |
| 上下文切换      | 1-2 μs      | ~10%     |
| **总 Overhead** | **7-18 μs** | **100%** |

## 优化策略

### 1. Kernel Fusion（内核融合）

**合并多个操作为一个 kernel**：

```python
# ❌ 多次 launch
x = x + bias      # kernel 1
x = torch.relu(x)  # kernel 2
x = x * scale      # kernel 3
# 3 次 launch，overhead: ~30μs

# ✅ 融合（理想情况）
x = fused_bias_relu_scale(x, bias, scale)  # kernel 1
# 1 次 launch，overhead: ~10μs
```

**PyTorch 示例**：
```python
# torch.nn.functional 中的融合操作
torch.addmm(bias, input, weight)  # bias + input @ weight（融合）
```

### 2. 操作合并

**合并相同类型的操作**：

```python
# ❌ 分离
q = linear_q(x)
k = linear_k(x)
v = linear_v(x)

# ✅ 合并
qkv = linear_qkv(x)
q, k, v = qkv.split(...)
```

### 3. 批处理（Batching）

```python
# ❌ 逐个处理
for i in range(batch_size):
    result[i] = process(data[i])  # batch_size 次 launch

# ✅ 批处理
result = process(data)  # 1 次 launch
```

### 4. 使用 torch.jit（JIT 编译）

```python
@torch.jit.script
def fused_operation(x, y, z):
    # JIT 编译器可能会融合这些操作
    a = x + y
    b = a * z
    return torch.relu(b)
```

### 5. 使用专用库

- **cuBLAS**：高度优化的 BLAS 操作
- **cuDNN**：深度学习专用库
- **Flash Attention**：融合的 attention kernel

## 何时 Overhead 最明显？

### 高 Overhead 场景

1. **小矩阵/小 batch**
   ```python
   # Overhead 占 50%+
   x = torch.randn(10, 10).cuda()
   y = torch.matmul(x, x)
   ```

2. **频繁的小操作**
   ```python
   # 循环中的小操作
   for i in range(1000):
       x = x + 1  # 1000 次 launch！
   ```

3. **Decode 阶段**（生成单个 token）
   ```python
   # seq_len=1，矩阵很小
   hidden_states = model(input_ids)  # [1, 1, hidden_size]
   ```

### 低 Overhead 场景

1. **大矩阵/大 batch**
   ```python
   # Overhead 占 1%
   x = torch.randn(1000, 1000).cuda()
   y = torch.matmul(x, x)
   ```

2. **Prefill 阶段**（处理整个 prompt）
   ```python
   # seq_len=100，矩阵较大
   hidden_states = model(input_ids)  # [1, 100, hidden_size]
   ```

## 在不同 Milestone 中的优化

### M1: 基础优化

```python
# 合并 QKV 投影
self.qkv_proj = nn.Linear(hidden_size, q_size + 2*kv_size)
```

### M3: Paged Attention

```python
# 减少 cache 操作的 kernel 数量
# 使用 slot mapping 一次性写入
```

### M4: Flash Attention

```python
# 极致的 kernel fusion
# 将 QK^T, softmax, @V 融合为单个 kernel
# 减少 HBM 访问，同时减少 kernel launch
```

## 性能分析工具

### 1. PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA]
) as prof:
    output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 2. NVIDIA Nsight Systems

```bash
nsys profile --stats=true python script.py
```

可以看到每个 kernel 的：
- Launch 次数
- 执行时间
- Overhead 时间

### 3. 简单计时

```python
import torch.cuda

torch.cuda.synchronize()  # 等待之前的操作完成
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# 你的代码
output = model(input)
end.record()

torch.cuda.synchronize()  # 等待完成
print(f"Time: {start.elapsed_time(end)} ms")
```

## 总结

**Kernel Launch Overhead**：
- 启动 GPU kernel 本身的时间开销（~10μs）
- 小操作时占比大，大操作时占比小
- 多次启动会累积

**优化方法**：
1. **Kernel Fusion**：合并多个操作
2. **操作合并**：如 QKV 投影合并
3. **批处理**：一次处理多个样本
4. **使用优化库**：cuBLAS, cuDNN, Flash Attention

**在 FoloVLLM 中**：
- M1: 合并 QKV 投影（18% 加速）
- M3: Paged Attention 减少 cache 操作
- M4: Flash Attention 极致融合（2-4x 加速）

**经验法则**：
- 小操作（<100μs）：努力减少 kernel launch
- 大操作（>1ms）：kernel launch overhead 可忽略
- Decode 阶段（小 batch）：overhead 影响大
- Prefill 阶段（大 batch）：overhead 影响小

