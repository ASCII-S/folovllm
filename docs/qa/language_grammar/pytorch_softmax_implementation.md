# PyTorch Softmax 底层实现原理

## 实现层次

PyTorch 的 softmax 实现分多层：

```
Python API (torch.nn.functional.softmax)
    ↓
ATen (C++ Tensor Library)
    ↓
Kernel Dispatch (CPU/CUDA/MPS...)
    ↓
具体实现 (CPU: Eigen/MKL, GPU: CUDA Kernel)
```

## 数值稳定的实现

### 朴素实现（会溢出）

```python
# ❌ 不稳定的实现
def naive_softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum()

# 问题：exp(1000) 会溢出为 inf
```

### 稳定实现（减去最大值）

```python
# ✅ PyTorch 实际使用的算法
def stable_softmax(x):
    max_x = x.max()
    exp_x = torch.exp(x - max_x)  # 防止溢出
    return exp_x / exp_x.sum()
```

**为什么有效**：
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
             = exp(x_i - C) / Σ exp(x_j - C)  # C 可以是任意常数
             = exp(x_i - max(x)) / Σ exp(x_j - max(x))  # 选 C = max(x)
```

减去最大值后：
- 最大的 x_i - max(x) = 0，exp(0) = 1（不会溢出）
- 其他 x_i - max(x) < 0，exp(负数) ∈ (0, 1)（不会溢出）

## CPU 实现（C++）

PyTorch CPU 版本使用 **向量化操作**：

```cpp
// 简化的 PyTorch ATen CPU 实现
template <typename scalar_t>
void softmax_kernel(
    scalar_t* output,
    const scalar_t* input,
    int64_t dim_size,
    int64_t outer_size,
    int64_t inner_size
) {
    for (int64_t i = 0; i < outer_size; i++) {
        for (int64_t k = 0; k < inner_size; k++) {
            // Step 1: 找最大值
            scalar_t max_val = input[i * dim_size * inner_size + k];
            for (int64_t j = 0; j < dim_size; j++) {
                int64_t idx = (i * dim_size + j) * inner_size + k;
                max_val = std::max(max_val, input[idx]);
            }
            
            // Step 2: 计算 exp(x - max) 和 sum
            scalar_t sum_exp = 0;
            for (int64_t j = 0; j < dim_size; j++) {
                int64_t idx = (i * dim_size + j) * inner_size + k;
                output[idx] = std::exp(input[idx] - max_val);
                sum_exp += output[idx];
            }
            
            // Step 3: 归一化
            for (int64_t j = 0; j < dim_size; j++) {
                int64_t idx = (i * dim_size + j) * inner_size + k;
                output[idx] /= sum_exp;
            }
        }
    }
}
```

**优化**：
- 使用 SIMD 指令（AVX/AVX512）向量化
- 利用 Intel MKL 或 Eigen 库加速
- 多线程并行处理

## GPU 实现（CUDA）

PyTorch GPU 使用 **高度优化的 CUDA kernel**：

### 朴素 CUDA 实现

```cuda
// 简化版本，实际 PyTorch 更复杂
template <typename T>
__global__ void softmax_kernel(T* output, const T* input, int dim_size) {
    int idx = blockIdx.x;  // 每个 block 处理一行
    
    // Shared memory for reduction
    extern __shared__ float shared[];
    
    // Step 1: 找最大值（parallel reduction）
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[idx * dim_size + i]);
    }
    shared[threadIdx.x] = thread_max;
    __syncthreads();
    
    // Reduction: 找到整行的最大值
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = shared[0];
    
    // Step 2: 计算 exp 和 sum（parallel reduction）
    float thread_sum = 0;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(input[idx * dim_size + i] - max_val);
        output[idx * dim_size + i] = val;
        thread_sum += val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduction: 求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_exp = shared[0];
    
    // Step 3: 归一化
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        output[idx * dim_size + i] /= sum_exp;
    }
}
```

### PyTorch 实际优化技巧

1. **Online Softmax**（一趟遍历）
   - 传统方法需要 3 次遍历（max, sum, normalize）
   - Online 算法只需 1-2 次遍历
   
2. **Warp-level primitives**
   - 使用 `__shfl_down_sync` 等指令加速 reduction
   - 充分利用 GPU warp 内并行

3. **Memory Coalescing**
   - 优化内存访问模式
   - 连续线程访问连续内存

4. **Kernel Fusion**
   - 与前后操作融合（如 scale + mask + softmax）
   - 减少内存读写

## Flash Attention 中的 Softmax

在 Flash Attention（M4 会用到）中，softmax 与 attention 计算融合：

```python
# 传统方式（3次 HBM 访问）
S = Q @ K.T               # 写 S 到 HBM
P = softmax(S)            # 读 S，写 P 到 HBM
O = P @ V                 # 读 P，写 O 到 HBM

# Flash Attention（1次 HBM 访问）
# Softmax 在 SRAM 中完成，不写回 HBM
for block in range(num_blocks):
    # 所有计算在片上完成
    S_block = Q_block @ K_block.T
    P_block = softmax(S_block)  # 在 SRAM 中
    O_block += P_block @ V_block
```

## 数据类型处理

### 混合精度计算

```cpp
// PyTorch 对 float16/bfloat16 的处理
if (input.dtype == torch.float16 || input.dtype == torch.bfloat16) {
    // Step 1: 转换为 float32
    auto input_fp32 = input.to(torch.float32);
    
    // Step 2: 计算 softmax（float32）
    auto output_fp32 = softmax_impl(input_fp32);
    
    // Step 3: 转回原类型
    return output_fp32.to(input.dtype);
}
```

**为什么要转换**：
- `exp()` 在 float16 容易溢出
- Softmax 要求精确的归一化（和为 1.0）

## 源码位置

在 PyTorch 源码中的位置：

```
pytorch/
├── aten/src/ATen/native/
│   ├── SoftMax.cpp              # CPU 实现调度
│   └── cuda/SoftMax.cu          # CUDA 实现
├── aten/src/ATen/native/cpu/
│   └── SoftMaxKernel.cpp        # CPU 具体实现
└── torch/nn/functional.py       # Python API
```

## 实际性能对比

以 `[1024, 4096]` tensor 为例：

| 实现方式         | CPU (ms) | GPU (ms) |
| ---------------- | -------- | -------- |
| 朴素 Python 循环 | ~500     | -        |
| NumPy            | ~50      | -        |
| PyTorch CPU      | ~5       | -        |
| PyTorch CUDA     | -        | ~0.1     |
| Fused Kernel     | -        | ~0.05    |

GPU 相比 CPU 快 **50-100倍**。

## 在 FoloVLLM 中的影响

### M1（当前）
```python
# 使用 PyTorch 自带的 softmax
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
```
- 单独的 softmax kernel
- 需要读写 HBM

### M4（Flash Attention）
```python
# Softmax 融合在 attention kernel 中
output = flash_attn_func(query, key, value)
# softmax 在 SRAM 中完成，不单独调用
```
- Softmax 与 matmul 融合
- 显著减少内存访问
- 预期加速 2-4x

## 关键要点总结

1. **数值稳定性**：减去最大值防止溢出
2. **并行化**：CPU 用 SIMD，GPU 用 thread block
3. **内存优化**：减少 HBM 访问次数
4. **精度控制**：float16 输入时用 float32 计算
5. **Kernel fusion**：与相邻操作融合提高性能

## 参考资料

- PyTorch ATen 源码：`pytorch/aten/src/ATen/native/cuda/SoftMax.cu`
- Online Softmax 论文："Online normalizer calculation for softmax"
- Flash Attention 论文：将 softmax 融合在 attention 中

