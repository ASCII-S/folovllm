# 技术对比与性能指标

## 各阶段优化技术对比

### 内存管理对比

| 方案              | 分配方式       | 碎片化 | 显存利用率 | 共享能力 | 实现复杂度 |
| ----------------- | -------------- | ------ | ---------- | -------- | ---------- |
| **连续分配** (M1) | 预分配最大长度 | 严重   | ~20%       | 无       | 低         |
| **动态扩展**      | 按需增长       | 中等   | ~50%       | 无       | 中         |
| **Paged KV** (M3) | Block 分配     | 最小   | ~100%      | 支持 COW | 高         |

### Attention 实现对比

| Backend             | 计算复杂度 | IO 复杂度 | 空间复杂度 | 最大序列长度 | 性能   |
| ------------------- | ---------- | --------- | ---------- | ------------ | ------ |
| **Naive** (M1)      | O(n²)      | O(n²)     | O(n²)      | ~2K          | 基线   |
| **Paged** (M3)      | O(n²)      | O(n²)     | O(n²)      | ~8K          | 1.0x   |
| **Flash Attn** (M4) | O(n²)      | O(n²/M)   | O(n)       | ~32K+        | 2-4x ↑ |

*M: SRAM block size*

### 批处理策略对比

| 策略                       | 调度单位       | GPU 利用率 | 吞吐量 | 延迟      | 实现难度 |
| -------------------------- | -------------- | ---------- | ------ | --------- | -------- |
| **Static Batch**           | Batch 粒度     | 低 (~30%)  | 基线   | 高 (等待) | 低       |
| **Continuous Batch** (M2)  | Iteration 粒度 | 高 (~70%)  | 3-5x ↑ | 中        | 中       |
| **+ Chunked Prefill** (M5) | Token 粒度     | 高 (~80%)  | 3-5x ↑ | 低        | 高       |

### 量化方案对比

| 方法          | Bits | 量化时间 | 推理速度 | 精度损失 | 显存占用 |
| ------------- | ---- | -------- | -------- | -------- | -------- |
| **FP16**      | 16   | -        | 基线     | 0%       | 基线     |
| **INT8 PTQ**  | 8    | 分钟级   | 1.2-1.5x | <0.5%    | 50% ↓    |
| **GPTQ** (M7) | 4    | 小时级   | 1.3-1.8x | <1%      | 75% ↓    |
| **AWQ**       | 4    | 小时级   | 1.5-2x   | <0.5%    | 75% ↓    |

---

## 性能指标定义

### 延迟指标

#### 1. TTFT (Time to First Token)
**定义**: 从请求发送到第一个 token 返回的时间

**影响因素**:
- Prefill 阶段时间 (主要)
- 排队等待时间
- 网络传输时间

**优化技术**:
- Chunked Prefill (M5): 减少阻塞
- Prefix Caching (M6): 复用前缀
- Flash Attention (M4): 加速 prefill

**目标值**:
- 短 prompt (<100 tokens): <50ms
- 中等 prompt (100-500): <200ms  
- 长 prompt (500-2000): <1s

---

#### 2. TPOT (Time Per Output Token)
**定义**: 平均每个输出 token 的生成时间

**公式**: 
```
TPOT = (总时间 - TTFT) / 输出token数
```

**影响因素**:
- Decode 阶段计算时间
- Attention 效率
- Batch size

**优化技术**:
- Flash Attention (M4): 加速 attention
- Continuous Batching (M2): 提升并行度

**目标值**:
- 单请求: ~10-20ms/token
- Batch (8-16): ~5-10ms/token
- Batch (32+): ~2-5ms/token

---

#### 3. E2E Latency (End-to-End)
**定义**: 整个请求的完成时间

**公式**:
```
E2E = TTFT + TPOT × 输出token数
```

**示例**:
```
Prompt: 100 tokens
Output: 50 tokens
TTFT: 100ms
TPOT: 10ms

E2E = 100 + 10 × 50 = 600ms
```

---

### 吞吐量指标

#### 1. Tokens/s (Token Throughput)
**定义**: 每秒处理的 token 数 (输入+输出)

**公式**:
```
Tokens/s = (∑输入tokens + ∑输出tokens) / 总时间
```

**影响因素**:
- Batch size (正相关)
- 序列长度 (负相关)
- Attention 效率

**目标值** (A100 40GB, Qwen3-0.6B):
- M1 (基础): ~2,000 tokens/s
- M2 (Continuous): ~8,000 tokens/s
- M4 (Flash): ~12,000 tokens/s

---

#### 2. Requests/s (Request Throughput)
**定义**: 每秒完成的请求数

**公式**:
```
Requests/s = 完成请求数 / 总时间
```

**影响因素**:
- 平均序列长度 (负相关)
- Batch 调度效率
- 并发能力

---

### 资源指标

#### 1. GPU Memory Usage
**测量点**:
- 模型权重 (固定)
- KV Cache (动态)
- Activations (临时)

**优化效果**:
- M3 (Paged KV): ~50% ↓
- M7 (GPTQ): ~75% ↓ (权重)

---

#### 2. GPU Utilization
**定义**: GPU 计算资源的使用率

**目标**: >70%

**影响因素**:
- Batch size
- Kernel 效率
- 调度策略

---

## 各阶段性能目标

### M1: 基础离线推理

**Baseline 建立**:
| 指标     | 目标值                      |
| -------- | --------------------------- |
| TTFT     | ~100ms (100 tokens)         |
| TPOT     | ~20ms                       |
| Tokens/s | ~2,000                      |
| GPU Util | ~40%                        |
| Memory   | ~2GB (model) + ~4GB (cache) |

---

### M2: 连续批处理

**相对 M1 提升**:
| 指标     | 提升   | 绝对值 |
| -------- | ------ | ------ |
| TTFT     | -      | ~100ms |
| TPOT     | 2-3x ↓ | ~8ms   |
| Tokens/s | 3-5x ↑ | ~8,000 |
| GPU Util | +30%   | ~70%   |
| Memory   | -      | ~6GB   |

**测试场景**:
- Batch size: 8-16
- 平均序列长度: 512
- Concurrent requests: 32

---

### M3: Paged KV Cache

**相对 M2 提升**:
| 指标      | 提升   | 说明       |
| --------- | ------ | ---------- |
| 吞吐量    | -      | 相同       |
| Memory    | 50% ↓  | 碎片消除   |
| Max Batch | 2x ↑   | 显存省出   |
| Tokens/s  | 1.5x ↑ | 更大 batch |

**显存对比**:
```
场景: 16 requests, 平均 512 tokens

传统方式:
- 预分配: 16 × 2048 × hidden_size = 大量浪费
- 实际用: 16 × 512 × hidden_size
- 利用率: 25%

Paged KV:
- 按需分配: 16 × 32 blocks × 16 tokens/block
- 利用率: ~96% (最后一个 block 可能不满)
```

---

### M4: Flash Attention

**相对 M3 提升**:
| 指标        | 提升     | 绝对值  |
| ----------- | -------- | ------- |
| TTFT        | 20-30% ↓ | ~70ms   |
| TPOT        | 15-25% ↓ | ~6ms    |
| Tokens/s    | 1.5-2x ↑ | ~15,000 |
| Max Seq Len | 4x ↑     | ~32K    |

**性能分解**:
- Prefill: 2-3x 快 (长序列更明显)
- Decode: 1.2-1.5x 快
- 显存: 10x ↓ (attention matrix)

---

### M5: Chunked Prefill

**相对 M4 提升**:
| 指标     | 提升   | 说明              |
| -------- | ------ | ----------------- |
| TTFT     | 显著 ↓ | 取决于 chunk size |
| 总吞吐   | -      | 基本不变          |
| P99 延迟 | 50% ↓  | 减少阻塞          |

**Chunk Size 影响**:
| Chunk Size  | TTFT (decode reqs) | Prefill 吞吐 |
| ----------- | ------------------ | ------------ |
| 全量 (2048) | 高 (~500ms)        | 最高         |
| 512         | 中 (~150ms)        | 高           |
| 256         | 低 (~80ms)         | 中           |
| 128         | 很低 (~40ms)       | 低           |

**最优值**: 通常 256-512

---

### M6: 前缀复用

**相对 M5 提升** (缓存命中时):
| 指标         | 提升    | 场景           |
| ------------ | ------- | -------------- |
| TTFT         | 3-10x ↓ | Few-shot, 对话 |
| Prefill 计算 | 跳过    | 命中部分       |
| 显存         | 复用    | 共享前缀       |

**缓存命中率影响**:
| 命中率 | TTFT 改善 | 典型场景 |
| ------ | --------- | -------- |
| 0%     | 无        | 独立请求 |
| 50%    | 2x ↓      | 部分共享 |
| 80%    | 5x ↓      | Few-shot |
| 95%+   | 10x ↓     | 多轮对话 |

**示例** (1000 token prompt):
```
无缓存: TTFT = 200ms
80% 命中: 
  - 800 tokens 复用 (0ms)
  - 200 tokens 计算 (40ms)
  - TTFT = 40ms (5x 快)
```

---

### M7: GPTQ 量化

**相对 M6 提升**:
| 指标         | 提升     | 说明     |
| ------------ | -------- | -------- |
| Model Memory | 75% ↓    | 权重压缩 |
| TPOT         | 20-50% ↓ | 部分场景 |
| Max Batch    | 2-3x ↑   | 显存节省 |

**精度对比**:
| 模型       | FP16 PPL | GPTQ-4bit PPL | 下降 |
| ---------- | -------- | ------------- | ---- |
| Qwen3-0.6B | 10.5     | 10.7          | 1.9% |

**显存占用**:
```
Qwen3-0.6B (600M 参数):

FP16: 600M × 2 bytes = 1.2GB
GPTQ-4bit: 600M × 0.5 bytes = 0.3GB

节省: 0.9GB (75%)
```

---

## 综合性能对比

### 完整流程性能 (Qwen3-0.6B, A100)

| 阶段 | TTFT   | TPOT | Tokens/s | GPU Util | Memory | 相对 M1 |
| ---- | ------ | ---- | -------- | -------- | ------ | ------- |
| M1   | 100ms  | 20ms | 2K       | 40%      | 6GB    | 1.0x    |
| M2   | 100ms  | 8ms  | 8K       | 70%      | 6GB    | 4.0x    |
| M3   | 100ms  | 8ms  | 12K      | 70%      | 3GB    | 6.0x    |
| M4   | 70ms   | 6ms  | 18K      | 75%      | 2GB    | 9.0x    |
| M5   | 40ms*  | 6ms  | 18K      | 80%      | 2GB    | 9.0x    |
| M6   | 10ms** | 6ms  | 18K      | 80%      | 2GB    | 9.0x    |
| M7   | 10ms** | 5ms  | 22K      | 85%      | 1GB    | 11.0x   |

*混合调度下的 decode 请求
**80%+ 缓存命中

---

## 性能测试方法

### 1. 延迟测试

**单请求测试**:
```python
# 测量 TTFT 和 TPOT
start = time.time()
first_token_time = None
for i, token in enumerate(llm.generate_stream(prompt)):
    if i == 0:
        first_token_time = time.time()
        ttft = first_token_time - start
end = time.time()

tpot = (end - first_token_time) / (num_tokens - 1)
```

**批量测试**:
```python
# 测量不同 batch size 下的延迟
for batch_size in [1, 2, 4, 8, 16, 32]:
    measure_latency(batch_size)
```

---

### 2. 吞吐量测试

**持续负载**:
```python
# 固定 QPS，测量稳态吞吐量
requests = generate_requests(num=1000)
start = time.time()

for req in requests:
    llm.add_request(req)
    
wait_all_complete()
end = time.time()

throughput = total_tokens / (end - start)
```

**最大吞吐**:
```python
# 批量提交，测量峰值
batch = generate_requests(num=100)
throughput = measure_batch_throughput(batch)
```

---

### 3. 显存测试

**峰值显存**:
```python
import torch

torch.cuda.reset_peak_memory_stats()
llm.generate(prompt)
peak_memory = torch.cuda.max_memory_allocated() / 1e9
```

**不同 batch 下显存**:
```python
for batch_size in [1, 8, 16, 32, 64]:
    memory = measure_memory(batch_size)
```

---

### 4. 缓存测试

**命中率**:
```python
# 构造共享前缀的请求
system_prompt = "You are a helpful assistant."
requests = [
    system_prompt + " " + query
    for query in user_queries
]

cache_hits = measure_cache_hits(requests)
hit_rate = cache_hits / len(requests)
```

---

## 对比 vLLM 官方

### 性能差距预期

| 指标     | FoloVLLM | vLLM Official | 差距 |
| -------- | -------- | ------------- | ---- |
| Tokens/s | ~20K     | ~25K          | ~80% |
| TTFT     | ~10ms    | ~8ms          | ~80% |
| Memory   | ~1GB     | ~0.8GB        | ~80% |

**差距原因**:
- 未实现 CUDA kernel 优化
- 未使用 C++ 扩展
- 部分功能简化
- 调度策略差异

**优势**:
- 代码简洁，易理解
- 渐进式，易复现
- 核心原理完整

---

## 性能调优建议

### 1. 针对吞吐量
- 增大 batch size
- 启用 Continuous Batching
- 使用 Paged KV Cache

### 2. 针对延迟  
- 启用 Chunked Prefill
- 使用 Prefix Caching
- 减小 batch size (trade-off)

### 3. 针对显存
- 使用 Paged KV Cache
- 启用 GPTQ 量化
- 调小 block size (更灵活)

### 4. 针对长上下文
- 必须: Flash Attention
- 可选: Sparse Attention
- KV Cache 压缩

---

## 参考基准

### ShareGPT 数据集
**特点**:
- 真实对话数据
- 长度分布广 (10-2000 tokens)
- 多样性高

**指标**:
- Input: avg 200, max 2048
- Output: avg 150, max 1024

### 合成数据集
**构造**:
```python
# 固定长度
inputs = [100, 200, 500, 1000, 2000]
outputs = [50, 100, 200]

# 泊松分布
input_len = np.random.poisson(200)
output_len = np.random.poisson(100)
```

---

**性能测试是验证优化效果的关键，每个阶段都必须有清晰的 baseline 和提升数据！**

