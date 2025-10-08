# Milestone 2: 连续批处理 - 技术原理笔记

**学习目标**: 深入理解连续批处理（Continuous Batching）的核心原理和实现技术

---

## 📚 核心概念

### 1. 连续批处理 (Continuous Batching)

#### 1.1 传统批处理的问题

**静态批处理 (Static Batching)**:
```
Batch 1: [Req1, Req2, Req3, Req4]
         ↓
    所有请求同时开始
         ↓
    等待最长请求完成
         ↓
    所有请求同时结束
```

**问题**:
- 短请求完成后仍需等待长请求
- GPU 大量时间处于空闲状态
- 吞吐量受最长请求限制
- 资源利用率低（通常 20-40%）

#### 1.2 连续批处理的解决方案

**迭代级调度 (Iteration-level Scheduling)**:
```
Iteration 1: [Req1, Req2, Req3, Req4] → 处理
Iteration 2: [Req1, Req2, Req5, Req6] → Req3,Req4完成，加入Req5,Req6
Iteration 3: [Req1, Req7, Req8, Req9] → Req2,Req5,Req6完成，加入新请求
...
```

**优势**:
- 完成的请求立即移除
- 新请求立即加入
- 动态维护满载批次
- GPU 利用率提升至 60-80%
- 吞吐量提升 3-5x

### 2. Prefill vs Decode 阶段

#### 2.1 Prefill 阶段（预填充）

**定义**: 处理输入 prompt 的所有 token，生成对应的 KV cache

**特征**:
```python
# 输入: 完整的 prompt tokens
input_tokens = [1, 15, 284, 318, 262, 3139, 286, 4881, 30]  # "What is the capital of France?"

# 输出: 所有位置的 KV cache + 下一个 token 的 logits
kv_cache = generate_kv_for_all_positions(input_tokens)
next_token_logits = model_forward(input_tokens, kv_cache)
```

**计算特性**:
- **并行度高**: 所有 token 位置可以并行计算 attention
- **计算密集**: 大量矩阵乘法运算
- **内存访问**: 主要是权重读取
- **时间复杂度**: O(n²) 其中 n 是 prompt 长度

#### 2.2 Decode 阶段（解码）

**定义**: 基于已有的 KV cache，逐个生成新 token

**特征**:
```python
# 输入: 单个新 token + 已有 KV cache
new_token = 464  # "Paris"
existing_kv_cache = [...] # 之前步骤的 KV cache

# 输出: 更新的 KV cache + 下一个 token 的 logits
updated_kv_cache = append_to_kv_cache(existing_kv_cache, new_token)
next_token_logits = model_forward([new_token], updated_kv_cache)
```

**计算特性**:
- **并行度低**: 只处理一个 token 位置
- **内存密集**: 大量 KV cache 读写
- **计算量小**: 相对较少的矩阵运算
- **时间复杂度**: O(n) 其中 n 是序列长度

#### 2.3 混合批次的挑战

**问题**: 同一批次中同时存在 prefill 和 decode 请求

```python
# 示例批次
batch = {
    "req-1": {"type": "prefill", "tokens": [1,2,3,4,5]},    # 新请求
    "req-2": {"type": "decode", "tokens": [42]},            # 继续生成
    "req-3": {"type": "decode", "tokens": [17]},            # 继续生成
    "req-4": {"type": "prefill", "tokens": [10,11,12]},    # 新请求
}
```

**挑战**:
- Prefill 需要大量计算资源
- Decode 需要大量内存带宽
- 资源需求不匹配导致利用率不均
- 需要精心设计调度策略

### 3. Token 预算管理

#### 3.1 调度约束

**两个关键限制**:
```python
class SchedulerConfig:
    max_num_seqs: int = 256           # 最大并发序列数
    max_num_batched_tokens: int = 2048  # 单次迭代最大 token 数
```

**约束原因**:
- `max_num_seqs`: GPU 内存限制（每个序列需要 KV cache）
- `max_num_batched_tokens`: 计算资源限制（单次前向传播的工作量）

#### 3.2 调度算法

**Token 预算分配**:
```python
def schedule_requests():
    total_tokens = 0
    
    # 1. 接纳新请求（Prefill）
    for request in waiting_queue:
        prompt_len = len(request.prompt_tokens)
        if total_tokens + prompt_len > max_num_batched_tokens:
            break  # 预算不足
        
        admit_request(request)
        total_tokens += prompt_len
    
    # 2. 调度运行中请求（Decode）
    for request in running_requests:
        if total_tokens + 1 > max_num_batched_tokens:
            break  # 预算不足
        
        schedule_decode(request)
        total_tokens += 1
```

**优先级策略**:
1. 新请求优先（避免饥饿）
2. 运行中请求保证（避免中断）
3. FCFS 公平调度

---

## 🏗️ 系统架构

### 1. 调度器架构

#### 1.1 请求队列管理

**队列状态转换**:
```
   add_request()
┌─────────────┐    schedule()    ┌─────────────┐    finish
│   WAITING   │ ──────────────→  │   RUNNING   │ ──────────→ FINISHED
└─────────────┘                  └─────────────┘
       ↑                                │
       │ preempt()                      │ update_from_output()
       └────────────────────────────────┘
```

**队列实现**:
```python
class Scheduler:
    def __init__(self):
        self.waiting: FCFSRequestQueue = FCFSRequestQueue()  # 等待队列
        self.running: List[Request] = []                     # 运行队列
        self.requests: Dict[str, Request] = {}               # 所有请求
        self.finished_req_ids: Set[str] = set()              # 已完成请求
```

#### 1.2 调度决策流程

**核心调度循环**:
```python
def schedule() -> SchedulerOutput:
    # 步骤1: 接纳新请求
    while (len(running) < max_num_seqs and 
           waiting and 
           budget_available()):
        request = waiting.pop()
        running.append(request)
        schedule_prefill(request)
    
    # 步骤2: 调度运行中请求
    for request in running:
        if budget_available():
            schedule_decode(request)
    
    # 步骤3: 构建调度输出
    return SchedulerOutput(...)
```

### 2. 批处理执行架构

#### 2.1 输入批次准备

**不定长序列处理**:
```python
# 原始输入（不定长）
raw_inputs = {
    "req-1": [1, 2, 3, 4, 5],      # 5 tokens
    "req-2": [10, 11],             # 2 tokens  
    "req-3": [20, 21, 22],         # 3 tokens
}

# 填充后的批次
padded_batch = {
    "token_ids": [
        [1, 2, 3, 4, 5],           # req-1: 无需填充
        [10, 11, 0, 0, 0],         # req-2: 填充3个0
        [20, 21, 22, 0, 0],        # req-3: 填充2个0
    ],
    "attention_mask": [
        [1, 1, 1, 1, 1],           # req-1: 全部有效
        [1, 1, 0, 0, 0],           # req-2: 前2个有效
        [1, 1, 1, 0, 0],           # req-3: 前3个有效
    ]
}
```

#### 2.2 KV Cache 管理

**M2 实现策略**:
```python
class ModelRunner:
    def __init__(self):
        # 每个请求独立的 KV cache
        self.request_caches: Dict[str, Any] = {}
    
    def execute_model_batch(self, input_batch):
        results = {}
        
        # 为每个请求单独执行（M2 限制）
        for req_id in input_batch.req_ids:
            # 获取该请求的 cache
            past_key_values = self.request_caches.get(req_id)
            
            # 单独执行模型
            outputs = self.model(
                input_ids=req_tokens,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # 更新该请求的 cache
            self.request_caches[req_id] = outputs.past_key_values
            results[req_id] = outputs.logits[:, -1, :]
        
        return results
```

**M3+ 改进方向**:
- PagedAttention: 真正的批处理 KV cache
- 块级管理: 精确的内存控制
- 共享机制: 前缀缓存支持

### 3. 引擎协调架构

#### 3.1 EngineCore 主循环

**连续批处理核心流程**:
```python
def step() -> Dict[str, RequestOutput]:
    # 1. 调度决策
    scheduler_output = self.scheduler.schedule()
    
    # 2. 准备输入
    input_batch = prepare_inputs_from_scheduler_output(scheduler_output)
    
    # 3. 执行模型
    logits_dict = self.executor.execute_model_batch(input_batch)
    
    # 4. 采样生成
    sampled_tokens = {}
    for req_id, logits in logits_dict.items():
        request = self.scheduler.requests[req_id]
        next_token = self.sampler.sample(logits, request.sampling_params)
        sampled_tokens[req_id] = next_token
    
    # 5. 更新状态
    outputs = self.scheduler.update_from_output(
        scheduler_output, sampled_tokens
    )
    
    # 6. 清理资源
    for req_id in scheduler_output.finished_req_ids:
        self.executor.free_request_cache(req_id)
    
    return outputs
```

---

## 🔧 关键算法

### 1. 动态批次组装算法

#### 1.1 贪心调度算法

```python
def greedy_schedule(waiting_queue, running_requests, constraints):
    """贪心调度算法：在约束下最大化资源利用"""
    
    scheduled_new = []
    scheduled_cached = []
    total_tokens = 0
    
    # 阶段1: 接纳新请求（按 FCFS 顺序）
    while waiting_queue:
        request = waiting_queue.peek()
        required_tokens = len(request.prompt_tokens)
        
        # 检查约束
        if (len(running_requests) + len(scheduled_new) >= max_num_seqs or
            total_tokens + required_tokens > max_num_batched_tokens):
            break
        
        # 接纳请求
        request = waiting_queue.pop()
        scheduled_new.append(request)
        total_tokens += required_tokens
    
    # 阶段2: 调度运行中请求（每个1 token）
    for request in running_requests:
        if total_tokens + 1 <= max_num_batched_tokens:
            scheduled_cached.append(request)
            total_tokens += 1
    
    return scheduled_new, scheduled_cached, total_tokens
```

#### 1.2 负载均衡考虑

**Prefill/Decode 比例平衡**:
```python
def balanced_schedule(waiting_queue, running_requests):
    """平衡 prefill 和 decode 的资源需求"""
    
    # 计算当前 decode 负载
    decode_load = len(running_requests)
    
    # 动态调整 prefill 接纳数量
    max_prefill_tokens = max_num_batched_tokens - decode_load
    
    # 优先保证 decode（避免中断）
    # 剩余预算分配给 prefill
    return schedule_with_budget(max_prefill_tokens)
```

### 2. 停止条件检测算法

#### 2.1 多种停止条件

```python
def check_stop_conditions(sequence, sampling_params):
    """检查序列是否应该停止生成"""
    
    # 1. 长度限制
    if (sampling_params.max_tokens and 
        sequence.get_output_len() >= sampling_params.max_tokens):
        return True, "length"
    
    # 2. EOS token
    if (not sampling_params.ignore_eos and 
        sequence.get_last_token_id() == eos_token_id):
        return True, "stop"
    
    # 3. 自定义停止 token
    if (sampling_params.stop_token_ids and 
        sequence.get_last_token_id() in sampling_params.stop_token_ids):
        return True, "stop"
    
    # 4. 停止字符串（需要解码后检查）
    if sampling_params.stop_strings:
        decoded_text = decode_tokens(sequence.get_output_tokens())
        for stop_str in sampling_params.stop_strings:
            if stop_str in decoded_text:
                return True, "stop"
    
    return False, None
```

### 3. 内存管理算法

#### 3.1 KV Cache 生命周期管理

```python
class KVCacheManager:
    """M2 简化版 KV Cache 管理"""
    
    def __init__(self):
        self.request_caches = {}
    
    def allocate_cache(self, req_id):
        """为新请求分配 cache"""
        self.request_caches[req_id] = None
    
    def update_cache(self, req_id, new_cache):
        """更新请求的 cache"""
        self.request_caches[req_id] = new_cache
    
    def free_cache(self, req_id):
        """释放完成请求的 cache"""
        if req_id in self.request_caches:
            del self.request_caches[req_id]
    
    def get_memory_usage(self):
        """获取内存使用情况（M3+ 需要）"""
        # M2: 无法精确计算
        # M3+: 基于块的精确统计
        return len(self.request_caches)
```

---

## 📊 性能分析

### 1. 理论性能模型

#### 1.1 吞吐量分析

**静态批处理吞吐量**:
```
T_static = B / max(L_1, L_2, ..., L_B)
```
其中：
- B: 批次大小
- L_i: 第 i 个请求的长度

**连续批处理吞吐量**:
```
T_continuous = Σ(tokens_per_iteration) / Σ(time_per_iteration)
```

**理论提升**:
- 最优情况: 5-10x（当请求长度差异很大时）
- 典型情况: 3-5x（实际工作负载）
- 最差情况: 1x（所有请求长度相同）

#### 1.2 延迟分析

**首 Token 延迟 (TTFT)**:
```
TTFT_continuous = TTFT_single + queue_wait_time + batch_overhead
```

**Token 间延迟 (TPOT)**:
```
TPOT_continuous ≈ TPOT_single  # 理想情况下相近
```

**权衡**:
- 吞吐量大幅提升
- 延迟略有增加（可接受）
- 整体效率显著改善

### 2. 资源利用率分析

#### 2.1 GPU 利用率

**计算利用率**:
```python
def compute_utilization():
    prefill_ops = sum(seq_len^2 for prefill_requests)
    decode_ops = sum(seq_len for decode_requests)
    total_ops = prefill_ops + decode_ops
    
    # GPU 峰值算力
    peak_ops = gpu_flops * time_per_iteration
    
    return total_ops / peak_ops
```

**内存带宽利用率**:
```python
def memory_utilization():
    kv_cache_reads = sum(seq_len * hidden_dim for decode_requests)
    weight_reads = model_size * num_requests
    total_memory_ops = kv_cache_reads + weight_reads
    
    # GPU 峰值带宽
    peak_bandwidth = gpu_bandwidth * time_per_iteration
    
    return total_memory_ops / peak_bandwidth
```

#### 2.2 瓶颈分析

**计算瓶颈场景**:
- 大量 prefill 请求
- 长 prompt 处理
- 模型参数量大

**内存瓶颈场景**:
- 大量 decode 请求
- 长序列生成
- KV cache 访问密集

**优化策略**:
- 动态调整 prefill/decode 比例
- 智能批次组装
- 异步处理流水线

---

## 🔮 M3+ 扩展方向

### 1. PagedAttention 原理

#### 1.1 虚拟内存思想

**传统 KV Cache**:
```
Request 1: [K1, V1, K2, V2, K3, V3, ...]  # 连续存储
Request 2: [K1, V1, K2, V2, ...]          # 连续存储
```

**PagedAttention**:
```
Block Pool: [Block0, Block1, Block2, Block3, ...]

Request 1 Block Table: [0, 2, 5, ...]  # 指向物理块
Request 2 Block Table: [1, 3, ...]     # 指向物理块
```

#### 1.2 核心优势

**内存效率**:
- 消除内部碎片
- 支持动态分配
- 内存利用率接近 100%

**共享能力**:
- 前缀共享（多个请求共享相同前缀）
- Copy-on-Write 机制
- 显著减少内存使用

### 2. 抢占和交换机制

#### 2.1 抢占策略

```python
def preemption_policy():
    """内存不足时的抢占策略"""
    
    # 候选抢占请求
    candidates = [req for req in running_requests 
                 if req.can_be_preempted()]
    
    # 抢占策略（多种选择）
    if policy == "LRU":
        return min(candidates, key=lambda r: r.last_access_time)
    elif policy == "shortest_remaining":
        return min(candidates, key=lambda r: r.remaining_tokens)
    elif policy == "lowest_priority":
        return min(candidates, key=lambda r: r.priority)
```

#### 2.2 交换机制

```python
def swap_out_request(request):
    """将请求的 KV cache 交换到 CPU"""
    
    # 1. 复制 KV cache 到 CPU
    cpu_cache = copy_to_cpu(request.kv_cache_blocks)
    
    # 2. 释放 GPU 内存
    free_gpu_blocks(request.kv_cache_blocks)
    
    # 3. 更新请求状态
    request.status = RequestStatus.SWAPPED
    request.cpu_cache = cpu_cache

def swap_in_request(request):
    """将请求的 KV cache 从 CPU 恢复到 GPU"""
    
    # 1. 分配 GPU 块
    gpu_blocks = allocate_gpu_blocks(request.num_blocks)
    
    # 2. 从 CPU 复制数据
    copy_from_cpu(request.cpu_cache, gpu_blocks)
    
    # 3. 更新请求状态
    request.status = RequestStatus.RUNNING
    request.kv_cache_blocks = gpu_blocks
```

---

## 📝 学习要点总结

### 核心概念掌握

1. **连续批处理 vs 静态批处理**
   - 迭代级调度的优势
   - 动态批次维护机制
   - 性能提升原理

2. **Prefill vs Decode**
   - 计算特性差异
   - 资源需求不同
   - 混合批次挑战

3. **调度算法**
   - Token 预算管理
   - 贪心调度策略
   - 负载均衡考虑

4. **系统架构**
   - 调度器设计
   - 批处理执行
   - 引擎协调

### 实现技巧

1. **队列管理**
   - FCFS 公平调度
   - 状态转换管理
   - 抢占恢复机制

2. **批次处理**
   - 不定长序列填充
   - Attention mask 生成
   - KV cache 独立管理

3. **资源优化**
   - 内存生命周期管理
   - 计算资源调度
   - 瓶颈识别和优化

### 扩展方向

1. **PagedAttention**
   - 虚拟内存思想
   - 块级管理机制
   - 前缀共享能力

2. **高级调度**
   - 抢占和交换
   - 优先级调度
   - SLA 感知调度

3. **性能优化**
   - 异步处理
   - 流水线并行
   - 硬件加速

---

**学习建议**:
1. 先理解连续批处理的核心思想
2. 深入分析 Prefill/Decode 的差异
3. 掌握调度算法的设计原理
4. 理解系统架构的协调机制
5. 思考 M3+ 的扩展方向

这些概念和技术是现代 LLM 推理系统的基础，掌握它们对于理解和开发高性能推理框架至关重要。
