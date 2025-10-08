# 快速开始指南

## 环境要求

### 硬件
- **GPU**: NVIDIA GPU with Compute Capability >= 7.0 (推荐 A100/V100/3090)
- **显存**: >= 8GB (Qwen3-0.6B)
- **内存**: >= 16GB

### 软件
- **操作系统**: Linux (Ubuntu 20.04+)
- **Python**: 3.10+
- **CUDA**: 11.8+ or 12.1+
- **PyTorch**: 2.0+

---

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/folovllm.git
cd folovllm
```

### 2. 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n folovllm python=3.10
conda activate folovllm

# 或使用 venv
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# Flash Attention (M4 阶段需要)
pip install flash-attn --no-build-isolation

# GPTQ 支持 (M7 阶段需要)
pip install auto-gptq
```

### 4. 验证安装

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 应该输出: True
```

---

## 下载模型

### Qwen3-0.6B

```bash
# 使用 HuggingFace CLI
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen2.5-0.6B

# 或在 Python 中自动下载
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
"
```

---

## 基础使用

### M1: 基础推理

```python
from folovllm import LLM
from folovllm.sampling_params import SamplingParams

# 初始化模型
llm = LLM(model="Qwen/Qwen3-0.6B")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# 单个请求推理
prompt = "你好，请介绍一下自己"
outputs = llm.generate(prompt, sampling_params)
print(outputs[0].text)
```

### M2: 批量推理

```python
# 批量请求
prompts = [
    "介绍一下北京",
    "什么是人工智能？",
    "解释一下机器学习"
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.text}")
    print("-" * 50)
```

### M3-M7: 高级功能

```python
from folovllm import LLM
from folovllm.config import EngineConfig

# 配置引擎
config = EngineConfig(
    # M3: Paged KV Cache
    enable_paged_kv=True,
    block_size=16,
    
    # M4: Flash Attention
    attention_backend="flash",
    
    # M5: Chunked Prefill
    enable_chunked_prefill=True,
    max_chunk_size=512,
    
    # M6: Prefix Caching
    enable_prefix_caching=True,
    
    # M7: GPTQ
    quantization="gptq",
    quantization_config="gptq_config.json"
)

llm = LLM(
    model="Qwen/Qwen3-0.6B-GPTQ",
    engine_config=config
)

# 使用所有优化的推理
outputs = llm.generate(prompts, sampling_params)
```

---

## 命令行使用

### 基础推理

```bash
python -m folovllm.run \
    --model Qwen/Qwen3-0.6B \
    --prompt "你好，请介绍一下自己" \
    --temperature 0.7 \
    --max-tokens 100
```

### 从文件读取

```bash
# prompts.txt
cat << EOF > prompts.txt
介绍一下北京
什么是人工智能？
解释一下机器学习
EOF

python -m folovllm.run \
    --model Qwen/Qwen3-0.6B \
    --input-file prompts.txt \
    --output-file results.txt
```

### 启用优化

```bash
python -m folovllm.run \
    --model Qwen/Qwen3-0.6B \
    --prompt "你好" \
    --enable-paged-kv \
    --enable-flash-attn \
    --enable-prefix-caching
```

---

## 性能测试

### Benchmark

```bash
# 单请求延迟测试
python tests/benchmark/latency_test.py \
    --model Qwen/Qwen3-0.6B \
    --prompt-len 100 \
    --output-len 50

# 吞吐量测试
python tests/benchmark/throughput_test.py \
    --model Qwen/Qwen3-0.6B \
    --num-requests 100 \
    --concurrent 16

# 不同 milestone 对比
python tests/benchmark/compare_milestones.py \
    --milestones m1,m2,m3,m4 \
    --model Qwen/Qwen3-0.6B
```

### 显存分析

```bash
# 显存占用分析
python tests/benchmark/memory_test.py \
    --model Qwen/Qwen3-0.6B \
    --batch-sizes 1,4,8,16,32
```

---

## 示例项目

### 1. 聊天机器人

```python
from folovllm import LLM
from folovllm.sampling_params import SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")

system_prompt = "你是一个有帮助的AI助手。"
conversation = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    # 构建对话历史
    conversation.append(f"User: {user_input}")
    prompt = system_prompt + "\n" + "\n".join(conversation) + "\nAssistant:"
    
    # 生成回复
    output = llm.generate(
        prompt,
        SamplingParams(temperature=0.7, max_tokens=200)
    )[0]
    
    assistant_reply = output.text
    conversation.append(f"Assistant: {assistant_reply}")
    print(f"Assistant: {assistant_reply}")
```

### 2. Few-shot 学习

```python
# 利用 Prefix Caching 加速 Few-shot
from folovllm import LLM
from folovllm.config import EngineConfig

# 启用前缀缓存
config = EngineConfig(enable_prefix_caching=True)
llm = LLM(model="Qwen/Qwen3-0.6B", engine_config=config)

# Few-shot examples (共享前缀)
examples = """
Q: 天空是什么颜色？
A: 蓝色

Q: 草地是什么颜色？
A: 绿色

Q: 太阳是什么颜色？
A: 黄色
"""

# 多个查询共享 examples
queries = [
    "Q: 雪是什么颜色？",
    "Q: 夜晚是什么颜色？",
    "Q: 火焰是什么颜色？"
]

for query in queries:
    prompt = examples + "\n" + query + "\nA:"
    output = llm.generate(prompt)[0]
    print(f"{query}\nA: {output.text}\n")
    # 第一次会计算 examples，后续会从缓存复用
```

### 3. 批量文本分类

```python
from folovllm import LLM

llm = LLM(model="Qwen/Qwen3-0.6B")

task_prompt = "请判断以下文本的情感（正面/负面）："

texts = [
    "这部电影太精彩了！",
    "服务态度很差，不推荐。",
    "产品质量很好，价格合理。",
    "浪费时间，完全不值得。"
]

prompts = [f"{task_prompt}\n文本: {text}\n情感:" for text in texts]
outputs = llm.generate(prompts)

for text, output in zip(texts, outputs):
    print(f"文本: {text}")
    print(f"情感: {output.text}\n")
```

---

## 开发模式

### 运行测试

```bash
# 所有测试
pytest tests/

# 特定 milestone
pytest tests/unit/test_m1_*.py

# 覆盖率
pytest --cov=folovllm --cov-report=html tests/
```

### 代码格式化

```bash
# Black
black folovllm/ tests/

# isort
isort folovllm/ tests/

# flake8
flake8 folovllm/ tests/
```

### 类型检查

```bash
mypy folovllm/
```

---

## 调试技巧

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from folovllm import LLM
llm = LLM(model="Qwen/Qwen3-0.6B", log_level="DEBUG")
```

### 性能分析

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    outputs = llm.generate(prompt)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 可视化 Attention

```python
from folovllm.utils import visualize_attention

outputs = llm.generate(prompt, return_attention=True)
visualize_attention(outputs[0].attention_weights, save_path="attention.png")
```

---

## 常见问题

### Q1: CUDA out of memory

**解决方案**:
1. 减小 batch size
2. 启用 Paged KV Cache
3. 使用 GPTQ 量化
4. 减小 max_model_len

```python
config = EngineConfig(
    max_model_len=2048,  # 减小最大长度
    enable_paged_kv=True,
    block_size=16
)
```

### Q2: 推理速度慢

**解决方案**:
1. 启用 Flash Attention
2. 增大 batch size (提升吞吐)
3. 使用 Chunked Prefill (降低延迟)

```python
config = EngineConfig(
    attention_backend="flash",
    enable_chunked_prefill=True
)
```

### Q3: 输出质量差

**检查**:
1. Sampling 参数设置
2. 模型权重是否正确加载
3. 量化精度损失

```python
# 使用更保守的参数
sampling_params = SamplingParams(
    temperature=0.7,  # 不要太高
    top_p=0.9,
    repetition_penalty=1.1
)
```

### Q4: 前缀缓存不生效

**检查**:
1. 是否启用了 prefix caching
2. 前缀是否真的相同 (token 级别)
3. 缓存是否被淘汰

```python
# 查看缓存统计
stats = llm.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"Cached blocks: {stats['cached_blocks']}")
```

---

## 性能优化建议

### 延迟优先

```python
config = EngineConfig(
    # Flash Attention
    attention_backend="flash",
    
    # Chunked Prefill (小 chunk)
    enable_chunked_prefill=True,
    max_chunk_size=256,
    
    # Prefix Caching
    enable_prefix_caching=True,
    
    # 小 batch
    max_batch_size=4
)
```

### 吞吐量优先

```python
config = EngineConfig(
    # Paged KV
    enable_paged_kv=True,
    block_size=16,
    
    # 大 batch
    max_batch_size=64,
    
    # Chunked Prefill (大 chunk)
    enable_chunked_prefill=True,
    max_chunk_size=1024,
    
    # Flash Attention
    attention_backend="flash"
)
```

### 显存优先

```python
config = EngineConfig(
    # GPTQ 量化
    quantization="gptq",
    
    # Paged KV
    enable_paged_kv=True,
    block_size=16,
    
    # 限制长度
    max_model_len=2048
)
```

---

## 下一步

1. **学习原理**: 阅读 [docs/learn/](learn/) 中的学习笔记
2. **查看示例**: 运行 [examples/](../examples/) 中的代码
3. **性能测试**: 使用 [tests/benchmark/](../tests/benchmark/) 脚本
4. **贡献代码**: 查看 [开发计划](development_plan.md)

---

## 资源链接

- [开发计划](development_plan.md)
- [技术路线图](roadmap.md)
- [面试准备](interview_guide.md)
- [技术对比](technical_comparison.md)
- [里程碑检查清单](milestone_checklist.md)

---

**祝你使用愉快！如有问题，请查看文档或提交 Issue。**

