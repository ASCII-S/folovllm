# 贡献指南

感谢你对 FoloVLLM 项目的关注！

## 📋 开发流程

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/folovllm.git
cd folovllm

# 创建虚拟环境
conda create -n folovllm python=3.10
conda activate folovllm

# 安装依赖（开发模式）
pip install -e ".[dev]"
```

### 2. 开发规范

#### 代码风格

- **格式化**: Black (line-length=100)
- **导入排序**: isort
- **代码检查**: flake8
- **类型标注**: mypy

运行格式化:
```bash
# 格式化所有代码
make format

# 或手动运行
black folovllm/ tests/ --line-length 100
isort folovllm/ tests/
```

#### 命名规范

- **文件**: `snake_case.py`
- **类**: `PascalCase`
- **函数/变量**: `snake_case`
- **常量**: `UPPER_CASE`

#### 文档字符串

使用 Google 风格:

```python
def function(arg1: int, arg2: str) -> bool:
    """One line summary.
    
    More detailed description if needed.
    
    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When something goes wrong.
    """
    pass
```

### 3. 测试要求

#### 单元测试

- 覆盖率 > 80%
- 测试文件: `tests/unit/test_*.py`
- 命名: `test_<functionality>`

```python
# tests/unit/test_scheduler.py
import pytest
from folovllm.core.scheduler import Scheduler

def test_scheduler_basic():
    scheduler = Scheduler()
    # ... test code
    
def test_scheduler_with_requests():
    # ... test code
```

#### 集成测试

- 端到端验证
- 测试文件: `tests/integration/test_*.py`

```python
# tests/integration/test_e2e.py
def test_basic_inference():
    from folovllm import LLM
    llm = LLM(model="Qwen/Qwen2.5-0.6B")
    output = llm.generate("Hello")
    assert len(output) > 0
```

#### 运行测试

```bash
# 所有测试
make test

# 单个文件
pytest tests/unit/test_scheduler.py

# 覆盖率报告
make coverage
```

### 4. 提交规范

#### Commit Message

遵循 [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**:
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

**示例**:
```
feat(scheduler): implement continuous batching

- Add request queue management
- Implement dynamic batch assembly
- Add preemption support

Closes #123
```

#### Branch 命名

- `milestone-X`: 功能开发
- `fix-<issue>`: Bug 修复
- `docs-<topic>`: 文档更新

### 5. Pull Request

#### PR 标题

```
[MX] Feature: Brief description
```

例如: `[M2] Feature: Implement continuous batching scheduler`

#### PR 描述模板

```markdown
## 变更说明
简要描述本 PR 的变更内容

## Milestone
- [ ] M1: 基础推理
- [x] M2: 连续批处理
- [ ] ...

## 变更类型
- [ ] 新功能
- [ ] Bug 修复
- [ ] 重构
- [ ] 文档
- [ ] 测试

## 检查清单
- [ ] 代码通过所有测试
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 代码符合规范
- [ ] 性能测试通过（如适用）

## 测试说明
如何测试本 PR 的变更

## 相关 Issue
Closes #123
```

---

## 🎯 开发 Milestone

### 当前阶段

请查看 [开发计划](docs/development_plan.md) 了解当前进度。

### 认领任务

1. 查看 [Issues](../../issues)
2. 评论表示认领
3. Fork 项目
4. 创建分支
5. 开发并提交 PR

---

## 📝 文档贡献

### 学习笔记

位置: `docs/learn/`

每个 Milestone 需要包含:
- 技术原理讲解
- 核心算法/数据结构
- 实现要点
- 面试问题 (至少 5 个)
- 参考资料

### 开发日志

位置: `docs/dev/`

每个 Milestone 需要包含:
- 功能清单
- 实现细节
- 遇到的问题和解决方案
- 代码结构说明
- 下一阶段接口预留

### API 文档

使用 docstring，自动生成:
```bash
make docs
```

---

## 🔍 Code Review 清单

### 功能性
- [ ] 实现了所有需求功能
- [ ] 边界条件处理正确
- [ ] 错误处理完善

### 代码质量
- [ ] 命名清晰有意义
- [ ] 逻辑简洁易懂
- [ ] 避免重复代码
- [ ] 合理的抽象层次

### 测试
- [ ] 单元测试充分
- [ ] 集成测试覆盖
- [ ] 性能测试（如需要）

### 文档
- [ ] Docstring 完整
- [ ] 复杂逻辑有注释
- [ ] README 更新（如需要）

### 性能
- [ ] 无明显性能问题
- [ ] 内存使用合理
- [ ] 避免不必要的计算

---

## 🚨 常见问题

### Q: 如何调试？

```python
# 设置日志级别
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用 pdb
import pdb; pdb.set_trace()

# 或使用 IPython
from IPython import embed; embed()
```

### Q: 如何添加新模型？

1. 在 `folovllm/model/` 添加模型文件
2. 实现统一的模型接口
3. 在 `model_loader.py` 注册模型
4. 添加测试

### Q: 如何添加新的 Attention Backend？

1. 在 `folovllm/attention/backends/` 添加实现
2. 继承 `AttentionBackend` 基类
3. 实现 `forward()` 方法
4. 在 `backends/__init__.py` 注册

### Q: 性能测试失败怎么办？

1. 检查 GPU 型号和驱动
2. 确认 CUDA 版本兼容
3. 查看是否有其他进程占用 GPU
4. 调整性能目标（可能因硬件而异）

---

## 📊 开发工具

### Makefile 命令

```bash
make format      # 格式化代码
make lint        # 代码检查
make test        # 运行测试
make coverage    # 生成覆盖率报告
make docs        # 生成文档
make clean       # 清理临时文件
```

### 推荐 IDE 配置

#### VS Code

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

#### PyCharm

- Black: Settings → Tools → Black
- isort: Settings → Tools → isort
- mypy: Settings → Tools → External Tools

---

## 🤝 社区

### 讨论

- GitHub Discussions: 技术讨论
- Issues: Bug 报告和功能请求

### 行为准则

- 尊重他人
- 建设性反馈
- 友好协作

---

## 📚 学习资源

### 推荐阅读

1. [vLLM 论文](https://arxiv.org/abs/2309.06180)
2. [Flash Attention 论文](https://arxiv.org/abs/2307.08691)
3. [GPTQ 论文](https://arxiv.org/abs/2210.17323)

### 代码参考

- [vLLM 源码](https://github.com/vllm-project/vllm)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

---

## ✅ Checklist

提交前确认:

- [ ] 代码通过 `make lint`
- [ ] 测试通过 `make test`
- [ ] 覆盖率符合要求
- [ ] 文档已更新
- [ ] Commit message 符合规范
- [ ] PR 描述完整

---

感谢你的贡献！🎉

