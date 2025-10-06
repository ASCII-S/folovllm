# 环境设置指南

本文档介绍如何设置 FoloVLLM 的开发环境。

---

## 🚀 快速开始

### Linux / macOS

```bash
# 一键设置环境（推荐）
bash setup_env.sh

# 后续使用：激活虚拟环境
source activate.sh
# 或
source venv/bin/activate
```

### Windows

```batch
# 一键设置环境（推荐）
setup_env.bat

# 后续使用：激活虚拟环境
activate.bat
# 或
venv\Scripts\activate.bat
```

---

## 📋 详细步骤

### 前置要求

- **Python 3.10+** (推荐 3.10 或 3.11)
- **CUDA 11.8+** (可选，用于 GPU 加速)
- **Git** (用于克隆代码)

### 1. 克隆项目

```bash
git clone https://github.com/your-org/folovllm.git
cd folovllm
```

### 2. 运行自动设置脚本

#### Linux / macOS

```bash
bash setup_env.sh
```

脚本会自动完成：
- ✅ 检查 Python 版本（>= 3.10）
- ✅ 创建虚拟环境 `venv/`
- ✅ 激活虚拟环境
- ✅ 升级 pip
- ✅ 安装所有依赖
- ✅ 安装项目（可编辑模式）
- ✅ 验证安装
- ✅ 创建快捷激活脚本

#### Windows

```batch
setup_env.bat
```

功能与 Linux 版本相同。

### 3. 验证安装

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate.bat  # Windows

# 运行示例
python examples/m0_basic_usage.py

# 运行测试
pytest tests/unit/test_m0_*.py -v
```

---

## 🔧 手动设置（可选）

如果自动脚本遇到问题，可以手动设置：

### 1. 创建虚拟环境

```bash
python3 -m venv venv
```

### 2. 激活虚拟环境

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```batch
venv\Scripts\activate.bat
```

### 3. 升级 pip

```bash
pip install --upgrade pip
```

### 4. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 安装项目（可编辑模式）
pip install -e .
```

### 5. 安装可选依赖（根据需要）

```bash
# Flash Attention 2 (需要 CUDA)
pip install flash-attn --no-build-isolation

# AutoGPTQ (用于量化)
pip install auto-gptq optimum
```

---

## 📦 依赖说明

### 核心依赖

- **PyTorch** (>= 2.0.0) - 深度学习框架
- **Transformers** (>= 4.36.0) - HuggingFace 模型库
- **Tokenizers** (>= 0.15.0) - 快速分词器

### 开发依赖

- **pytest** - 测试框架
- **pytest-cov** - 代码覆盖率
- **black** - 代码格式化
- **flake8** - 代码检查

### 可选依赖

- **flash-attn** - Flash Attention 2（需要 CUDA，M4 需要）
- **auto-gptq** - GPTQ 量化（M7 需要）

---

## ❓ 常见问题

### Q: Python 版本不符合要求怎么办？

**A:** 安装 Python 3.10 或更高版本：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv

# macOS (使用 Homebrew)
brew install python@3.10

# Windows
# 从 https://www.python.org/downloads/ 下载安装
```

### Q: CUDA 不可用怎么办？

**A:** 
- **如果有 GPU**: 安装对应版本的 CUDA Toolkit (11.8+) 和 cuDNN
- **如果没有 GPU**: 项目可以在 CPU 上运行，但速度较慢

检查 CUDA：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: 依赖安装失败怎么办？

**A:** 
1. 确保 pip 是最新版本：`pip install --upgrade pip`
2. 如果是网络问题，使用国内镜像：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. 如果是编译问题（如 flash-attn），跳过可选依赖

### Q: Flash Attention 安装失败？

**A:** Flash Attention 需要：
- CUDA 11.8+
- GCC 7+
- 约 10-20 分钟编译时间

如果不需要，可以跳过。项目会使用朴素的 attention 实现。

### Q: 如何切换 Python 版本？

**A:** 
```bash
# 删除旧的虚拟环境
rm -rf venv

# 使用指定版本创建
python3.10 -m venv venv

# 重新运行设置脚本
bash setup_env.sh
```

### Q: 如何更新依赖？

**A:** 
```bash
# 激活虚拟环境
source venv/bin/activate

# 更新所有依赖
pip install -r requirements.txt --upgrade

# 重新安装项目
pip install -e .
```

---

## 🧪 验证环境

运行以下命令验证环境设置正确：

```bash
# 激活虚拟环境
source venv/bin/activate

# 检查 Python 版本
python --version

# 检查包导入
python -c "import folovllm; print('✓ FoloVLLM 安装成功')"

# 检查 PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"

# 检查 CUDA
python -c "import torch; print('✓ CUDA 可用' if torch.cuda.is_available() else '✗ CUDA 不可用')"

# 运行测试
pytest tests/unit/test_m0_*.py -v
```

预期输出：
```
✓ FoloVLLM 安装成功
✓ PyTorch 2.x.x
✓ CUDA 可用
====== 42 passed in 6.72s ======
```

---

## 🔄 日常使用

### 激活虚拟环境

**快捷方式（推荐）:**
```bash
source activate.sh     # Linux/macOS
activate.bat          # Windows
```

**标准方式:**
```bash
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate.bat    # Windows
```

### 退出虚拟环境

```bash
deactivate
```

### 运行示例

```bash
python examples/m0_basic_usage.py
```

### 运行测试

```bash
# 所有测试
pytest tests/ -v

# 特定 milestone 测试
pytest tests/unit/test_m0_*.py -v

# 带覆盖率
pytest tests/unit/test_m0_*.py --cov=folovllm
```

---

## 📚 相关文档

- [项目 README](README.md) - 项目概述
- [开发计划](docs/development_plan.md) - 完整开发路线
- [M0 开发日志](docs/dev/milestone_0.md) - 第一阶段实现细节
- [贡献指南](CONTRIBUTING.md) - 如何贡献代码

---

## 💡 提示

1. **使用虚拟环境**: 始终在虚拟环境中工作，避免污染系统 Python
2. **检查 CUDA**: 如果有 GPU，确保 CUDA 可用以获得最佳性能
3. **更新依赖**: 定期运行 `pip install -r requirements.txt --upgrade`
4. **运行测试**: 每次修改代码后运行测试确保没有破坏功能

---

**需要帮助？** 查看 [常见问题](#-常见问题) 或提交 Issue。

