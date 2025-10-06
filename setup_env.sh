#!/bin/bash

# FoloVLLM 虚拟环境设置脚本
# 用于创建和配置 Python 虚拟环境

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目配置
PROJECT_NAME="folovllm"
PYTHON_VERSION="3.10"
VENV_DIR="venv"

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印标题
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# 检查 Python 版本
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
            else
                print_warning "Python 版本过低 ($PYTHON_VER < 3.10)"
            fi
        fi
    done
    
    print_error "未找到 Python 3.10 或更高版本"
    print_info "请安装 Python 3.10+ 后重试"
    exit 1
}

# 创建虚拟环境
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

# 激活虚拟环境
activate_venv() {
    print_header "激活虚拟环境"
    
    if [ -f "$VENV_DIR/bin/activate" ]; then
        print_info "激活虚拟环境..."
        source $VENV_DIR/bin/activate
        print_success "虚拟环境已激活"
        print_info "Python 路径: $(which python)"
        print_info "Python 版本: $(python --version)"
    else
        print_error "虚拟环境激活脚本不存在"
        exit 1
    fi
}

# 升级 pip
upgrade_pip() {
    print_header "升级 pip"
    
    print_info "升级 pip 到最新版本..."
    python -m pip install --upgrade pip -q
    print_success "pip 升级成功"
    print_info "pip 版本: $(pip --version)"
}

# 安装依赖
install_dependencies() {
    print_header "安装依赖"
    
    # 检查 requirements.txt 是否存在
    if [ ! -f "requirements.txt" ]; then
        print_error "未找到 requirements.txt"
        exit 1
    fi
    
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

# 安装项目（可编辑模式）
install_project() {
    print_header "安装项目"
    
    print_info "以可编辑模式安装项目..."
    pip install -e . -q
    print_success "项目安装成功"
}

# 验证安装
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

# 创建激活脚本
create_activate_script() {
    print_header "创建快捷脚本"
    
    # 创建 activate.sh
    cat > activate.sh << 'EOF'
#!/bin/bash
# 快速激活虚拟环境
source venv/bin/activate
echo "✓ FoloVLLM 虚拟环境已激活"
echo "Python: $(python --version)"
echo "位置: $(which python)"
EOF
    
    chmod +x activate.sh
    print_success "创建激活脚本: ./activate.sh"
}

# 打印使用说明
print_usage_info() {
    print_header "设置完成"
    
    echo "虚拟环境已成功设置！"
    echo ""
    echo "使用方法："
    echo ""
    echo "  1. 激活虚拟环境："
    echo "     ${GREEN}source venv/bin/activate${NC}"
    echo "     或"
    echo "     ${GREEN}source activate.sh${NC}"
    echo ""
    echo "  2. 运行示例："
    echo "     ${GREEN}python examples/m0_basic_usage.py${NC}"
    echo ""
    echo "  3. 运行测试："
    echo "     ${GREEN}pytest tests/unit/test_m0_*.py -v${NC}"
    echo ""
    echo "  4. 退出虚拟环境："
    echo "     ${GREEN}deactivate${NC}"
    echo ""
    print_success "祝您使用愉快！"
}

# 主函数
main() {
    print_header "FoloVLLM 虚拟环境设置"
    
    print_info "项目目录: $(pwd)"
    echo ""
    
    # 执行各个步骤
    check_python_version
    create_venv
    activate_venv
    upgrade_pip
    install_dependencies
    install_project
    verify_installation
    create_activate_script
    print_usage_info
}

# 运行主函数
main

