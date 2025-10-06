@echo off
REM FoloVLLM 虚拟环境设置脚本 (Windows)
REM 用于创建和配置 Python 虚拟环境

setlocal enabledelayedexpansion

echo ==========================================
echo FoloVLLM 虚拟环境设置 (Windows)
echo ==========================================
echo.

REM 项目配置
set VENV_DIR=venv
set PYTHON_MIN_VERSION=3.10

REM 检查 Python 版本
echo [1/7] 检查 Python 版本...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python
    echo 请安装 Python 3.10+ 并添加到 PATH
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo 找到 Python: %PYTHON_VER%

REM 简单版本检查（检查是否包含 3.1x）
echo %PYTHON_VER% | findstr /R "3\.1[0-9]" >nul
if errorlevel 1 (
    echo %PYTHON_VER% | findstr /R "3\.9" >nul
    if not errorlevel 1 (
        echo [WARNING] Python 版本过低，建议使用 3.10+
    )
)

echo [SUCCESS] Python 版本检查完成
echo.

REM 创建虚拟环境
echo [2/7] 创建虚拟环境...
if exist %VENV_DIR% (
    echo [WARNING] 虚拟环境已存在: %VENV_DIR%
    set /p "choice=是否删除并重新创建？(y/N): "
    if /i "!choice!"=="y" (
        echo 删除旧的虚拟环境...
        rmdir /s /q %VENV_DIR%
    ) else (
        echo 跳过虚拟环境创建
        goto :activate
    )
)

echo 创建虚拟环境: %VENV_DIR%
python -m venv %VENV_DIR%
if errorlevel 1 (
    echo [ERROR] 虚拟环境创建失败
    pause
    exit /b 1
)
echo [SUCCESS] 虚拟环境创建成功
echo.

:activate
REM 激活虚拟环境
echo [3/7] 激活虚拟环境...
call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] 虚拟环境激活失败
    pause
    exit /b 1
)
echo [SUCCESS] 虚拟环境已激活
echo Python 路径: %VENV_DIR%\Scripts\python.exe
echo.

REM 升级 pip
echo [4/7] 升级 pip...
python -m pip install --upgrade pip -q
echo [SUCCESS] pip 升级成功
python -m pip --version
echo.

REM 安装依赖
echo [5/7] 安装依赖...
if not exist requirements.txt (
    echo [ERROR] 未找到 requirements.txt
    pause
    exit /b 1
)

echo 安装基础依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] 依赖安装失败
    pause
    exit /b 1
)
echo [SUCCESS] 基础依赖安装成功
echo.

echo 可选依赖:
echo   1^) Flash Attention 2 (需要 CUDA，编译时间较长)
echo   2^) AutoGPTQ (用于 M7 量化支持)
echo   3^) 跳过可选依赖 (默认)
set /p "opt_choice=请选择 (1/2/3): "

if "%opt_choice%"=="1" (
    echo 安装 Flash Attention 2...
    echo [WARNING] 这可能需要 10-20 分钟...
    pip install flash-attn --no-build-isolation
    echo [SUCCESS] Flash Attention 2 安装成功
) else if "%opt_choice%"=="2" (
    echo 安装 AutoGPTQ...
    pip install auto-gptq optimum
    echo [SUCCESS] AutoGPTQ 安装成功
) else (
    echo 跳过可选依赖
)
echo.

REM 安装项目
echo [6/7] 安装项目...
pip install -e . -q
if errorlevel 1 (
    echo [ERROR] 项目安装失败
    pause
    exit /b 1
)
echo [SUCCESS] 项目安装成功
echo.

REM 验证安装
echo [7/7] 验证安装...
python -c "import folovllm; print('✓ folovllm 导入成功')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
echo.

echo 检查 CUDA 可用性...
python -c "import torch; print('✓ CUDA 可用' if torch.cuda.is_available() else '✗ CUDA 不可用')"
python -c "import torch; torch.cuda.is_available() and print(f'  GPU: {torch.cuda.get_device_name(0)}')"
python -c "import torch; torch.cuda.is_available() and print(f'  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
echo.

echo [SUCCESS] 安装验证通过
echo.

REM 创建激活脚本
echo 创建快捷脚本...
echo @echo off > activate.bat
echo call venv\Scripts\activate.bat >> activate.bat
echo echo ✓ FoloVLLM 虚拟环境已激活 >> activate.bat
echo python --version >> activate.bat
echo [SUCCESS] 创建激活脚本: activate.bat
echo.

REM 打印使用说明
echo ==========================================
echo 设置完成！
echo ==========================================
echo.
echo 使用方法:
echo.
echo   1. 激活虚拟环境:
echo      venv\Scripts\activate.bat
echo      或
echo      activate.bat
echo.
echo   2. 运行示例:
echo      python examples\m0_basic_usage.py
echo.
echo   3. 运行测试:
echo      pytest tests\unit\test_m0_*.py -v
echo.
echo   4. 退出虚拟环境:
echo      deactivate
echo.
echo 祝您使用愉快！
echo.

pause

