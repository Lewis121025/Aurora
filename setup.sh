#!/bin/bash
# AURORA 快速设置脚本
# 创建虚拟环境并安装依赖

set -e

echo "🚀 AURORA 快速设置"
echo

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip -q

# 安装依赖
echo "安装 AURORA 及依赖..."
pip install -e ".[ark,api,dev]" -q

echo
echo "✅ 设置完成!"
echo
echo "要激活虚拟环境，运行:"
echo "  source venv/bin/activate"
echo
echo "要测试火山方舟集成，运行:"
echo "  python test_ark_integration.py"
echo
echo "要启动 API 服务，运行:"
echo "  aurora serve --port 8000"
