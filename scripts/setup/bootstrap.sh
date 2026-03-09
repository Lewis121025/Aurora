#!/bin/bash
# AURORA 快速设置脚本
# 创建虚拟环境并安装依赖

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

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
echo "安装 AURORA 本地开发依赖..."
pip install -e ".[api,bailian,dev]" -q

echo
echo "✅ 设置完成!"
echo
echo "要激活虚拟环境，运行:"
echo "  source venv/bin/activate"
echo
echo "默认已包含百炼依赖。"
echo "如需接入火山方舟，额外运行:"
echo "  pip install -e '.[ark]'"
echo
echo "要启动 API 服务，运行:"
echo "  aurora serve --port 8000"
