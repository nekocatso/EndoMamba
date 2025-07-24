#!/bin/bash

# EndoMamba 环境设置脚本
# 该脚本设置正确的环境变量以执行 EndoMamba

# 获取 PyTorch 库路径
PYTORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

# 将 PyTorch 库路径添加到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$PYTORCH_LIB_PATH:$LD_LIBRARY_PATH"

echo "环境变量已设置："
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""
echo "现在您可以执行 EndoMamba 代码了。"
echo "例如：python endomamba_demo.py"
