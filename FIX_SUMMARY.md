# EndoMamba 运行问题修复总结

## 问题描述

在执行 `python endomamba_demo.py` 时遇到了以下错误：

```
TypeError: 'NoneType' object is not callable
```

主要原因是：
1. `libc10.so: cannot open shared object file` - PyTorch 库路径未正确设置
2. `mamba_inner_fn_no_out_proj` 函数导入失败，返回 `None`

## 解决方案

### 1. 修复库路径问题

**问题原因：** CUDA 模块（`selective_scan_cuda`, `causal_conv1d_cuda`）无法找到 PyTorch 的共享库文件 `libc10.so`。

**解决方法：** 将 PyTorch 库路径添加到 `LD_LIBRARY_PATH` 环境变量中：

```bash
export LD_LIBRARY_PATH="$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
```

### 2. 修复硬编码路径问题

**问题原因：** `mamba_simple.py` 中有硬编码的路径 `/home/tqy/endomamba/videomamba/_mamba/mamba_ssm`，导致导入失败。

**解决方法：** 修改 `videomamba/_mamba/mamba_ssm/modules/mamba_simple.py`：

```python
# 原代码（有问题）
sys.path.append('/home/tqy/endomamba/videomamba/_mamba/mamba_ssm')

# 修复后的代码
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## 提供的便捷工具

### 1. 环境设置脚本 (setup_environment.sh)

```bash
source setup_environment.sh
cd videomamba/tests
python endomamba_demo.py
```

### 2. Python 运行器 (run_endomamba.py)

```bash
python run_endomamba.py
```

### 3. 手动设置（一次性）

```bash
export LD_LIBRARY_PATH="$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
cd videomamba/tests
python endomamba_demo.py
```

## 验证结果

修复后，`endomamba_demo.py` 成功执行，输出：

```
Error_max =  tensor(2.2352e-07, device='cuda:0', grad_fn=<MaxBackward1>)
Error_mean =  tensor(5.7270e-08, device='cuda:0', grad_fn=<MeanBackward0>)
Error_std =  tensor(4.6732e-08, device='cuda:0', grad_fn=<StdBackward0>)

The outputs of recurrent and parallel modes are equivalent.
```

这表明并行模式和递归模式的输出在数值上是等价的（误差在可接受范围内）。

## 文件修改列表

1. `/root/EndoMamba-main/videomamba/_mamba/mamba_ssm/modules/mamba_simple.py` - 修复硬编码路径
2. `/root/EndoMamba-main/setup_environment.sh` - 新增环境设置脚本
3. `/root/EndoMamba-main/run_endomamba.py` - 新增 Python 运行器
4. `/root/EndoMamba-main/README.md` - 更新文档，添加快速开始指南

## 注意事项

- 确保已安装正确版本的 PyTorch (2.4.1+cu121)
- 确保 CUDA 环境正确配置
- 如果还遇到问题，可以尝试重新编译 mamba-ssm 和 causal-conv1d 库
