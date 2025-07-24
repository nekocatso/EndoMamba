#!/usr/bin/env python3
"""
EndoMamba 环境修复脚本
该脚本自动设置正确的环境变量并执行 EndoMamba 代码
"""

import os
import sys
import torch

def setup_environment():
    """设置 EndoMamba 执行环境"""
    # 获取 PyTorch 库路径
    pytorch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    
    # 更新 LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if pytorch_lib_path not in current_ld_path:
        new_ld_path = f"{pytorch_lib_path}:{current_ld_path}" if current_ld_path else pytorch_lib_path
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
        print(f"已更新 LD_LIBRARY_PATH: {new_ld_path}")
    else:
        print("LD_LIBRARY_PATH 已正确设置")
    
    # 添加当前目录到 Python 路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("环境设置完成")

def run_demo():
    """执行 EndoMamba 演示"""
    setup_environment()
    
    # 设置正确的 Python 路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    videomamba_path = os.path.join(project_root, 'videomamba')
    
    if videomamba_path not in sys.path:
        sys.path.insert(0, videomamba_path)
    
    # 切换到正确的目录
    demo_dir = os.path.join(project_root, 'videomamba', 'tests')
    os.chdir(demo_dir)
    
    print(f"正在执行演示，工作目录: {os.getcwd()}")
    print(f"Python 路径包含: {videomamba_path}")
    
    # 执行演示脚本
    exec(open('endomamba_demo.py').read())

if __name__ == "__main__":
    run_demo()
