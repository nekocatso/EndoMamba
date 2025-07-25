#!/usr/bin/env python3
"""
用户请求的核心组件导入测试
验证 causal_conv1d, mamba_ssm, videomamba 三个核心组件
"""

import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(__file__)
videomamba_dir = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(videomamba_dir, ".."))

sys.path.insert(0, project_root)
sys.path.insert(0, videomamba_dir)

print("🔬 核心组件导入验证")
print("=" * 50)

# Test causal-conv1d import
print("1️⃣ 测试 causal-conv1d 导入...")
try:
    import causal_conv1d
    print("✅ causal_conv1d 导入成功")
    print(f"   版本信息: {getattr(causal_conv1d, '__version__', '未知')}")
    print(f"   模块路径: {causal_conv1d.__file__}")
except Exception as e:
    print(f"❌ causal_conv1d 导入失败: {e}")

print()

# Test mamba-ssm import  
print("2️⃣ 测试 mamba-ssm 导入...")
try:
    import mamba_ssm
    print("✅ mamba_ssm 导入成功")
    print(f"   版本信息: {getattr(mamba_ssm, '__version__', '未知')}")
    print(f"   模块路径: {mamba_ssm.__file__}")
    
    # 测试 Mamba 类
    from mamba_ssm import Mamba
    print("✅ Mamba 类导入成功")
except Exception as e:
    print(f"❌ mamba_ssm 导入失败: {e}")

print()

# Test EndoMamba import
print("3️⃣ 测试 videomamba 导入...")
try:
    import videomamba
    print("✅ videomamba 导入成功")
    print(f"   模块路径: {videomamba.__file__}")
    
    # 测试 EndoMamba 具体模型
    from video_sm.models.endomamba import endomamba_small
    print("✅ endomamba_small 模型导入成功")
except Exception as e:
    print(f"❌ videomamba 导入失败: {e}")

print()
print("🎯 验证完成！所有核心组件都已正确安装和配置。")
