#!/usr/bin/env python3
"""
EndoMamba 核心组件验证脚本
验证所有必要的组件是否正确安装和可导入
"""

import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(__file__)
videomamba_dir = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(videomamba_dir, ".."))

sys.path.insert(0, project_root)
sys.path.insert(0, videomamba_dir)

def test_component_import(component_name, import_statement):
    """测试组件导入"""
    print(f"🔍 测试 {component_name} 导入...")
    try:
        exec(import_statement)
        print(f"✅ {component_name} 导入成功")
        return True
    except ImportError as e:
        print(f"❌ {component_name} 导入失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {component_name} 导入时出现其他错误: {e}")
        return False

def main():
    """主验证函数"""
    print("=" * 60)
    print("🚀 EndoMamba 核心组件验证")
    print("=" * 60)
    
    # 要测试的组件列表
    components = [
        ("causal-conv1d", "import causal_conv1d"),
        ("mamba-ssm", "import mamba_ssm"),
        ("videomamba", "import videomamba"),
        ("torch", "import torch"),
        ("einops", "import einops"),
        ("timm", "import timm"),
    ]
    
    # 详细的子模块测试
    detailed_tests = [
        ("causal_conv1d.causal_conv1d_fn", "from causal_conv1d import causal_conv1d_fn"),
        ("mamba_ssm.Mamba", "from mamba_ssm import Mamba"),
        ("mamba_ssm.modules", "from mamba_ssm.modules import mamba_simple"),
        ("EndoMamba 模型", "from video_sm.models.endomamba import endomamba_small"),
        ("配置模块", "from config.paths import MODEL_PATH, MODEL_CONFIGS"),
    ]
    
    success_count = 0
    total_count = len(components) + len(detailed_tests)
    
    # 基本组件测试
    print("\n📦 基本组件测试:")
    print("-" * 40)
    for name, import_stmt in components:
        if test_component_import(name, import_stmt):
            success_count += 1
    
    # 详细子模块测试
    print("\n🔧 详细子模块测试:")
    print("-" * 40)
    for name, import_stmt in detailed_tests:
        if test_component_import(name, import_stmt):
            success_count += 1
    
    # 功能性测试
    print("\n⚡ 功能性测试:")
    print("-" * 40)
    try:
        print("🔍 测试 EndoMamba 模型创建...")
        from video_sm.models.endomamba import endomamba_small
        
        # 创建模型但不加载预训练权重
        model = endomamba_small(
            num_classes=7, 
            pretrained=False,  # 不加载预训练权重以加快测试
            return_last_state=True, 
            with_head=True,
        )
        print("✅ EndoMamba 模型创建成功")
        success_count += 1
        total_count += 1
        
    except Exception as e:
        print(f"❌ EndoMamba 模型创建失败: {e}")
        total_count += 1
    
    # GPU 可用性测试
    print("\n🖥️  GPU 可用性测试:")
    print("-" * 40)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用，设备数量: {torch.cuda.device_count()}")
            print(f"   当前设备: {torch.cuda.current_device()}")
            print(f"   设备名称: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA 不可用，将使用 CPU")
    except Exception as e:
        print(f"❌ GPU 测试失败: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 验证总结:")
    print(f"   成功: {success_count}/{total_count}")
    print(f"   成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 所有组件验证通过！EndoMamba 环境设置正确。")
        return 0
    else:
        print("⚠️  部分组件验证失败，请检查安装。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
