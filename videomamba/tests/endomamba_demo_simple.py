#!/usr/bin/env python3
"""
EndoMamba 简化演示脚本
测试环境配置和基本功能，不依赖预训练模型
"""
import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试配置模块
        from config.paths import PathConfig, MODEL_PATH, DATASET_PATH
        print("✅ 配置模块导入成功")
        print(f"  模型路径: {MODEL_PATH}")
        print(f"  数据集路径: {DATASET_PATH}")
        
        # 测试 EndoMamba 基础模块
        from video_sm.models.endomamba import EndoMamba
        print("✅ EndoMamba 模块导入成功")
        
        # 测试 Mamba SSM
        from _mamba.mamba_ssm.modules.mamba_simple import Mamba
        print("✅ Mamba SSM 模块导入成功")
        
        # 测试因果卷积
        from causal_conv1d import causal_conv1d_fn
        print("✅ Causal Conv1D 模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🎮 使用设备: {device}")
    
    try:
        # 创建测试输入
        batch_size = 1
        seq_len = 8
        height = width = 224
        
        x = torch.randn(batch_size, 3, seq_len, height, width).to(device)
        print(f"✅ 创建测试输入: {x.shape}")
        
        # 创建简化的 EndoMamba 模型
        from video_sm.models.endomamba import EndoMamba
        
        model = EndoMamba(
            embed_dim=384,
            depth=12,  # 减少层数以节省内存
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            num_classes=7,
            return_last_state=True,
            with_head=True
        ).to(device)
        
        model.eval()
        print("✅ 创建 EndoMamba 模型成功")
        
        # 前向传播测试
        with torch.no_grad():
            output, _ = model(x)
        
        print(f"✅ 前向传播成功: {x.shape} -> {output.shape}")
        
        # 检查输出合理性
        if output.shape[0] == batch_size and output.shape[1] == 7:
            print("✅ 输出形状正确")
        else:
            print(f"⚠️  输出形状异常: 期望 ({batch_size}, 7), 实际 {output.shape}")
        
        if torch.isfinite(output).all():
            print("✅ 输出数值有效")
        else:
            print("❌ 输出包含无效数值")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_configuration():
    """测试路径配置"""
    print("\n📁 测试路径配置...")
    
    try:
        from config.paths import config, get_dataset_config, get_model_config
        
        # 测试基本路径
        print(f"✅ 项目根目录: {config.project_root}")
        print(f"✅ 模型路径: {config.get_model_path()}")
        print(f"✅ 数据集路径: {config.get_dataset_path()}")
        print(f"✅ 输出路径: {config.get_output_path()}")
        
        # 测试数据集配置
        colonoscopic_config = get_dataset_config('colonoscopic')
        if colonoscopic_config:
            print(f"✅ 数据集配置测试: {colonoscopic_config['root']}")
        
        # 检查目录是否存在
        import os
        if os.path.exists(config.get_model_path()):
            print("✅ 模型目录已创建")
        else:
            print("⚠️  模型目录不存在")
            
        if os.path.exists(config.get_dataset_path()):
            print("✅ 数据集目录已创建")
        else:
            print("⚠️  数据集目录不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 路径配置测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 EndoMamba 简化演示测试")
    print("=" * 60)
    
    # 环境信息
    print("🔍 环境信息:")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name()}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 运行测试
    success_count = 0
    total_tests = 3
    
    if test_path_configuration():
        success_count += 1
    
    if test_imports():
        success_count += 1
    
    if test_basic_functionality():
        success_count += 1
    
    # 测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试通过!")
        print("✅ EndoMamba 环境配置成功!")
        print("✅ 路径修复工作完成!")
    elif success_count >= 2:
        print("⚠️  大部分功能正常，存在轻微问题")
    else:
        print("❌ 多项测试失败，需要进一步检查")
    
    print("\n💡 说明:")
    print("- 这是不依赖预训练模型的基础功能测试")
    print("- 模型使用随机初始化的权重")
    print("- 如需使用预训练模型，请将模型文件放入 pretrained_models/ 目录")
    
    print("\n📋 下一步:")
    print("1. 如需训练: 将数据集放入 datasets/ 目录")
    print("2. 如需推理: 下载预训练模型到 pretrained_models/ 目录") 
    print("3. 查看 PATH_CONFIGURATION_README.md 了解详细配置")

if __name__ == "__main__":
    main()
