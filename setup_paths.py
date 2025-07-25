#!/usr/bin/env python3
"""
EndoMamba 目录结构设置脚本
"""
import os
import json
from pathlib import Path

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        'pretrained_models/endomamba',
        'pretrained_models/videomae', 
        'datasets/endoscopy/Colonoscopic',
        'datasets/endoscopy/LDPolypVideo',
        'datasets/endoscopy/Hyper-Kvasir',
        'datasets/endoscopy/Kvasir-Capsule',
        'datasets/endoscopy/EndoFM',
        'datasets/endoscopy/SUN-SEG',
        'datasets/endoscopy/GLENDA_v1.0',
        'datasets/endoscopy/EndoMapper',
        'datasets/surgery/CholecT45',
        'datasets/surgery/ROBUST-MIS',
        'datasets/surgery/AutoLaparo/AutoLaparo_Task1',
        'datasets/classification/PolypDiag',
        'outputs/pretraining',
        'outputs/segmentation',
        'outputs/classification',
        'outputs/surgical_phase',
        'outputs/logs/wandb',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def create_env_file():
    """创建环境变量文件"""
    current_dir = Path.cwd().absolute()
    env_content = f"""# EndoMamba 环境变量配置
export ENDOMAMBA_MODEL_PATH="{current_dir}/pretrained_models"
export ENDOMAMBA_DATASET_PATH="{current_dir}/datasets"
export ENDOMAMBA_OUTPUT_PATH="{current_dir}/outputs"
export ENDOMAMBA_WANDB_PATH="{current_dir}/outputs/logs/wandb"
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ 创建环境文件: .env")

def create_config_json():
    """创建配置 JSON 文件"""
    current_dir = str(Path.cwd().absolute())
    config = {
        "model_path": f"{current_dir}/pretrained_models",
        "dataset_path": f"{current_dir}/datasets",
        "output_path": f"{current_dir}/outputs",
        "wandb_path": f"{current_dir}/outputs/logs/wandb"
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ 创建配置文件: config.json")

if __name__ == "__main__":
    print("🔧 设置 EndoMamba 目录结构...")
    setup_directories()
    create_env_file()
    create_config_json()
    print("\n🎉 目录设置完成!")
    print("\n📋 后续步骤:")
    print("1. 运行: source .env")
    print("2. 将数据集放入 datasets/ 相应目录")
    print("3. 将预训练模型放入 pretrained_models/ 目录")
    print("4. 开始使用 EndoMamba!")
