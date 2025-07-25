# EndoMamba 路径配置指南

本文档详细说明了 EndoMamba 项目中所有硬编码路径的类型、用途以及如何正确配置。

## 🚨 硬编码路径问题总结

项目中发现了 **111 个硬编码路径**，分布在 **40 个文件**中。这些硬编码路径导致了环境兼容性问题，需要根据您的实际环境进行配置。

## 📁 路径类型分类

### 1. 系统路径 (System Paths)
用于模块导入和系统级配置的硬编码路径。

#### 问题文件：
- `videomamba/_mamba/mamba_ssm/modules/mamba_simple.py`
- `videomamba/downstream/SurgicalPhase/Surgformer/downstream_phase/smart_test.py`
- `videomamba/downstream/SurgicalPhase/Surgformer/model/surgformer_base.py`

#### 修复方法：
```python
# ❌ 错误的硬编码路径
sys.path.append('/home/tqy/endomamba/videomamba/_mamba/mamba_ssm')

# ✅ 正确的相对路径
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
```

### 2. 模型路径 (Model Paths)
存储预训练模型检查点的路径。

#### 涉及的路径：
- `/data/tqy/endomamba_pretrain/` - 主要预训练模型目录
- `/mnt/tqy/checkpoints/` - 其他模型检查点目录

#### 需要配置的文件：
```
videomamba/video_sm/models/endomamba.py
videomamba/video_sm/models/endomamba_pretrain.py
videomamba/video_sm/models/endomamba_two_heads.py
videomamba/downstream/CVC-12kSegmentation/networks/endomamba_seg_modeling.py
videomamba/downstream/PolypDiagClassification/models/endomamba_classification.py
videomamba/downstream/SurgicalPhase/Surgformer/model/endomamba.py
videomamba/video_sm/models/videomae_v2.py
videomamba/video_sm/models/videomamba_custom.py
videomamba/video_sm/models/videomamba.py
```

#### 建议的目录结构：
```
📁 pretrained_models/
├── endomamba/
│   ├── endomamba_small_b48_seqlen16_withteacher_MIX12/
│   │   └── checkpoint-499.pth
│   ├── endomamba_medium/
│   └── endomamba_large/
├── videomae/
└── other_models/
```

### 3. 数据集路径 (Dataset Paths)
存储训练和测试数据集的路径。

#### 数据集列表：
| 数据集名称 | 硬编码路径 | 用途 |
|-----------|------------|------|
| Colonoscopic | `/mnt/tqy/Colonoscopic/` | 结肠镜检查视频 |
| LDPolypVideo | `/mnt/tqy/LDPolypVideo/` | 息肉检测视频 |
| Hyper-Kvasir | `/mnt/tqy/Hyper-Kvasir/` | 内窥镜图像数据集 |
| Kvasir-Capsule | `/mnt/tqy/Kvasir-Capsule/` | 胶囊内镜数据 |
| CholecT45 | `/mnt/tqy/CholecT45/` | 腹腔镜手术视频 |
| EndoFM | `/mnt/tqy/EndoFM/` | 内窥镜基础模型数据 |
| SUN-SEG | `/mnt/tqy/SUN-SEG/` | 分割标注数据 |
| GLENDA | `/mnt/tqy/GLENDA_v1.0/` | 内窥镜数据集 |
| EndoMapper | `/mnt/tqy/EndoMapper/` | 内窥镜映射数据 |
| ROBUST-MIS | `/mnt/tqy/ROBUST-MIS/` | 微创手术数据 |
| AutoLaparo | `/mnt/tqy/AutoLaparo/AutoLaparo_Task1/` | 腹腔镜手术阶段识别 |
| CVC-ClinicVideoDB | `/mnt/tqy/CVC-ClinicVideoDB/` | 临床视频数据库 |
| PolypDiag | `/data/tqy/PolypDiag/` | 息肉诊断数据 |

#### 建议的数据集目录结构：
```
📁 datasets/
├── endoscopy/
│   ├── Colonoscopic/
│   ├── LDPolypVideo/
│   ├── Hyper-Kvasir/
│   ├── Kvasir-Capsule/
│   ├── EndoFM/
│   ├── SUN-SEG/
│   ├── GLENDA_v1.0/
│   ├── EndoMapper/
│   └── CVC-ClinicVideoDB/
├── surgery/
│   ├── CholecT45/
│   ├── ROBUST-MIS/
│   └── AutoLaparo/
└── classification/
    └── PolypDiag/
```

### 4. 输出路径 (Output Paths)
存储训练输出、日志和结果的路径。

#### 硬编码的输出路径：
- `/mnt/tqy/out/` - 主要输出目录
- `/data/tqy/out/` - 备用输出目录
- `/mnt/tqy/wandb/` - Weights & Biases 日志目录
- `/home/tqy/out/` - 用户特定输出目录

#### 建议的输出目录结构：
```
📁 outputs/
├── pretraining/
│   └── endomamba_pretrain/
├── segmentation/
│   └── EndoMamba_NF8_s1/
├── classification/
│   └── EndoMamba_NF32_s3/
├── surgical_phase/
│   └── AutoLaparo/
└── logs/
    └── wandb/
```

## 🔧 配置解决方案

### 方案 1: 环境变量配置 (推荐)

创建环境配置文件 `config/paths.py`：

```python
import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 环境变量配置
def get_model_path():
    """获取模型路径"""
    return os.environ.get('ENDOMAMBA_MODEL_PATH', 
                         str(PROJECT_ROOT / 'pretrained_models'))

def get_dataset_path():
    """获取数据集路径"""
    return os.environ.get('ENDOMAMBA_DATASET_PATH', 
                         str(PROJECT_ROOT / 'datasets'))

def get_output_path():
    """获取输出路径"""
    return os.environ.get('ENDOMAMBA_OUTPUT_PATH', 
                         str(PROJECT_ROOT / 'outputs'))

def get_wandb_path():
    """获取 Wandb 日志路径"""
    return os.environ.get('ENDOMAMBA_WANDB_PATH', 
                         str(PROJECT_ROOT / 'outputs' / 'logs' / 'wandb'))

# 路径常量
PATHS = {
    'MODEL_PATH': get_model_path(),
    'DATASET_PATH': get_dataset_path(),
    'OUTPUT_PATH': get_output_path(),
    'WANDB_PATH': get_wandb_path(),
}

# 数据集特定路径
DATASET_CONFIGS = {
    'colonoscopic': {
        'root': os.path.join(PATHS['DATASET_PATH'], 'endoscopy/Colonoscopic'),
        'setting': 'train_list.txt',
    },
    'ldpolypvideo': {
        'root': os.path.join(PATHS['DATASET_PATH'], 'endoscopy/LDPolypVideo'),
        'setting': 'train_list.txt',
    },
    'hyper_kvasir': {
        'root': os.path.join(PATHS['DATASET_PATH'], 'endoscopy/Hyper-Kvasir'),
        'setting': 'train_list.txt',
    },
    'autolaparo': {
        'root': os.path.join(PATHS['DATASET_PATH'], 'surgery/AutoLaparo/AutoLaparo_Task1'),
        'setting': 'annotation.txt',
    },
    'polypdiag': {
        'root': os.path.join(PATHS['DATASET_PATH'], 'classification/PolypDiag'),
        'setting': 'train_list.txt',
    },
}

# 模型检查点路径
MODEL_CONFIGS = {
    'videomamba_s16_in1k': os.path.join(
        PATHS['MODEL_PATH'], 
        'endomamba/endomamba_small_b48_seqlen16_withteacher_MIX12/checkpoint-499.pth'
    ),
}
```

### 方案 2: 创建符号链接

如果您已经有现有的数据集目录，可以创建符号链接：

```bash
# 创建项目目录结构
mkdir -p pretrained_models datasets outputs

# 创建符号链接到现有数据集
ln -s /your/existing/dataset/path datasets/endoscopy
ln -s /your/existing/model/path pretrained_models/endomamba
ln -s /your/existing/output/path outputs/pretraining
```

### 方案 3: 配置脚本

创建一键配置脚本 `setup_paths.py`：

```python
#!/usr/bin/env python3
"""
EndoMamba 路径配置脚本
自动创建必要的目录结构并配置路径
"""
import os
import json
from pathlib import Path

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        'pretrained_models/endomamba',
        'pretrained_models/videomae',
        'datasets/endoscopy',
        'datasets/surgery',
        'datasets/classification',
        'outputs/pretraining',
        'outputs/segmentation',
        'outputs/classification',
        'outputs/surgical_phase',
        'outputs/logs/wandb',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def create_config_file():
    """创建配置文件"""
    config = {
        "model_path": "./pretrained_models",
        "dataset_path": "./datasets", 
        "output_path": "./outputs",
        "wandb_path": "./outputs/logs/wandb"
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ 创建配置文件: config.json")

def create_env_file():
    """创建环境变量文件"""
    env_content = """# EndoMamba 环境变量配置
export ENDOMAMBA_MODEL_PATH="$(pwd)/pretrained_models"
export ENDOMAMBA_DATASET_PATH="$(pwd)/datasets"
export ENDOMAMBA_OUTPUT_PATH="$(pwd)/outputs"
export ENDOMAMBA_WANDB_PATH="$(pwd)/outputs/logs/wandb"
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ 创建环境文件: .env")

if __name__ == "__main__":
    print("🔧 设置 EndoMamba 路径配置...")
    setup_directories()
    create_config_file()
    create_env_file()
    print("\n🎉 路径配置完成!")
    print("\n📋 后续步骤:")
    print("1. 将您的数据集复制到 datasets/ 目录")
    print("2. 将预训练模型复制到 pretrained_models/ 目录")
    print("3. 运行: source .env")
    print("4. 开始使用 EndoMamba!")
```

## 🚀 快速开始

### 步骤 1: 运行路径配置
```bash
cd /root/lanyun-tmp/EndoMamba-main
python setup_paths.py
source .env
```

### 步骤 2: 下载必需文件
```bash
# 下载预训练模型 (示例)
# 请根据您的需求下载相应的模型检查点
mkdir -p pretrained_models/endomamba/endomamba_small_b48_seqlen16_withteacher_MIX12/
# wget [model_url] -O pretrained_models/endomamba/endomamba_small_b48_seqlen16_withteacher_MIX12/checkpoint-499.pth
```

### 步骤 3: 修复关键文件
需要手动修复以下关键文件中的硬编码路径：

1. **修复 mamba_simple.py**:
```bash
# 将在下一步提供自动修复脚本
```

## 📝 需要修复的文件列表

### 高优先级 (立即修复)
- [x] `videomamba/_mamba/mamba_ssm/modules/mamba_simple.py` - 系统路径
- [ ] `videomamba/video_sm/models/endomamba.py` - 模型路径
- [ ] `videomamba/video_sm/datasets/build.py` - 数据集路径

### 中优先级 (使用前修复)
- [ ] 所有 downstream 任务中的路径配置
- [ ] 训练脚本中的输出路径
- [ ] 评估脚本中的模型路径

### 低优先级 (可选修复)
- [ ] 数据预处理脚本中的路径
- [ ] 工具脚本中的临时路径

## 🛠️ 自动修复工具

我们将为您提供自动修复脚本来批量处理这些硬编码路径问题。

## 📞 技术支持

如果在路径配置过程中遇到问题，请检查：

1. **权限问题**: 确保对目标目录有读写权限
2. **磁盘空间**: 确保有足够空间存储数据集和模型
3. **路径长度**: 某些系统对路径长度有限制
4. **符号链接**: 确保系统支持符号链接

## 📚 相关文档

- [EndoMamba 安装指南](README.md)
- [数据集准备指南](DATASET.md)
- [模型训练指南](TRAINING.md)
- [故障排除指南](TROUBLESHOOTING.md)

---

**注意**: 本指南基于对项目代码的分析生成。在实际使用前，请根据您的具体环境和需求调整路径配置。
