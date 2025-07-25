"""
EndoMamba 路径配置模块
提供统一的路径管理和配置
"""
import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

class PathConfig:
    """路径配置类"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self._load_env_config()
    
    def _load_env_config(self):
        """加载环境变量配置"""
        self.model_path = os.environ.get('ENDOMAMBA_MODEL_PATH', 
                                       str(self.project_root / 'pretrained_models'))
        self.dataset_path = os.environ.get('ENDOMAMBA_DATASET_PATH', 
                                         str(self.project_root / 'datasets'))
        self.output_path = os.environ.get('ENDOMAMBA_OUTPUT_PATH', 
                                        str(self.project_root / 'outputs'))
        self.wandb_path = os.environ.get('ENDOMAMBA_WANDB_PATH', 
                                       str(self.project_root / 'outputs' / 'logs' / 'wandb'))
    
    def get_model_path(self, model_name: str = "") -> str:
        """获取模型路径"""
        if model_name:
            return os.path.join(self.model_path, model_name)
        return self.model_path
    
    def get_dataset_path(self, dataset_name: str = "") -> str:
        """获取数据集路径"""
        if dataset_name:
            return os.path.join(self.dataset_path, dataset_name)
        return self.dataset_path
    
    def get_output_path(self, task_name: str = "") -> str:
        """获取输出路径"""
        if task_name:
            return os.path.join(self.output_path, task_name)
        return self.output_path

# 全局配置实例
config = PathConfig()

# 兼容性常量
MODEL_PATH = config.model_path
DATASET_PATH = config.dataset_path
OUTPUT_PATH = config.output_path
WANDB_PATH = config.wandb_path

# 数据集配置字典
DATASET_CONFIGS = {
    'colonoscopic': {
        'root': os.path.join(config.dataset_path, 'endoscopy/Colonoscopic'),
        'setting': os.path.join(config.dataset_path, 'endoscopy/Colonoscopic/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'endoscopy/Colonoscopic'),
    },
    'ldpolypvideo': {
        'root': os.path.join(config.dataset_path, 'endoscopy/LDPolypVideo'),
        'setting': os.path.join(config.dataset_path, 'endoscopy/LDPolypVideo/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'endoscopy/LDPolypVideo'),
    },
    'hyper_kvasir': {
        'root': os.path.join(config.dataset_path, 'endoscopy/Hyper-Kvasir'),
        'setting': os.path.join(config.dataset_path, 'endoscopy/Hyper-Kvasir/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'endoscopy/Hyper-Kvasir'),
    },
    'kvasir_capsule': {
        'root': os.path.join(config.dataset_path, 'endoscopy/Kvasir-Capsule'),
        'setting': os.path.join(config.dataset_path, 'endoscopy/Kvasir-Capsule/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'endoscopy/Kvasir-Capsule'),
    },
    'cholect45': {
        'root': os.path.join(config.dataset_path, 'surgery/CholecT45'),
        'setting': os.path.join(config.dataset_path, 'surgery/CholecT45/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'surgery/CholecT45'),
    },
    'endofm': {
        'root': os.path.join(config.dataset_path, 'endoscopy/EndoFM'),
        'setting': os.path.join(config.dataset_path, 'endoscopy/EndoFM/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'endoscopy/EndoFM'),
    },
    'sun_seg': {
        'root': os.path.join(config.dataset_path, 'endoscopy/SUN-SEG'),
        'setting': os.path.join(config.dataset_path, 'endoscopy/SUN-SEG/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'endoscopy/SUN-SEG'),
    },
    'glenda': {
        'root': os.path.join(config.dataset_path, 'endoscopy/GLENDA_v1.0'),
        'setting': os.path.join(config.dataset_path, 'endoscopy/GLENDA_v1.0/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'endoscopy/GLENDA_v1.0'),
    },
    'autolaparo': {
        'root': os.path.join(config.dataset_path, 'surgery/AutoLaparo/AutoLaparo_Task1'),
        'setting': os.path.join(config.dataset_path, 'surgery/AutoLaparo/AutoLaparo_Task1/annotation.txt'),
        'prefix': os.path.join(config.dataset_path, 'surgery/AutoLaparo/AutoLaparo_Task1'),
    },
    'polypdiag': {
        'root': os.path.join(config.dataset_path, 'classification/PolypDiag'),
        'setting': os.path.join(config.dataset_path, 'classification/PolypDiag/train_list.txt'),
        'prefix': os.path.join(config.dataset_path, 'classification/PolypDiag'),
    },
}

# 模型配置字典（带路径验证）
def _validate_path(path):
    if not os.path.exists(path):
        print(f"⚠️ 警告: 模型路径不存在: {path}")
    return path

MODEL_CONFIGS = {
    'videomamba_s16_in1k': _validate_path(os.path.join(
        config.model_path, 
        'endomamba/endomamba_small_b48_seqlen16_withteacher_MIX12/checkpoint-499.pth'
    )),
    'videomamba_t16_in1k': _validate_path(os.path.join(
        config.model_path,
        'endomamba/endomamba_tiny/checkpoint.pth'
    )),
    'videomamba_m16_in1k': _validate_path(os.path.join(
        config.model_path,
        'endomamba/endomamba_medium/checkpoint.pth'
    )),
}

def get_model_config(model_name: str) -> str:
    """获取模型配置路径"""
    return MODEL_CONFIGS.get(model_name, "")

def get_dataset_config(dataset_name: str) -> dict:
    """获取数据集配置"""
    return DATASET_CONFIGS.get(dataset_name, {})
