#!/usr/bin/env python3
"""
EndoMamba 路径自动修复脚本
自动修复项目中的硬编码路径问题
"""
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

class PathFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_applied = []
        self.backup_dir = self.project_root / "path_fix_backups"
        
    def create_backup(self, file_path: Path) -> Path:
        """为文件创建备份"""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)
        
        backup_path = self.backup_dir / f"{file_path.name}.backup"
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def fix_mamba_simple_path(self) -> bool:
        """修复 mamba_simple.py 中的硬编码路径"""
        file_path = self.project_root / "videomamba/_mamba/mamba_ssm/modules/mamba_simple.py"
        
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return False
        
        # 创建备份
        backup_path = self.create_backup(file_path)
        print(f"📁 创建备份: {backup_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找硬编码路径
            old_pattern = r"sys\.path\.append\(['\"]\/home\/tqy\/endomamba\/videomamba\/_mamba\/mamba_ssm['\"] \)"
            new_code = """import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))"""
            
            # 替换硬编码路径
            if "sys.path.append('/home/tqy/endomamba/videomamba/_mamba/mamba_ssm')" in content:
                content = content.replace(
                    "sys.path.append('/home/tqy/endomamba/videomamba/_mamba/mamba_ssm')",
                    "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\")))"
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixes_applied.append(f"✅ 修复 mamba_simple.py 系统路径")
                return True
            else:
                print("⚠️  mamba_simple.py 中未找到需要修复的路径")
                return False
                
        except Exception as e:
            print(f"❌ 修复 mamba_simple.py 失败: {e}")
            # 恢复备份
            shutil.copy2(backup_path, file_path)
            return False
    
    def create_config_module(self) -> bool:
        """创建统一的配置模块"""
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        config_content = '''"""
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

# 模型配置字典
MODEL_CONFIGS = {
    'videomamba_s16_in1k': os.path.join(
        config.model_path, 
        'endomamba/endomamba_small_b48_seqlen16_withteacher_MIX12/checkpoint-499.pth'
    ),
    'videomamba_t16_in1k': os.path.join(
        config.model_path,
        'endomamba/endomamba_tiny/checkpoint.pth'
    ),
    'videomamba_m16_in1k': os.path.join(
        config.model_path,
        'endomamba/endomamba_medium/checkpoint.pth'
    ),
}

def get_model_config(model_name: str) -> str:
    """获取模型配置路径"""
    return MODEL_CONFIGS.get(model_name, "")

def get_dataset_config(dataset_name: str) -> dict:
    """获取数据集配置"""
    return DATASET_CONFIGS.get(dataset_name, {})
'''
        
        config_file = config_dir / "paths.py" 
        config_file.write_text(config_content, encoding='utf-8')
        
        # 创建 __init__.py
        init_file = config_dir / "__init__.py"
        init_file.write_text('from .paths import *\n', encoding='utf-8')
        
        self.fixes_applied.append("✅ 创建统一配置模块: config/paths.py")
        return True
    
    def fix_model_files(self) -> bool:
        """修复模型文件中的硬编码路径"""
        model_files = [
            "videomamba/video_sm/models/endomamba.py",
            "videomamba/video_sm/models/endomamba_pretrain.py", 
            "videomamba/video_sm/models/endomamba_two_heads.py",
        ]
        
        fixed_count = 0
        
        for file_rel_path in model_files:
            file_path = self.project_root / file_rel_path
            
            if not file_path.exists():
                print(f"⚠️  文件不存在: {file_path}")
                continue
            
            try:
                # 创建备份
                backup_path = self.create_backup(file_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 替换硬编码的 MODEL_PATH
                old_pattern = r"MODEL_PATH = ['\"][^'\"]*['\"]"
                new_line = "from config.paths import MODEL_PATH, MODEL_CONFIGS"
                
                if "MODEL_PATH = '/data/tqy/endomamba_pretrain/'" in content:
                    # 添加导入语句
                    import_pattern = r"(import\s+.*\n)*"
                    content = re.sub(
                        r"^(import\s+.*\n|from\s+.*\n)*",
                        lambda m: m.group(0) + "from config.paths import MODEL_PATH, MODEL_CONFIGS\n",
                        content,
                        count=1,
                        flags=re.MULTILINE
                    )
                    
                    # 删除硬编码的 MODEL_PATH 定义
                    content = re.sub(
                        r"MODEL_PATH = ['\"][^'\"]*['\"]",
                        "# MODEL_PATH 现在从 config.paths 导入",
                        content
                    )
                    
                    # 替换 _MODELS 字典中的硬编码路径
                    content = re.sub(
                        r'_MODELS = \{[^}]*\}',
                        '_MODELS = MODEL_CONFIGS',
                        content,
                        flags=re.DOTALL
                    )
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    fixed_count += 1
                    self.fixes_applied.append(f"✅ 修复模型文件: {file_rel_path}")
                
            except Exception as e:
                print(f"❌ 修复 {file_rel_path} 失败: {e}")
                # 恢复备份
                shutil.copy2(backup_path, file_path)
        
        return fixed_count > 0
    
    def create_setup_script(self) -> bool:
        """创建目录设置脚本"""
        setup_content = '''#!/usr/bin/env python3
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
    print("\\n🎉 目录设置完成!")
    print("\\n📋 后续步骤:")
    print("1. 运行: source .env")
    print("2. 将数据集放入 datasets/ 相应目录")
    print("3. 将预训练模型放入 pretrained_models/ 目录")
    print("4. 开始使用 EndoMamba!")
'''
        
        setup_file = self.project_root / "setup_paths.py"
        setup_file.write_text(setup_content, encoding='utf-8')
        setup_file.chmod(0o755)  # 添加执行权限
        
        self.fixes_applied.append("✅ 创建目录设置脚本: setup_paths.py")
        return True
    
    def run_all_fixes(self) -> bool:
        """运行所有修复"""
        print("🔧 开始修复 EndoMamba 硬编码路径问题...")
        
        success = True
        
        # 1. 修复关键的 mamba_simple.py 
        if not self.fix_mamba_simple_path():
            success = False
        
        # 2. 创建统一配置模块
        if not self.create_config_module():
            success = False
        
        # 3. 修复模型文件
        if not self.fix_model_files():
            success = False
        
        # 4. 创建设置脚本
        if not self.create_setup_script():
            success = False
        
        return success
    
    def print_summary(self):
        """打印修复总结"""
        print("\\n" + "="*60)
        print("📊 路径修复总结")
        print("="*60)
        
        for fix in self.fixes_applied:
            print(fix)
        
        print(f"\\n📁 备份目录: {self.backup_dir}")
        print("💡 如需回滚，请从备份目录恢复文件")
        
        print("\\n📋 下一步操作:")
        print("1. 运行: python setup_paths.py")
        print("2. 运行: source .env")
        print("3. 将数据集和模型文件放入相应目录")
        print("4. 测试修复效果")

def main():
    """主函数"""
    project_root = Path.cwd()
    fixer = PathFixer(str(project_root))
    
    if fixer.run_all_fixes():
        print("\\n🎉 路径修复完成!")
    else:
        print("\\n⚠️  部分修复失败，请查看详细信息")
    
    fixer.print_summary()

if __name__ == "__main__":
    main()
