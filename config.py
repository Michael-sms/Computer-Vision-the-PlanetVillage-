"""
配置文件 - 项目超参数和路径配置
"""

import yaml
import os

# 默认配置
DEFAULT_CONFIG = {
    # ========== 路径配置 ==========
    'paths': {
        'dataset_root': './datasets/plantvillage dataset/color',
        'splits_dir': './datasets/splits',
        'checkpoints_dir': './checkpoints',
        'results_dir': './results',
        'logs_dir': './logs',
    },
    
    # ========== 数据配置 ==========
    'data': {
        'num_classes': 38,
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        
        # 数据划分比例
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        
        # ImageNet标准化参数
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    
    # ========== 模型配置 ==========
    'model': {
        'name': 'MobileNetV2WithSE',
        'pretrained': True,
        'freeze_layers': 10,
        'dropout_rate': 0.2,
        'se_reduction': 16,
    },
    
    # ========== 训练配置 ==========
    'training': {
        'epochs': 100,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        
        # 学习率调度
        'scheduler': 'cosine',
        'warmup_epochs': 0,
        'warmup_lr': 1e-6,
        
        # Label Smoothing
        'label_smoothing': 0.1,
        
        # 正则化
        'grad_clip': 1.0,
        
        # 早停
        'patience': 10,
        
        # 日志
        'log_interval': 50,
        'save_interval': 10,
        
        # AMP (混合精度训练) - RTX4060推荐启用
        'use_amp': False,  # 可通过 --amp 启用
    },
    
    # ========== 数据增强配置 ==========
    'augmentation': {
        'train': {
            'random_rotation': 20,
            'random_flip': True,
            'random_scale': [0.8, 1.2],
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1,
            },
            'cutout': {
                'enable': True,
                'size': 16,
                'prob': 0.2,
            },
        },
        'val': {
            'center_crop': True,
        },
        'test': {
            'center_crop': True,
        },
    },
    
    # ========== 设备配置 ==========
    'device': {
        'cuda': True,
        'gpu_id': 0,
        'mixed_precision': False,
    },
    
    # ========== 随机种子 ==========
    'seed': 42,
}


def save_config(config, save_path='config.yaml'):
    """
    保存配置到YAML文件
    
    参数:
        config (dict): 配置字典
        save_path (str): 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"配置已保存至: {save_path}")


def load_config(config_path='config.yaml'):
    """
    从YAML文件加载配置
    
    参数:
        config_path (str): 配置文件路径
    
    返回:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        print(f"配置文件不存在，使用默认配置")
        return DEFAULT_CONFIG
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"配置已加载: {config_path}")
    return config


def print_config(config):
    """
    打印配置信息
    
    参数:
        config (dict): 配置字典
    """
    print("\n" + "=" * 60)
    print("配置信息".center(60))
    print("=" * 60)
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    print_dict(config)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 生成默认配置文件
    save_config(DEFAULT_CONFIG, 'config.yaml')
    
    # 加载并打印配置
    config = load_config('config.yaml')
    print_config(config)
