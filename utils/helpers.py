"""
工具函数模块
包含设备管理、随机种子设置等通用功能
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    设置随机种子以保证可复现性
    
    参数:
        seed (int): 随机种子，默认42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[随机种子] 已设置为 {seed}")


def get_device(cuda=True, gpu_id=0):
    """
    获取计算设备
    
    参数:
        cuda (bool): 是否使用CUDA，默认True
        gpu_id (int): GPU编号，默认0
    
    返回:
        torch.device: 计算设备
    """
    if cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[设备] 使用 GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  显存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"[设备] 使用 CPU")
    
    return device


def count_parameters(model, trainable_only=False):
    """
    统计模型参数量
    
    参数:
        model: PyTorch模型
        trainable_only (bool): 仅统计可训练参数，默认False
    
    返回:
        int: 参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_lr(optimizer):
    """
    获取当前学习率
    
    参数:
        optimizer: PyTorch优化器
    
    返回:
        float: 当前学习率
    """
    return optimizer.param_groups[0]['lr']


def ensure_dir(path):
    """
    确保目录存在，不存在则创建
    
    参数:
        path (str): 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[创建目录] {path}")


def format_time(seconds):
    """
    格式化时间显示
    
    参数:
        seconds (float): 秒数
    
    返回:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def save_dict_to_json(data, save_path):
    """
    保存字典到JSON文件
    
    参数:
        data (dict): 数据字典
        save_path (str): 保存路径
    """
    import json
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"[保存] JSON文件: {save_path}")


def load_dict_from_json(load_path):
    """
    从JSON文件加载字典
    
    参数:
        load_path (str): JSON文件路径
    
    返回:
        dict: 数据字典
    """
    import json
    
    with open(load_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[加载] JSON文件: {load_path}")
    return data


class AverageMeter:
    """
    计算并存储平均值和当前值
    
    用于训练过程中的指标统计
    """
    
    def __init__(self, name='metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        """重置所有统计量"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        更新统计量
        
        参数:
            val: 当前值
            n (int): 批次大小
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


if __name__ == "__main__":
    # 测试工具函数
    print("=" * 60)
    print("工具函数测试")
    print("=" * 60)
    
    # 测试随机种子
    set_seed(42)
    
    # 测试设备
    device = get_device()
    
    # 测试时间格式化
    print(f"\n[时间格式化]")
    print(f"  30秒: {format_time(30)}")
    print(f"  90秒: {format_time(90)}")
    print(f"  3700秒: {format_time(3700)}")
    
    # 测试AverageMeter
    print(f"\n[AverageMeter测试]")
    meter = AverageMeter('loss')
    for i in range(5):
        meter.update(1.0 - i * 0.1)
        print(f"  更新 {i+1}: {meter}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
