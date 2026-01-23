"""
数据增强模块
定义训练集和验证/测试集的数据增强策略
"""

import torch
from torchvision import transforms
from typing import Callable, Optional
import random
import numpy as np
from PIL import Image

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class GaussianNoise:
    """添加高斯噪声"""
    
    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.5):
        """
        Args:
            mean: 噪声均值
            std: 噪声标准差
            p: 应用概率
        """
        self.mean = mean
        self.std = std
        self.p = p
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0., 1.)
        return tensor
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"


class Cutout:
    """
    Cutout数据增强
    随机遮挡图像的一个正方形区域
    """
    
    def __init__(self, size: int = 16, p: float = 0.5):
        """
        Args:
            size: 遮挡区域的边长
            p: 应用概率
        """
        self.size = size
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        
        h, w = img.shape[1], img.shape[2]
        
        # 随机选择遮挡区域的中心点
        y = random.randint(0, h)
        x = random.randint(0, w)
        
        # 计算遮挡区域的边界
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)
        
        # 用0填充遮挡区域
        img[:, y1:y2, x1:x2] = 0
        
        return img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, p={self.p})"


def get_train_transforms() -> Callable:
    """
    获取训练集的数据增强变换
    
    包含：
    - 随机旋转(±20°)
    - 随机水平/垂直翻转
    - 随机缩放和裁剪
    - 颜色抖动
    - 高斯噪声
    - Cutout
    - ImageNet标准化
    
    Returns:
        transforms.Compose 变换组合
    """
    return transforms.Compose([
        # 调整大小（稍大于目标尺寸，为后续裁剪做准备）
        transforms.Resize((256, 256)),
        
        # 随机旋转 ±20度
        transforms.RandomRotation(degrees=20),
        
        # 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 随机垂直翻转
        transforms.RandomVerticalFlip(p=0.5),
        
        # 随机缩放和裁剪到目标尺寸
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        
        # 颜色抖动
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        
        # 转换为Tensor
        transforms.ToTensor(),
        
        # 添加高斯噪声
        GaussianNoise(mean=0.0, std=0.02, p=0.3),
        
        # Cutout
        Cutout(size=16, p=0.2),
        
        # ImageNet标准化
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> Callable:
    """
    获取验证集/测试集的数据增强变换
    
    仅包含：
    - 调整大小
    - 中心裁剪
    - ImageNet标准化
    
    Returns:
        transforms.Compose 变换组合
    """
    return transforms.Compose([
        # 调整大小
        transforms.Resize((256, 256)),
        
        # 中心裁剪到目标尺寸
        transforms.CenterCrop(IMAGE_SIZE),
        
        # 转换为Tensor
        transforms.ToTensor(),
        
        # ImageNet标准化
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_visualization_transforms() -> Callable:
    """
    获取用于可视化的数据变换（不包含标准化）
    
    Returns:
        transforms.Compose 变换组合
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    反标准化，将ImageNet标准化后的张量还原
    
    Args:
        tensor: 标准化后的图像张量 [C, H, W] 或 [B, C, H, W]
        
    Returns:
        还原后的张量，值域 [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0., 1.)
    
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将张量转换为PIL图像
    
    Args:
        tensor: 图像张量 [C, H, W]，值域 [0, 1]
        
    Returns:
        PIL.Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.cpu().detach()
    tensor = torch.clamp(tensor, 0., 1.)
    
    # 转换为numpy数组并调整维度顺序
    np_img = tensor.permute(1, 2, 0).numpy()
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)


# 导出
__all__ = [
    'get_train_transforms',
    'get_val_transforms',
    'get_visualization_transforms',
    'denormalize',
    'tensor_to_pil',
    'GaussianNoise',
    'Cutout',
]
