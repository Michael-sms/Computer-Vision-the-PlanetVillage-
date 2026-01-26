"""
训练模块初始化文件
包含损失函数、学习率调度器等训练相关组件
"""

from .losses import LabelSmoothingCrossEntropy
from .lr_scheduler import get_cosine_scheduler

__all__ = ['LabelSmoothingCrossEntropy', 'get_cosine_scheduler']
