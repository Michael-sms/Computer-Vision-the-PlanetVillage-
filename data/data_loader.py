"""
数据加载模块
定义PlantVillage数据集类和DataLoader生成函数
"""

import os
from typing import Callable, Dict, List, Optional, Tuple
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    SPLIT_DATASET_ROOT,
    NUM_CLASSES,
    IDX_TO_CLASS,
    CLASS_DISPLAY_NAMES,
)
from data.augmentation import get_train_transforms, get_val_transforms


class PlantVillageDataset(Dataset):
    """
    PlantVillage数据集类
    
    从划分文件中加载图像路径和标签，支持自定义数据增强
    """
    
    def __init__(
        self,
        split_file: str,
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        """
        初始化数据集
        
        Args:
            split_file: 划分文件路径（train.txt / val.txt / test.txt）
            transform: 数据变换（增强）函数
            return_path: 是否同时返回图像路径
        """
        self.transform = transform
        self.return_path = return_path
        
        # 加载划分文件
        self.samples: List[Tuple[str, int]] = []
        self._load_split_file(split_file)
        
        # 统计类别分布
        self.class_counts = self._count_classes()
    
    def _load_split_file(self, split_file: str) -> None:
        """从划分文件加载数据"""
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"划分文件不存在: {split_file}")
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    img_path, label = parts
                    self.samples.append((img_path, int(label)))
    
    def _count_classes(self) -> Dict[int, int]:
        """统计各类别样本数"""
        labels = [label for _, label in self.samples]
        return dict(Counter(labels))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, str]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (image, label) 或 (image, label, path) 如果 return_path=True
        """
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 返回一个黑色图像作为占位
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重（用于处理类不平衡问题）
        
        使用逆频率加权：weight[c] = total_samples / (num_classes * count[c])
        
        Returns:
            类别权重张量，形状 [NUM_CLASSES]
        """
        total = len(self.samples)
        weights = []
        
        for c in range(NUM_CLASSES):
            count = self.class_counts.get(c, 1)  # 避免除0
            weight = total / (NUM_CLASSES * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        计算每个样本的权重（用于WeightedRandomSampler）
        
        Returns:
            样本权重张量，形状 [num_samples]
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() for _, label in self.samples]
        return torch.tensor(sample_weights, dtype=torch.float32)
    
    def get_labels(self) -> List[int]:
        """获取所有样本的标签列表"""
        return [label for _, label in self.samples]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        获取类别分布（使用类别名称）
        
        Returns:
            字典 {类别名称: 样本数}
        """
        distribution = {}
        for idx, count in self.class_counts.items():
            class_name = IDX_TO_CLASS.get(idx, f"Unknown_{idx}")
            distribution[class_name] = count
        return distribution


def get_dataloader(
    split: str = "train",
    split_file: Optional[str] = None,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    return_path: bool = False
) -> DataLoader:
    """
    获取DataLoader
    
    Args:
        split: 数据集划分 ("train", "val", "test")
        split_file: 自定义划分文件路径（指定后split参数将被忽略）
        batch_size: 批大小
        shuffle: 是否打乱（默认训练集打乱，验证/测试集不打乱）
        num_workers: 数据加载进程数
        pin_memory: 是否锁页内存
        use_weighted_sampler: 是否使用加权采样器（处理类不平衡）
        return_path: 是否返回图像路径
        
    Returns:
        DataLoader实例
    """
    # 确定划分文件路径
    if split_file is None:
        split_file = os.path.join(SPLIT_DATASET_ROOT, f"{split}.txt")
    
    # 选择数据变换
    if split == "train":
        transform = get_train_transforms()
        if shuffle is None:
            shuffle = True
    else:
        transform = get_val_transforms()
        if shuffle is None:
            shuffle = False
    
    # 创建数据集
    dataset = PlantVillageDataset(
        split_file=split_file,
        transform=transform,
        return_path=return_path
    )
    
    # 配置采样器
    sampler = None
    if use_weighted_sampler and split == "train":
        sample_weights = dataset.get_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False  # 使用sampler时不能shuffle
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=(split == "train"),  # 训练时丢弃不完整的batch
    )
    
    return dataloader


def get_all_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取训练集、验证集、测试集的DataLoader
    
    Args:
        batch_size: 批大小
        num_workers: 数据加载进程数
        use_weighted_sampler: 是否对训练集使用加权采样器
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_loader = get_dataloader(
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler
    )
    
    val_loader = get_dataloader(
        split="val",
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    test_loader = get_dataloader(
        split="test",
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def print_dataset_info(dataloader: DataLoader, name: str = "Dataset") -> None:
    """打印数据集信息"""
    dataset = dataloader.dataset
    print(f"\n{name} 信息:")
    print(f"  样本数: {len(dataset)}")
    print(f"  批大小: {dataloader.batch_size}")
    print(f"  批次数: {len(dataloader)}")
    
    if hasattr(dataset, 'class_counts'):
        print(f"  类别数: {len(dataset.class_counts)}")


# 导出
__all__ = [
    'PlantVillageDataset',
    'get_dataloader',
    'get_all_dataloaders',
    'print_dataset_info',
]


if __name__ == "__main__":
    # 测试代码
    print("测试DataLoader...")
    
    try:
        train_loader, val_loader, test_loader = get_all_dataloaders(batch_size=32)
        
        print_dataset_info(train_loader, "训练集")
        print_dataset_info(val_loader, "验证集")
        print_dataset_info(test_loader, "测试集")
        
        # 测试加载一个batch
        print("\n测试加载一个batch...")
        images, labels = next(iter(train_loader))
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        
        print("\n✓ DataLoader测试通过!")
        
    except FileNotFoundError as e:
        print(f"\n✗ 错误: {e}")
        print("请先运行 split_dataset.py 生成划分文件")
