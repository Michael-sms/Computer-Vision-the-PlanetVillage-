"""
样本可视化模块
可视化数据集样本、数据增强效果、类别分布等
结果保存到 results 目录
"""

import os
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
from PIL import Image
import torch
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    DATASET_ROOT,
    SPLIT_DATASET_ROOT,
    RESULTS_DIR,
    CLASS_NAMES,
    CLASS_DISPLAY_NAMES,
    IDX_TO_CLASS,
    NUM_CLASSES,
    RANDOM_SEED,
)
from data.augmentation import (
    get_train_transforms,
    get_val_transforms,
    denormalize,
    tensor_to_pil,
)
from data.data_loader import PlantVillageDataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def ensure_results_dir() -> str:
    """确保结果目录存在"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def visualize_class_samples(
    num_samples_per_class: int = 3,
    num_classes_to_show: int = 12,
    save_path: Optional[str] = None
) -> None:
    """
    可视化每个类别的样本图像
    
    Args:
        num_samples_per_class: 每个类别展示的样本数
        num_classes_to_show: 展示的类别数
        save_path: 保存路径，默认保存到results目录
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "class_samples.png")
    
    random.seed(RANDOM_SEED)
    
    # 选择要展示的类别（均匀采样）
    step = max(1, NUM_CLASSES // num_classes_to_show)
    selected_classes = list(range(0, NUM_CLASSES, step))[:num_classes_to_show]
    
    fig, axes = plt.subplots(
        num_classes_to_show, 
        num_samples_per_class, 
        figsize=(num_samples_per_class * 3, num_classes_to_show * 2.5)
    )
    
    for row, class_idx in enumerate(selected_classes):
        class_name = CLASS_NAMES[class_idx]
        display_name = CLASS_DISPLAY_NAMES[class_idx]
        class_dir = os.path.join(DATASET_ROOT, class_name)
        
        if not os.path.exists(class_dir):
            continue
        
        # 获取该类别的图像
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        for col in range(num_samples_per_class):
            ax = axes[row, col] if num_classes_to_show > 1 else axes[col]
            
            if col < len(images):
                img_path = os.path.join(class_dir, images[col])
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
            
            ax.axis('off')
            
            # 第一列显示类别名称
            if col == 0:
                ax.set_ylabel(display_name, fontsize=10, rotation=0, ha='right', va='center')
    
    plt.suptitle("PlantVillage 数据集样本展示", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 类别样本可视化已保存: {save_path}")


def visualize_augmentation_effects(
    num_samples: int = 4,
    num_augmented: int = 5,
    save_path: Optional[str] = None
) -> None:
    """
    可视化数据增强效果
    
    Args:
        num_samples: 原始样本数量
        num_augmented: 每个样本的增强版本数量
        save_path: 保存路径
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "augmentation_effects.png")
    
    random.seed(RANDOM_SEED)
    
    # 从数据集随机选择样本
    train_transform = get_train_transforms()
    
    # 随机选择几个类别
    selected_classes = random.sample(CLASS_NAMES, num_samples)
    
    fig, axes = plt.subplots(
        num_samples, 
        num_augmented + 1,  # +1 for original
        figsize=((num_augmented + 1) * 2.5, num_samples * 2.5)
    )
    
    for row, class_name in enumerate(selected_classes):
        class_dir = os.path.join(DATASET_ROOT, class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
        
        img_path = os.path.join(class_dir, random.choice(images))
        original_img = Image.open(img_path).convert('RGB')
        
        # 显示原始图像
        ax = axes[row, 0]
        ax.imshow(original_img)
        ax.set_title("原图" if row == 0 else "", fontsize=10)
        ax.axis('off')
        
        # 显示增强后的图像
        for col in range(1, num_augmented + 1):
            ax = axes[row, col]
            
            # 应用数据增强
            augmented = train_transform(original_img)
            # 反标准化以便显示
            augmented = denormalize(augmented)
            augmented_pil = tensor_to_pil(augmented)
            
            ax.imshow(augmented_pil)
            if row == 0:
                ax.set_title(f"增强 {col}", fontsize=10)
            ax.axis('off')
    
    plt.suptitle("数据增强效果展示", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 数据增强效果可视化已保存: {save_path}")


def visualize_class_distribution(save_path: Optional[str] = None) -> None:
    """
    可视化数据集各类别样本分布
    
    Args:
        save_path: 保存路径
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    
    # 统计各类别样本数
    class_counts = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATASET_ROOT, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            count = 0
        class_counts.append(count)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(NUM_CLASSES)
    bars = ax.bar(x, class_counts, color='steelblue', edgecolor='navy', alpha=0.8)
    
    # 添加数值标签
    for bar, count in zip(bars, class_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(count),
            ha='center',
            va='bottom',
            fontsize=7,
            rotation=90
        )
    
    ax.set_xlabel("类别索引", fontsize=12)
    ax.set_ylabel("样本数量", fontsize=12)
    ax.set_title("PlantVillage 数据集类别分布", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=8)
    
    # 添加统计信息
    total = sum(class_counts)
    mean_count = np.mean(class_counts)
    min_count = min(class_counts)
    max_count = max(class_counts)
    
    stats_text = f"总样本数: {total}\n平均: {mean_count:.0f}\n最小: {min_count}\n最大: {max_count}"
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 类别分布可视化已保存: {save_path}")


def visualize_split_distribution(save_path: Optional[str] = None) -> None:
    """
    可视化训练/验证/测试集的划分分布
    
    Args:
        save_path: 保存路径
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "split_distribution.png")
    
    # 加载划分文件并统计
    splits = ['train', 'val', 'test']
    split_counts = {}
    split_class_counts = {}
    
    for split in splits:
        split_file = os.path.join(SPLIT_DATASET_ROOT, f"{split}.txt")
        if not os.path.exists(split_file):
            print(f"警告: 划分文件不存在 {split_file}")
            continue
        
        labels = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    labels.append(int(parts[1]))
        
        split_counts[split] = len(labels)
        split_class_counts[split] = Counter(labels)
    
    if not split_counts:
        print("错误: 没有找到划分文件，请先运行 split_dataset.py")
        return
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 划分比例饼图
    ax1 = axes[0]
    labels = list(split_counts.keys())
    sizes = list(split_counts.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    explode = (0.05, 0, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title("数据集划分比例", fontsize=12, fontweight='bold')
    
    # 图2: 各划分的类别分布对比
    ax2 = axes[1]
    
    x = np.arange(NUM_CLASSES)
    width = 0.25
    
    for i, (split, color) in enumerate(zip(splits, colors)):
        if split in split_class_counts:
            counts = [split_class_counts[split].get(c, 0) for c in range(NUM_CLASSES)]
            ax2.bar(x + i * width, counts, width, label=split, color=color, alpha=0.8)
    
    ax2.set_xlabel("类别索引", fontsize=10)
    ax2.set_ylabel("样本数量", fontsize=10)
    ax2.set_title("各划分的类别分布", fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 划分分布可视化已保存: {save_path}")


def visualize_sample_grid(
    split: str = "train",
    grid_size: Tuple[int, int] = (4, 8),
    save_path: Optional[str] = None
) -> None:
    """
    可视化数据集的样本网格
    
    Args:
        split: 数据集划分 ("train", "val", "test")
        grid_size: 网格大小 (行, 列)
        save_path: 保存路径
    """
    ensure_results_dir()
    
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"{split}_sample_grid.png")
    
    split_file = os.path.join(SPLIT_DATASET_ROOT, f"{split}.txt")
    if not os.path.exists(split_file):
        print(f"错误: 划分文件不存在 {split_file}")
        return
    
    # 加载数据集
    transform = get_val_transforms()
    dataset = PlantVillageDataset(split_file, transform=transform)
    
    # 随机选择样本
    random.seed(RANDOM_SEED)
    num_samples = grid_size[0] * grid_size[1]
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # 创建图表
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1] * 2, grid_size[0] * 2))
    
    for i, idx in enumerate(indices):
        row = i // grid_size[1]
        col = i % grid_size[1]
        ax = axes[row, col] if grid_size[0] > 1 else axes[col]
        
        image, label = dataset[idx]
        # 反标准化
        image = denormalize(image)
        image_pil = tensor_to_pil(image)
        
        ax.imshow(image_pil)
        ax.set_title(f"{CLASS_DISPLAY_NAMES[label][:8]}", fontsize=8)
        ax.axis('off')
    
    split_names = {'train': '训练集', 'val': '验证集', 'test': '测试集'}
    plt.suptitle(f"{split_names.get(split, split)} 样本展示", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {split}样本网格已保存: {save_path}")


def generate_all_visualizations() -> None:
    """生成所有可视化结果"""
    print("\n" + "=" * 60)
    print("生成数据可视化结果")
    print("=" * 60)
    
    print("\n[1/5] 生成类别样本展示...")
    visualize_class_samples()
    
    print("\n[2/5] 生成数据增强效果展示...")
    visualize_augmentation_effects()
    
    print("\n[3/5] 生成类别分布图...")
    visualize_class_distribution()
    
    print("\n[4/5] 生成划分分布图...")
    visualize_split_distribution()
    
    print("\n[5/5] 生成样本网格...")
    visualize_sample_grid(split="train")
    
    print("\n" + "=" * 60)
    print(f"✓ 所有可视化结果已保存至: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_visualizations()
