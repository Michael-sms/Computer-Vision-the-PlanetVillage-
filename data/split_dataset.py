"""
数据集划分模块
将PlantVillage数据集按70%-15%-15%比例划分为训练集、验证集、测试集
使用分层抽样确保每个类别的比例一致
"""

import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    DATASET_ROOT,
    SPLIT_DATASET_ROOT,
    CLASS_NAMES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def get_all_images(dataset_root: str) -> Dict[str, List[str]]:
    """
    获取所有图像文件路径，按类别组织
    
    Args:
        dataset_root: 数据集根目录
        
    Returns:
        字典 {类别名: [图像路径列表]}
    """
    images_by_class = defaultdict(list)
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.exists(class_dir):
            print(f"警告: 类别目录不存在 - {class_name}")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, img_name)
                images_by_class[class_name].append(img_path)
    
    return dict(images_by_class)


def split_dataset(
    images_by_class: Dict[str, List[str]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    分层抽样划分数据集
    
    Args:
        images_by_class: 按类别组织的图像路径字典
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        (train_data, val_data, test_data)，每个元素为 (图像路径, 类别索引) 的列表
    """
    random.seed(seed)
    
    train_data = []
    val_data = []
    test_data = []
    
    from utils.constants import CLASS_TO_IDX
    
    for class_name, img_paths in images_by_class.items():
        # 随机打乱
        shuffled_paths = img_paths.copy()
        random.shuffle(shuffled_paths)
        
        # 计算划分点
        n = len(shuffled_paths)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # 获取类别索引
        class_idx = CLASS_TO_IDX[class_name]
        
        # 划分数据
        for path in shuffled_paths[:train_end]:
            train_data.append((path, class_idx))
        for path in shuffled_paths[train_end:val_end]:
            val_data.append((path, class_idx))
        for path in shuffled_paths[val_end:]:
            test_data.append((path, class_idx))
    
    # 再次打乱（打破类别顺序）
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def save_split_files(
    train_data: List[Tuple[str, int]],
    val_data: List[Tuple[str, int]],
    test_data: List[Tuple[str, int]],
    output_dir: str
) -> None:
    """
    保存划分结果到文本文件
    
    Args:
        train_data: 训练集数据
        val_data: 验证集数据
        test_data: 测试集数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def write_split(data: List[Tuple[str, int]], filename: str):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for img_path, label in data:
                f.write(f"{img_path}\t{label}\n")
        print(f"已保存: {filepath} ({len(data)} 条记录)")
    
    write_split(train_data, "train.txt")
    write_split(val_data, "val.txt")
    write_split(test_data, "test.txt")


def print_split_statistics(
    train_data: List[Tuple[str, int]],
    val_data: List[Tuple[str, int]],
    test_data: List[Tuple[str, int]]
) -> None:
    """
    打印数据集划分统计信息
    """
    from utils.constants import IDX_TO_CLASS
    
    print("\n" + "=" * 60)
    print("数据集划分统计")
    print("=" * 60)
    
    total = len(train_data) + len(val_data) + len(test_data)
    print(f"\n总样本数: {total}")
    print(f"训练集: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"验证集: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"测试集: {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    
    # 统计各类别样本数
    print("\n各类别样本数统计:")
    print("-" * 60)
    
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for _, label in train_data:
        train_counts[label] += 1
    for _, label in val_data:
        val_counts[label] += 1
    for _, label in test_data:
        test_counts[label] += 1
    
    print(f"{'类别':<45} {'训练':>8} {'验证':>8} {'测试':>8} {'总计':>8}")
    print("-" * 60)
    
    for idx in range(len(IDX_TO_CLASS)):
        class_name = IDX_TO_CLASS[idx]
        # 截断过长的类别名
        display_name = class_name[:42] + "..." if len(class_name) > 45 else class_name
        train_n = train_counts[idx]
        val_n = val_counts[idx]
        test_n = test_counts[idx]
        total_n = train_n + val_n + test_n
        print(f"{display_name:<45} {train_n:>8} {val_n:>8} {test_n:>8} {total_n:>8}")
    
    print("=" * 60)


def main():
    """主函数：执行数据集划分"""
    print("开始划分PlantVillage数据集...")
    print(f"数据集路径: {DATASET_ROOT}")
    print(f"划分比例: 训练集{TRAIN_RATIO*100:.0f}% / 验证集{VAL_RATIO*100:.0f}% / 测试集{TEST_RATIO*100:.0f}%")
    print(f"随机种子: {RANDOM_SEED}")
    
    # 1. 获取所有图像
    print("\n[1/3] 扫描图像文件...")
    images_by_class = get_all_images(DATASET_ROOT)
    total_images = sum(len(v) for v in images_by_class.values())
    print(f"找到 {len(images_by_class)} 个类别，共 {total_images} 张图像")
    
    # 2. 划分数据集
    print("\n[2/3] 执行分层抽样划分...")
    train_data, val_data, test_data = split_dataset(images_by_class)
    
    # 3. 保存划分结果
    print("\n[3/3] 保存划分结果...")
    save_split_files(train_data, val_data, test_data, SPLIT_DATASET_ROOT)
    
    # 打印统计信息
    print_split_statistics(train_data, val_data, test_data)
    
    print(f"\n✓ 数据集划分完成！划分文件已保存至: {SPLIT_DATASET_ROOT}")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    main()
