"""
混淆矩阵分析模块
绘制混淆矩阵热力图，分析易混淆类别对，识别高低准确率类别
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import EVALUATION_RESULTS_DIR, CLASS_NAMES, CLASS_DISPLAY_NAMES, NUM_CLASSES

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (20, 18),
    cmap: str = 'Blues',
    title: str = '混淆矩阵 (Confusion Matrix)'
) -> str:
    """
    绘制混淆矩阵热力图
    
    Args:
        cm: 混淆矩阵，shape=[num_classes, num_classes]
        class_names: 类别名称列表
        save_path: 保存路径，默认保存到results目录
        normalize: 是否按行归一化（转为百分比）
        figsize: 图像尺寸
        cmap: 颜色映射
        title: 图表标题
        
    Returns:
        保存的文件路径
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(cm.shape[0])]
    
    if save_path is None:
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(EVALUATION_RESULTS_DIR, "confusion_matrix.png")
    
    # 归一化
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        cm_display = cm_normalized
        fmt = '.1%'
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = 'd'
        vmin, vmax = 0, cm.max()
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(
        cm_display,
        annot=False,  # 38x38太大，不显示数值
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': '比例' if normalize else '样本数'}
    )
    
    ax.set_xlabel('预测类别 (Predicted)', fontsize=12)
    ax.set_ylabel('真实类别 (Actual)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 旋转标签
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 混淆矩阵已保存: {save_path}")
    return save_path


def plot_confusion_matrix_simplified(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    top_n: int = 15
) -> str:
    """
    绘制简化的混淆矩阵（仅显示错误率最高的类别）
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        top_n: 显示错误率最高的N个类别
        
    Returns:
        保存的文件路径
    """
    if class_names is None:
        class_names = CLASS_DISPLAY_NAMES
    
    if save_path is None:
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(EVALUATION_RESULTS_DIR, "confusion_matrix_top_errors.png")
    
    # 计算每个类别的错误率
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    error_rates = 1 - np.diag(cm_normalized)
    
    # 选择错误率最高的类别
    top_indices = np.argsort(error_rates)[-top_n:][::-1]
    
    # 提取子矩阵
    cm_sub = cm_normalized[np.ix_(top_indices, top_indices)]
    names_sub = [class_names[i][:15] for i in top_indices]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm_sub,
        annot=True,
        fmt='.1%',
        cmap='Reds',
        xticklabels=names_sub,
        yticklabels=names_sub,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': '比例'}
    )
    
    ax.set_xlabel('预测类别', fontsize=12)
    ax.set_ylabel('真实类别', fontsize=12)
    ax.set_title(f'错误率最高的{top_n}个类别混淆矩阵', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 简化混淆矩阵已保存: {save_path}")
    return save_path


def find_confused_pairs(
    cm: np.ndarray,
    class_names: List[str] = None,
    top_k: int = 10
) -> List[Dict]:
    """
    找出最容易混淆的类别对
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        top_k: 返回Top-K个混淆对
        
    Returns:
        混淆对列表 [{"true_class": str, "pred_class": str, "count": int, "rate": float}, ...]
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    num_classes = cm.shape[0]
    
    # 归一化
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-10
    cm_normalized = cm.astype('float') / row_sums
    
    # 收集所有非对角线的误分类
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class_id": i,
                    "pred_class_id": j,
                    "true_class": class_names[i],
                    "pred_class": class_names[j],
                    "count": int(cm[i, j]),
                    "rate": float(cm_normalized[i, j])
                })
    
    # 按误分类率排序
    confused_pairs.sort(key=lambda x: x["rate"], reverse=True)
    
    return confused_pairs[:top_k]


def find_best_worst_classes(
    cm: np.ndarray,
    class_names: List[str] = None,
    top_k: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    找出准确率最高和最低的类别
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        top_k: 返回Top-K个类别
        
    Returns:
        (最佳类别列表, 最差类别列表)
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    # 计算每个类别的准确率
    row_sums = cm.sum(axis=1) + 1e-10
    accuracies = np.diag(cm) / row_sums
    
    # 创建类别信息
    class_info = []
    for i in range(cm.shape[0]):
        class_info.append({
            "class_id": i,
            "class_name": class_names[i],
            "accuracy": float(accuracies[i]),
            "correct": int(np.diag(cm)[i]),
            "total": int(row_sums[i])
        })
    
    # 按准确率排序
    class_info.sort(key=lambda x: x["accuracy"])
    
    worst_classes = class_info[:top_k]
    best_classes = class_info[-top_k:][::-1]
    
    return best_classes, worst_classes


def generate_confusion_analysis_report(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None
) -> str:
    """
    生成混淆矩阵分析报告
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 报告保存路径
        
    Returns:
        报告文本
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if save_path is None:
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(EVALUATION_RESULTS_DIR, "confusion_analysis_report.txt")
    
    lines = []
    lines.append("=" * 80)
    lines.append("混淆矩阵分析报告 (Confusion Matrix Analysis Report)")
    lines.append("=" * 80)
    
    # 总体统计
    total_samples = cm.sum()
    correct_samples = np.trace(cm)
    overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0
    
    lines.append(f"\n【总体统计】")
    lines.append(f"总样本数: {total_samples:,}")
    lines.append(f"正确分类: {correct_samples:,}")
    lines.append(f"错误分类: {total_samples - correct_samples:,}")
    lines.append(f"总体准确率: {overall_accuracy*100:.2f}%")
    
    # 准确率最高的类别
    best_classes, worst_classes = find_best_worst_classes(cm, class_names, top_k=5)
    
    lines.append(f"\n【准确率最高的5个类别】")
    lines.append("-" * 60)
    for i, c in enumerate(best_classes, 1):
        lines.append(f"{i}. {c['class_name'][:40]:<40} "
                    f"准确率: {c['accuracy']*100:.2f}% ({c['correct']}/{c['total']})")
    
    lines.append(f"\n【准确率最低的5个类别】")
    lines.append("-" * 60)
    for i, c in enumerate(worst_classes, 1):
        lines.append(f"{i}. {c['class_name'][:40]:<40} "
                    f"准确率: {c['accuracy']*100:.2f}% ({c['correct']}/{c['total']})")
    
    # 最容易混淆的类别对
    confused_pairs = find_confused_pairs(cm, class_names, top_k=10)
    
    lines.append(f"\n【最容易混淆的10个类别对】")
    lines.append("-" * 80)
    lines.append(f"{'真实类别':<30} -> {'预测类别':<30} {'误分类率':>10} {'数量':>8}")
    lines.append("-" * 80)
    
    for pair in confused_pairs:
        true_name = pair['true_class'][:28]
        pred_name = pair['pred_class'][:28]
        lines.append(f"{true_name:<30} -> {pred_name:<30} "
                    f"{pair['rate']*100:>9.2f}% {pair['count']:>8}")
    
    lines.append("\n" + "=" * 80)
    
    report_text = "\n".join(lines)
    
    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 混淆矩阵分析报告已保存: {save_path}")
    
    return report_text


def save_confusion_data(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None
) -> str:
    """
    保存混淆矩阵数据为JSON格式
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        
    Returns:
        保存的文件路径
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if save_path is None:
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(EVALUATION_RESULTS_DIR, "confusion_matrix_data.json")
    
    # 计算各项统计
    best_classes, worst_classes = find_best_worst_classes(cm, class_names, top_k=5)
    confused_pairs = find_confused_pairs(cm, class_names, top_k=10)
    
    data = {
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "best_classes": best_classes,
        "worst_classes": worst_classes,
        "confused_pairs": confused_pairs,
        "statistics": {
            "total_samples": int(cm.sum()),
            "correct_samples": int(np.trace(cm)),
            "overall_accuracy": float(np.trace(cm) / (cm.sum() + 1e-10))
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 混淆矩阵数据已保存: {save_path}")
    return save_path


# 导出
__all__ = [
    'plot_confusion_matrix',
    'plot_confusion_matrix_simplified',
    'find_confused_pairs',
    'find_best_worst_classes',
    'generate_confusion_analysis_report',
    'save_confusion_data',
]
