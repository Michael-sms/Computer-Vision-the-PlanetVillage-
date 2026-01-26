"""
评估可视化工具
- 训练曲线 (loss/acc/lr)
- 各类别指标对比 (F1/Precision/Recall)
"""

import os
import json
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import EVALUATION_RESULTS_DIR, CHECKPOINTS_DIR, CLASS_DISPLAY_NAMES

# 配置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_history(
    history_path: Optional[str] = None,
    save_dir: str = EVALUATION_RESULTS_DIR
) -> str:
    """
    绘制训练历史曲线 (loss/acc/lr)
    
    Args:
        history_path: 训练历史JSON路径，默认 `checkpoints/training_history.json`
        save_dir: 图像保存目录
        
    Returns:
        保存的图像路径
    """
    if history_path is None:
        history_path = os.path.join(CHECKPOINTS_DIR, 'training_history.json')
    _ensure_dir(save_dir)
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"训练历史文件不存在: {history_path}")
    
    history = load_json(history_path)
    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(epochs, history.get('train_loss', []), label='Train Loss')
    axes[0].plot(epochs, history.get('val_loss', []), label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(epochs, history.get('train_acc', []), label='Train Acc')
    axes[1].plot(epochs, history.get('val_acc', []), label='Val Acc')
    axes[1].set_title('Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    
    # Learning rate
    axes[2].plot(epochs, history.get('learning_rate', []), color='tab:green')
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 训练曲线已保存: {save_path}")
    return save_path


def _extract_class_metrics(class_metrics: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将class_metrics拆分为数组"""
    num_classes = len(class_metrics)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for cid_str, metrics in class_metrics.items():
        cid = int(cid_str)
        precision[cid] = metrics.get('precision', 0.0)
        recall[cid] = metrics.get('recall', 0.0)
        f1[cid] = metrics.get('f1', 0.0)
    
    return precision, recall, f1


def plot_class_metrics(
    metrics_path: Optional[str] = None,
    save_dir: str = EVALUATION_RESULTS_DIR,
    top_k: int = 10
) -> str:
    """
    绘制各类别Precision/Recall/F1的对比条形图
    
    Args:
        metrics_path: 指标JSON路径，默认 `results/evaluated/metrics_report.json`
        save_dir: 图像保存目录
        top_k: 展示Top-K最佳与最差类别
        
    Returns:
        保存的图像路径
    """
    if metrics_path is None:
        metrics_path = os.path.join(EVALUATION_RESULTS_DIR, 'metrics_report.json')
    _ensure_dir(save_dir)
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"指标文件不存在: {metrics_path}")
    
    metrics = load_json(metrics_path)
    class_metrics = metrics.get('class_metrics', {})
    precision, recall, f1 = _extract_class_metrics(class_metrics)
    
    indices = np.arange(len(f1))
    names = np.array(CLASS_DISPLAY_NAMES)
    
    # 选择Top-K最佳和最差F1类别
    sorted_idx = np.argsort(f1)
    worst_idx = sorted_idx[:top_k]
    best_idx = sorted_idx[-top_k:][::-1]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    bar_width = 0.25
    
    def _plot_group(ax, idx_list, title):
        x_pos = np.arange(len(idx_list))  # 使用连续的x轴位置
        ax.bar(x_pos - bar_width, precision[idx_list], width=bar_width, label='Precision')
        ax.bar(x_pos, recall[idx_list], width=bar_width, label='Recall')
        ax.bar(x_pos + bar_width, f1[idx_list], width=bar_width, label='F1')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names[idx_list], rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    _plot_group(axes[0], worst_idx, f'F1最低的{top_k}个类别')
    _plot_group(axes[1], best_idx, f'F1最高的{top_k}个类别')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'class_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 类别指标对比图已保存: {save_path}")
    return save_path


__all__ = [
    'plot_training_history',
    'plot_class_metrics',
    'load_json'
]
