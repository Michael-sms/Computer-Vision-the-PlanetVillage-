"""
评估指标计算模块
计算准确率、精确率、召回率、F1-score等分类评估指标
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import torch


def compute_accuracy(predictions: Union[np.ndarray, torch.Tensor], 
                     labels: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算总体准确率
    
    Args:
        predictions: 预测类别索引，shape=[N]
        labels: 真实类别索引，shape=[N]
        
    Returns:
        准确率 (0.0 ~ 1.0)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    correct = (predictions == labels).sum()
    total = len(labels)
    
    return correct / total if total > 0 else 0.0


def compute_precision_recall_f1(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int
) -> Dict[int, Dict[str, float]]:
    """
    计算每个类别的精确率、召回率和F1-score
    
    Args:
        predictions: 预测类别索引，shape=[N]
        labels: 真实类别索引，shape=[N]
        num_classes: 类别总数
        
    Returns:
        字典 {class_id: {"precision": float, "recall": float, "f1": float}}
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    metrics = {}
    
    for c in range(num_classes):
        # True Positive: 预测为c且真实为c
        tp = ((predictions == c) & (labels == c)).sum()
        # False Positive: 预测为c但真实不是c
        fp = ((predictions == c) & (labels != c)).sum()
        # False Negative: 预测不是c但真实是c
        fn = ((predictions != c) & (labels == c)).sum()
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((labels == c).sum())  # 该类别的样本数
        }
    
    return metrics


def compute_confusion_matrix(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int
) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        predictions: 预测类别索引，shape=[N]
        labels: 真实类别索引，shape=[N]
        num_classes: 类别总数
        
    Returns:
        混淆矩阵，shape=[num_classes, num_classes]
        cm[i, j] 表示真实类别为i，预测类别为j的样本数
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for true_label, pred_label in zip(labels, predictions):
        cm[true_label, pred_label] += 1
    
    return cm


def compute_macro_metrics(class_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """
    计算宏平均指标（所有类别平均）
    
    Args:
        class_metrics: 每类指标字典（来自compute_precision_recall_f1）
        
    Returns:
        {"macro_precision": float, "macro_recall": float, "macro_f1": float}
    """
    precisions = [m["precision"] for m in class_metrics.values()]
    recalls = [m["recall"] for m in class_metrics.values()]
    f1s = [m["f1"] for m in class_metrics.values()]
    
    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s))
    }


def compute_weighted_metrics(class_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """
    计算加权平均指标（按样本数加权）
    
    Args:
        class_metrics: 每类指标字典（来自compute_precision_recall_f1）
        
    Returns:
        {"weighted_precision": float, "weighted_recall": float, "weighted_f1": float}
    """
    total_support = sum(m["support"] for m in class_metrics.values())
    
    if total_support == 0:
        return {"weighted_precision": 0.0, "weighted_recall": 0.0, "weighted_f1": 0.0}
    
    weighted_precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / total_support
    weighted_recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_support
    weighted_f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_support
    
    return {
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1)
    }


def compute_top_k_accuracy(
    probabilities: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    k: int = 5
) -> float:
    """
    计算Top-K准确率
    
    Args:
        probabilities: 预测概率，shape=[N, num_classes]
        labels: 真实类别索引，shape=[N]
        k: Top-K的K值
        
    Returns:
        Top-K准确率 (0.0 ~ 1.0)
    """
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 获取Top-K预测
    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
    
    # 检查真实标签是否在Top-K中
    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_preds[i]:
            correct += 1
    
    return correct / len(labels) if len(labels) > 0 else 0.0


def compute_all_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    probabilities: Union[np.ndarray, torch.Tensor] = None
) -> Dict:
    """
    计算所有评估指标
    
    Args:
        predictions: 预测类别索引，shape=[N]
        labels: 真实类别索引，shape=[N]
        num_classes: 类别总数
        probabilities: 预测概率（可选），用于计算Top-K准确率
        
    Returns:
        包含所有指标的字典
    """
    # 基础指标
    accuracy = compute_accuracy(predictions, labels)
    class_metrics = compute_precision_recall_f1(predictions, labels, num_classes)
    confusion_matrix = compute_confusion_matrix(predictions, labels, num_classes)
    
    # 宏平均和加权平均
    macro_metrics = compute_macro_metrics(class_metrics)
    weighted_metrics = compute_weighted_metrics(class_metrics)
    
    result = {
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),  # 转为列表便于JSON序列化
        **macro_metrics,
        **weighted_metrics
    }
    
    # Top-K准确率（如果提供了概率）
    if probabilities is not None:
        result["top_3_accuracy"] = compute_top_k_accuracy(probabilities, labels, k=3)
        result["top_5_accuracy"] = compute_top_k_accuracy(probabilities, labels, k=5)
    
    return result


def print_metrics_report(metrics: Dict, class_names: List[str] = None) -> str:
    """
    生成可读的指标报告
    
    Args:
        metrics: compute_all_metrics返回的字典
        class_names: 类别名称列表（可选）
        
    Returns:
        格式化的报告字符串
    """
    lines = []
    lines.append("=" * 80)
    lines.append("评估报告 (Evaluation Report)")
    lines.append("=" * 80)
    
    # 总体指标
    lines.append("\n【总体指标 Overall Metrics】")
    lines.append("-" * 40)
    lines.append(f"准确率 (Accuracy):          {metrics['accuracy']*100:.2f}%")
    
    if "top_3_accuracy" in metrics:
        lines.append(f"Top-3 准确率:               {metrics['top_3_accuracy']*100:.2f}%")
    if "top_5_accuracy" in metrics:
        lines.append(f"Top-5 准确率:               {metrics['top_5_accuracy']*100:.2f}%")
    
    lines.append(f"\n宏平均精确率 (Macro Precision): {metrics['macro_precision']*100:.2f}%")
    lines.append(f"宏平均召回率 (Macro Recall):    {metrics['macro_recall']*100:.2f}%")
    lines.append(f"宏平均F1 (Macro F1):            {metrics['macro_f1']*100:.2f}%")
    
    lines.append(f"\n加权精确率 (Weighted Precision): {metrics['weighted_precision']*100:.2f}%")
    lines.append(f"加权召回率 (Weighted Recall):    {metrics['weighted_recall']*100:.2f}%")
    lines.append(f"加权F1 (Weighted F1):            {metrics['weighted_f1']*100:.2f}%")
    
    # 每类指标
    lines.append("\n【各类别指标 Per-Class Metrics】")
    lines.append("-" * 80)
    lines.append(f"{'类别':<35} {'精确率':>10} {'召回率':>10} {'F1':>10} {'样本数':>10}")
    lines.append("-" * 80)
    
    class_metrics = metrics["class_metrics"]
    # 处理 key 可能是 int 或 str 的情况
    sorted_keys = sorted(class_metrics.keys(), key=lambda x: int(x))
    for class_id_key in sorted_keys:
        m = class_metrics[class_id_key]
        class_id = int(class_id_key)
        if class_names and class_id < len(class_names):
            name = class_names[class_id][:32]
        else:
            name = f"Class_{class_id}"
        
        lines.append(f"{name:<35} {m['precision']*100:>9.2f}% {m['recall']*100:>9.2f}% "
                    f"{m['f1']*100:>9.2f}% {m['support']:>10}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# 导出
__all__ = [
    'compute_accuracy',
    'compute_precision_recall_f1',
    'compute_confusion_matrix',
    'compute_macro_metrics',
    'compute_weighted_metrics',
    'compute_top_k_accuracy',
    'compute_all_metrics',
    'print_metrics_report',
]
