"""
测试集评估脚本
加载训练好的模型，在测试集上进行评估，生成完整的评估报告
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mobilenetv2_se import MobileNetV2WithSE
from data.data_loader import get_dataloader, PlantVillageDataset
from data.augmentation import get_val_transforms
from utils.constants import (
    CHECKPOINTS_DIR, EVALUATION_RESULTS_DIR, SPLIT_DATASET_ROOT,
    CLASS_NAMES, CLASS_DISPLAY_NAMES, NUM_CLASSES
)

from evaluation.metrics import (
    compute_accuracy, compute_precision_recall_f1,
    compute_confusion_matrix, compute_macro_metrics,
    compute_weighted_metrics, compute_top_k_accuracy,
    compute_all_metrics, print_metrics_report
)
from evaluation.confusion_analysis import (
    plot_confusion_matrix, plot_confusion_matrix_simplified,
    generate_confusion_analysis_report, save_confusion_data
)
from evaluation.gradcam import (
    GradCAM, generate_gradcam_samples, create_gradcam_summary
)


DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4


def load_model(checkpoint_path: str, device: torch.device) -> MobileNetV2WithSE:
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型权重路径
        device: 计算设备
        
    Returns:
        加载好权重的模型
    """
    print(f"加载模型: {checkpoint_path}")
    
    # 创建模型
    model = MobileNetV2WithSE(
        num_classes=NUM_CLASSES,
        pretrained=False,  # 不需要预训练，直接加载权重
        se_reduction=16
    )
    
    # 加载权重
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 处理不同格式的checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"  - Best Val Acc: {checkpoint.get('best_val_acc', 'N/A')}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        print("✓ 模型加载成功")
    else:
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_on_testset(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    在测试集上进行评估
    
    Args:
        model: 模型
        test_loader: 测试集DataLoader
        device: 计算设备
        
    Returns:
        包含所有预测结果的字典
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("在测试集上进行评估...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    eval_time = time.time() - start_time
    
    results = {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'eval_time': eval_time,
        'num_samples': len(all_labels)
    }
    
    print(f"✓ 评估完成 | 样本数: {results['num_samples']} | 耗时: {eval_time:.2f}秒")
    
    return results


def generate_evaluation_report(
    results: dict,
    save_dir: str
) -> dict:
    """
    生成完整的评估报告
    
    Args:
        results: 评估结果字典
        save_dir: 保存目录
        
    Returns:
        评估报告字典
    """
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    
    print("\n" + "="*60)
    print("生成评估报告")
    print("="*60)
    
    # 1. 计算所有指标
    print("\n1. 计算评估指标...")
    metrics = compute_all_metrics(
        predictions, labels, NUM_CLASSES,
        probabilities=probabilities
    )
    
    # 打印指标报告
    metrics_report_text = print_metrics_report(metrics, CLASS_DISPLAY_NAMES)
    print(metrics_report_text)
    
    # 2. 保存指标到JSON
    metrics_path = os.path.join(save_dir, 'metrics_report.json')
    
    # 转换numpy数组为Python列表
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, dict):
            metrics_serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            metrics_serializable[key] = value
    
    # 添加元信息
    metrics_serializable['metadata'] = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': results['num_samples'],
        'num_classes': NUM_CLASSES,
        'eval_time_seconds': results['eval_time'],
        'class_names': CLASS_NAMES,
        'class_display_names': CLASS_DISPLAY_NAMES
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, ensure_ascii=False, indent=2)
    print(f"✓ 指标报告已保存: {metrics_path}")
    
    # 3. 生成混淆矩阵
    print("\n2. 生成混淆矩阵...")
    confusion_matrix = compute_confusion_matrix(predictions, labels, NUM_CLASSES)
    
    # 完整混淆矩阵
    cm_full_path = os.path.join(save_dir, 'confusion_matrix_full.png')
    plot_confusion_matrix(
        confusion_matrix, CLASS_DISPLAY_NAMES,
        save_path=cm_full_path,
        title='混淆矩阵 (完整)'
    )
    
    # 简化混淆矩阵（显示主要错误）
    cm_simple_path = os.path.join(save_dir, 'confusion_matrix_simplified.png')
    plot_confusion_matrix_simplified(
        confusion_matrix, CLASS_DISPLAY_NAMES,
        save_path=cm_simple_path,
        top_n=15
    )
    
    # 保存混淆矩阵数据
    cm_data_path = os.path.join(save_dir, 'confusion_matrix_data.json')
    save_confusion_data(confusion_matrix, CLASS_NAMES, cm_data_path)
    
    # 4. 混淆分析
    print("\n3. 生成混淆分析...")
    analysis_path = os.path.join(save_dir, 'confusion_analysis.txt')
    generate_confusion_analysis_report(
        confusion_matrix, CLASS_DISPLAY_NAMES,
        save_path=analysis_path
    )
    
    return metrics_serializable


def generate_gradcam_visualizations(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str
):
    """
    生成Grad-CAM可视化
    
    Args:
        model: 模型
        test_loader: 测试集DataLoader
        device: 计算设备
        save_dir: 保存目录
    """
    print("\n4. 生成Grad-CAM可视化...")
    
    # 获取目标层（MobileNetV2的最后一个卷积层）
    target_layer = model.features[-1]
    
    # 生成汇总图
    summary_path = os.path.join(save_dir, 'gradcam_summary.png')
    create_gradcam_summary(
        model, test_loader, target_layer, device,
        num_samples=16, save_path=summary_path
    )
    
    # 生成各类别样本
    gradcam_dir = os.path.join(save_dir, 'gradcam_samples')
    generate_gradcam_samples(
        model, test_loader, target_layer, device,
        num_samples_per_class=3,
        num_classes_to_show=10,
        save_dir=gradcam_dir
    )


def main():
    parser = argparse.ArgumentParser(description='植物病害分类模型测试评估')
    parser.add_argument('--checkpoint', type=str, 
                       default=os.path.join(CHECKPOINTS_DIR, 'best_model.pth'),
                       help='模型权重路径')
    parser.add_argument('--test-split', type=str,
                       default=os.path.join(SPLIT_DATASET_ROOT, 'test.txt'),
                       help='测试集划分文件路径')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
                       help='DataLoader的工作进程数')
    parser.add_argument('--output-dir', type=str, default=EVALUATION_RESULTS_DIR,
                       help='结果输出目录')
    parser.add_argument('--no-gradcam', action='store_true',
                       help='跳过Grad-CAM可视化')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 加载测试集
    print(f"\n加载测试集: {args.test_split}")
    if os.path.isfile(args.test_split):
        # 自定义划分文件
        val_transform = get_val_transforms()
        dataset = PlantVillageDataset(
            split_file=args.test_split,
            transform=val_transform,
            return_path=True
        )
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        # 使用默认split名称
        test_loader = get_dataloader(
            split="test",
            split_file=None,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            return_path=True
        )
    print(f"✓ 测试集加载完成 | 批次数: {len(test_loader)}")
    
    # 在测试集上评估
    results = evaluate_on_testset(model, test_loader, device)
    
    # 生成评估报告
    metrics = generate_evaluation_report(results, args.output_dir)
    
    # 生成Grad-CAM可视化
    if not args.no_gradcam:
        generate_gradcam_visualizations(model, test_loader, device, args.output_dir)
    
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)
    print(f"\n结果已保存到: {args.output_dir}")
    print(f"  - metrics_report.json: 完整评估指标")
    print(f"  - confusion_matrix_full.png: 完整混淆矩阵")
    print(f"  - confusion_matrix_simplified.png: 简化混淆矩阵")
    print(f"  - confusion_analysis.txt: 混淆分析报告")
    if not args.no_gradcam:
        print(f"  - gradcam_summary.png: Grad-CAM汇总图")
        print(f"  - gradcam_samples/: 各类别Grad-CAM样本")
    
    # 打印关键指标摘要
    print(f"\n关键指标摘要:")
    print(f"  - 测试集准确率: {metrics.get('accuracy', 0)*100:.2f}%")
    if 'top_3_accuracy' in metrics:
        print(f"  - Top-3 准确率: {metrics['top_3_accuracy']*100:.2f}%")
    if 'top_5_accuracy' in metrics:
        print(f"  - Top-5 准确率: {metrics['top_5_accuracy']*100:.2f}%")
    print(f"  - 宏平均 F1: {metrics.get('macro_f1', 0):.4f}")
    print(f"  - 加权平均 F1: {metrics.get('weighted_f1', 0):.4f}")


if __name__ == '__main__':
    main()
