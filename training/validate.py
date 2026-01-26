"""
验证模块

实现模型在验证集上的评估，计算损失和准确率等指标。
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def validate(model, val_loader, criterion, device):
    """
    在验证集上评估模型性能
    
    参数:
        model: 待评估的模型
        val_loader: 验证集DataLoader
        criterion: 损失函数
        device: 计算设备 ('cuda' 或 'cpu')
    
    返回:
        dict: 包含以下键值对:
            - 'loss': 平均验证损失
            - 'accuracy': Top-1准确率
            - 'top5_accuracy': Top-5准确率
    
    示例:
        >>> metrics = validate(model, val_loader, criterion, device)
        >>> print(f"验证损失: {metrics['loss']:.4f}")
        >>> print(f"准确率: {metrics['accuracy']:.2f}%")
    """
    model.eval()  # 设置为评估模式（关闭Dropout、BN使用移动平均）
    
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    # 禁用梯度计算，节省显存和加速推理
    with torch.no_grad():
        # 使用tqdm显示进度条
        pbar = tqdm(val_loader, desc='验证中', leave=False)
        
        for images, labels in pbar:
            # 数据移至设备
            images = images.to(device)
            labels = labels.to(device)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item() * batch_size
            
            # 计算Top-1准确率
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()
            
            # 计算Top-5准确率
            _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(labels.view(1, -1).expand_as(pred_top5)).sum().item()
            
            # 更新进度条信息
            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'acc': 100.0 * correct_top1 / total_samples
            })
    
    # 计算平均指标
    avg_loss = total_loss / total_samples
    top1_accuracy = 100.0 * correct_top1 / total_samples
    top5_accuracy = 100.0 * correct_top5 / total_samples
    
    return {
        'loss': avg_loss,
        'accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy
    }


def validate_with_class_accuracy(model, val_loader, criterion, device, num_classes=38):
    """
    在验证集上评估模型性能（包含每类准确率）
    
    参数:
        model: 待评估的模型
        val_loader: 验证集DataLoader
        criterion: 损失函数
        device: 计算设备
        num_classes (int): 类别数，默认38
    
    返回:
        dict: 包含以下键值对:
            - 'loss': 平均验证损失
            - 'accuracy': 总体准确率
            - 'top5_accuracy': Top-5准确率
            - 'class_correct': 每类正确预测数 (list)
            - 'class_total': 每类总样本数 (list)
            - 'class_accuracy': 每类准确率 (list)
    """
    model.eval()
    
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    # 每类统计
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='详细验证中', leave=False)
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * batch_size
            
            # Top-1预测
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()
            
            # Top-5预测
            _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(labels.view(1, -1).expand_as(pred_top5)).sum().item()
            
            # 统计每类准确率
            for i in range(batch_size):
                label = labels[i].item()
                class_total[label] += 1
                if pred_top1[i] == label:
                    class_correct[label] += 1
            
            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'acc': 100.0 * correct_top1 / total_samples
            })
    
    # 计算每类准确率
    class_accuracy = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
        else:
            acc = 0.0
        class_accuracy.append(acc)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': 100.0 * correct_top1 / total_samples,
        'top5_accuracy': 100.0 * correct_top5 / total_samples,
        'class_correct': class_correct,
        'class_total': class_total,
        'class_accuracy': class_accuracy
    }


def test_validate():
    """测试验证函数"""
    print("=" * 60)
    print("验证函数测试")
    print("=" * 60)
    
    # 创建模拟模型和数据
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(3*224*224, 38)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建模拟数据加载器
    class FakeDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return torch.randn(3, 224, 224), torch.randint(0, 38, (1,)).item()
    
    val_loader = torch.utils.data.DataLoader(
        FakeDataset(), 
        batch_size=16, 
        shuffle=False
    )
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 测试基础验证
    print("\n测试1: 基础验证")
    print("-" * 60)
    metrics = validate(model, val_loader, criterion, device)
    print(f"  验证损失: {metrics['loss']:.4f}")
    print(f"  Top-1准确率: {metrics['accuracy']:.2f}%")
    print(f"  Top-5准确率: {metrics['top5_accuracy']:.2f}%")
    
    # 测试带类别准确率的验证
    print("\n测试2: 详细验证（含每类准确率）")
    print("-" * 60)
    metrics = validate_with_class_accuracy(model, val_loader, criterion, device, num_classes=38)
    print(f"  验证损失: {metrics['loss']:.4f}")
    print(f"  总体准确率: {metrics['accuracy']:.2f}%")
    print(f"  Top-5准确率: {metrics['top5_accuracy']:.2f}%")
    print(f"  每类准确率 (前5类): {[f'{acc:.1f}%' for acc in metrics['class_accuracy'][:5]]}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    test_validate()
