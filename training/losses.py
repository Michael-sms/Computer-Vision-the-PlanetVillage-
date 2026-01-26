"""
损失函数模块

包含Label Smoothing交叉熵损失函数，用于缓解过拟合和提升模型泛化能力。

Label Smoothing原理:
    将one-hot标签(0,1)软化为(ε/(K-1), 1-ε)
    避免模型对预测过于自信，增强鲁棒性
    
公式:
    y_smooth = (1 - ε) * y_true + ε / K
    其中 ε 为平滑参数，K 为类别数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    
    标签平滑交叉熵损失函数，通过软化标签分布来提升模型泛化能力。
    
    参数:
        smoothing (float): 平滑系数ε，范围[0, 1)
            - 0: 等价于标准交叉熵
            - 0.1: 常用值，推荐用于图像分类
            - 越大越平滑，但可能降低准确率
        num_classes (int): 分类类别数
        reduction (str): 损失归约方式 ('mean' | 'sum' | 'none')
    
    输入:
        inputs: Tensor, shape=[B, num_classes] (未经Softmax的logits)
        targets: Tensor, shape=[B] (类别索引，取值范围[0, num_classes-1])
    
    输出:
        loss: Tensor, 标量损失值（如果reduction='mean'或'sum'）
    
    示例:
        >>> criterion = LabelSmoothingCrossEntropy(smoothing=0.1, num_classes=38)
        >>> logits = torch.randn(4, 38)  # 4个样本，38类
        >>> targets = torch.randint(0, 38, (4,))  # 真实标签
        >>> loss = criterion(logits, targets)
        >>> print(loss.item())
    
    数学原理:
        标准CE:  L = -log(p_true)
        平滑后:  L = -(1-ε)log(p_true) - ε·Σlog(p_i)/K
    """
    
    def __init__(self, smoothing=0.1, num_classes=38, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        
        assert 0 <= smoothing < 1, f"smoothing必须在[0,1)范围内，当前值: {smoothing}"
        
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.reduction = reduction
        
        # 计算平滑后的置信度
        self.confidence = 1.0 - smoothing  # 真实类别的目标概率
        self.smoothing_value = smoothing / (num_classes - 1)  # 其他类别的目标概率
    
    def forward(self, inputs, targets):
        """
        前向传播
        
        流程:
            1. 将logits转换为log概率分布
            2. 构造平滑标签分布
            3. 计算KL散度损失
        """
        # 计算log softmax: log(exp(x_i) / Σexp(x_j))
        log_probs = F.log_softmax(inputs, dim=-1)  # [B, num_classes]
        
        # 构造平滑标签分布
        # 先创建全为smoothing_value的矩阵
        smooth_labels = torch.full_like(log_probs, self.smoothing_value)  # [B, num_classes]
        
        # 将真实类别位置的值设为confidence
        smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # 计算负对数似然损失（NLL Loss）
        # loss = -Σ(smooth_labels * log_probs)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)  # [B]
        
        # 应用归约策略
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"smoothing={self.smoothing}, "
                f"num_classes={self.num_classes}, "
                f"reduction='{self.reduction}')")


def test_label_smoothing():
    """测试Label Smoothing损失函数"""
    print("=" * 60)
    print("Label Smoothing Cross Entropy 测试")
    print("=" * 60)
    
    # 创建损失函数
    criterion_smooth = LabelSmoothingCrossEntropy(smoothing=0.1, num_classes=38)
    criterion_standard = nn.CrossEntropyLoss()
    
    print(f"\n损失函数配置:")
    print(f"  平滑系数(ε): {criterion_smooth.smoothing}")
    print(f"  类别数: {criterion_smooth.num_classes}")
    print(f"  真实类别置信度: {criterion_smooth.confidence:.4f}")
    print(f"  其他类别置信度: {criterion_smooth.smoothing_value:.6f}")
    
    # 测试用例1: 完美预测
    print("\n" + "-" * 60)
    print("测试1: 完美预测（真实类别概率=1.0）")
    logits = torch.zeros(4, 38)
    logits[:, 0] = 10.0  # 第0类的logit很大
    targets = torch.zeros(4, dtype=torch.long)  # 真实标签都是类别0
    
    loss_smooth = criterion_smooth(logits, targets)
    loss_standard = criterion_standard(logits, targets)
    
    print(f"  Label Smoothing损失: {loss_smooth.item():.6f}")
    print(f"  标准交叉熵损失:     {loss_standard.item():.6f}")
    print(f"  损失差异: {(loss_smooth - loss_standard).item():.6f}")
    
    # 测试用例2: 随机预测
    print("\n" + "-" * 60)
    print("测试2: 随机预测")
    logits = torch.randn(4, 38)
    targets = torch.randint(0, 38, (4,))
    
    loss_smooth = criterion_smooth(logits, targets)
    loss_standard = criterion_standard(logits, targets)
    
    print(f"  Label Smoothing损失: {loss_smooth.item():.6f}")
    print(f"  标准交叉熵损失:     {loss_standard.item():.6f}")
    print(f"  损失比例: {loss_smooth.item() / loss_standard.item():.4f}")
    
    # 测试用例3: 不同smoothing值的影响
    print("\n" + "-" * 60)
    print("测试3: 不同平滑系数的影响")
    logits = torch.randn(8, 38)
    targets = torch.randint(0, 38, (8,))
    
    smoothing_values = [0.0, 0.05, 0.1, 0.2, 0.3]
    for eps in smoothing_values:
        criterion = LabelSmoothingCrossEntropy(smoothing=eps, num_classes=38)
        loss = criterion(logits, targets)
        print(f"  ε={eps:.2f}: 损失={loss.item():.6f}")
    
    # 测试用例4: 梯度反向传播
    print("\n" + "-" * 60)
    print("测试4: 梯度反向传播")
    logits = torch.randn(4, 38, requires_grad=True)
    targets = torch.randint(0, 38, (4,))
    
    loss = criterion_smooth(logits, targets)
    loss.backward()
    
    print(f"  损失值: {loss.item():.6f}")
    print(f"  梯度形状: {logits.grad.shape}")
    print(f"  梯度范围: [{logits.grad.min().item():.6f}, {logits.grad.max().item():.6f}]")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    test_label_smoothing()
