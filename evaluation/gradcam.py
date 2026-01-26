"""
Grad-CAM 可视化模块
生成类激活热力图，可视化模型在分类决策时关注的图像区域
验证模型是否真正学到了病害特征而非背景噪声
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Optional, Tuple, List, Union
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    EVALUATION_RESULTS_DIR, CLASS_NAMES, CLASS_DISPLAY_NAMES, 
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
)


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) 可视化类
    
    通过计算目标类别对特征图的梯度来生成热力图，
    显示模型在做出分类决策时关注的图像区域。
    
    参考论文: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
    
    Args:
        model: 训练好的PyTorch模型
        target_layer: 目标特征层（通常是最后一个卷积层）
    
    示例:
        >>> model = MobileNetV2WithSE(num_classes=38)
        >>> gradcam = GradCAM(model, model.features[-1])
        >>> cam = gradcam.generate_cam(input_tensor, class_id=5)
        >>> gradcam.visualize(original_image, cam, save_path="cam.png")
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        
        # 存储前向传播的特征图和梯度
        self.activations = None
        self.gradients = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        class_id: Optional[int] = None
    ) -> np.ndarray:
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入图像张量，shape=[1, 3, H, W]，已标准化
            class_id: 目标类别ID，如果为None则使用预测类别
            
        Returns:
            热力图，shape=[H, W]，值域[0, 1]
        """
        self.model.eval()
        
        # 确保输入需要梯度
        input_tensor = input_tensor.clone()
        input_tensor.requires_grad_(True)
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 如果没有指定类别，使用预测类别
        if class_id is None:
            class_id = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_id] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算梯度的全局平均池化权重
        # gradients shape: [1, C, H, W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和
        # activations shape: [1, C, H, W]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU激活（只保留正值）
        cam = F.relu(cam)
        
        # 上采样到输入图像尺寸
        cam = F.interpolate(
            cam, 
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        # 归一化到[0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        original_image: Union[np.ndarray, Image.Image, torch.Tensor],
        cam: np.ndarray,
        save_path: Optional[str] = None,
        alpha: float = 0.5,
        colormap: str = 'jet',
        title: str = None
    ) -> np.ndarray:
        """
        将热力图叠加到原始图像上进行可视化
        
        Args:
            original_image: 原始图像 (未标准化)
            cam: Grad-CAM热力图，shape=[H, W]，值域[0, 1]
            save_path: 保存路径
            alpha: 热力图透明度
            colormap: 颜色映射
            title: 图像标题
            
        Returns:
            叠加后的图像，shape=[H, W, 3]
        """
        # 转换原始图像格式
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.cpu().numpy()
            if original_image.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                original_image = original_image.transpose(1, 2, 0)
            # 反标准化
            original_image = original_image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            original_image = np.clip(original_image, 0, 1)
        elif isinstance(original_image, Image.Image):
            original_image = np.array(original_image) / 255.0
        
        # 确保图像在[0, 1]范围
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # 调整CAM尺寸匹配原始图像
        if cam.shape != original_image.shape[:2]:
            cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (original_image.shape[1], original_image.shape[0]),
                Image.BILINEAR
            )) / 255.0
        else:
            cam_resized = cam
        
        # 应用颜色映射
        cmap = plt.get_cmap(colormap)
        cam_colored = cmap(cam_resized)[:, :, :3]  # 去掉alpha通道
        
        # 叠加
        overlay = (1 - alpha) * original_image + alpha * cam_colored
        overlay = np.clip(overlay, 0, 1)
        
        # 保存或显示
        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_image)
            axes[0].set_title('原始图像', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(cam_resized, cmap=colormap)
            axes[1].set_title('Grad-CAM 热力图', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('叠加结果', fontsize=12)
            axes[2].axis('off')
            
            if title:
                fig.suptitle(title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return (overlay * 255).astype(np.uint8)


def generate_gradcam_samples(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_layer: torch.nn.Module,
    device: torch.device,
    num_samples_per_class: int = 3,
    num_classes_to_show: int = 10,
    save_dir: Optional[str] = None,
    random_seed: int = 42
) -> str:
    """
    为每个类别生成Grad-CAM可视化样本
    
    Args:
        model: 训练好的模型
        dataloader: 测试集DataLoader（需要return_path=True）
        target_layer: 目标特征层
        device: 计算设备
        num_samples_per_class: 每个类别的样本数
        num_classes_to_show: 展示的类别数
        save_dir: 保存目录
        random_seed: 随机种子
        
    Returns:
        保存目录路径
    """
    if save_dir is None:
        save_dir = os.path.join(EVALUATION_RESULTS_DIR, "gradcam_samples")
    os.makedirs(save_dir, exist_ok=True)
    
    random.seed(random_seed)
    model.eval()
    
    # 创建GradCAM对象
    gradcam = GradCAM(model, target_layer)
    
    # 收集每个类别的样本
    class_samples = {i: [] for i in range(len(CLASS_NAMES))}
    
    print("收集样本...")
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [None] * len(labels)
            
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            for i, (img, label, pred, path) in enumerate(zip(images, labels, preds, paths)):
                label = label.item()
                pred = pred.item()
                
                if len(class_samples[label]) < num_samples_per_class * 2:  # 多收集一些
                    class_samples[label].append({
                        'image': img.cpu(),
                        'label': label,
                        'pred': pred,
                        'correct': label == pred,
                        'path': path
                    })
    
    # 选择要展示的类别（均匀采样）
    step = max(1, len(CLASS_NAMES) // num_classes_to_show)
    selected_classes = list(range(0, len(CLASS_NAMES), step))[:num_classes_to_show]
    
    print(f"生成Grad-CAM可视化 (共{num_classes_to_show}个类别)...")
    
    for class_id in selected_classes:
        samples = class_samples[class_id]
        if not samples:
            continue
        
        # 随机选择样本
        random.shuffle(samples)
        samples = samples[:num_samples_per_class]
        
        class_name = CLASS_DISPLAY_NAMES[class_id]
        class_dir = os.path.join(save_dir, f"class_{class_id:02d}_{class_name[:10]}")
        os.makedirs(class_dir, exist_ok=True)
        
        for idx, sample in enumerate(samples):
            img_tensor = sample['image'].unsqueeze(0).to(device)
            
            # 生成CAM
            cam = gradcam.generate_cam(img_tensor, class_id=sample['label'])
            
            # 反标准化图像用于可视化
            img_vis = sample['image'].numpy().transpose(1, 2, 0)
            img_vis = img_vis * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img_vis = np.clip(img_vis, 0, 1)
            
            # 生成标题
            pred_name = CLASS_DISPLAY_NAMES[sample['pred']]
            status = "✓" if sample['correct'] else "✗"
            title = f"真实: {class_name} | 预测: {pred_name} {status}"
            
            # 保存
            save_path = os.path.join(class_dir, f"sample_{idx+1}.png")
            gradcam.visualize(img_vis, cam, save_path=save_path, title=title)
    
    print(f"✓ Grad-CAM可视化已保存: {save_dir}")
    return save_dir


def create_gradcam_summary(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_layer: torch.nn.Module,
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[str] = None
) -> str:
    """
    创建Grad-CAM汇总图（一张图展示多个样本）
    
    Args:
        model: 训练好的模型
        dataloader: 测试集DataLoader
        target_layer: 目标特征层
        device: 计算设备
        num_samples: 展示的样本数
        save_path: 保存路径
        
    Returns:
        保存路径
    """
    if save_path is None:
        save_path = os.path.join(EVALUATION_RESULTS_DIR, "gradcam_summary.png")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    gradcam = GradCAM(model, target_layer)
    
    # 收集样本
    samples = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            for img, label, pred in zip(images, labels, preds):
                samples.append({
                    'image': img.cpu(),
                    'label': label.item(),
                    'pred': pred.item()
                })
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
    
    # 创建汇总图
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=(grid_size * 4, grid_size * 2))
    
    for idx, sample in enumerate(samples):
        if idx >= grid_size * grid_size:
            break
        
        row = idx // grid_size
        col = (idx % grid_size) * 2
        
        img_tensor = sample['image'].unsqueeze(0).to(device)
        cam = gradcam.generate_cam(img_tensor, class_id=sample['label'])
        
        # 反标准化
        img_vis = sample['image'].numpy().transpose(1, 2, 0)
        img_vis = img_vis * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_vis = np.clip(img_vis, 0, 1)
        
        # 叠加
        cmap = plt.get_cmap('jet')
        cam_colored = cmap(cam)[:, :, :3]
        overlay = 0.5 * img_vis + 0.5 * cam_colored
        overlay = np.clip(overlay, 0, 1)
        
        # 原图
        axes[row, col].imshow(img_vis)
        axes[row, col].axis('off')
        
        # CAM叠加
        axes[row, col + 1].imshow(overlay)
        axes[row, col + 1].axis('off')
        
        # 标题
        status = "✓" if sample['label'] == sample['pred'] else "✗"
        axes[row, col].set_title(f"{CLASS_DISPLAY_NAMES[sample['label']][:8]}", fontsize=8)
        axes[row, col + 1].set_title(f"预测: {CLASS_DISPLAY_NAMES[sample['pred']][:6]} {status}", fontsize=8)
    
    # 隐藏空白子图
    for idx in range(len(samples), grid_size * grid_size):
        row = idx // grid_size
        col = (idx % grid_size) * 2
        if row < len(axes) and col + 1 < len(axes[row]):
            axes[row, col].axis('off')
            axes[row, col + 1].axis('off')
    
    plt.suptitle('Grad-CAM 可视化汇总', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Grad-CAM汇总图已保存: {save_path}")
    return save_path


# 导出
__all__ = [
    'GradCAM',
    'generate_gradcam_samples',
    'create_gradcam_summary',
]
