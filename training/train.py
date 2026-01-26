"""
训练主程序

实现完整的训练流程，包括：
- 训练循环
- 验证评估
- 模型保存
- 早停机制
- TensorBoard日志
"""

import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import MobileNetV2WithSE
from training.losses import LabelSmoothingCrossEntropy
from training.lr_scheduler import get_cosine_scheduler
from training.validate import validate


class Trainer:
    """
    训练器类，封装完整的训练流程
    
    参数:
        model: 待训练的模型
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
        config (dict): 训练配置字典
        device: 计算设备
    """
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 初始化训练组件
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_tracking()
        
        # TensorBoard
        log_dir = os.path.join(config['save_dir'], 'logs', 
                               datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)
        print(f"[TensorBoard] 日志目录: {log_dir}")
    
    def _setup_criterion(self):
        """设置损失函数"""
        if self.config.get('label_smoothing', 0.0) > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=self.config['label_smoothing'],
                num_classes=self.config['num_classes']
            )
            print(f"[损失函数] Label Smoothing CE (ε={self.config['label_smoothing']})")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print(f"[损失函数] 标准交叉熵")
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        print(f"[优化器] Adam (lr={self.config['learning_rate']}, "
              f"wd={self.config.get('weight_decay', 1e-5)})")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        self.scheduler = get_cosine_scheduler(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config.get('min_lr', 1e-6),
            warmup_epochs=self.config.get('warmup_epochs', 0),
            warmup_lr=self.config.get('warmup_lr', 1e-6)
        )
        
        print(f"[调度器] Cosine Annealing (T_max={self.config['epochs']}, "
              f"eta_min={self.config.get('min_lr', 1e-6)})")
    
    def _setup_tracking(self):
        """设置训练追踪变量"""
        self.start_epoch = 0
        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        返回:
            dict: 包含训练损失和准确率
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        # 进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # 数据移至设备
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（可选）
            if self.config.get('grad_clip', 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'acc': 100.0 * correct / total_samples,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # TensorBoard记录（每N步）
            if batch_idx % self.config.get('log_interval', 50) == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self):
        """
        完整训练流程
        """
        print("\n" + "=" * 70)
        print("开始训练".center(70))
        print("=" * 70)
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            epoch_start_time = time.time()
            
            # ===== 训练阶段 =====
            train_metrics = self.train_epoch(epoch)
            
            # ===== 验证阶段 =====
            val_metrics = validate(
                self.model, 
                self.val_loader, 
                self.criterion, 
                self.device
            )
            
            # ===== 更新学习率 =====
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # ===== 记录历史 =====
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # ===== TensorBoard记录 =====
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_acc', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_acc', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)
            
            # ===== 打印摘要 =====
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} - {epoch_time:.2f}s")
            print(f"  训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
                  f"Top5: {val_metrics['top5_accuracy']:.2f}%")
            print(f"  学习率: {current_lr:.8f}")
            
            # ===== 模型保存 =====
            is_best = val_metrics['accuracy'] > self.best_acc
            
            if is_best:
                self.best_acc = val_metrics['accuracy']
                self.best_loss = val_metrics['loss']
                self.epochs_no_improve = 0
                
                # 保存最佳模型
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ 最佳模型已保存 (Acc: {self.best_acc:.2f}%)")
            else:
                self.epochs_no_improve += 1
            
            # 定期保存checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # ===== 早停检查 =====
            patience = self.config.get('patience', 10)
            if self.epochs_no_improve >= patience:
                print(f"\n早停: 验证准确率连续{patience}轮未提升")
                break
            
            print("-" * 70)
        
        # 训练结束
        self.writer.close()
        self.save_training_history()
        
        print("\n" + "=" * 70)
        print("训练完成".center(70))
        print("=" * 70)
        print(f"最佳验证准确率: {self.best_acc:.2f}%")
        print(f"最佳验证损失: {self.best_loss:.4f}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存模型checkpoint
        
        参数:
            epoch (int): 当前轮数
            is_best (bool): 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # 保存最佳模型
        if is_best:
            save_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, save_path)
        
        # 保存最新checkpoint
        save_path = os.path.join(self.config['save_dir'], 'last_checkpoint.pth')
        torch.save(checkpoint, save_path)
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config['save_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"[保存] 训练历史: {history_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载checkpoint继续训练
        
        参数:
            checkpoint_path (str): checkpoint文件路径
        """
        if not os.path.exists(checkpoint_path):
            print(f"[警告] checkpoint不存在: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        self.best_loss = checkpoint['best_loss']
        
        print(f"[加载] Checkpoint: epoch={self.start_epoch}, best_acc={self.best_acc:.2f}%")


def get_default_config():
    """
    获取默认训练配置
    
    返回:
        dict: 配置字典
    """
    return {
        # 模型参数
        'num_classes': 38,
        'pretrained': True,
        'freeze_layers': 10,
        'dropout_rate': 0.2,
        
        # 训练参数
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        
        # 学习率调度
        'warmup_epochs': 0,
        'warmup_lr': 1e-6,
        
        # Label Smoothing
        'label_smoothing': 0.1,
        
        # 正则化
        'grad_clip': 1.0,
        
        # 早停
        'patience': 10,
        
        # 保存设置
        'save_dir': './checkpoints',
        'save_interval': 10,
        'log_interval': 50,
    }


if __name__ == "__main__":
    print("训练模块已加载")
    print("请使用以下方式启动训练:")
    print("""
    from training.train import Trainer, get_default_config
    from models import MobileNetV2WithSE
    from data import get_all_dataloaders
    
    # 加载数据
    train_loader, val_loader, _ = get_all_dataloaders(batch_size=32)
    
    # 创建模型
    model = MobileNetV2WithSE(num_classes=38, pretrained=True)
    
    # 获取配置
    config = get_default_config()
    
    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # 开始训练
    trainer.train()
    """)
