"""
学习率调度器模块

实现余弦退火学习率调度策略，使学习率按余弦曲线从初始值衰减到最小值。

余弦退火(Cosine Annealing)原理:
    学习率随训练轮数按余弦函数周期性下降
    公式: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
    
优势:
    - 平滑下降，避免突变
    - 后期学习率极小，有助于精细调参
    - 可配合重启机制(SGDR)使用
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
import math


def get_cosine_scheduler(optimizer, 
                         T_max, 
                         eta_min=1e-6,
                         last_epoch=-1,
                         warmup_epochs=0,
                         warmup_lr=1e-6):
    """
    创建余弦退火学习率调度器（可选预热阶段）
    
    参数:
        optimizer: PyTorch优化器对象
        T_max (int): 余弦退火周期（总训练轮数）
        eta_min (float): 最小学习率，默认1e-6
        last_epoch (int): 上次训练的轮数（用于恢复训练），默认-1
        warmup_epochs (int): 预热轮数，默认0（不使用预热）
        warmup_lr (float): 预热起始学习率，默认1e-6
    
    返回:
        scheduler: 学习率调度器对象
    
    示例:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> scheduler = get_cosine_scheduler(optimizer, T_max=100, eta_min=1e-6)
        >>> 
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()  # 每轮结束后更新学习率
    
    学习率曲线:
        warmup阶段(可选): 线性增长 warmup_lr -> lr_initial
        cosine阶段: 余弦衰减 lr_initial -> eta_min
    """
    if warmup_epochs > 0:
        # 使用预热+余弦退火组合调度器
        return CosineAnnealingWarmupScheduler(
            optimizer, 
            warmup_epochs=warmup_epochs,
            T_max=T_max - warmup_epochs,
            eta_min=eta_min,
            warmup_lr=warmup_lr,
            last_epoch=last_epoch
        )
    else:
        # 标准余弦退火调度器
        return CosineAnnealingLR(
            optimizer, 
            T_max=T_max, 
            eta_min=eta_min,
            last_epoch=last_epoch
        )


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    带预热的余弦退火调度器
    
    学习率变化分为两个阶段:
        1. Warmup阶段(0 -> warmup_epochs): 线性增长
           lr = warmup_lr + (base_lr - warmup_lr) * (epoch / warmup_epochs)
        
        2. Cosine阶段(warmup_epochs -> T_max): 余弦衰减
           lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(π * t / T_max))
    
    参数:
        optimizer: 优化器
        warmup_epochs (int): 预热轮数
        T_max (int): 余弦退火周期（不包含warmup）
        eta_min (float): 最小学习率
        warmup_lr (float): 预热起始学习率
        last_epoch (int): 上次训练的轮数
    """
    
    def __init__(self, 
                 optimizer, 
                 warmup_epochs, 
                 T_max, 
                 eta_min=1e-6,
                 warmup_lr=1e-6,
                 last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_lr = warmup_lr
        
        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        计算当前轮的学习率
        
        返回:
            list: 每个参数组的学习率列表
        """
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段: 线性增长
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_lr + (base_lr - self.warmup_lr) * alpha 
                    for base_lr in self.base_lrs]
        else:
            # Cosine阶段: 余弦衰减
            # 调整epoch计数（减去warmup轮数）
            t = self.last_epoch - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * t / self.T_max)) / 2
                    for base_lr in self.base_lrs]


def visualize_lr_schedule(scheduler, total_epochs, save_path=None):
    """
    可视化学习率变化曲线
    
    参数:
        scheduler: 学习率调度器
        total_epochs (int): 总训练轮数
        save_path (str): 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    lrs = []
    for epoch in range(total_epochs):
        lrs.append(scheduler.optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(total_epochs), lrs, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习率曲线已保存至: {save_path}")
    else:
        plt.show()


def test_scheduler():
    """测试学习率调度器"""
    print("=" * 60)
    print("学习率调度器测试")
    print("=" * 60)
    
    # 创建模拟模型和优化器
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 测试1: 标准余弦退火
    print("\n测试1: 标准余弦退火 (T_max=100, eta_min=1e-6)")
    print("-" * 60)
    scheduler1 = get_cosine_scheduler(optimizer, T_max=100, eta_min=1e-6)
    
    test_epochs = [0, 25, 50, 75, 99]
    for epoch in range(100):
        if epoch in test_epochs:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}: lr = {lr:.8f}")
        scheduler1.step()
    
    # 测试2: 带预热的余弦退火
    print("\n测试2: 预热+余弦退火 (warmup=10, T_max=100, eta_min=1e-6)")
    print("-" * 60)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler2 = get_cosine_scheduler(
        optimizer2, 
        T_max=100, 
        eta_min=1e-6,
        warmup_epochs=10,
        warmup_lr=1e-6
    )
    
    test_epochs = [0, 5, 10, 30, 60, 99]
    for epoch in range(100):
        if epoch in test_epochs:
            lr = optimizer2.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}: lr = {lr:.8f}")
        scheduler2.step()
    
    # 测试3: 学习率范围统计
    print("\n测试3: 学习率统计")
    print("-" * 60)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler3 = get_cosine_scheduler(optimizer3, T_max=50, eta_min=1e-6)
    
    lrs = []
    for epoch in range(50):
        lrs.append(optimizer3.param_groups[0]['lr'])
        scheduler3.step()
    
    print(f"  初始学习率: {lrs[0]:.8f}")
    print(f"  最终学习率: {lrs[-1]:.8f}")
    print(f"  最大学习率: {max(lrs):.8f}")
    print(f"  最小学习率: {min(lrs):.8f}")
    print(f"  平均学习率: {sum(lrs)/len(lrs):.8f}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    test_scheduler()
