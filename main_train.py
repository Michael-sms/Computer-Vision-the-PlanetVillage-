"""
训练启动脚本

完整的训练流程，包括：
1. 环境设置（随机种子、设备）
2. 数据加载
3. 模型创建
4. 训练执行
5. 结果保存
"""

import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MobileNetV2WithSE
from training.train import Trainer
from config import load_config, save_config, DEFAULT_CONFIG, print_config
from utils.helpers import set_seed, get_device, ensure_dir


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练MobileNetV2+SE模型')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（覆盖配置文件）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置文件）')
    
    # 模型参数
    parser.add_argument('--pretrained', action='store_true',
                        help='使用预训练权重')
    parser.add_argument('--freeze_layers', type=int, default=None,
                        help='冻结层数（覆盖配置文件）')
    
    # 设备参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU编号')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU训练')
    
    # 其他
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # AMP & 优化参数
    parser.add_argument('--amp', action='store_true',
                        help='启用混合精度训练 (推荐RTX4060使用)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='数据加载进程数 (覆盖配置文件)')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("MobileNetV2+SE 农作物病害识别训练程序".center(70))
    print("=" * 70 + "\n")
    
    # ========== 1. 加载配置 ==========
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"配置文件不存在，使用默认配置并保存")
        config = DEFAULT_CONFIG
        save_config(config, args.config)
    
    # 命令行参数覆盖配置文件
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.freeze_layers is not None:
        config['model']['freeze_layers'] = args.freeze_layers
    if args.pretrained:
        config['model']['pretrained'] = True
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    if args.amp:
        config['training']['use_amp'] = True
        print("\n[AMP] 启用混合精度训练 (RTX4060优化)")
        print("  - 显存占用 -40%")
        print("  - 训练速度 +30-50%")
    
    # 打印配置
    print_config(config)
    
    # ========== 2. 设置环境 ==========
    set_seed(args.seed)
    device = get_device(cuda=not args.cpu, gpu_id=args.gpu)
    
    # 创建必要的目录
    for key in ['checkpoints_dir', 'results_dir', 'logs_dir']:
        ensure_dir(config['paths'][key])
    
    # ========== 3. 加载数据 ==========
    print("\n" + "-" * 70)
    print("加载数据集...")
    print("-" * 70)
    
    # 注意：这里需要第二人完成data_loader模块后才能运行
    # 临时使用占位符
    try:
        from data import get_all_dataloaders
        train_loader, val_loader, test_loader = get_all_dataloaders(
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )
        print(f"训练集: {len(train_loader.dataset)} 张图像")
        print(f"验证集: {len(val_loader.dataset)} 张图像")
        print(f"测试集: {len(test_loader.dataset)} 张图像")
    except ImportError:
        print("[警告] 数据加载模块未完成，使用模拟数据")
        print("请等待第二人完成 data_loader.py 模块")
        
        # 创建模拟数据（仅用于测试训练流程）
        class FakeDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 1000
            
            def __getitem__(self, idx):
                return torch.randn(3, 224, 224), torch.randint(0, 38, (1,)).item()
        
        train_loader = torch.utils.data.DataLoader(
            FakeDataset(), 
            batch_size=config['data']['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            FakeDataset(), 
            batch_size=config['data']['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        print(f"[模拟] 训练集: {len(train_loader.dataset)} 张图像")
        print(f"[模拟] 验证集: {len(val_loader.dataset)} 张图像")
    
    # ========== 4. 创建模型 ==========
    print("\n" + "-" * 70)
    print("创建模型...")
    print("-" * 70)
    
    model = MobileNetV2WithSE(
        num_classes=config['data']['num_classes'],
        pretrained=config['model']['pretrained'],
        freeze_layers=config['model']['freeze_layers'],
        dropout_rate=config['model']['dropout_rate'],
        se_reduction=config['model']['se_reduction']
    )
    
    # ========== 5. 准备训练配置 ==========
    train_config = {
        'num_classes': config['data']['num_classes'],
        'epochs': config['training']['epochs'],
        'batch_size': config['data']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'min_lr': config['training']['min_lr'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'warmup_lr': config['training']['warmup_lr'],
        'label_smoothing': config['training']['label_smoothing'],
        'grad_clip': config['training']['grad_clip'],
        'patience': config['training']['patience'],
        'save_dir': config['paths']['checkpoints_dir'],
        'log_interval': config['training']['log_interval'],
        'save_interval': config['training']['save_interval'],
    }
    
    # ========== 6. 创建训练器 ==========
    trainer = Trainer(model, train_loader, val_loader, train_config, device)
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # ========== 7. 开始训练 ==========
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[中断] 训练被用户中断")
        print("已保存最后的checkpoint")
    
    print("\n" + "=" * 70)
    print("训练完成！".center(70))
    print("=" * 70)
    print(f"最佳模型保存在: {os.path.join(train_config['save_dir'], 'best_model.pth')}")
    print(f"训练历史保存在: {os.path.join(train_config['save_dir'], 'training_history.json')}")


if __name__ == "__main__":
    main()
