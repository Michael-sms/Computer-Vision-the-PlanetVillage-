"""
SE-Block (Squeeze-and-Excitation Block) 注意力机制模块

SE-Block通过全局平均池化和通道注意力机制，自适应地重新校准通道特征响应。

架构：
    输入 -> GlobalAvgPool -> FC(降维r) -> ReLU -> FC(升维) -> Sigmoid -> 通道加权 -> 输出
    
公式：
    SE(x) = x ⊗ σ(FC_2(ReLU(FC_1(GAP(x)))))
    其中 r 为缩放因子(通常为16)
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (SE-Block)
    
    参数:
        in_channels (int): 输入特征图的通道数
        reduction_ratio (int): 降维缩放比例，默认为16
            - 压缩后通道数 = in_channels // reduction_ratio
            - 更大的ratio会减少参数量但可能降低性能
    
    输入:
        x: Tensor, shape=[B, C, H, W]
    
    输出:
        out: Tensor, shape=[B, C, H, W] (经过通道注意力加权后的特征图)
    
    示例:
        >>> se = SEBlock(in_channels=96, reduction_ratio=16)
        >>> x = torch.randn(4, 96, 28, 28)
        >>> out = se(x)
        >>> print(out.shape)  # torch.Size([4, 96, 28, 28])
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        
        # 确保压缩后的通道数至少为1
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Squeeze: 全局平均池化 (B, C, H, W) -> (B, C, 1, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: 两层全连接网络
        self.excitation = nn.Sequential(
            # 降维：减少计算量和参数量
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            
            # 升维：恢复到原始通道数
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()  # 输出[0,1]之间的注意力权重
        )
    
    def forward(self, x):
        """
        前向传播
        
        流程:
            1. Squeeze: 全局平均池化 -> (B, C, 1, 1)
            2. Flatten: (B, C, 1, 1) -> (B, C)
            3. Excitation: FC降维 -> ReLU -> FC升维 -> Sigmoid -> (B, C)
            4. Reshape: (B, C) -> (B, C, 1, 1)
            5. Scale: 原始特征图逐通道加权
        """
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: 全局平均池化
        # (B, C, H, W) -> (B, C, 1, 1)
        squeeze = self.squeeze(x)
        
        # Flatten: (B, C, 1, 1) -> (B, C)
        squeeze = squeeze.view(batch_size, channels)
        
        # Excitation: 计算通道注意力权重
        # (B, C) -> (B, C//r) -> (B, C) -> [0,1]
        excitation = self.excitation(squeeze)
        
        # Reshape: (B, C) -> (B, C, 1, 1)
        excitation = excitation.view(batch_size, channels, 1, 1)
        
        # Scale: 逐通道加权融合
        # (B, C, H, W) * (B, C, 1, 1) -> (B, C, H, W)
        return x * excitation.expand_as(x)


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("SE-Block 模块测试")
    print("=" * 50)
    
    # 测试不同通道数的SE-Block
    test_cases = [
        (32, 8, 8),   # 较小特征图
        (96, 28, 28),  # 中等特征图
        (320, 7, 7),   # 较大通道数
    ]
    
    for channels, height, width in test_cases:
        print(f"\n测试配置: channels={channels}, size={height}x{width}")
        
        # 创建SE-Block
        se_block = SEBlock(in_channels=channels, reduction_ratio=16)
        
        # 生成随机输入
        x = torch.randn(4, channels, height, width)
        
        # 前向传播
        output = se_block(x)
        
        # 计算参数量
        num_params = sum(p.numel() for p in se_block.parameters())
        
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  参数量: {num_params}")
        print(f"  压缩后通道数: {channels // 16}")
        
        # 验证输出形状
        assert output.shape == x.shape, "输出形状不匹配！"
    
    print("\n" + "=" * 50)
    print("所有测试通过！✓")
    print("=" * 50)
