"""
MobileNetV2 + SE-Block 模型架构

基于预训练的MobileNetV2骨干网络，在最后4个MBConv块后插入SE-Block注意力机制，
用于PlantVillage数据集的38类农作物病害分类任务。

技术方案:
    - 骨干网络: MobileNetV2 (ImageNet预训练)
    - 注意力机制: SE-Block (r=16)
    - 冻结策略: 冻结前10层，训练后8层+分类头
    - 分类头: Dropout(0.2) + FC(1280 -> 38)
"""

import torch
import torch.nn as nn
from torchvision import models
from .attention import SEBlock


class MobileNetV2WithSE(nn.Module):
    """
    MobileNetV2 + SE-Block 模型
    
    架构流程:
        输入图像(224×224×3) 
        -> MobileNetV2骨干网络(18个MBConv块)
        -> SE-Block插入最后4个块
        -> 全局平均池化
        -> Dropout(0.2) + FC(1280->38)
        -> Softmax输出
    
    参数:
        num_classes (int): 分类类别数，默认38
        pretrained (bool): 是否使用ImageNet预训练权重，默认True
        freeze_layers (int): 冻结前n层，默认10
        dropout_rate (float): Dropout比例，默认0.2
        se_reduction (int): SE-Block降维比例，默认16
    
    示例:
        >>> model = MobileNetV2WithSE(num_classes=38, pretrained=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([4, 38])
    """
    
    def __init__(self, 
                 num_classes=38, 
                 pretrained=True, 
                 freeze_layers=10,
                 dropout_rate=0.2,
                 se_reduction=16):
        super(MobileNetV2WithSE, self).__init__()
        
        self.num_classes = num_classes
        self.freeze_layers = freeze_layers
        
        # ===== 1. 加载预训练的MobileNetV2骨干网络 =====
        print(f"[模型构建] 加载MobileNetV2骨干网络 (预训练={pretrained})")
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # 提取特征提取部分 (不包含分类头)
        # mobilenet.features 包含18个MBConv块
        self.features = mobilenet.features
        
        # ===== 2. 插入SE-Block到最后4个MBConv块 =====
        # MobileNetV2的features是Sequential结构，包含19层(0-18)
        # 层索引: [0]: Conv, [1-18]: InvertedResidual块
        # 在最后4个块(索引15,16,17,18)后插入SE-Block
        print(f"[模型构建] 在最后4个MBConv块后插入SE-Block (r={se_reduction})")
        
        # 获取各个MBConv块的输出通道数
        self.se_positions = [15, 16, 17, 18]  # 插入SE的位置
        self.se_blocks = nn.ModuleDict()
        
        for idx in self.se_positions:
            # 获取该层的输出通道数
            out_channels = self._get_layer_channels(idx)
            self.se_blocks[str(idx)] = SEBlock(out_channels, se_reduction)
            print(f"  位置{idx}: 通道数={out_channels}, SE压缩后={out_channels//se_reduction}")
        
        # ===== 3. 构建分类头 =====
        # MobileNetV2最后输出1280通道
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 (B,1280,7,7) -> (B,1280,1,1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)
        
        # Xavier初始化分类头权重
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # ===== 4. 冻结前N层参数 =====
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        print(f"[模型构建] 分类头: 1280 -> Dropout({dropout_rate}) -> {num_classes}")
        self._print_model_info()
    
    def _get_layer_channels(self, layer_idx):
        """
        获取指定层的输出通道数
        
        MobileNetV2各层输出通道数:
            [0]: 32, [1]: 16, [2-3]: 24, [4-6]: 32, 
            [7-10]: 64, [11-13]: 96, [14-16]: 160, [17]: 320, [18]: 1280
        """
        channels_map = {
            0: 32,
            1: 16,
            2: 24, 3: 24,
            4: 32, 5: 32, 6: 32,
            7: 64, 8: 64, 9: 64, 10: 64,
            11: 96, 12: 96, 13: 96,
            14: 160, 15: 160, 16: 160,
            17: 320,
            18: 1280
        }
        return channels_map[layer_idx]
    
    def _freeze_layers(self, num_layers):
        """
        冻结前num_layers层的参数（不计算梯度）
        
        参数:
            num_layers (int): 要冻结的层数（从0开始）
        """
        frozen_params = 0
        total_params = 0
        
        for idx, layer in enumerate(self.features):
            total_params += sum(p.numel() for p in layer.parameters())
            if idx < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen_params += sum(p.numel() for p in layer.parameters())
        
        print(f"[参数冻结] 冻结前{num_layers}层，冻结参数: {frozen_params:,} / {total_params:,} " 
              f"({frozen_params/total_params*100:.1f}%)")
    
    def forward(self, x):
        """
        前向传播
        
        输入:
            x: Tensor, shape=[B, 3, 224, 224]
        
        输出:
            out: Tensor, shape=[B, num_classes] (未经Softmax的logits)
        
        流程:
            1. 逐层通过MobileNetV2的features
            2. 在特定位置插入SE-Block
            3. 全局平均池化
            4. Dropout + 全连接层
        """
        # 逐层前向传播，在指定位置插入SE-Block
        for idx, layer in enumerate(self.features):
            x = layer(x)
            
            # 如果当前层需要插入SE-Block
            if idx in self.se_positions:
                x = self.se_blocks[str(idx)](x)
        
        # 全局平均池化: (B, 1280, 7, 7) -> (B, 1280, 1, 1)
        x = self.avgpool(x)
        
        # Flatten: (B, 1280, 1, 1) -> (B, 1280)
        x = torch.flatten(x, 1)
        
        # Dropout + 分类
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def _print_model_info(self):
        """打印模型信息统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print("\n" + "=" * 60)
        print("MobileNetV2+SE 模型信息")
        print("=" * 60)
        print(f"总参数量:       {total_params:>12,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数:     {trainable_params:>12,} ({trainable_params/1e6:.2f}M)")
        print(f"冻结参数:       {frozen_params:>12,} ({frozen_params/1e6:.2f}M)")
        print(f"训练参数比例:   {trainable_params/total_params*100:>11.1f}%")
        print(f"分类类别数:     {self.num_classes:>12}")
        print(f"SE-Block数量:   {len(self.se_blocks):>12}")
        print("=" * 60 + "\n")
    
    def get_trainable_params(self):
        """
        获取可训练参数列表（用于优化器）
        
        返回:
            list: 可训练的参数列表
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def unfreeze_all(self):
        """解冻所有层的参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("[参数解冻] 所有层参数已解冻")
    
    def freeze_backbone(self):
        """冻结整个骨干网络（仅训练分类头）"""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.se_blocks.parameters():
            param.requires_grad = False
        print("[参数冻结] 骨干网络和SE-Block已冻结，仅训练分类头")


def test_model():
    """模型测试代码"""
    print("\n" + "=" * 60)
    print("MobileNetV2+SE 模型测试")
    print("=" * 60 + "\n")
    
    # 创建模型（使用预训练权重）
    model = MobileNetV2WithSE(
        num_classes=38, 
        pretrained=False,  # 测试时不下载预训练权重
        freeze_layers=10,
        dropout_rate=0.2
    )
    
    # 设置为评估模式
    model.eval()
    
    # 生成随机输入 (batch_size=4, channels=3, height=224, width=224)
    x = torch.randn(4, 3, 224, 224)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输出样例 (前5个类别): {output[0, :5]}")
    
    # 验证输出维度
    assert output.shape == (4, 38), f"输出形状错误: {output.shape}"
    
    # 测试梯度流
    print("\n检查梯度流...")
    model.train()
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    # 检查哪些层有梯度
    has_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"有梯度的参数: {has_grad}/{total_trainable}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
