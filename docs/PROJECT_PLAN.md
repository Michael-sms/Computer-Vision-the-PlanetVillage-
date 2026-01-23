# 农作物病害识别分类项目 - 完整工作计划文档

## 1. 项目概述

### 项目目标
基于PlantVillage数据集（50000+张图像，14种作物，含健康及多种病害样本），构建高效的农作物病害识别系统。采用迁移学习范式，使用MobileNetV2作为骨干网络，融合SE-Block注意力机制，实现实时、高精度的病害分类。

### 数据集说明
- **数据规模**：50000+ 张RGB和灰度图像
- **作物种类**：14种（苹果、葡萄、玉米、番茄、土豆、樱桃等）
- **类别数**：38个（每种作物含健康和多种病害状态）
- **数据分布**：color/、grayscale/、segmented/ 三个子目录
- **划分方案**：训练集70%、验证集15%、测试集15%（分层抽样）

### 技术方案总结
- **骨干网络**：MobileNetV2（1.0倍通道，ImageNet预训练）
- **注意力机制**：SE-Block（压缩激励块）增强特征判别性
- **训练策略**：Fine-tuning（冻结前10层，训练后续层）
- **优化器**：Adam（lr=1e-4），余弦退火学习率调度
- **损失函数**：CrossEntropyLoss + Label Smoothing

---

## 2. 技术架构

### MobileNetV2 + SE-Block 网络设计

```
输入图像(224×224×3)
    ↓
[预处理模块]：标准化(ImageNet均值/方差)
    ↓
[MobileNetV2骨干网络]：
  - Conv 3×3 (32通道)
  - 18个瓶颈残差块 (MBConv)
  - 深度可分离卷积(DSC)
  - 步长：[1,2,2,2,1,2,1,1,1,1,1,1,2,1,1,1]
  - 输出通道数逐步增加：32→64→96→160→320→1280
    ↓
[SE-Block注意力模块]（应用于最后4个MBConv块）：
  - 全局平均池化(GAP)
  - FC层降维(缩放因子r=16)
  - ReLU激活
  - FC层升维 + Sigmoid
  - 逐通道加权融合
    ↓
[全局平均池化层]
    ↓
[分类头]：
  - Dropout (p=0.2)
  - 全连接层(1280 → 38)
  - Softmax激活
    ↓
输出：38个类别的概率分布
```

### 关键设计要点
- SE-Block插入位置：MobileNetV2最后4个MBConv块之后
- SE-Block结构：$\text{SE}(x) = x \otimes \sigma(FC_2(ReLU(FC_1(GAP(x)))))$
- 低秩分解比例：r=16（通道数压缩至原来的1/16）

---

## 3. 数据处理流程

### 数据加载模块（`data_loader.py`）
```python
# 类：PlantVillageDataset
# 功能：继承torch.utils.data.Dataset
# 方法：
#   - __init__：初始化数据路径、类别映射、分割方案
#   - __len__：返回样本总数
#   - __getitem__：单样本加载+预处理
#   - get_class_weights()：计算不平衡类别权重
```

### 预处理策略
- **图像尺寸**：统一resize至224×224（MobileNetV2标准输入）
- **像素值范围**：[0,1]归一化
- **标准化**：ImageNet均值(0.485, 0.456, 0.406)和方差(0.229, 0.224, 0.225)

### 数据增强策略（`augmentation.py`）
**训练集**：
- 随机旋转(±20°)
- 随机翻转(水平/竖直，概率50%)
- 随机缩放(0.8-1.2)
- 随机裁剪(224×224)
- 颜色抖动(亮度0.2，对比度0.2)
- Cutout(掩码大小16×16，概率20%)

**验证/测试集**：
- 中心裁剪(224×224)
- 无其他增强

### 数据集划分（`split_dataset.py`）
```python
# 分层抽样：按类别比例划分
# 输出：train_split.txt, val_split.txt, test_split.txt
# 格式：image_path label
```

---

## 4. 模型实现细节

### 迁移学习方案（`models/mobilenetv2_se.py`）

**模型架构类**：`MobileNetV2WithSE`
```python
# 继承：nn.Module
# 核心模块：
#   - backbone：torchvision预加载的MobileNetV2
#   - se_blocks：4个SE-Block（插入最后4个MBConv）
#   - classifier：全连接分类头
#   - forward()：前向传播逻辑
```

**迁移学习参数配置**：
- 冻结层数：前10层（第0-9层MBConv块）
- 可训练层数：后8层MBConv + SE-Block + 分类头
- 初始化：分类头权重用Xavier初始化

### 训练策略（`train.py`）
```python
# 函数签名：train_epoch(model, train_loader, criterion, optimizer, device, epoch)
# 功能：单轮训练循环，计算损失、反向传播、参数更新

# 超参数设置：
#   - 批大小(batch_size)：32（GPU显存考虑）
#   - 学习率(lr_initial)：0.0001
#   - 学习率调度：CosineAnnealingLR(T_max=100, eta_min=1e-6)
#   - 优化器：Adam(betas=(0.9, 0.999), weight_decay=1e-5)
#   - 轮数(epochs)：100
#   - Label Smoothing：epsilon=0.1
#   - 早停(Early Stopping)：验证损失3轮不下降则停止
```

### 验证逻辑（`validate.py`）
```python
# 函数：evaluate_model(model, val_loader, criterion, device)
# 输出：
#   - 验证损失(val_loss)
#   - 准确率(accuracy)
#   - 每类精确率/召回率/F1-score字典
```

---

## 5. 评估方案

### 定量指标计算（`metrics.py`）
```python
# 函数列表：
#   - compute_accuracy(predictions, labels)：总体准确率
#   - compute_precision_recall_f1(predictions, labels, num_classes)：
#     返回字典{class_id: {precision, recall, f1}}
#   - compute_confusion_matrix(predictions, labels, num_classes)：
#     返回38×38混淆矩阵(numpy数组)
```

### 混淆矩阵分析（`confusion_analysis.py`）
- 绘制热力图：`plot_confusion_matrix(cm, class_names, save_path)`
- 识别易混淆类别对：提取cm中Top-5最高误分类率对
- 按类别统计：识别准确率最低/最高的3种类别

### Grad-CAM可视化（`gradcam.py`）
```python
# 类：GradCAM
# 方法：
#   - __init__(model, target_layer)：绑定目标层(最后一个MBConv)
#   - generate_cam(input_tensor, class_id)：
#     返回(224, 224)的热力图
#   - visualize(image, cam, alpha=0.5)：叠加原图和热力图
# 输出：为测试集中每类随机抽取5张样本的Grad-CAM图
```

### 测试集评估（`test.py`）
```python
# 流程：
# 1. 加载最佳模型(best_model.pth)
# 2. 在测试集上计算所有指标
# 3. 生成报告：metrics_report.json
# 4. 保存结果可视化
```

---

## 6. 两人分工方案

### **第一个人：模型与训练模块负责人**

**职责模块**：
1. **模型架构实现**（`models/mobilenetv2_se.py`）
   - MobileNetV2预加载和冻结层管理
   - SE-Block模块设计与集成
   - 分类头(FC+Dropout)实现
   - forward()前向传播逻辑

2. **训练流程实现**（`train.py`、`validate.py`）
   - 训练循环、反向传播、参数更新
   - 学习率调度(CosineAnnealingLR)实现
   - 早停机制
   - 模型checkpoint保存/加载逻辑
   - TensorBoard日志记录

3. **超参数调优**
   - 学习率、batch_size、weight_decay等参数调试
   - 冻结层数量的实验验证
   - 学习率预热(Warm-up)策略设计

4. **协作点**：
   - 与第二人共同定义 `config.yaml`（统一超参数配置）
   - 接收第二人的 `PlantVillageDataset` 类，集成到训练流程
   - 提供训练好的 `best_model.pth`、`training_log.json` 给第二人进行评估

---

### **第二个人：数据处理与评估模块负责人**

**职责模块**：
1. **数据处理流程**（`data_loader.py`、`split_dataset.py`）
   - PlantVillageDataset类实现（数据加载、预处理、增强）
   - 分层数据集划分（70%-15%-15%）
   - 类别权重计算（处理类不平衡问题）
   - DataLoader生成函数

2. **数据增强策略**（`augmentation.py`）
   - Compose()管道设计
   - 训练/验证/测试增强策略差异化实现
   - 可视化数据增强效果（plot_augmentation_samples()）

3. **评估体系建设**（`metrics.py`、`confusion_analysis.py`、`gradcam.py`、`test.py`）
   - 精确率/召回率/F1-score计算
   - 混淆矩阵绘制与热力图
   - 易混淆类别分析与可视化
   - Grad-CAM热力图生成与叠加显示
   - 测试集推理与完整报告生成

4. **结果可视化**（`visualization.py`）
   - 训练曲线绘制(loss/accuracy)
   - 类别级性能对比图
   - 模型误分类样本统计

5. **协作点**：
   - 接收第一人的 `best_model.pth`，进行评估
   - 与第一人共同定义 `config.yaml`
   - 提供 `PlantVillageDataset` 给第一人，支撑训练流程
   - 生成最终评估报告 `evaluation_report.md`

---

### **分工合理性分析**
- **独立性强**：两人工作界面清晰，最小化依赖冲突
- **并行进度**：第一人可在第二人完成DataLoader后立即开始训练；第二人可在第一人训练进行中继续完善评估模块
- **工作量平衡**：模型实现与评估体系工作量相当
- **技术难度**：第一人偏重深度学习训练细节，第二人偏重数据工程与可视化分析

---

## 7. 项目工作流程与里程碑（一周时间表）

| 阶段 | 时间 | 第一人 | 第二人 | 交付物 |
|------|------|--------|--------|--------|
| 环境搭建与需求分析 | Day 1 上午 | 确定网络架构、配置环境 | 数据集下载、分析统计 | `config.yaml`, 环境配置 |
| 数据准备与模型框架 | Day 1 下午-Day 2 | 搭建模型架构(MobileNetV2+SE-Block) | 数据划分、增强实现、DataLoader | `mobilenetv2_se.py`, `data_loader.py` |
| 训练流程开发 | Day 3 | 训练循环、验证函数、损失函数 | 配合测试DataLoader、准备评估框架 | `train.py`, `validate.py` |
| 模型训练与调优 | Day 4-5 | 执行训练、监控loss、调整超参数 | 完善评估指标、混淆矩阵模块 | `best_model.pth`, 训练日志 |
| 评估与可视化 | Day 6 | 辅助分析训练结果 | Grad-CAM实现、完整评估报告 | `evaluation_report.md`, 可视化图表 |
| 文档整理与汇报准备 | Day 7 | 模型设计文档、PPT制作 | 实验结果整理、PPT制作 | `PROJECT_REPORT.md`, 汇报材料 |

### 关键里程碑
- **Day 1 结束**：环境配置完成、数据集下载并划分、模型架构代码完成
- **Day 2 结束**：DataLoader调通、模型能成功前向传播、训练脚本基本完成
- **Day 3 结束**：能够启动训练、验证流程正常运行
- **Day 5 结束**：训练完成、保存最佳模型权重
- **Day 6 结束**：所有评估指标计算完成、可视化结果生成
- **Day 7 结束**：项目文档、汇报材料准备完毕

### 时间紧凑应对策略
1. **简化训练轮数**：从100轮减少到30-50轮，使用更大的学习率(2e-4)加快收敛
2. **减少冻结层调试**：直接采用冻结前10层的标准方案，不做过多实验
3. **并行工作**：Day 4-5 训练期间，第二人同步完成评估模块代码
4. **快速迭代**：如Day 4训练效果不佳，Day 5立即调整超参数重新训练
5. **提前准备**：Day 1-3 期间，两人利用碎片时间准备文档模板和PPT框架

---

## 8. 推荐的文件目录结构

```
final-project/
├── config.yaml                 # 超参数配置文件（两人共维护）
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明
│
├── datasets/                   # 数据目录
│   └── plantvillage dataset/
│       ├── color/
│       ├── grayscale/
│       └── segmented/
│
├── data/                       # 数据处理相关
│   ├── split_dataset.py       # 数据集划分(第二人)
│   ├── data_loader.py         # 数据加载类(第二人)
│   ├── augmentation.py        # 数据增强策略(第二人)
│   └── visualize_samples.py   # 样本可视化(第二人)
│
├── models/                     # 模型定义
│   ├── __init__.py
│   ├── mobilenetv2_se.py      # MobileNetV2+SE(第一人)
│   └── attention.py           # SE-Block模块(第一人)
│
├── training/                   # 训练流程
│   ├── train.py               # 训练主循环(第一人)
│   ├── validate.py            # 验证函数(第一人)
│   ├── losses.py              # 损失函数定义(第一人)
│   └── lr_scheduler.py        # 学习率调度(第一人)
│
├── evaluation/                 # 评估模块
│   ├── metrics.py             # 指标计算(第二人)
│   ├── confusion_analysis.py  # 混淆矩阵分析(第二人)
│   ├── gradcam.py             # Grad-CAM可视化(第二人)
│   ├── test.py                # 测试集评估(第二人)
│   └── visualization.py       # 结果可视化(第二人)
│
├── checkpoints/               # 模型保存目录
│   ├── best_model.pth         # 最佳模型(第一人生成)
│   └── training_log.json      # 训练日志
│
├── results/                   # 评估结果输出
│   ├── evaluation_report.md   # 评估报告(第二人)
│   ├── confusion_matrix.png   # 混淆矩阵热力图
│   ├── metrics_by_class.csv   # 类别级指标表
│   └── gradcam_samples/       # Grad-CAM可视化样本
│
└── utils/                      # 工具函数
    ├── constants.py           # 类别标签定义(两人共用)
    ├── helpers.py             # 通用函数(两人共用)
    └── device.py              # GPU设备管理(两人共用)
```

---

## 9. 基座模型选择与权重获取

### MobileNetV2预训练权重
- **来源**：torchvision官方
- **获取方式**：
```python
from torchvision import models
mobilenetv2 = models.mobilenet_v2(pretrained=True)
# 自动下载ImageNet预训练权重(约14MB)
# 默认存储位置：~/.cache/torch/hub/
```

### 权重特性
- **训练数据**：ImageNet-1k（1000个类别）
- **Top-1精度**：71.89%
- **参数量**：3.5M（轻量级设计）
- **推理速度**：~40ms (CPU), ~3ms (GPU)

### 迁移学习权重冻结策略
```python
# 冻结前10层MBConv块
for i, layer in enumerate(model.features):
    if i < 10:
        for param in layer.parameters():
            param.requires_grad = False
            
# 解冻后续层+分类头，参数量约220K（可训练）
```
