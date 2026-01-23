# 数据处理模块使用说明

本文档介绍如何使用数据处理模块完成数据集划分、数据加载和可视化等任务。

---

## 目录结构

```
data/
├── __init__.py           # 模块初始化
├── split_dataset.py      # 数据集划分
├── augmentation.py       # 数据增强策略
├── data_loader.py        # 数据加载器
└── visualize_samples.py  # 样本可视化

datasets/
├── plantvillage dataset/  # 原始数据集
│   └── color/            # 彩色图像（38个类别文件夹）
└── splits/               # 划分后的数据集索引
    ├── train.txt         # 训练集（70%）
    ├── val.txt           # 验证集（15%）
    └── test.txt          # 测试集（15%）

results/                  # 可视化结果保存目录
├── class_samples.png
├── augmentation_effects.png
├── class_distribution.png
├── split_distribution.png
└── train_sample_grid.png
```

---

## 快速开始

### 1. 划分数据集

首次使用时，需要先运行数据集划分脚本：

```bash
python -m data.split_dataset
```

这将在 `datasets/splits/` 目录下生成三个划分文件：
- `train.txt` - 训练集（约37,997张图像）
- `val.txt` - 验证集（约8,129张图像）
- `test.txt` - 测试集（约8,179张图像）

**划分文件格式**：
```
图像绝对路径\t类别索引
```

示例：
```
C:\...\datasets\plantvillage dataset\color\Apple___Apple_scab\image001.jpg	0
C:\...\datasets\plantvillage dataset\color\Tomato___healthy\image002.jpg	30
```

---

### 2. 使用 DataLoader 加载数据

#### 基本用法

```python
from data import get_all_dataloaders

# 获取训练集、验证集、测试集的DataLoader
train_loader, val_loader, test_loader = get_all_dataloaders(
    batch_size=32,
    num_workers=4
)

# 遍历训练数据
for images, labels in train_loader:
    # images: [B, 3, 224, 224] 的张量
    # labels: [B] 的标签张量
    print(f"图像形状: {images.shape}, 标签形状: {labels.shape}")
    break
```

#### 单独获取某个划分的DataLoader

```python
from data import get_dataloader

# 只获取训练集
train_loader = get_dataloader(
    split="train",      # "train", "val", "test"
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 获取测试集（不打乱）
test_loader = get_dataloader(
    split="test",
    batch_size=64,
    shuffle=False
)
```

#### 使用加权采样器（处理类不平衡）

```python
from data import get_dataloader

# 使用加权采样器，使每个类别被采样的概率相等
train_loader = get_dataloader(
    split="train",
    batch_size=32,
    use_weighted_sampler=True  # 启用加权采样
)
```

---

### 3. 直接使用 Dataset 类

```python
from data import PlantVillageDataset
from data import get_train_transforms, get_val_transforms

# 创建训练集Dataset
train_dataset = PlantVillageDataset(
    split_file="datasets/splits/train.txt",
    transform=get_train_transforms(),
    return_path=False  # 是否同时返回图像路径
)

# 获取单个样本
image, label = train_dataset[0]
print(f"图像形状: {image.shape}, 标签: {label}")

# 获取数据集大小
print(f"训练集样本数: {len(train_dataset)}")

# 获取类别权重（用于损失函数加权）
class_weights = train_dataset.get_class_weights()
print(f"类别权重形状: {class_weights.shape}")  # [38]

# 获取类别分布
distribution = train_dataset.get_class_distribution()
for class_name, count in list(distribution.items())[:5]:
    print(f"  {class_name}: {count}")
```

---

### 4. 数据增强

#### 获取预定义的数据变换

```python
from data import get_train_transforms, get_val_transforms

# 训练集变换（包含数据增强）
train_transform = get_train_transforms()
# 包含：随机旋转、翻转、缩放裁剪、颜色抖动、高斯噪声、Cutout、ImageNet标准化

# 验证/测试集变换（仅预处理，无增强）
val_transform = get_val_transforms()
# 包含：Resize、中心裁剪、ImageNet标准化
```

#### 自定义数据增强

```python
from torchvision import transforms
from data.augmentation import GaussianNoise, Cutout
from utils.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

# 自定义训练变换
custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(30),  # 更大的旋转角度
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    GaussianNoise(mean=0.0, std=0.03, p=0.5),  # 自定义噪声
    Cutout(size=32, p=0.3),  # 更大的Cutout
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

#### 反标准化（用于可视化）

```python
from data import denormalize, tensor_to_pil
import matplotlib.pyplot as plt

# 假设 image 是经过标准化的张量 [3, 224, 224]
image_denorm = denormalize(image)  # 反标准化
image_pil = tensor_to_pil(image_denorm)  # 转为PIL图像

# 显示图像
plt.imshow(image_pil)
plt.show()
```

---

### 5. 生成可视化结果

```bash
python -m data.visualize_samples
```

这将在 `results/` 目录下生成以下可视化图像：

| 文件名 | 内容 |
|--------|------|
| `class_samples.png` | 各类别样本展示 |
| `augmentation_effects.png` | 数据增强效果对比 |
| `class_distribution.png` | 38个类别的样本数量分布 |
| `split_distribution.png` | 训练/验证/测试集划分比例 |
| `train_sample_grid.png` | 训练集样本网格展示 |

#### 在代码中调用可视化函数

```python
from data.visualize_samples import (
    visualize_class_samples,
    visualize_augmentation_effects,
    visualize_class_distribution,
    visualize_split_distribution,
    visualize_sample_grid,
)

# 生成类别样本展示图
visualize_class_samples(
    num_samples_per_class=3,
    num_classes_to_show=12,
    save_path="results/my_class_samples.png"
)

# 生成数据增强效果图
visualize_augmentation_effects(
    num_samples=4,
    num_augmented=5,
    save_path="results/my_augmentation.png"
)

# 生成指定划分的样本网格
visualize_sample_grid(
    split="val",  # 可选 "train", "val", "test"
    grid_size=(4, 8),
    save_path="results/val_samples.png"
)
```

---

### 6. 常量和配置

所有项目常量定义在 `utils/constants.py` 中：

```python
from utils.constants import (
    # 路径
    PROJECT_ROOT,        # 项目根目录
    DATASET_ROOT,        # 原始数据集路径
    SPLIT_DATASET_ROOT,  # 划分文件保存路径
    RESULTS_DIR,         # 结果保存路径
    CHECKPOINTS_DIR,     # 模型检查点保存路径
    
    # 类别信息
    CLASS_NAMES,         # 38个类别的英文名称列表
    CLASS_DISPLAY_NAMES, # 38个类别的中文显示名称
    NUM_CLASSES,         # 类别数量 = 38
    CLASS_TO_IDX,        # 类别名 -> 索引 的映射字典
    IDX_TO_CLASS,        # 索引 -> 类别名 的映射字典
    
    # 图像处理参数
    IMAGE_SIZE,          # 图像尺寸 = 224
    IMAGENET_MEAN,       # ImageNet均值 [0.485, 0.456, 0.406]
    IMAGENET_STD,        # ImageNet标准差 [0.229, 0.224, 0.225]
    
    # 数据集划分比例
    TRAIN_RATIO,         # 训练集比例 = 0.70
    VAL_RATIO,           # 验证集比例 = 0.15
    TEST_RATIO,          # 测试集比例 = 0.15
    
    # 随机种子
    RANDOM_SEED,         # 随机种子 = 42
)
```

---

## 完整训练示例

```python
import torch
import torch.nn as nn
from torch.optim import Adam

from data import get_all_dataloaders
from utils.constants import NUM_CLASSES

# 1. 加载数据
train_loader, val_loader, test_loader = get_all_dataloaders(
    batch_size=32,
    num_workers=4,
    use_weighted_sampler=True  # 处理类不平衡
)

# 2. 获取类别权重（用于损失函数）
class_weights = train_loader.dataset.get_class_weights()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# 3. 定义损失函数（带类别权重）
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 4. 训练循环
model = ...  # 你的模型
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f"Epoch {epoch+1}: Val Accuracy = {accuracy:.2f}%")
```

---

## 常见问题

### Q1: 运行时提示"划分文件不存在"

**解决方案**：先运行数据集划分脚本
```bash
python -m data.split_dataset
```

### Q2: 数据加载速度慢

**解决方案**：
1. 增加 `num_workers` 参数（Windows下建议设为0或2）
2. 使用 `pin_memory=True`（需要GPU）
3. 确保数据集在SSD上

```python
train_loader = get_dataloader(
    split="train",
    batch_size=32,
    num_workers=0,  # Windows下设为0避免多进程问题
    pin_memory=True
)
```

### Q3: 内存不足 (OOM)

**解决方案**：减小 `batch_size`
```python
train_loader = get_dataloader(split="train", batch_size=16)  # 减小batch_size
```

### Q4: 类别不平衡问题

**解决方案**：
1. 使用加权采样器
2. 使用带权重的损失函数

```python
# 方法1：加权采样
train_loader = get_dataloader(split="train", use_weighted_sampler=True)

# 方法2：带权重的损失函数
class_weights = train_loader.dataset.get_class_weights().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## API 参考

### `get_dataloader()`

```python
def get_dataloader(
    split: str = "train",           # 数据集划分: "train", "val", "test"
    batch_size: int = 32,           # 批大小
    shuffle: bool = None,           # 是否打乱，默认训练集打乱
    num_workers: int = 4,           # 数据加载进程数
    pin_memory: bool = True,        # 是否锁页内存
    use_weighted_sampler: bool = False,  # 是否使用加权采样
    return_path: bool = False       # 是否返回图像路径
) -> DataLoader
```

### `PlantVillageDataset`

```python
class PlantVillageDataset(Dataset):
    def __init__(
        self,
        split_file: str,            # 划分文件路径
        transform: Callable = None, # 数据变换
        return_path: bool = False   # 是否返回图像路径
    )
    
    def __len__(self) -> int
    def __getitem__(self, idx) -> Tuple[Tensor, int]
    def get_class_weights(self) -> Tensor      # 获取类别权重
    def get_sample_weights(self) -> Tensor     # 获取样本权重
    def get_labels(self) -> List[int]          # 获取所有标签
    def get_class_distribution(self) -> Dict   # 获取类别分布
```
