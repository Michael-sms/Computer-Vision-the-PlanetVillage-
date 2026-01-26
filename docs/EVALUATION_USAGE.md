# 评估模块使用说明

本文档介绍如何使用 `evaluation` 模块对训练好的植物病害分类模型进行评估。

---

## 模块结构

```
evaluation/
├── __init__.py           # 模块导出
├── metrics.py            # 评估指标计算
├── confusion_analysis.py # 混淆矩阵分析与可视化
├── gradcam.py            # Grad-CAM 可解释性可视化
├── visualization.py      # 训练曲线与类别指标可视化
└── test.py               # 测试集评估主脚本
```

---

## 快速开始

### 1. 运行完整评估（推荐）

使用 `test.py` 一键完成测试集评估，生成所有报告和可视化：

```powershell
# 使用默认参数（GPU自动检测，结果保存到 results/evaluated/）
python -m evaluation.test

# 指定参数
python -m evaluation.test \
    --checkpoint checkpoints/best_model.pth \
    --test-split datasets/splits/test.txt \
    --output-dir results/evaluated \
    --batch-size 32 \
    --num-workers 4
```

### 2. 跳过 Grad-CAM（加速评估）

Grad-CAM 可视化较耗时，如需快速获取指标，可跳过：

```powershell
python -m evaluation.test --no-gradcam
```

### 3. 指定计算设备

```powershell
# 使用CPU
python -m evaluation.test --device cpu

# 使用GPU
python -m evaluation.test --device cuda
```

---

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | `checkpoints/best_model.pth` | 模型权重路径 |
| `--test-split` | str | `datasets/splits/test.txt` | 测试集划分文件 |
| `--batch-size` | int | 32 | 批次大小 |
| `--num-workers` | int | 4 | DataLoader 工作进程数 |
| `--output-dir` | str | `results/evaluated` | 结果输出目录 |
| `--no-gradcam` | flag | - | 跳过 Grad-CAM 可视化 |
| `--device` | str | `auto` | 计算设备 (auto/cpu/cuda) |

---

## 输出文件说明

评估完成后，`results/evaluated/` 目录下将生成以下文件：

| 文件 | 说明 |
|------|------|
| `metrics_report.json` | 完整评估指标（JSON格式） |
| `confusion_matrix_full.png` | 38类完整混淆矩阵热力图 |
| `confusion_matrix_simplified.png` | 错误率最高的15类简化混淆矩阵 |
| `confusion_matrix_data.json` | 混淆矩阵原始数据 |
| `confusion_analysis.txt` | 混淆分析文本报告 |
| `gradcam_summary.png` | Grad-CAM 汇总图（16个样本） |
| `gradcam_samples/` | 各类别 Grad-CAM 样本（每类3张） |
| `training_curves.png` | 训练曲线（Loss/Acc/LR） |
| `class_metrics.png` | 类别指标对比图 |

> **注意**：数据处理可视化结果保存在 `results/data_processing/` 目录下。

---

## 评估指标说明

### 总体指标

- **Accuracy（准确率）**: 正确分类样本数 / 总样本数
- **Top-3/5 Accuracy**: 真实类别在预测概率前3/5名的比例
- **Macro Precision/Recall/F1**: 各类别指标的简单平均
- **Weighted Precision/Recall/F1**: 按样本数加权平均

### 每类指标

- **Precision（精确率）**: TP / (TP + FP)
- **Recall（召回率）**: TP / (TP + FN)
- **F1-Score**: 精确率和召回率的调和平均
- **Support**: 该类别的样本数

---

## 单独使用各子模块

### 计算评估指标

```python
from evaluation.metrics import compute_all_metrics, print_metrics_report

# predictions: 预测类别索引数组
# labels: 真实类别索引数组
# probabilities: 预测概率矩阵（可选，用于Top-K）
metrics = compute_all_metrics(predictions, labels, num_classes=38, probabilities=probabilities)

# 打印报告
report = print_metrics_report(metrics, class_names=CLASS_DISPLAY_NAMES)
print(report)
```

### 绘制混淆矩阵

```python
from evaluation.confusion_analysis import (
    plot_confusion_matrix, 
    plot_confusion_matrix_simplified,
    generate_confusion_analysis_report
)

# 绘制完整混淆矩阵
plot_confusion_matrix(confusion_matrix, class_names, save_path='cm.png')

# 绘制简化混淆矩阵（错误率最高的N类）
plot_confusion_matrix_simplified(confusion_matrix, class_names, top_n=15)

# 生成文本分析报告
generate_confusion_analysis_report(confusion_matrix, class_names, save_path='report.txt')
```

### Grad-CAM 可视化

```python
from evaluation.gradcam import GradCAM, generate_gradcam_samples

# 创建 GradCAM 对象
gradcam = GradCAM(model, model.features[-1])  # 目标层为最后一个卷积层

# 生成单张图像的 CAM
cam = gradcam.generate_cam(input_tensor, class_id=5)

# 可视化
gradcam.visualize(original_image, cam, save_path='cam.png', alpha=0.5)

# 批量生成各类别样本
generate_gradcam_samples(
    model, test_loader, target_layer, device,
    num_samples_per_class=3,
    num_classes_to_show=10,
    save_dir='gradcam_output'
)
```

### 训练曲线可视化

```python
from evaluation.visualization import plot_training_history, plot_class_metrics

# 绘制训练曲线（Loss/Acc/LR）
plot_training_history(
    history_path='checkpoints/training_history.json',
    save_dir='results'
)

# 绘制类别指标对比图
plot_class_metrics(
    metrics_path='results/metrics_report.json',
    save_dir='results',
    top_k=10  # 展示最佳/最差各10类
)
```

---

## 示例：完整评估流程

```python
import torch
from models.mobilenetv2_se import MobileNetV2WithSE
from data.data_loader import get_dataloader
from evaluation.test import load_model, evaluate_on_testset, generate_evaluation_report
from evaluation.visualization import plot_training_history, plot_class_metrics

# 1. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 加载模型
model = load_model('checkpoints/best_model.pth', device)

# 3. 加载测试集
test_loader = get_dataloader(split='test', batch_size=32, shuffle=False)

# 4. 评估
results = evaluate_on_testset(model, test_loader, device)

# 5. 生成报告
metrics = generate_evaluation_report(results, save_dir='results')

# 6. 生成可视化
plot_training_history()
plot_class_metrics()

print(f"测试集准确率: {metrics['accuracy']*100:.2f}%")
```

---

## 常见问题

### Q1: 报错 "模型文件不存在"
确保 `checkpoints/best_model.pth` 存在，或通过 `--checkpoint` 指定正确路径。

### Q2: 报错 "划分文件不存在"
请先运行数据划分脚本生成 `datasets/splits/test.txt`：
```powershell
python -m data.split_dataset
```

### Q3: Grad-CAM 生成很慢
Grad-CAM 需要对每个样本进行反向传播，较耗时。可以：
- 使用 `--no-gradcam` 跳过
- 减少 `num_samples_per_class` 参数

### Q4: 中文乱码
确保系统安装了 SimHei 或 Microsoft YaHei 字体。Linux 用户可安装：
```bash
sudo apt install fonts-wqy-microhei
```

---

## 目标指标（参考）

根据项目计划，目标测试集准确率 ≥ 95%。评估时请关注：

1. **总体准确率**: 是否达到 95% 以上
2. **Top-5 准确率**: 是否接近 99%
3. **低 F1 类别**: 混淆分析中准确率最低的类别，可能需要更多数据或针对性优化
4. **Grad-CAM 热力图**: 验证模型是否关注病害区域而非背景

---

## 更新日志

- **2026-01-26**: 初版评估模块完成
  - 支持准确率、精确率、召回率、F1-Score、Top-K 准确率
  - 混淆矩阵可视化与分析
  - Grad-CAM 可解释性可视化
  - 训练曲线与类别指标可视化
