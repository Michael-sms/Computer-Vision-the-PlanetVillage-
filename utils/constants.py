"""
常量定义模块
包含类别标签、数据集路径等项目级常量
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集路径
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "plantvillage dataset", "color")
SPLIT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "splits")

# 结果保存路径
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_PROCESSING_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "data_processing")
EVALUATION_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "evaluated")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# 38个类别名称（与文件夹名称对应）
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

# 类别数量
NUM_CLASSES = len(CLASS_NAMES)

# 类别名称到索引的映射
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 索引到类别名称的映射
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# 简化的类别显示名称（用于可视化）
CLASS_DISPLAY_NAMES = [
    "苹果-黑星病", "苹果-黑腐病", "苹果-雪松锈病", "苹果-健康",
    "蓝莓-健康",
    "樱桃-健康", "樱桃-白粉病",
    "玉米-灰斑病", "玉米-锈病", "玉米-健康", "玉米-叶枯病",
    "葡萄-黑腐病", "葡萄-黑麻疹", "葡萄-健康", "葡萄-叶斑病",
    "橙子-黄龙病",
    "桃子-细菌斑点病", "桃子-健康",
    "甜椒-细菌斑点病", "甜椒-健康",
    "土豆-早疫病", "土豆-健康", "土豆-晚疫病",
    "树莓-健康",
    "大豆-健康",
    "南瓜-白粉病",
    "草莓-健康", "草莓-叶焦病",
    "番茄-细菌斑点病", "番茄-早疫病", "番茄-健康", "番茄-晚疫病",
    "番茄-叶霉病", "番茄-叶斑病", "番茄-红蜘蛛", "番茄-靶斑病",
    "番茄-花叶病毒", "番茄-黄化曲叶病毒",
]

# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 图像尺寸
IMAGE_SIZE = 224

# 数据集划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 随机种子（保证可复现性）
RANDOM_SEED = 42
