import kagglehub
import os

# 设置下载路径为项目下的 datasets 目录
project_dir = os.path.dirname(os.path.abspath(__file__))
download_path = os.path.join(project_dir, "datasets")

# 创建 datasets 目录(如果不存在)
os.makedirs(download_path, exist_ok=True)
'''
# 检查 Kaggle API 配置
kaggle_config = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
if not os.path.exists(kaggle_config):
    print("⚠️  未找到 Kaggle API 配置文件!")
    print("请按以下步骤设置:")
    print("1. 访问 https://www.kaggle.com/account")
    print("2. 点击 'Create New API Token' 下载 kaggle.json")
    print(f"3. 将文件放到: {os.path.dirname(kaggle_config)}")
    exit(1)
'''
print("正在下载数据集...")
try:
    # 下载数据集到指定目录
    # 使用可用的 PlantVillage 数据集: mohitsingh1804/plantvillage
    path = kagglehub.dataset_download("mohitsingh1804/plantvillage", path=download_path)
    print(f"✓ 数据集下载成功!")
    print(f"Path to dataset files: {path}")
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print("\n可能的解决方案:")
    print("1. 确保已登录 Kaggle 并接受数据集条款")
    print("2. 检查数据集 URL 是否正确")
    print("3. 尝试其他 PlantVillage 数据集:")
    print("   - emmarex/plantdisease")
    print("   - vipoooool/new-plant-diseases-dataset")