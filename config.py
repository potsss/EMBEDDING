"""
项目配置文件
"""
import torch
import os
from datetime import datetime

# 模块级变量，用于缓存已确定的实验目录路径
_ACTUAL_EXPERIMENT_DIR = None

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")

def get_experiment_dir(experiment_name, allow_existing_without_timestamp=False, force_recalculate=False):
    """
    获取实验目录。
    - 如果 allow_existing_without_timestamp 为 True 且基础目录存在，则使用它。
    - 否则，如果基础目录存在，则添加时间戳。
    - 通过缓存确保在单次运行中路径的一致性（除非 force_recalculate）。
    Args:
        experiment_name: 基础实验名称。
        allow_existing_without_timestamp: 是否允许使用已存在的不带时间戳的目录。
        force_recalculate: 是否强制重新计算路径。
    """
    global _ACTUAL_EXPERIMENT_DIR
    if force_recalculate or not _ACTUAL_EXPERIMENT_DIR:
        base_experiment_path = os.path.join(EXPERIMENT_DIR, experiment_name)
        
        if allow_existing_without_timestamp and os.path.exists(base_experiment_path):
            _ACTUAL_EXPERIMENT_DIR = base_experiment_path
        elif os.path.exists(base_experiment_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _ACTUAL_EXPERIMENT_DIR = f"{base_experiment_path}_{timestamp}"
        else:
            _ACTUAL_EXPERIMENT_DIR = base_experiment_path
            
    return _ACTUAL_EXPERIMENT_DIR

def get_experiment_paths(experiment_name, allow_existing_without_timestamp=False, force_recalculate=False):
    """
    获取实验相关的所有路径
    """
    experiment_dir_path = get_experiment_dir(experiment_name, allow_existing_without_timestamp, force_recalculate)
    return {
        'PROCESSED_DATA_PATH': os.path.join(experiment_dir_path, "processed_data"),
        'CHECKPOINT_DIR': os.path.join(experiment_dir_path, "checkpoints"),
        'MODEL_SAVE_PATH': os.path.join(experiment_dir_path, "models"),
        'LOG_DIR': os.path.join(experiment_dir_path, "logs"),
        'TENSORBOARD_DIR': os.path.join(experiment_dir_path, "runs"),
        'VISUALIZATION_DIR': os.path.join(experiment_dir_path, "visualizations")
    }

class Config:
    """
    配置类
    包含所有模型训练和评估的参数
    """
    # 实验配置
    EXPERIMENT_NAME = "edu1"
    
    # 基础路径配置
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    EXPERIMENT_DIR = EXPERIMENT_DIR
    
    # 数据相关配置
    DATA_PATH = "data/edu.csv"
    
    # 模型相关配置
    EMBEDDING_DIM = 128
    WINDOW_SIZE = 5
    MIN_COUNT = 5
    NEGATIVE_SAMPLES = 5
    LEARNING_RATE = 0.001
    EPOCHS = 0
    BATCH_SIZE = 1024
    
    # 训练相关配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EARLY_STOPPING_PATIENCE = 50
    
    # 评估相关配置
    EVAL_INTERVAL = 2
    TOP_K = 10
    
    # 可视化相关配置
    CLUSTER_NUM = 10
    
    # 随机种子
    RANDOM_SEED = 42

# 初始化实验路径的逻辑将移到 main.py 中，以便根据模式进行更灵活的控制
# 此处不再自动调用 get_experiment_paths 和 setattr
