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
    EXPERIMENT_NAME = "three_vector_test"
    
    # 基础路径配置
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    EXPERIMENT_DIR = EXPERIMENT_DIR
    
    # 数据相关配置
    DATA_PATH = "data/test_user_behavior.csv"  # 使用我们创建的测试数据
    ATTRIBUTE_DATA_PATH = "data/sample_user_attributes.tsv"
    
    # 行为向量相关配置
    MODEL_TYPE = "node2vec"  # 可选 "item2vec" 或 "node2vec"
    EMBEDDING_DIM = 128
    WINDOW_SIZE = 5
    MIN_COUNT = 5
    NEGATIVE_SAMPLES = 5
    
    # 属性向量相关配置
    ENABLE_ATTRIBUTES = True  # 启用属性向量
    ATTRIBUTE_EMBEDDING_DIM = 64  # 属性嵌入维度
    FUSION_HIDDEN_DIM = 256  # 融合层隐藏维度
    FINAL_USER_EMBEDDING_DIM = 256  # 最终用户嵌入维度
    
    # 位置相关配置
    ENABLE_LOCATION = True   # 启用位置向量
    LOCATION_DATA_PATH = "data/sample_user_base_stations.tsv"
    LOCATION_FEATURES_PATH = "data/sample_base_station_features.tsv"
    LOCATION_EMBEDDING_DIM = 128
    LOCATION_MIN_CONNECTIONS = 2  # 用户最少需要连接的基站数量
    
    # 基站特征使用模式
    BASE_STATION_FEATURE_MODE = "none"  # "none" 或 "text_embedding"
    TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 预训练语言模型
    TEXT_EMBEDDING_DIM = 384  # 文本嵌入维度
    
    # 位置模型训练参数
    LOCATION_LEARNING_RATE = 0.001
    LOCATION_EPOCHS = 5
    LOCATION_MODEL_TYPE = "item2vec"  # "item2vec" 或 "node2vec"
    LOCATION_WINDOW_SIZE = 5
    LOCATION_NEGATIVE_SAMPLES = 5
    LOCATION_MIN_COUNT = 1
    LOCATION_BATCH_SIZE = 64
    
    # 数值属性处理配置
    NUMERICAL_STANDARDIZATION = True  # 是否对数值属性进行标准化
    CATEGORICAL_MIN_FREQ = 5  # 类别属性最小频次（低频类别会被归为'其他'）
    
    # Node2Vec specific parameters
    P_PARAM = 1.0  # Return parameter
    Q_PARAM = 1.0  # In-out parameter
    WALK_LENGTH = 20  # Length of each random walk
    NUM_WALKS = 4  # Number of walks per node
    NODE2VEC_PRECOMPUTE = False  # True: 预计算边转移概率(慢启动，快游走); False: 动态计算(快启动，稍慢游走)
    
    # Node2Vec 缓存配置
    USE_WALKS_CACHE = True  # 是否使用随机游走缓存
    FORCE_REGENERATE_WALKS = False  # 是否强制重新生成随机游走（忽略缓存）
    
    # 行为训练相关配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 0.001
    EPOCHS = 10              # 减少训练轮数用于快速测试
    BATCH_SIZE = 256 # 4096
    RANDOM_SEED = 42
    NUM_WORKERS = 0  # DataLoader的num_workers, Windows默认为0, Linux可尝试 os.cpu_count() // 2
    PIN_MEMORY = True if DEVICE == "cuda" else False # DataLoader的pin_memory

    # 属性训练相关配置
    ATTRIBUTE_LEARNING_RATE = 0.001  # 属性训练学习率
    ATTRIBUTE_EPOCHS = 8     # 属性模型训练轮数
    ATTRIBUTE_BATCH_SIZE = 512  # 属性训练批次大小
    MASKING_RATIO = 0.15  # 掩码比例
    ATTRIBUTE_EARLY_STOPPING_PATIENCE = 10  # 属性训练早停耐心值
    
    # 可视化相关配置
    CLUSTER_NUM = 10
    
    # 其他配置（lsx）
    EARLY_STOPPING_PATIENCE = 10  # 假设早停的耐心值为10
    EVAL_INTERVAL = 2  # 假设每1个epoch进行一次评估
