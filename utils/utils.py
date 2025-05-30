"""
工具函数模块
包含一些通用的辅助函数
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import Config

def setup_logging(log_file=None):
    """
    设置日志
    """
    if log_file is None:
        log_file = os.path.join(Config.LOG_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_json(data, file_path):
    """
    保存JSON文件
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path):
    """
    加载JSON文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(data, file_path):
    """
    保存pickle文件
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    """
    加载pickle文件
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_sample_data(output_path, n_users=1000, n_domains=100, n_records=100000, n_interests=5, interest_ratio=0.5):
    """
    创建示例数据用于测试，包含用户兴趣领域
    """
    print(f"创建示例数据: {n_users}个用户, {n_domains}个域名, {n_records}条记录, {n_interests}个兴趣领域")
    
    # 生成域名列表并分配到不同兴趣领域
    domains = [f"domain{i}.com" for i in range(n_domains)]
    np.random.shuffle(domains)
    domains_per_interest = n_domains // n_interests
    interest_domains = {i: domains[i*domains_per_interest:(i+1)*domains_per_interest] for i in range(n_interests)}
    
    # 为用户分配兴趣领域
    user_interests = {f"user_{i}": np.random.randint(0, n_interests) for i in range(n_users)}
    
    # 生成用户行为数据
    data = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    for _ in range(n_records):
        user_id = f"user_{np.random.randint(0, n_users)}"
        user_interest = user_interests[user_id]
        
        if np.random.random() < interest_ratio:
            # 用户访问其兴趣领域内的域名
            domain = np.random.choice(interest_domains[user_interest])
        else:
            # 随机访问任意域名
            domain = np.random.choice(domains)
        
        url = f"https://{domain}/page{np.random.randint(1, 100)}"
        # 生成 'YYYY-MM-DD' 格式的日期字符串
        random_days = np.random.randint(0, date_range_days + 1)
        timestamp_dt = start_date + timedelta(days=random_days)
        timestamp_str = timestamp_dt.strftime("%Y-%m-%d")
        weight = np.random.exponential(2.0)  # 指数分布的权重
        
        data.append([user_id, url, timestamp_str, weight])
    
    # 保存为CSV文件
    df = pd.DataFrame(data, columns=['user_id', 'url', 'timestamp_str', 'weight'])
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"示例数据已保存到: {output_path}")
    return df

def calculate_metrics(predictions, targets):
    """
    计算各种评估指标
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 准确率
    accuracy = np.mean(predictions == targets)
    
    # 精确率、召回率、F1分数（针对二分类）
    if len(np.unique(targets)) == 2:
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return {'accuracy': accuracy}

def cosine_similarity(a, b):
    """
    计算余弦相似度
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def euclidean_distance(a, b):
    """
    计算欧几里得距离
    """
    return np.linalg.norm(a - b)

def normalize_embeddings(embeddings):
    """
    归一化嵌入向量
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    return embeddings / norms

def get_top_k_similar(query_embedding, embeddings, k=10, exclude_indices=None):
    """
    获取最相似的top-k个嵌入向量
    """
    if exclude_indices is None:
        exclude_indices = set()
    
    similarities = []
    for i, embedding in enumerate(embeddings):
        if i not in exclude_indices:
            sim = cosine_similarity(query_embedding, embedding)
            similarities.append((i, sim))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 返回top-k
    top_k = similarities[:k]
    indices = [item[0] for item in top_k]
    scores = [item[1] for item in top_k]
    
    return indices, scores

def print_model_info(model):
    """
    打印模型信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("模型信息:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数数量: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

def format_time(seconds):
    """
    格式化时间
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def check_data_quality(df):
    """
    检查数据质量
    """
    print("数据质量检查:")
    print(f"  数据形状: {df.shape}")
    print(f"  缺失值数量: {df.isnull().sum().sum()}")
    print(f"  重复行数量: {df.duplicated().sum()}")
    
    if 'user_id' in df.columns:
        print(f"  唯一用户数: {df['user_id'].nunique()}")
    
    if 'url' in df.columns:
        print(f"  唯一URL数: {df['url'].nunique()}")
    
    if 'weight' in df.columns:
        print(f"  权重统计:")
        print(f"    最小值: {df['weight'].min():.4f}")
        print(f"    最大值: {df['weight'].max():.4f}")
        print(f"    平均值: {df['weight'].mean():.4f}")
        print(f"    标准差: {df['weight'].std():.4f}")

def create_experiment_config(config_dict, save_path=None):
    """
    创建实验配置文件
    """
    experiment_config = {
        'timestamp': datetime.now().isoformat(),
        'config': config_dict,
        'git_commit': get_git_commit(),  # 如果使用git
        'python_version': get_python_version(),
        'dependencies': get_package_versions()
    }
    
    if save_path:
        save_json(experiment_config, save_path)
        print(f"实验配置已保存到: {save_path}")
        
    return experiment_config

def get_git_commit():
    """
    获取当前的git commit hash
    """
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        return commit
    except:
        return "N/A"

def get_python_version():
    """
    获取Python版本
    """
    import sys
    return sys.version.split()[0]

def get_package_versions():
    """
    获取主要依赖包的版本
    """
    try:
        import torch
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import tqdm
        
        versions = {
            'torch': torch.__version__,
            'pandas': pandas.__version__,
            'numpy': numpy.__version__,
            'scikit-learn': sklearn.__version__,
            'matplotlib': matplotlib.__version__,
            'tqdm': tqdm.__version__
        }
        return versions
    except ImportError:
        return {"error": "Could not retrieve all package versions."}

def validate_config(config):
    """
    验证配置项的有效性
    """
    required_attrs = [
        'EXPERIMENT_NAME', 'DATA_PATH', 'EMBEDDING_DIM', 'WINDOW_SIZE', 
        'MIN_COUNT', 'NEGATIVE_SAMPLES', 'LEARNING_RATE', 'EPOCHS', 'BATCH_SIZE',
        'P_PARAM', 'Q_PARAM', 'WALK_LENGTH', 'NUM_WALKS' # Node2Vec参数
    ]
    
    missing = [attr for attr in required_attrs if not hasattr(config, attr)]
    if missing:
        raise ValueError(f"配置缺失必要属性: {', '.join(missing)}")
    
    if config.EMBEDDING_DIM <= 0:
        raise ValueError("EMBEDDING_DIM 必须为正整数")
    if config.WINDOW_SIZE <= 0:
        raise ValueError("WINDOW_SIZE 必须为正整数")
    # 可以添加更多验证逻辑
    
    print("配置验证通过。")

if __name__ == "__main__":
    # 示例：创建示例数据
    # create_sample_data("data/sample_user_behavior_with_interest.csv", n_records=50000)

    # 示例：设置日志
    # logger = setup_logging()
    # logger.info("这是一条日志信息")

    # 示例: 验证Config (假设Config类已定义且可访问)
    # try:
    #     validate_config(Config)
    # except ValueError as e:
    #     print(f"配置错误: {e}")
    pass 