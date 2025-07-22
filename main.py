"""
主程序
整合所有模块，提供完整的训练和评估流程
"""
import os
import argparse
import torch
import numpy as np
import random
import json
import pickle
from datetime import datetime
from config import Config, get_experiment_dir, get_experiment_paths
from data_preprocessing import DataPreprocessor, LocationProcessor
import pandas as pd
from model import Item2Vec, UserEmbedding
from trainer import Trainer, AttributeTrainer, train_location_model
from trainer import load_attribute_models
from visualizer import Visualizer

# 导入Node2Vec相关的模块
from model import Node2Vec
from utils.node2vec_utils import build_graph_from_sequences, generate_node2vec_walks, generate_node2vec_walks_precompute, generate_node2vec_walks_with_cache

# 导入属性相关的模块
from model import AttributeEmbeddingModel, UserFusionModel, EnhancedUserEmbedding

# 导入位置相关的模块
from model import UserLocationEmbedding

def set_random_seed(seed):
    """
    设置随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_training_entities(url_mappings, base_station_mappings, processed_data_path):
    """
    保存训练时的实体记录，用于新用户推理时的过滤
    
    Args:
        url_mappings: URL映射字典
        base_station_mappings: 基站映射字典（可选）
        processed_data_path: 处理数据保存路径
    """
    training_entities = {
        'urls': set(url_mappings['url_to_id'].keys()),
        'url_to_id': url_mappings['url_to_id'],
        'id_to_url': url_mappings['id_to_url']
    }
    
    # 添加基站信息（如果可用）
    if base_station_mappings:
        training_entities['base_stations'] = set(base_station_mappings['base_station_to_id'].keys())
        training_entities['base_station_to_id'] = base_station_mappings['base_station_to_id']
        training_entities['id_to_base_station'] = base_station_mappings['id_to_base_station']
    
    # 保存到文件
    entities_path = os.path.join(processed_data_path, 'training_entities.pkl')
    with open(entities_path, 'wb') as f:
        pickle.dump(training_entities, f)
    
    print(f"训练实体记录已保存: {entities_path}")
    print(f"  训练URL数量: {len(training_entities['urls'])}")
    if base_station_mappings:
        print(f"  训练基站数量: {len(training_entities['base_stations'])}")

def load_training_entities(processed_data_path):
    """
    加载训练时的实体记录
    
    Args:
        processed_data_path: 处理数据路径
        
    Returns:
        训练实体记录字典
    """
    entities_path = os.path.join(processed_data_path, 'training_entities.pkl')
    if os.path.exists(entities_path):
        with open(entities_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"警告：未找到训练实体记录文件: {entities_path}")
        return None

def initialize_experiment_paths(experiment_name_override=None, mode=None):
    """
    初始化实验相关的路径，并设置到Config类上。
    根据运行模式决定是否允许使用已存在的不带时间戳的目录。
    """
    current_experiment_name = experiment_name_override if experiment_name_override else Config.EXPERIMENT_NAME
    
    force_recalc = True # 默认为True，确保在新的main.py执行开始时重新计算
    allow_existing = False

    if experiment_name_override:
        # 如果通过命令行指定了实验名称
        Config.EXPERIMENT_NAME = experiment_name_override
        # 对于推理相关模式，允许使用已存在的目录
        if mode in ['compute_new_users', 'visualize', 'compute_embeddings']:
            allow_existing = True
        else:
            allow_existing = False 
    elif mode and mode not in ['preprocess', 'all']:
        # 如果是分步执行（非preprocess/all），且未指定新实验名，则尝试使用已存在的目录
        allow_existing = True
        force_recalc = True # 仍然需要重新计算，但会优先使用已存在的
    
    paths = get_experiment_paths(current_experiment_name, 
                                 allow_existing_without_timestamp=allow_existing, 
                                 force_recalculate=force_recalc)
    for key, value in paths.items():
        setattr(Config, key, value)
    
    # 确保DEVICE_OBJ在路径确定后设置
    Config.DEVICE_OBJ = torch.device(Config.DEVICE)

def create_directories():
    """
    创建必要的目录。此时Config中的路径应该已经被正确设置了。
    """
    # 获取已确定的实验目录路径
    # 注意：这里的get_experiment_dir调用不应再修改_ACTUAL_EXPERIMENT_DIR，因为它应该已经被initialize_experiment_paths正确设置
    # 所以，理想情况下，我们应该直接从Config中获取experiment_dir，或者确保get_experiment_dir在不带force_recalculate时返回缓存值
    experiment_dir_to_create = get_experiment_dir(Config.EXPERIMENT_NAME) # 这会返回缓存的路径
    
    print(f"\n确保实验目录存在: {experiment_dir_to_create}")
    os.makedirs(experiment_dir_to_create, exist_ok=True)
    
    # 子目录现在直接从Config中获取
    sub_directories = [
        Config.PROCESSED_DATA_PATH,
        Config.CHECKPOINT_DIR,
        Config.MODEL_SAVE_PATH,
        Config.LOG_DIR,
        Config.TENSORBOARD_DIR,
        Config.VISUALIZATION_DIR,
        # Config.DATA_DIR # DATA_DIR 通常是项目级别的，不应在每个实验下创建
    ]
    # 确保项目级数据目录存在
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        print(f"创建项目数据目录: {Config.DATA_DIR}")

    for directory in sub_directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建子目录: {directory}")
    
    # 保存实验配置信息
    def convert_to_serializable(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool, list, dict, tuple)):
            return obj
        elif obj is None:
            return None
        return str(obj)
    
    config_dict = {}
    for key in dir(Config):
        if not key.startswith('_') and not callable(getattr(Config, key)) and key != 'DEVICE_OBJ': # 排除DEVICE_OBJ，因为它不是原始配置项
            value = getattr(Config, key)
            config_dict[key] = convert_to_serializable(value)
    
    config_info = {
        'experiment_name': Config.EXPERIMENT_NAME,
        'actual_experiment_dir': experiment_dir_to_create, # 使用实际创建/使用的目录
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': config_dict
    }
    
    config_path = os.path.join(experiment_dir_to_create, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=4, ensure_ascii=False)
    
    print(f"\n实验配置已保存到: {config_path}")
    print("="*50)

def preprocess_data(data_path=None):
    """
    数据预处理
    返回:
        user_sequences: 用户访问序列
        url_mappings: 域名到ID的映射
        user_attributes: 用户属性数据
        attribute_info: 属性信息
        user_location_sequences: 用户位置序列
        base_station_mappings: 基站到ID的映射
        location_weights: 位置权重信息
    """
    print("="*50)
    print("开始数据预处理")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    
    if data_path:
        user_sequences = preprocessor.preprocess(data_path)
    else:
        # 尝试加载已处理的数据
        try:
            user_sequences = preprocessor.load_processed_data()
            print("已加载处理后的数据")
        except:
            print("未找到处理后的数据，请提供原始数据路径")
            return None, None, None, None, None, None, None
    
    url_mappings = {
        'url_to_id': preprocessor.url_to_id,
        'id_to_url': preprocessor.id_to_url
    }
    
    print(f"数据预处理完成:")
    print(f"  用户数量: {len(user_sequences)}")
    print(f"  物品数量: {len(url_mappings['url_to_id'])}")
    
    # 保存训练实体记录（在属性和位置数据加载后进行）
    # 这里先声明，后面会在所有数据加载完成后调用
    
    # 加载属性数据（如果启用）
    user_attributes = None
    attribute_info = None
    if Config.ENABLE_ATTRIBUTES and preprocessor.attribute_processor:
        try:
            user_attributes, attribute_info = preprocessor.attribute_processor.load_processed_attributes()
            if user_attributes is not None:
                print(f"  属性用户数量: {len(user_attributes)}")
                print(f"  属性数量: {len(attribute_info)}")
            else:
                print("  属性数据未找到或处理失败")
        except:
            print("  无法加载属性数据")
    
    # 加载位置数据（如果启用）
    user_location_sequences = None
    base_station_mappings = None
    location_weights = None
    if Config.ENABLE_LOCATION and preprocessor.location_processor:
        try:
            location_data = preprocessor.location_processor.load_processed_data(Config.PROCESSED_DATA_PATH)
            if location_data:
                user_location_sequences = location_data['user_sequences']
                base_station_mappings = location_data['base_station_mappings']
                location_weights = location_data['user_weights']
                print(f"位置数据加载成功，用户数量: {len(user_location_sequences)}")
                print(f"  位置用户数量: {len(user_location_sequences)}")
                print(f"  基站数量: {len(base_station_mappings['base_station_to_id'])}")
            else:
                print("位置数据加载成功，用户数量: 0")
                print("  位置用户数量: 0")
                print("  基站数量: 0")
        except Exception as e:
            print(f"位置数据加载成功，用户数量: 0")
            print(f"  位置用户数量: 0")
            print(f"  基站数量: 0")
            print(f"  无法加载位置数据: {e}")
    
    # 保存训练实体记录（用于新用户推理时的过滤）
    save_training_entities(url_mappings, base_station_mappings, Config.PROCESSED_DATA_PATH)
    
    return user_sequences, url_mappings, user_attributes, attribute_info, user_location_sequences, base_station_mappings, location_weights

def train_model(user_sequences, url_mappings, resume=False):
    """
    训练模型
    """
    print("="*50)
    print("开始模型训练")
    print("="*50)
    
    # 创建模型
    vocab_size = len(url_mappings['url_to_id'])
    model = Item2Vec(vocab_size, Config.EMBEDDING_DIM)
    
    print(f"模型参数:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  嵌入维度: {Config.EMBEDDING_DIM}")
    print(f"  设备: {Config.DEVICE}")
    
    # 创建训练器
    trainer = Trainer(model)
    
    # 开始训练
    trainer.train(user_sequences, resume_from_checkpoint=resume)
    
    return model, trainer

def train_node2vec_model(user_sequences, url_mappings, resume=False):
    """
    训练 Node2Vec 模型
    """
    print("="*50)
    print("开始 Node2Vec 模型训练")
    print("="*50)

    # 1. 从用户序列构建图
    # Node2Vec 通常在无向图上效果更好，权重可以来自共现频率
    item_graph = build_graph_from_sequences(user_sequences, directed=False)
    if not item_graph:
        print("错误:未能从用户序列构建图。请检查数据。")
        return None, None

    # 2. 生成随机游走
    print("生成 Node2Vec 随机游走...")
    
    # 使用带缓存的随机游走生成器
    node2vec_walks = generate_node2vec_walks_with_cache(
        graph=item_graph,
        num_walks=Config.NUM_WALKS,
        walk_length=Config.WALK_LENGTH,
        p=Config.P_PARAM,
        q=Config.Q_PARAM,
        use_cache=Config.USE_WALKS_CACHE,
        force_regenerate=Config.FORCE_REGENERATE_WALKS
    )
    if not node2vec_walks:
        print("错误:未能生成 Node2Vec 随机游走。")
        return None, None
    
    print(f"已生成 {len(node2vec_walks)} 条随机游走。")

    # 3. 创建 Node2Vec 模型
    vocab_size = len(url_mappings['url_to_id']) # 词汇表大小与Item2Vec相同
    model = Node2Vec(vocab_size, Config.EMBEDDING_DIM)
    
    print(f"Node2Vec 模型参数:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  嵌入维度: {Config.EMBEDDING_DIM}")
    print(f"  P参数: {Config.P_PARAM}")
    print(f"  Q参数: {Config.Q_PARAM}")
    print(f"  游走长度: {Config.WALK_LENGTH}")
    print(f"  每个节点的游走次数: {Config.NUM_WALKS}")
    print(f"  设备: {Config.DEVICE}")

    # 4. 创建训练器
    trainer = Trainer(model) # Trainer应该可以复用

    # 5. 开始训练 (使用生成的随机游走作为输入序列)
    # 注意: Trainer 的 create_dataloader 和 SkipGramDataset 需要能够处理这些游走
    # SkipGramDataset 期望的是一个序列列表，这与 node2vec_walks 的输出格式一致
    print("开始使用生成的游走训练 Node2Vec 模型...")
    trainer.train(node2vec_walks, resume_from_checkpoint=resume) # 将游走序列传递给训练器
    
    return model, trainer

def visualize_results(model, user_sequences, url_mappings):
    """
    可视化结果
    进行用户和物品嵌入向量的 t-SNE 可视化
    """
    print("="*50)
    print("开始嵌入向量可视化")
    print("="*50)
    
    # 创建可视化器
    visualizer = Visualizer(model, user_sequences, url_mappings)
    
    # 可视化用户嵌入向量
    print("\n1. 用户嵌入向量可视化")
    visualizer.visualize_user_embeddings(
        sample_size=500,  # 采样500个用户
        perplexity=30,    # t-SNE 困惑度
        n_iter=1000       # t-SNE 迭代次数
    )
    
    # 可视化物品嵌入向量
    print("\n2. 物品嵌入向量可视化")
    visualizer.visualize_item_embeddings(
        sample_size=1000,  # 采样1000个物品
        perplexity=30,     # t-SNE 困惑度
        n_iter=1000        # t-SNE 迭代次数
    )
    
    print("="*50)
    print("可视化完成")
    print("="*50)

def compute_user_embeddings(model, user_sequences, url_mappings, save_path=None):
    """
    计算并保存用户嵌入向量
    """
    print("="*50)
    print("计算用户嵌入向量")
    print("="*50)
    
    # 创建用户嵌入计算器
    user_embedding = UserEmbedding(model, user_sequences, url_mappings)
    
    # 计算用户嵌入
    user_embeddings = user_embedding.compute_user_embeddings()
    
    print(f"已计算 {len(user_embeddings)} 个用户的嵌入向量")
    
    # 保存用户嵌入
    if save_path:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(user_embeddings, f)
        print(f"用户嵌入已保存到: {save_path}")
    
    return user_embeddings

def load_trained_model(model_path, vocab_size):
    """
    加载训练好的模型
    """
    model = Item2Vec(vocab_size, Config.EMBEDDING_DIM)
    
    if os.path.exists(model_path):
        # Add Config to safe globals for weights_only=True loading
        torch.serialization.add_safe_globals([Config])
        checkpoint = torch.load(model_path, map_location=Config.DEVICE_OBJ, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载训练好的模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return None
    
    return model

def compute_location_embeddings(location_model, user_base_stations, base_station_mappings, location_processor=None, save_path=None):
    """
    计算用户位置嵌入
    
    Args:
        location_model: 训练好的位置嵌入模型
        user_base_stations: 用户基站连接数据 {user_id: {'base_stations': [...], 'weights': [...], 'total_duration': ...}}
        base_station_mappings: 基站映射字典
        location_processor: 位置数据处理器（用于特征处理）
        save_path: 保存路径
    
    Returns:
        用户位置嵌入字典
    """
    if not location_model or not user_base_stations:
        return {}
    
    print("开始计算用户位置嵌入...")
    
    # 获取基站嵌入
    base_station_embeddings = {}
    
    # 对于PyTorch模型，直接从模型权重获取嵌入
    for bs_id in base_station_mappings['base_station_to_id'].keys():
        idx = base_station_mappings['base_station_to_id'][bs_id]
        if hasattr(location_model, 'in_embeddings'):
            # Item2Vec和Node2Vec使用in_embeddings
            embedding = location_model.in_embeddings.weight[idx].detach()
            base_station_embeddings[bs_id] = embedding
        elif hasattr(location_model, 'embeddings'):
            # 其他模型可能使用embeddings
            embedding = location_model.embeddings.weight[idx].detach()
            base_station_embeddings[bs_id] = embedding
        else:
            # 如果模型结构不同，使用其他方法获取嵌入
            # 这里可以根据具体模型结构调整
            print(f"警告：无法从模型中获取基站 {bs_id} 的嵌入")
            continue
    
    # 创建用户位置嵌入计算器
    location_embedding_calculator = UserLocationEmbedding(
        Config, base_station_embeddings, location_processor
    )
    
    # 计算用户位置嵌入
    user_location_embeddings = {}
    for user_id, data in user_base_stations.items():
        base_stations = data['base_stations']
        weights = data['weights']
        
        # 计算位置嵌入
        location_embedding = location_embedding_calculator(base_stations, weights)
        user_location_embeddings[user_id] = location_embedding.detach().numpy()
    
    print(f"完成计算 {len(user_location_embeddings)} 个用户的位置嵌入")
    
    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, user_location_embeddings)
        print(f"位置嵌入已保存到: {save_path}")
    
    return user_location_embeddings

def train_attribute_models(behavior_model, user_sequences, user_attributes, attribute_info, url_mappings):
    """
    训练属性模型
    """
    if not Config.ENABLE_ATTRIBUTES or user_attributes is None or attribute_info is None:
        print("属性训练未启用或属性数据不可用，跳过属性训练")
        return None, None
    
    print("="*50)
    print("开始属性模型训练")
    print("="*50)
    
    # 创建属性训练器
    attribute_trainer = AttributeTrainer(
        behavior_model, user_sequences, user_attributes, 
        attribute_info, url_mappings, Config
    )
    
    # 开始训练
    attribute_trainer.train()
    
    return attribute_trainer.attribute_model, attribute_trainer.fusion_model

def compute_enhanced_user_embeddings(behavior_model, attribute_model, fusion_model, 
                                   user_sequences, user_attributes, url_mappings, attribute_info, 
                                   location_model=None, user_location_sequences=None, 
                                   base_station_mappings=None, location_weights=None, 
                                   location_processor=None, save_path=None):
    """
    计算增强的用户嵌入向量（行为+属性+位置）
    """
    if not Config.ENABLE_ATTRIBUTES or attribute_model is None or fusion_model is None:
        print("属性模型不可用，使用基础用户嵌入")
        return compute_user_embeddings(behavior_model, user_sequences, url_mappings, save_path)
    
    print("="*50)
    print("计算增强用户嵌入向量（行为+属性+位置）")
    print("="*50)
    
    # 创建增强用户嵌入计算器
    enhanced_user_embedding = EnhancedUserEmbedding(
        behavior_model=behavior_model,
        attribute_model=attribute_model,
        fusion_model=fusion_model,
        user_sequences=user_sequences,
        user_attributes=user_attributes,
        url_mappings=url_mappings,
        attribute_info=attribute_info,
        location_model=location_model,
        user_location_sequences=user_location_sequences,
        base_station_mappings=base_station_mappings,
        location_weights=location_weights,
        location_processor=location_processor
    )
    
    # 计算增强嵌入
    enhanced_embeddings = enhanced_user_embedding.compute_enhanced_user_embeddings()
    
    print(f"已计算 {len(enhanced_embeddings)} 个用户的增强嵌入向量")
    
    # 保存增强嵌入
    if save_path:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(enhanced_embeddings, f)
        print(f"增强用户嵌入已保存到: {save_path}")
    
    return enhanced_embeddings

def compute_new_user_embeddings(behavior_model, attribute_model, fusion_model,
                              url_mappings, attribute_info, base_station_mappings=None,
                              location_model=None, location_processor=None,
                              new_user_behavior_path=None, new_user_attribute_path=None, 
                              new_user_location_path=None, save_path=None):
    """
    为新用户计算向量表示
    
    Args:
        behavior_model: 训练好的行为模型
        attribute_model: 训练好的属性模型
        fusion_model: 训练好的融合模型
        url_mappings: URL映射字典
        attribute_info: 属性信息字典
        base_station_mappings: 基站映射字典
        location_model: 训练好的位置模型
        location_processor: 位置数据处理器
        new_user_behavior_path: 新用户行为数据路径
        new_user_attribute_path: 新用户属性数据路径
        new_user_location_path: 新用户位置数据路径
        save_path: 保存路径
    
    Returns:
        新用户向量字典
    """
    print("="*50)
    print("计算新用户向量表示")
    print("="*50)
    
    # 使用配置文件中的默认路径
    if new_user_behavior_path is None:
        new_user_behavior_path = Config.NEW_USER_BEHAVIOR_PATH
    if new_user_attribute_path is None:
        new_user_attribute_path = Config.NEW_USER_ATTRIBUTE_PATH
    if new_user_location_path is None:
        new_user_location_path = Config.NEW_USER_LOCATION_PATH
    
    # 加载新用户数据
    new_user_data = load_new_user_data(
        behavior_path=new_user_behavior_path,
        attribute_path=new_user_attribute_path,
        location_path=new_user_location_path,
        url_mappings=url_mappings,
        attribute_info=attribute_info,
        base_station_mappings=base_station_mappings,
        location_processor=location_processor
    )
    
    if not new_user_data:
        print("没有找到新用户数据")
        return {}
    
    new_user_sequences = new_user_data['user_sequences']
    new_user_attributes = new_user_data['user_attributes']
    new_user_location_data = new_user_data['user_location_data']
    
    print(f"加载了 {len(new_user_sequences)} 个新用户的行为数据")
    print(f"加载了 {len(new_user_attributes)} 个新用户的属性数据")
    location_user_count = len(new_user_location_data.get('user_location_sequences', {}))
    print(f"加载了 {location_user_count} 个新用户的位置数据")
    
    # 计算新用户向量
    if (Config.ENABLE_ATTRIBUTES and attribute_model is not None and fusion_model is not None 
        and len(new_user_attributes) > 0):
        # 使用增强嵌入（行为+属性+位置）
        new_user_embeddings = compute_enhanced_user_embeddings(
            behavior_model=behavior_model,
            attribute_model=attribute_model,
            fusion_model=fusion_model,
            user_sequences=new_user_sequences,
            user_attributes=new_user_attributes,
            url_mappings=url_mappings,
            attribute_info=attribute_info,
            location_model=location_model,
            user_location_sequences=new_user_location_data.get('user_location_sequences'),
            base_station_mappings=base_station_mappings,
            location_weights=new_user_location_data.get('location_weights'),
            location_processor=location_processor,
            save_path=None  # 暂时不保存，最后统一保存
        )
    else:
        # 仅使用行为嵌入
        new_user_embeddings = compute_user_embeddings(
            model=behavior_model,
            user_sequences=new_user_sequences,
            url_mappings=url_mappings,
            save_path=None
        )
    
    print(f"成功计算 {len(new_user_embeddings)} 个新用户的向量表示")
    
    # 保存结果
    if save_path:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(new_user_embeddings, f)
        print(f"新用户向量已保存到: {save_path}")
    
    return new_user_embeddings

def load_new_user_data(behavior_path, attribute_path, location_path,
                      url_mappings, attribute_info, base_station_mappings=None, 
                      location_processor=None):
    """
    加载新用户的所有数据，并根据训练实体进行过滤
    
    Args:
        behavior_path: 新用户行为数据路径
        attribute_path: 新用户属性数据路径
        location_path: 新用户位置数据路径
        url_mappings: URL映射字典
        attribute_info: 属性信息字典
        base_station_mappings: 基站映射字典
        location_processor: 位置数据处理器
    
    Returns:
        包含所有新用户数据的字典
    """
    result = {
        'user_sequences': {},
        'user_attributes': {},
        'user_location_data': {}
    }
    
    # 加载训练实体记录用于过滤
    training_entities = load_training_entities(Config.PROCESSED_DATA_PATH)
    if training_entities is None:
        print("警告：无法加载训练实体记录，将尝试使用所有新用户数据（可能导致错误）")
    else:
        print(f"已加载训练实体记录: {len(training_entities['urls'])} 个URL")
        if 'base_stations' in training_entities:
            print(f"  {len(training_entities['base_stations'])} 个基站")
    
    # 初始化过滤统计变量
    unknown_urls = set()
    unknown_base_stations = set()
    
    # 1. 加载新用户行为数据
    if os.path.exists(behavior_path):
        print(f"加载新用户行为数据: {behavior_path}")
        try:
            # 使用现有的数据预处理器
            preprocessor = DataPreprocessor(Config)
            
            # 直接加载并处理行为数据
            df = pd.read_csv(behavior_path, sep='\t')
            print(f"新用户行为数据形状: {df.shape}")
            
            # 处理行为序列，根据训练实体记录进行过滤
            user_sequences = {}
            url_to_id = url_mappings['url_to_id']
            
            # 统计过滤信息
            total_records = len(df)
            filtered_records = 0
            
            # 按用户分组处理
            for user_id, group in df.groupby('user_id'):
                sequence = []
                for _, row in group.iterrows():
                    url = row['url']
                    
                    # 检查URL是否在训练实体记录中
                    if training_entities and url not in training_entities['urls']:
                        unknown_urls.add(url)
                        filtered_records += 1
                        continue
                    
                    if url in url_to_id:  # 只处理训练时见过的URL
                        sequence.append(url_to_id[url])
                    else:
                        filtered_records += 1
                
                if sequence:  # 只保留有有效URL的用户
                    user_sequences[user_id] = sequence
            
            result['user_sequences'] = user_sequences
            
            # 输出过滤统计信息
            print(f"成功处理 {len(user_sequences)} 个新用户的行为序列")
            if filtered_records > 0:
                print(f"  过滤了 {filtered_records}/{total_records} 条记录（URL不在训练数据中）")
                if unknown_urls:
                    print(f"  未知URL示例: {list(unknown_urls)[:5]}{'...' if len(unknown_urls) > 5 else ''}")
            
        except Exception as e:
            print(f"加载新用户行为数据时出错: {e}")
    
    # 2. 加载新用户属性数据
    if Config.ENABLE_ATTRIBUTES and attribute_path and os.path.exists(attribute_path):
        print(f"加载新用户属性数据: {attribute_path}")
        try:
            # 使用现有的属性处理逻辑
            preprocessor = DataPreprocessor(Config)
            
            # 加载属性数据
            attr_df = pd.read_csv(attribute_path, sep='\t')
            print(f"新用户属性数据形状: {attr_df.shape}")
            
            # 处理属性数据，使用训练时的编码器
            user_attributes = {}
            
            for _, row in attr_df.iterrows():
                user_id = row['user_id']
                user_attrs = {}
                
                for attr_name, attr_info_item in attribute_info.items():
                    if attr_name in row:
                        attr_value = row[attr_name]
                        
                        if attr_info_item['type'] == 'categorical':
                            # 对于类别属性，使用训练时的编码
                            # 由于训练数据中所有类别属性都被编码为0（Other类别），
                            # 新用户的类别属性也使用0
                            user_attrs[attr_name] = 0
                        else:
                            # 数值属性
                            user_attrs[attr_name] = float(attr_value)
                
                if user_attrs:
                    user_attributes[user_id] = user_attrs
            
            result['user_attributes'] = user_attributes
            print(f"成功处理 {len(user_attributes)} 个新用户的属性数据")
            
        except Exception as e:
            print(f"加载新用户属性数据时出错: {e}")
    
    # 3. 加载新用户位置数据
    if Config.ENABLE_LOCATION and location_path and os.path.exists(location_path) and base_station_mappings:
        print(f"加载新用户位置数据: {location_path}")
        try:
            # 使用位置处理器
            if location_processor is None:
                location_processor = LocationProcessor(Config)
            
            # 加载位置数据
            location_df = pd.read_csv(location_path, sep='\t')
            print(f"新用户位置数据形状: {location_df.shape}")
            
            # 处理位置数据
            user_location_sequences = {}
            location_weights = {}
            base_station_to_id = base_station_mappings['base_station_to_id']
            
            # 统计过滤信息
            total_location_records = len(location_df)
            filtered_location_records = 0
            
            for user_id, group in location_df.groupby('user_id'):
                # 计算每个基站的权重（基于停留时间）
                base_station_durations = group.groupby('base_station_id')['duration'].sum()
                
                # 根据训练实体记录过滤基站
                valid_stations = []
                valid_weights = []
                
                for bs_id, duration in base_station_durations.items():
                    # 检查基站是否在训练实体记录中
                    if training_entities and 'base_stations' in training_entities and bs_id not in training_entities['base_stations']:
                        unknown_base_stations.add(bs_id)
                        filtered_location_records += len(group[group['base_station_id'] == bs_id])
                        continue
                    
                    if bs_id in base_station_to_id:
                        valid_stations.append(base_station_to_id[bs_id])
                        valid_weights.append(duration)
                    else:
                        filtered_location_records += len(group[group['base_station_id'] == bs_id])
                
                if len(valid_stations) >= Config.LOCATION_MIN_CONNECTIONS:
                    # 生成位置序列（按时间排序）
                    user_location_sequences[user_id] = valid_stations
                    
                    # 计算权重（归一化）
                    total_duration = sum(valid_weights)
                    # 注意：这里使用基站ID而不是基站名称作为键
                    normalized_weights = {base_station_to_id[bs_id]: weight/total_duration 
                                        for bs_id, weight in base_station_durations.items() if bs_id in base_station_to_id}
                    location_weights[user_id] = normalized_weights
            
            result['user_location_data'] = {
                'user_location_sequences': user_location_sequences,
                'location_weights': location_weights
            }
            print(f"成功处理 {len(user_location_sequences)} 个新用户的位置数据")
            if filtered_location_records > 0:
                print(f"  过滤了 {filtered_location_records}/{total_location_records} 条位置记录（基站不在训练数据中）")
                if unknown_base_stations:
                    print(f"  未知基站示例: {list(unknown_base_stations)[:5]}{'...' if len(unknown_base_stations) > 5 else ''}")
            
        except Exception as e:
            print(f"加载新用户位置数据时出错: {e}")
    
    # 生成过滤报告
    if training_entities:
        # 生成报告保存路径
        report_save_path = os.path.join(Config.PROCESSED_DATA_PATH, 'new_user_compatibility_report.json')
        generate_filtering_report(result, training_entities, behavior_path, location_path, 
                                unknown_urls, unknown_base_stations, report_save_path)
    
    return result

def generate_filtering_report(new_user_data, training_entities, behavior_path, location_path, 
                            unknown_urls, unknown_base_stations, save_path=None):
    """
    生成新用户数据过滤报告
    
    Args:
        new_user_data: 新用户数据字典
        training_entities: 训练实体记录
        behavior_path: 行为数据路径
        location_path: 位置数据路径
        unknown_urls: 未知URL集合
        unknown_base_stations: 未知基站集合
        save_path: 报告保存路径（可选）
    """
    print("\n" + "="*50)
    print("🔍 新用户数据过滤报告")
    print("="*50)
    
    # 基本统计
    print(f"📊 处理结果统计:")
    print(f"  ✅ 成功处理用户数量: {len(new_user_data['user_sequences'])}")
    
    # URL过滤统计
    if unknown_urls:
        print(f"\n🌐 URL过滤统计:")
        print(f"  ❌ 未知URL数量: {len(unknown_urls)}")
        print(f"  📝 训练URL数量: {len(training_entities['urls'])}")
        print(f"  📋 未知URL列表: {sorted(unknown_urls)}")
        
        # 建议
        print(f"\n💡 建议:")
        print(f"  • 如果这些URL很重要，考虑将它们添加到训练数据中")
        print(f"  • 或者可以将它们映射到相似的已知URL")
    
    # 基站过滤统计
    if unknown_base_stations:
        print(f"\n📡 基站过滤统计:")
        print(f"  ❌ 未知基站数量: {len(unknown_base_stations)}")
        print(f"  📝 训练基站数量: {len(training_entities['base_stations'])}")
        print(f"  📋 未知基站列表: {sorted(unknown_base_stations)}")
        
        # 建议
        print(f"\n💡 建议:")
        print(f"  • 如果这些基站很重要，考虑将它们添加到训练数据中")
        print(f"  • 或者检查基站ID的命名规范是否一致")
    
    # 数据质量评估
    total_users = len(new_user_data['user_sequences'])
    if total_users > 0:
        print(f"\n📈 数据质量评估:")
        
        # 行为数据质量
        avg_behavior_length = sum(len(seq) for seq in new_user_data['user_sequences'].values()) / total_users
        print(f"  📱 平均行为序列长度: {avg_behavior_length:.1f}")
        
        # 位置数据质量
        if 'user_location_data' in new_user_data and new_user_data['user_location_data']:
            location_data = new_user_data['user_location_data']
            if 'user_location_sequences' in location_data:
                location_users = len(location_data['user_location_sequences'])
                print(f"  📍 有位置数据的用户比例: {location_users/total_users*100:.1f}%")
        
        # 属性数据质量
        if 'user_attributes' in new_user_data:
            attr_users = len(new_user_data['user_attributes'])
            print(f"  👤 有属性数据的用户比例: {attr_users/total_users*100:.1f}%")
    
    print("="*50)
    
    # 保存报告到文件
    if save_path:
        save_compatibility_report_to_file(new_user_data, training_entities, behavior_path, location_path,
                                         unknown_urls, unknown_base_stations, save_path)

def save_compatibility_report_to_file(new_user_data, training_entities, behavior_path, location_path,
                                     unknown_urls, unknown_base_stations, save_path):
    """
    将兼容性报告保存到文件
    """
    import json
    from datetime import datetime
    
    # 计算统计信息
    total_users = len(new_user_data['user_sequences'])
    
    # 行为数据统计
    behavior_stats = None
    if unknown_urls is not None:
        behavior_records = 0
        filtered_behavior_records = 0
        try:
            if os.path.exists(behavior_path):
                import pandas as pd
                df = pd.read_csv(behavior_path, sep='\t')
                behavior_records = len(df)
                filtered_behavior_records = len(df[~df['url'].isin(training_entities['urls'])])
        except:
            pass
            
        behavior_stats = {
            'total_records': behavior_records,
            'filtered_records': filtered_behavior_records,
            'unknown_urls_count': len(unknown_urls),
            'unknown_urls': sorted(list(unknown_urls)),
            'known_urls_count': len(training_entities['urls']),
            'coverage': (len(training_entities['urls']) - len(unknown_urls)) / len(training_entities['urls']) if training_entities['urls'] else 0
        }
    
    # 位置数据统计
    location_stats = None
    if unknown_base_stations is not None and 'base_stations' in training_entities:
        location_records = 0
        filtered_location_records = 0
        try:
            if os.path.exists(location_path):
                import pandas as pd
                df = pd.read_csv(location_path, sep='\t')
                location_records = len(df)
                filtered_location_records = len(df[~df['base_station_id'].isin(training_entities['base_stations'])])
        except:
            pass
            
        location_stats = {
            'total_records': location_records,
            'filtered_records': filtered_location_records,
            'unknown_base_stations_count': len(unknown_base_stations),
            'unknown_base_stations': sorted(list(unknown_base_stations)),
            'known_base_stations_count': len(training_entities['base_stations']),
            'coverage': (len(training_entities['base_stations']) - len(unknown_base_stations)) / len(training_entities['base_stations']) if training_entities['base_stations'] else 0
        }
    
    # 数据质量评估
    avg_behavior_length = sum(len(seq) for seq in new_user_data['user_sequences'].values()) / total_users if total_users > 0 else 0
    
    location_users = 0
    if 'user_location_data' in new_user_data and new_user_data['user_location_data']:
        location_data = new_user_data['user_location_data']
        if 'user_location_sequences' in location_data:
            location_users = len(location_data['user_location_sequences'])
    
    attr_users = len(new_user_data.get('user_attributes', {}))
    
    # 计算总体兼容性评分
    total_score = 0
    max_score = 0
    if behavior_stats:
        total_score += behavior_stats['coverage'] * 50
        max_score += 50
    if location_stats:
        total_score += location_stats['coverage'] * 50
        max_score += 50
    
    final_score = total_score / max_score if max_score > 0 else 1
    
    # 生成报告数据
    report = {
        'timestamp': datetime.now().isoformat(),
        'experiment_info': {
            'behavior_data_path': behavior_path,
            'location_data_path': location_path,
            'training_entities_count': {
                'urls': len(training_entities['urls']),
                'base_stations': len(training_entities.get('base_stations', []))
            }
        },
        'processing_results': {
            'total_users_processed': total_users,
            'avg_behavior_sequence_length': round(avg_behavior_length, 2),
            'users_with_location_data': location_users,
            'users_with_attribute_data': attr_users,
            'location_coverage_percent': round(location_users / total_users * 100, 1) if total_users > 0 else 0,
            'attribute_coverage_percent': round(attr_users / total_users * 100, 1) if total_users > 0 else 0
        },
        'behavior_data_analysis': behavior_stats,
        'location_data_analysis': location_stats,
        'compatibility_assessment': {
            'overall_score_percent': round(final_score * 100, 1),
            'rating': (
                'excellent' if final_score >= 0.8 else
                'good' if final_score >= 0.6 else
                'fair' if final_score >= 0.4 else
                'poor'
            ),
            'recommendations': []
        }
    }
    
    # 添加建议
    recommendations = []
    if behavior_stats and behavior_stats['unknown_urls_count'] > 0:
        recommendations.append("考虑将重要的未知URL添加到训练数据中")
        recommendations.append("或者将未知URL映射到相似的已知URL")
    
    if location_stats and location_stats['unknown_base_stations_count'] > 0:
        recommendations.append("检查基站ID的命名规范是否一致")
        recommendations.append("考虑将重要的未知基站添加到训练数据中")
    
    report['compatibility_assessment']['recommendations'] = recommendations
    
    # 保存到JSON文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 兼容性报告已保存到: {save_path}")

def load_training_config(experiment_dir):
    """
    加载训练时保存的配置，用于推理阶段
    """
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    if os.path.exists(config_path):
        print(f"加载训练时的配置: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        # 更新Config类的属性
        if 'config' in saved_config:
            training_config = saved_config['config']
            for key, value in training_config.items():
                if hasattr(Config, key):
                    # 跳过路径相关的配置，因为这些会在initialize_experiment_paths中设置
                    if key.endswith('_PATH') or key.endswith('_DIR'):
                        continue
                    setattr(Config, key, value)
                    print(f"  更新配置: {key} = {value}")
        
        # 重新设置设备对象
        Config.DEVICE_OBJ = torch.device(Config.DEVICE)
        
        print("训练时配置加载完成")
        return True
    else:
        print(f"警告：未找到训练时的配置文件: {config_path}")
        return False

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Item2Vec/Node2Vec用户表示向量训练项目')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'visualize', 'compute_embeddings', 'compute_new_users', 'all'], 
                       default='all', help='运行模式 (preprocess, train, visualize, compute_embeddings, compute_new_users, all)')
    parser.add_argument('--data_path', type=str, help='原始数据路径 (例如: data/user_behavior.csv)')
    parser.add_argument('--model_path', type=str, help='已训练模型的路径 (用于visualize/compute_embeddings模式)')
    parser.add_argument('--resume', action='store_true', help='从最新的检查点恢复训练')
    parser.add_argument('--no_train', action='store_true', help='跳过训练，直接使用已有模型 (与visualize/compute_embeddings模式结合)')
    parser.add_argument('--experiment_name', type=str, help='自定义实验名称 (默认为config.py中的EXPERIMENT_NAME)')
    parser.add_argument('--no_cache', action='store_true', help='禁用随机游走缓存')
    parser.add_argument('--force_regenerate', action='store_true', help='强制重新生成随机游走（忽略缓存）')
    parser.add_argument('--enable_attributes', action='store_true', help='启用属性向量训练')
    parser.add_argument('--attribute_data_path', type=str, help='用户属性数据文件路径')
    
    # 新用户向量计算相关参数
    parser.add_argument('--new_user_behavior_path', type=str, help='新用户行为数据路径')
    parser.add_argument('--new_user_attribute_path', type=str, help='新用户属性数据路径')
    parser.add_argument('--new_user_location_path', type=str, help='新用户位置数据路径')
    
    args = parser.parse_args()
    
    # 1. 初始化实验路径 (这是关键改动)
    initialize_experiment_paths(experiment_name_override=args.experiment_name, mode=args.mode)
    
    # 2. 对于推理相关模式，加载训练时的配置
    if args.mode in ['compute_new_users', 'visualize', 'compute_embeddings']:
        experiment_dir = get_experiment_dir(Config.EXPERIMENT_NAME)
        load_training_config(experiment_dir)
    
    # 3. 设置随机种子
    set_random_seed(Config.RANDOM_SEED)
    
    # 4. 创建目录 (现在它会使用initialize_experiment_paths确定的路径)
    create_directories()
    
    # 更新 DATA_PATH 如果通过命令行指定 (通常用于 preprocess)
    if args.data_path:
        Config.DATA_PATH = args.data_path
        print(f"使用命令行指定的数据路径: {Config.DATA_PATH}")

    # 处理缓存相关参数
    if args.no_cache:
        Config.USE_WALKS_CACHE = False
        print("已禁用随机游走缓存")
    
    if args.force_regenerate:
        Config.FORCE_REGENERATE_WALKS = True
        print("将强制重新生成随机游走（忽略缓存）")

    # 处理属性相关参数
    if args.enable_attributes:
        Config.ENABLE_ATTRIBUTES = True
        print("已启用属性向量训练")
    
    if args.attribute_data_path:
        Config.ATTRIBUTE_DATA_PATH = args.attribute_data_path
        print(f"使用命令行指定的属性数据路径: {Config.ATTRIBUTE_DATA_PATH}")

    print(f"\nItem2Vec/Node2Vec用户表示向量训练项目")
    print(f"实验名称 (Config): {Config.EXPERIMENT_NAME}")
    # 使用 get_experiment_dir() 来获取缓存的/最终确定的路径进行显示
    print(f"实际实验目录: {get_experiment_dir(Config.EXPERIMENT_NAME)}") 
    print(f"运行模式: {args.mode}")
    print(f"设备: {Config.DEVICE_OBJ}") # 使用DEVICE_OBJ
    print(f"模型类型 (来自Config): {Config.MODEL_TYPE}")
    if Config.MODEL_TYPE == "node2vec":
        print(f"Node2Vec 缓存: {'启用' if Config.USE_WALKS_CACHE else '禁用'}")
        if Config.USE_WALKS_CACHE and Config.FORCE_REGENERATE_WALKS:
            print("缓存模式: 强制重新生成")
    print("="*50)
    
    user_sequences = None
    url_mappings = None
    user_attributes = None
    attribute_info = None
    user_location_sequences = None
    base_station_mappings = None
    location_weights = None
    model = None
    location_model = None
    attribute_model = None
    fusion_model = None
    location_processor = None
    
    # 数据预处理
    if args.mode in ['preprocess', 'all']:
        user_sequences, url_mappings, user_attributes, attribute_info, user_location_sequences, base_station_mappings, location_weights = preprocess_data(Config.DATA_PATH) # 使用Config.DATA_PATH
        if user_sequences is None:
            print("数据预处理失败，程序退出")
            return
    
    # 如果不是预处理模式，需要加载已处理的数据
    # 确保从正确的PROCESSED_DATA_PATH加载
    if args.mode not in ['preprocess'] and user_sequences is None:
        print(f"尝试从 {Config.PROCESSED_DATA_PATH} 加载已处理数据...")
        user_sequences, url_mappings, user_attributes, attribute_info, user_location_sequences, base_station_mappings, location_weights = preprocess_data() # preprocessor内部会使用Config.PROCESSED_DATA_PATH
        if user_sequences is None:
            print("无法加载数据，请确保已运行预处理或提供了正确的数据路径。程序退出")
            return
    
    # 模型训练
    if args.mode in ['train', 'all'] and not args.no_train:
        if user_sequences is None or url_mappings is None:
            print("数据未加载，无法开始训练。请先运行 preprocess 模式。")
            return
        
        if Config.MODEL_TYPE == 'item2vec':
            model, trainer = train_model(user_sequences, url_mappings, args.resume)
        elif Config.MODEL_TYPE == 'node2vec':
            model, trainer = train_node2vec_model(user_sequences, url_mappings, args.resume)
            if model is None: # 如果Node2Vec训练中途失败（例如图构建或游走生成失败）
                print("Node2Vec模型训练失败，程序退出。")
                return
        else:
            print(f"未知的模型类型 (来自Config): {Config.MODEL_TYPE}")
            return
        
        # 行为模型训练完成后，训练位置模型（如果启用）
        if Config.ENABLE_LOCATION and model is not None:
            location_processor = LocationProcessor(Config)
            
            # 加载基站特征数据
            location_processor.load_base_station_features(Config.LOCATION_FEATURES_PATH)
            
            # 训练位置模型
            location_model, base_station_mappings = train_location_model(
                Config, location_processor
            )
            
            # 计算用户位置嵌入
            if location_model is not None:
                user_base_stations = location_processor.process_user_base_stations(Config.LOCATION_DATA_PATH)
                user_location_embeddings = compute_location_embeddings(
                    location_model, user_base_stations, base_station_mappings, location_processor
                )
            else:
                user_location_embeddings = {}
        else:
            location_processor = None
            user_location_embeddings = {}
            user_base_stations = {}
        
        # 行为模型训练完成后，训练属性模型（如果启用）
        if Config.ENABLE_ATTRIBUTES and model is not None:
            # 为属性模型创建支持位置的融合模型
            behavior_dim = Config.EMBEDDING_DIM
            attribute_dim = Config.ATTRIBUTE_EMBEDDING_DIM
            location_dim = Config.LOCATION_EMBEDDING_DIM if Config.ENABLE_LOCATION else None
            
            # 创建支持多模态的融合模型
            fusion_model = UserFusionModel(
                behavior_dim, attribute_dim, location_dim, Config
            )
            
            # 使用现有的属性训练器，但传入位置信息
            attribute_trainer = AttributeTrainer(
                behavior_model=model,
                user_sequences=user_sequences,
                user_attributes=user_attributes,
                attribute_info=attribute_info,
                url_mappings=url_mappings,
                config=Config
            )
            
            # 如果有位置信息，将其传递给训练器
            if location_model is not None and user_location_embeddings:
                attribute_trainer.location_model = location_model
                attribute_trainer.user_location_sequences = user_base_stations
                attribute_trainer.base_station_mappings = base_station_mappings
                attribute_trainer.location_weights = user_location_embeddings
                attribute_trainer.location_processor = location_processor
                attribute_trainer.user_base_stations = user_base_stations
            
            # 设置多模态融合模型
            attribute_trainer.fusion_model = fusion_model
            
            # 开始训练
            attribute_trainer.train()
            
            attribute_model = attribute_trainer.attribute_model
            fusion_model = attribute_trainer.fusion_model
    
    # 加载已训练的模型 (如果需要)
    # 确保从正确的MODEL_SAVE_PATH加载
    if (args.no_train and args.mode in ['train', 'all']) or args.mode in ['visualize', 'compute_embeddings']:
        if model is None: # 避免重复加载
            model_save_subdir = Config.MODEL_SAVE_PATH # 路径现在由Config动态确定
            
            # 根据模型类型确定模型文件名
            model_filename = 'item2vec_model.pth' if Config.MODEL_TYPE == 'item2vec' else 'node2vec_model.pth'
            user_embeddings_filename = 'user_embeddings.pkl' if Config.MODEL_TYPE == 'item2vec' else 'user_embeddings_node2vec.pkl'

            model_load_path = args.model_path or os.path.join(model_save_subdir, model_filename)
            
            if url_mappings is None: # 评估、可视化或计算嵌入时可能需要先加载映射
                print("URL mappings 未加载，尝试从预处理数据中获取...")
                _, url_mappings_temp, user_attributes_temp, attribute_info_temp = preprocess_data() # 再次调用以获取映射
                if url_mappings_temp is None:
                    print("无法获取URL mappings，后续步骤可能受限。")
                    #可以选择退出或继续，取决于后续步骤是否严格需要它
                else:
                    url_mappings = url_mappings_temp
                    if user_attributes is None:
                        user_attributes = user_attributes_temp
                    if attribute_info is None:
                        attribute_info = attribute_info_temp
            
            if url_mappings is None and args.mode not in ['train']: # 对于非训练的后续步骤，词汇表大小是必须的
                 print("无法确定词汇表大小 (url_mappings is None)，无法加载模型。请确保已进行预处理。")
                 return
            vocab_size = len(url_mappings['url_to_id']) if url_mappings else 0
            if vocab_size == 0 and args.mode not in ['train']: # 再次检查
                print("词汇表大小为0，无法加载模型。")
                return
            
            # 修改: 根据模型类型加载不同的模型
            if Config.MODEL_TYPE == 'item2vec':
                model = load_trained_model(model_load_path, vocab_size) # load_trained_model 内部使用 Item2Vec
            elif Config.MODEL_TYPE == 'node2vec':
                # 需要一个类似load_trained_model的函数来加载Node2Vec，或者修改load_trained_model使其通用
                # 暂时简单处理，假设Node2Vec模型保存和Item2Vec一样
                # 注意: 实际应用中，可能需要确保加载的是正确的模型类型
                temp_node2vec_model = Node2Vec(vocab_size, Config.EMBEDDING_DIM)
                if os.path.exists(model_load_path):
                    torch.serialization.add_safe_globals([Config])
                    checkpoint = torch.load(model_load_path, map_location=Config.DEVICE_OBJ, weights_only=True)
                    temp_node2vec_model.load_state_dict(checkpoint['model_state_dict'])
                    model = temp_node2vec_model
                    print(f"已加载训练好的Node2Vec模型: {model_load_path}")
                else:
                    print(f"Node2Vec模型文件不存在: {model_load_path}")
                    model = None
            else:
                print(f"加载模型时遇到未知模型类型 (来自Config): {Config.MODEL_TYPE}")
                return

            if model is None:
                print(f"无法从 {model_load_path} 加载模型，程序退出")
                return
        
        # 加载属性模型（如果启用且存在）
        if Config.ENABLE_ATTRIBUTES and attribute_model is None and attribute_info is not None:
            attribute_model_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_attribute_models.pth')
            if os.path.exists(attribute_model_path):
                try:
                    attribute_model, fusion_model = load_attribute_models(attribute_model_path, attribute_info, Config)
                    print(f"已加载属性模型: {attribute_model_path}")
                except Exception as e:
                    print(f"加载属性模型时出错: {e}")
                    attribute_model = None
                    fusion_model = None
            else:
                print(f"属性模型文件不存在: {attribute_model_path}")
    
    # 结果可视化
    if args.mode in ['visualize', 'all']:
        if model is not None and user_sequences is not None and url_mappings is not None:
            visualize_results(model, user_sequences, url_mappings)
        else:
            print("模型或数据未准备好，跳过可视化。请确保已训练模型并加载了数据。")
    
    # 计算用户嵌入 (通常在'all'模式或者需要最终嵌入时运行)
    # 现在也为 'compute_embeddings' 模式启用
    if args.mode in ['all', 'compute_embeddings'] and model is not None and user_sequences is not None and url_mappings is not None:
        if Config.ENABLE_ATTRIBUTES and attribute_model is not None and fusion_model is not None and user_attributes is not None:
            # 计算增强用户嵌入（支持位置信息）
            enhanced_embeddings_filename = f'enhanced_user_embeddings_{Config.MODEL_TYPE}.pkl'
            enhanced_embeddings_path = os.path.join(Config.MODEL_SAVE_PATH, enhanced_embeddings_filename)
            enhanced_embeddings = compute_enhanced_user_embeddings(
                behavior_model=model,
                attribute_model=attribute_model,
                fusion_model=fusion_model,
                user_sequences=user_sequences,
                user_attributes=user_attributes,
                url_mappings=url_mappings,
                attribute_info=attribute_info,
                location_model=location_model,
                user_location_sequences=user_location_sequences,
                base_station_mappings=base_station_mappings,
                location_weights=location_weights,
                location_processor=location_processor,
                save_path=enhanced_embeddings_path
            )
        else:
            # 计算基础用户嵌入
            user_embeddings_filename = 'user_embeddings.pkl' if Config.MODEL_TYPE == 'item2vec' else 'user_embeddings_node2vec.pkl'
            user_embeddings_path = os.path.join(Config.MODEL_SAVE_PATH, user_embeddings_filename)
            compute_user_embeddings(model, user_sequences, url_mappings, user_embeddings_path)
    elif args.mode == 'compute_embeddings': # 如果是compute_embeddings模式但条件未满足，给出提示
        print("模型、用户序列或URL映射未准备好，无法计算用户嵌入。请确保：")
        print("1. 已运行数据预处理。")
        print("2. 已有训练好的模型（或通过 --model_path 指定）。")
    
    # 新用户向量计算模式
    elif args.mode == 'compute_new_users':
        print("="*50)
        print("新用户向量计算模式")
        print("="*50)
        
        # 如果模型未加载，尝试加载已训练的模型
        if model is None or url_mappings is None:
            print("尝试加载已训练的模型...")
            
            # 加载URL映射
            url_mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "url_mappings.pkl")
            if os.path.exists(url_mappings_path):
                with open(url_mappings_path, 'rb') as f:
                    url_mappings = pickle.load(f)
                print(f"URL映射加载成功，包含 {len(url_mappings['url_to_id'])} 个URL")
            else:
                print(f"错误：未找到URL映射文件 {url_mappings_path}")
                return
            
            # 加载行为模型
            behavior_model_path = os.path.join(Config.MODEL_SAVE_PATH, f"best_{Config.MODEL_TYPE}_model.pth")
            if not os.path.exists(behavior_model_path):
                behavior_model_path = os.path.join(Config.MODEL_SAVE_PATH, f"{Config.MODEL_TYPE}_model.pth")
            
            if os.path.exists(behavior_model_path):
                vocab_size = len(url_mappings['url_to_id'])
                
                if Config.MODEL_TYPE == 'item2vec':
                    model = Item2Vec(vocab_size, Config.EMBEDDING_DIM)
                else:  # node2vec
                    model = Node2Vec(vocab_size, Config.EMBEDDING_DIM)
                
                # 加载模型权重
                checkpoint = torch.load(behavior_model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                print(f"行为模型 ({Config.MODEL_TYPE}) 加载成功")
            else:
                print(f"错误：未找到行为模型文件 {behavior_model_path}")
                return
            
            # 加载属性相关模型（如果启用）
            if Config.ENABLE_ATTRIBUTES:
                attribute_info_path = os.path.join(Config.PROCESSED_DATA_PATH, "attribute_info.pkl")
                if os.path.exists(attribute_info_path):
                    with open(attribute_info_path, 'rb') as f:
                        attribute_info = pickle.load(f)
                    print("属性信息加载成功")
                    
                    # 加载属性模型
                    attribute_model_path = os.path.join(Config.MODEL_SAVE_PATH, "best_attribute_models.pth")
                    if not os.path.exists(attribute_model_path):
                        attribute_model_path = os.path.join(Config.MODEL_SAVE_PATH, "attribute_models.pth")
                    
                    if os.path.exists(attribute_model_path):
                        attribute_model, fusion_model = load_attribute_models(attribute_model_path, attribute_info)
                        print("属性模型和融合模型加载成功")
                    else:
                        print("警告：未找到属性模型文件，将跳过属性向量计算")
                else:
                    print("警告：未找到属性信息文件，将跳过属性向量计算")
            
            # 加载位置相关模型（如果启用）
            if Config.ENABLE_LOCATION:
                # 加载基站映射
                base_station_mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "base_station_mappings.pkl")
                if os.path.exists(base_station_mappings_path):
                    with open(base_station_mappings_path, 'rb') as f:
                        base_station_mappings = pickle.load(f)
                    print(f"基站映射加载成功，包含 {len(base_station_mappings['base_station_to_id'])} 个基站")
                    
                    # 加载位置模型
                    location_model_path = os.path.join(Config.MODEL_SAVE_PATH, f"location_{Config.LOCATION_MODEL_TYPE}_model.pth")
                    if os.path.exists(location_model_path):
                        vocab_size = len(base_station_mappings['base_station_to_id'])
                        
                        if Config.LOCATION_MODEL_TYPE == 'item2vec':
                            location_model = Item2Vec(vocab_size, Config.LOCATION_EMBEDDING_DIM)
                        else:  # node2vec
                            location_model = Node2Vec(vocab_size, Config.LOCATION_EMBEDDING_DIM)
                        
                        # 加载位置模型权重
                        checkpoint = torch.load(location_model_path, map_location='cpu')
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            location_model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            location_model.load_state_dict(checkpoint)
                        
                        location_model.eval()
                        print(f"位置模型 ({Config.LOCATION_MODEL_TYPE}) 加载成功")
                        
                        # 创建位置处理器
                        location_processor = LocationProcessor(Config)
                        if Config.BASE_STATION_FEATURE_MODE != "none":
                            location_processor.load_base_station_features(Config.LOCATION_FEATURES_PATH)
                    else:
                        print(f"警告：未找到位置模型文件 {location_model_path}")
                else:
                    print("警告：未找到基站映射文件，将跳过位置向量计算")
        
        # 计算新用户向量
        new_user_embeddings_filename = f'new_user_embeddings_{Config.MODEL_TYPE}.pkl'
        new_user_embeddings_path = os.path.join(Config.MODEL_SAVE_PATH, new_user_embeddings_filename)
        
        new_user_embeddings = compute_new_user_embeddings(
            behavior_model=model,
            attribute_model=attribute_model,
            fusion_model=fusion_model,
            url_mappings=url_mappings,
            attribute_info=attribute_info,
            base_station_mappings=base_station_mappings,
            location_model=location_model,
            location_processor=location_processor,
            new_user_behavior_path=args.new_user_behavior_path,
            new_user_attribute_path=args.new_user_attribute_path,
            new_user_location_path=args.new_user_location_path,
            save_path=new_user_embeddings_path
        )
        
        if new_user_embeddings:
            print(f"新用户向量计算完成！共计算了 {len(new_user_embeddings)} 个新用户的向量")
        else:
            print("未能计算任何新用户向量，请检查数据文件是否存在且格式正确")
    
    print("="*50)
    print("程序执行完成！")
    print("="*50)

if __name__ == "__main__":
    main() 