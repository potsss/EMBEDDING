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
from datetime import datetime
from config import Config, get_experiment_dir, get_experiment_paths
from data_preprocessing import DataPreprocessor, LocationProcessor
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

def initialize_experiment_paths(experiment_name_override=None, mode=None):
    """
    初始化实验相关的路径，并设置到Config类上。
    根据运行模式决定是否允许使用已存在的不带时间戳的目录。
    """
    current_experiment_name = experiment_name_override if experiment_name_override else Config.EXPERIMENT_NAME
    
    force_recalc = True # 默认为True，确保在新的main.py执行开始时重新计算
    allow_existing = False

    if experiment_name_override:
        # 如果通过命令行指定了实验名称，则强制重新计算，并且不查找已存在的非时间戳目录（除非新名称的目录已存在，此时会加时间戳）
        Config.EXPERIMENT_NAME = experiment_name_override
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

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Item2Vec/Node2Vec用户表示向量训练项目')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'visualize', 'compute_embeddings', 'all'], 
                       default='all', help='运行模式 (preprocess, train, visualize, compute_embeddings, all)')
    parser.add_argument('--data_path', type=str, help='原始数据路径 (例如: data/user_behavior.csv)')
    parser.add_argument('--model_path', type=str, help='已训练模型的路径 (用于visualize/compute_embeddings模式)')
    parser.add_argument('--resume', action='store_true', help='从最新的检查点恢复训练')
    parser.add_argument('--no_train', action='store_true', help='跳过训练，直接使用已有模型 (与visualize/compute_embeddings模式结合)')
    parser.add_argument('--experiment_name', type=str, help='自定义实验名称 (默认为config.py中的EXPERIMENT_NAME)')
    parser.add_argument('--no_cache', action='store_true', help='禁用随机游走缓存')
    parser.add_argument('--force_regenerate', action='store_true', help='强制重新生成随机游走（忽略缓存）')
    parser.add_argument('--enable_attributes', action='store_true', help='启用属性向量训练')
    parser.add_argument('--attribute_data_path', type=str, help='用户属性数据文件路径')
    
    args = parser.parse_args()
    
    # 1. 初始化实验路径 (这是关键改动)
    initialize_experiment_paths(experiment_name_override=args.experiment_name, mode=args.mode)
    
    # 2. 设置随机种子
    set_random_seed(Config.RANDOM_SEED)
    
    # 3. 创建目录 (现在它会使用initialize_experiment_paths确定的路径)
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
            from data_preprocessing import LocationProcessor
            
            # 创建位置处理器
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
    
    print("="*50)
    print("程序执行完成！")
    print("="*50)

if __name__ == "__main__":
    main() 