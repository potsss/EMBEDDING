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
from data_preprocessing import DataPreprocessor
from model import Item2Vec, UserEmbedding
from trainer import Trainer
from evaluator import Evaluator
from visualizer import Visualizer

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
    user_sequences: 用户访问序列
    url_mappings: 域名到ID的映射
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
            return None, None
    
    url_mappings = {
        'url_to_id': preprocessor.url_to_id,
        'id_to_url': preprocessor.id_to_url
    }
    
    print(f"数据预处理完成:")
    print(f"  用户数量: {len(user_sequences)}")
    print(f"  物品数量: {len(url_mappings['url_to_id'])}")
    
    return user_sequences, url_mappings

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

def evaluate_model(model, user_sequences, url_mappings):
    """
    评估模型
    """
    print("="*50)
    print("开始模型评估")
    print("="*50)
    
    # 创建评估器
    evaluator = Evaluator(model, user_sequences, url_mappings)
    
    # 综合评估
    results = evaluator.comprehensive_evaluation()
    
    # 打印结果
    evaluator.print_evaluation_results(results)
    
    return results

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

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Item2Vec用户表示向量训练项目')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'evaluate', 'visualize', 'compute_embeddings', 'all'], 
                       default='all', help='运行模式 (preprocess, train, evaluate, visualize, compute_embeddings, all)')
    parser.add_argument('--data_path', type=str, help='原始数据路径 (例如: data/user_behavior.csv)')
    parser.add_argument('--model_path', type=str, help='已训练模型的路径 (用于evaluate/visualize模式)')
    parser.add_argument('--resume', action='store_true', help='从最新的检查点恢复训练')
    parser.add_argument('--no_train', action='store_true', help='跳过训练，直接使用已有模型 (与evaluate/visualize模式结合)')
    parser.add_argument('--experiment_name', type=str, help='自定义实验名称 (默认为config.py中的EXPERIMENT_NAME)')
    
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

    print(f"\nItem2Vec用户表示向量训练项目")
    print(f"实验名称 (Config): {Config.EXPERIMENT_NAME}")
    # 使用 get_experiment_dir() 来获取缓存的/最终确定的路径进行显示
    print(f"实际实验目录: {get_experiment_dir(Config.EXPERIMENT_NAME)}") 
    print(f"运行模式: {args.mode}")
    print(f"设备: {Config.DEVICE_OBJ}") # 使用DEVICE_OBJ
    print("="*50)
    
    user_sequences = None
    url_mappings = None
    model = None
    
    # 数据预处理
    if args.mode in ['preprocess', 'all']:
        user_sequences, url_mappings = preprocess_data(Config.DATA_PATH) # 使用Config.DATA_PATH
        if user_sequences is None:
            print("数据预处理失败，程序退出")
            return
    
    # 如果不是预处理模式，需要加载已处理的数据
    # 确保从正确的PROCESSED_DATA_PATH加载
    if args.mode not in ['preprocess'] and user_sequences is None:
        print(f"尝试从 {Config.PROCESSED_DATA_PATH} 加载已处理数据...")
        user_sequences, url_mappings = preprocess_data() # preprocessor内部会使用Config.PROCESSED_DATA_PATH
        if user_sequences is None:
            print("无法加载数据，请确保已运行预处理或提供了正确的数据路径。程序退出")
            return
    
    # 模型训练
    if args.mode in ['train', 'all'] and not args.no_train:
        if user_sequences is None or url_mappings is None:
            print("数据未加载，无法开始训练。请先运行 preprocess 模式。")
            return
        model, trainer = train_model(user_sequences, url_mappings, args.resume)
    
    # 加载已训练的模型 (如果需要)
    # 确保从正确的MODEL_SAVE_PATH加载
    if (args.no_train and args.mode in ['train', 'all']) or args.mode in ['evaluate', 'visualize', 'compute_embeddings']:
        if model is None: # 避免重复加载
            model_load_path = args.model_path or os.path.join(Config.MODEL_SAVE_PATH, 'item2vec_model.pth')
            if url_mappings is None: # 评估、可视化或计算嵌入时可能需要先加载映射
                print("URL mappings 未加载，尝试从预处理数据中获取...")
                _, url_mappings_temp = preprocess_data() # 再次调用以获取映射
                if url_mappings_temp is None:
                    print("无法获取URL mappings，后续步骤可能受限。")
                    #可以选择退出或继续，取决于后续步骤是否严格需要它
                else:
                    url_mappings = url_mappings_temp
            
            if url_mappings is None and args.mode not in ['train']: # 对于非训练的后续步骤，词汇表大小是必须的
                 print("无法确定词汇表大小 (url_mappings is None)，无法加载模型。请确保已进行预处理。")
                 return
            vocab_size = len(url_mappings['url_to_id']) if url_mappings else 0
            if vocab_size == 0 and args.mode not in ['train']: # 再次检查
                print("词汇表大小为0，无法加载模型。")
                return
                
            model = load_trained_model(model_load_path, vocab_size)
            if model is None:
                print(f"无法从 {model_load_path} 加载模型，程序退出")
                return
    
    # 模型评估
    if args.mode in ['evaluate', 'all']:
        if model is not None and user_sequences is not None and url_mappings is not None:
            results = evaluate_model(model, user_sequences, url_mappings)
        else:
            print("模型或数据未准备好，跳过评估。请确保已训练模型并加载了数据。")
    
    # 结果可视化
    if args.mode in ['visualize', 'all']:
        if model is not None and user_sequences is not None and url_mappings is not None:
            visualize_results(model, user_sequences, url_mappings)
        else:
            print("模型或数据未准备好，跳过可视化。请确保已训练模型并加载了数据。")
    
    # 计算用户嵌入 (通常在'all'模式或者需要最终嵌入时运行)
    # 现在也为 'compute_embeddings' 模式启用
    if args.mode in ['all', 'compute_embeddings'] and model is not None and user_sequences is not None and url_mappings is not None:
        user_embeddings_path = os.path.join(Config.MODEL_SAVE_PATH, 'user_embeddings.pkl')
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