"""
新用户向量计算脚本
用于为未参与训练的新用户计算向量表示
"""
import os
import argparse
import torch
import pickle
import numpy as np
from config import Config, get_experiment_paths
from data_preprocessing import DataPreprocessor, LocationProcessor
from model import Item2Vec, Node2Vec, AttributeEmbeddingModel, UserFusionModel
from main import compute_new_user_embeddings, load_new_user_data
from trainer import load_attribute_models
import pandas as pd

def load_trained_models(experiment_path):
    """
    加载训练好的模型和相关数据
    
    Args:
        experiment_path: 实验路径
        
    Returns:
        包含所有模型和映射的字典
    """
    print("加载训练好的模型和映射数据...")
    
    result = {
        'behavior_model': None,
        'attribute_model': None,
        'fusion_model': None,
        'location_model': None,
        'url_mappings': None,
        'attribute_info': None,
        'base_station_mappings': None,
        'location_processor': None
    }
    
    try:
        # 1. 加载行为模型
        model_save_path = os.path.join(experiment_path, "models")
        
        # 加载URL映射
        url_mappings_path = os.path.join(experiment_path, "processed_data", "url_mappings.pkl")
        if os.path.exists(url_mappings_path):
            with open(url_mappings_path, 'rb') as f:
                result['url_mappings'] = pickle.load(f)
            print(f"URL映射加载成功，包含 {len(result['url_mappings']['url_to_id'])} 个URL")
        else:
            print(f"警告：未找到URL映射文件 {url_mappings_path}")
            return None
        
        # 加载行为模型
        behavior_model_path = os.path.join(model_save_path, f"best_{Config.MODEL_TYPE}_model.pth")
        # 如果best模型不存在，尝试加载普通模型
        if not os.path.exists(behavior_model_path):
            behavior_model_path = os.path.join(model_save_path, f"{Config.MODEL_TYPE}_model.pth")
        if os.path.exists(behavior_model_path):
            vocab_size = len(result['url_mappings']['url_to_id'])
            
            if Config.MODEL_TYPE == 'item2vec':
                model = Item2Vec(vocab_size, Config.EMBEDDING_DIM)
            else:  # node2vec
                model = Node2Vec(vocab_size, Config.EMBEDDING_DIM)
            
            # 加载模型，处理不同的保存格式
            checkpoint = torch.load(behavior_model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            result['behavior_model'] = model
            print(f"行为模型 ({Config.MODEL_TYPE}) 加载成功")
        else:
            print(f"警告：未找到行为模型文件 {behavior_model_path}")
            return None
        
        # 2. 加载属性相关模型（如果启用）
        if Config.ENABLE_ATTRIBUTES:
            # 加载属性信息
            attribute_info_path = os.path.join(experiment_path, "processed_data", "attribute_info.pkl")
            if os.path.exists(attribute_info_path):
                with open(attribute_info_path, 'rb') as f:
                    result['attribute_info'] = pickle.load(f)
                print("属性信息加载成功")
                
                # 加载属性模型
                attribute_models = load_attribute_models(model_save_path, result['attribute_info'])
                if attribute_models:
                    result['attribute_model'] = attribute_models['attribute_model']
                    result['fusion_model'] = attribute_models['fusion_model']
                    print("属性模型和融合模型加载成功")
            else:
                print("警告：未找到属性信息文件，将跳过属性向量计算")
        
        # 3. 加载位置相关模型（如果启用）
        if Config.ENABLE_LOCATION:
            # 加载基站映射
            base_station_mappings_path = os.path.join(experiment_path, "processed_data", "base_station_mappings.pkl")
            if os.path.exists(base_station_mappings_path):
                with open(base_station_mappings_path, 'rb') as f:
                    result['base_station_mappings'] = pickle.load(f)
                print(f"基站映射加载成功，包含 {len(result['base_station_mappings']['base_station_to_id'])} 个基站")
                
                # 加载位置模型
                location_model_path = os.path.join(model_save_path, f"best_location_{Config.LOCATION_MODEL_TYPE}_model.pth")
                # 如果best位置模型不存在，尝试加载普通位置模型
                if not os.path.exists(location_model_path):
                    location_model_path = os.path.join(model_save_path, f"location_{Config.LOCATION_MODEL_TYPE}_model.pth")
                # 如果还是不存在，尝试直接使用模型类型名称
                if not os.path.exists(location_model_path):
                    location_model_path = os.path.join(model_save_path, f"{Config.LOCATION_MODEL_TYPE}_model.pth")
                if os.path.exists(location_model_path):
                    vocab_size = len(result['base_station_mappings']['base_station_to_id'])
                    
                    if Config.LOCATION_MODEL_TYPE == 'item2vec':
                        location_model = Item2Vec(vocab_size, Config.LOCATION_EMBEDDING_DIM)
                    else:  # node2vec
                        location_model = Node2Vec(vocab_size, Config.LOCATION_EMBEDDING_DIM)
                    
                    # 加载位置模型，处理不同的保存格式
                    checkpoint = torch.load(location_model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        location_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        location_model.load_state_dict(checkpoint)
                    
                    location_model.eval()
                    result['location_model'] = location_model
                    print(f"位置模型 ({Config.LOCATION_MODEL_TYPE}) 加载成功")
                
                # 创建位置处理器
                result['location_processor'] = LocationProcessor(Config)
                if Config.BASE_STATION_FEATURE_MODE != "none":
                    # 如果使用基站特征，需要加载特征数据
                    result['location_processor'].load_base_station_features(Config.LOCATION_FEATURES_PATH)
            else:
                print("警告：未找到基站映射文件，将跳过位置向量计算")
                # 关闭位置功能
                Config.ENABLE_LOCATION = False
        
        return result
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='新用户向量计算工具')
    parser.add_argument('--experiment_name', type=str, required=True, 
                       help='实验名称（与训练时使用的实验名称一致）')
    parser.add_argument('--new_user_behavior_path', type=str, required=True,
                       help='新用户行为数据文件路径')
    parser.add_argument('--new_user_attribute_path', type=str,
                       help='新用户属性数据文件路径（可选）')
    parser.add_argument('--new_user_location_path', type=str,
                       help='新用户位置数据文件路径（可选）')
    parser.add_argument('--output_path', type=str,
                       help='输出文件路径（默认保存到实验目录下）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("新用户向量计算工具")
    print("="*60)
    
    # 设置实验路径
    Config.EXPERIMENT_NAME = args.experiment_name
    experiment_paths = get_experiment_paths(args.experiment_name, allow_existing_without_timestamp=True)
    for key, value in experiment_paths.items():
        setattr(Config, key, value)
    
    # 初始化设备对象
    Config.DEVICE_OBJ = torch.device(Config.DEVICE)
    
    # 检查实验目录是否存在
    experiment_dir = os.path.dirname(Config.PROCESSED_DATA_PATH)
    if not os.path.exists(experiment_dir):
        print(f"错误：实验目录不存在: {experiment_dir}")
        print("请确保已经运行过训练流程，并使用正确的实验名称")
        return
    
    # 检查新用户数据文件
    if not os.path.exists(args.new_user_behavior_path):
        print(f"错误：新用户行为数据文件不存在: {args.new_user_behavior_path}")
        return
    
    # 加载训练好的模型
    models_data = load_trained_models(experiment_dir)
    if models_data is None:
        print("模型加载失败，程序退出")
        return
    
    # 设置输出路径
    if args.output_path:
        output_path = args.output_path
    else:
        output_filename = f'new_user_embeddings_{Config.MODEL_TYPE}.pkl'
        output_path = os.path.join(Config.MODEL_SAVE_PATH, output_filename)
    
    # 计算新用户向量
    print("\n开始计算新用户向量...")
    new_user_embeddings = compute_new_user_embeddings(
        behavior_model=models_data['behavior_model'],
        attribute_model=models_data['attribute_model'],
        fusion_model=models_data['fusion_model'],
        url_mappings=models_data['url_mappings'],
        attribute_info=models_data['attribute_info'],
        base_station_mappings=models_data['base_station_mappings'],
        location_model=models_data['location_model'],
        location_processor=models_data['location_processor'],
        new_user_behavior_path=args.new_user_behavior_path,
        new_user_attribute_path=args.new_user_attribute_path,
        new_user_location_path=args.new_user_location_path,
        save_path=output_path
    )
    
    if new_user_embeddings:
        print(f"\n✅ 成功计算了 {len(new_user_embeddings)} 个新用户的向量表示")
        print(f"📁 结果已保存到: {output_path}")
        
        # 显示一些统计信息
        embedding_dim = len(list(new_user_embeddings.values())[0])
        print(f"📊 向量维度: {embedding_dim}")
        print(f"👥 新用户ID列表: {list(new_user_embeddings.keys())[:10]}{'...' if len(new_user_embeddings) > 10 else ''}")
        
    else:
        print("❌ 未能计算任何新用户向量")
        print("请检查：")
        print("1. 新用户数据文件格式是否正确")
        print("2. 新用户访问的URL是否在训练数据中出现过")
        print("3. 新用户属性值是否在训练数据的取值范围内")
    
    print("\n" + "="*60)
    print("程序执行完成！")
    print("="*60)

if __name__ == "__main__":
    main() 