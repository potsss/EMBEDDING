"""
新用户向量计算示例脚本
演示如何为新用户计算向量表示，并与已有用户进行相似度比较
"""
import os
import pickle
import numpy as np
from compute_new_users import load_trained_models
from main import compute_new_user_embeddings
from config import Config, get_experiment_paths

def load_existing_user_embeddings(experiment_name):
    """
    加载已有用户的向量表示
    """
    # 设置实验路径
    Config.EXPERIMENT_NAME = experiment_name
    experiment_paths = get_experiment_paths(experiment_name, allow_existing_without_timestamp=True)
    for key, value in experiment_paths.items():
        setattr(Config, key, value)
    
    # 加载已有用户向量
    existing_embeddings_path = os.path.join(Config.MODEL_SAVE_PATH, f'enhanced_user_embeddings_{Config.MODEL_TYPE}.pkl')
    
    if os.path.exists(existing_embeddings_path):
        with open(existing_embeddings_path, 'rb') as f:
            existing_embeddings = pickle.load(f)
        print(f"加载了 {len(existing_embeddings)} 个已有用户的向量")
        return existing_embeddings
    else:
        print(f"未找到已有用户向量文件: {existing_embeddings_path}")
        return None

def compute_user_similarity(user_embedding, existing_embeddings, top_k=5):
    """
    计算新用户与已有用户的相似度
    """
    similarities = []
    user_ids = []
    
    for user_id, embedding in existing_embeddings.items():
        # 计算余弦相似度
        similarity = np.dot(user_embedding, embedding) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(embedding)
        )
        similarities.append(similarity)
        user_ids.append(user_id)
    
    # 排序并返回top_k
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    similar_users = [(user_ids[i], similarities[i]) for i in sorted_indices]
    
    return similar_users

def main():
    """
    主函数：演示新用户向量计算和相似度分析
    """
    print("="*60)
    print("新用户向量计算和相似度分析示例")
    print("="*60)
    
    # 配置参数
    experiment_name = "three_vector_test"  # 使用你的实验名称
    new_user_behavior_path = "data/new_user_behavior.csv"
    new_user_attribute_path = "data/new_user_attributes.tsv"
    new_user_location_path = "data/new_user_base_stations.tsv"
    
    # 检查数据文件是否存在
    if not all(os.path.exists(path) for path in [new_user_behavior_path, new_user_attribute_path, new_user_location_path]):
        print("错误：新用户数据文件不完整")
        print("请确保以下文件存在：")
        print(f"- {new_user_behavior_path}")
        print(f"- {new_user_attribute_path}")
        print(f"- {new_user_location_path}")
        return
    
    # 设置实验路径
    Config.EXPERIMENT_NAME = experiment_name
    experiment_paths = get_experiment_paths(experiment_name, allow_existing_without_timestamp=True)
    for key, value in experiment_paths.items():
        setattr(Config, key, value)
    
    # 检查实验目录
    experiment_dir = os.path.dirname(Config.PROCESSED_DATA_PATH)
    if not os.path.exists(experiment_dir):
        print(f"错误：实验目录不存在: {experiment_dir}")
        print("请先运行完整的训练流程")
        return
    
    # 1. 加载训练好的模型
    print("步骤1: 加载训练好的模型...")
    models_data = load_trained_models(experiment_dir)
    if models_data is None:
        print("模型加载失败")
        return
    
    # 2. 计算新用户向量
    print("\n步骤2: 计算新用户向量...")
    new_user_embeddings = compute_new_user_embeddings(
        behavior_model=models_data['behavior_model'],
        attribute_model=models_data['attribute_model'],
        fusion_model=models_data['fusion_model'],
        url_mappings=models_data['url_mappings'],
        attribute_info=models_data['attribute_info'],
        base_station_mappings=models_data['base_station_mappings'],
        location_model=models_data['location_model'],
        location_processor=models_data['location_processor'],
        new_user_behavior_path=new_user_behavior_path,
        new_user_attribute_path=new_user_attribute_path,
        new_user_location_path=new_user_location_path
    )
    
    if not new_user_embeddings:
        print("新用户向量计算失败")
        return
    
    # 3. 加载已有用户向量
    print("\n步骤3: 加载已有用户向量...")
    existing_embeddings = load_existing_user_embeddings(experiment_name)
    if existing_embeddings is None:
        print("无法加载已有用户向量，跳过相似度分析")
        return
    
    # 4. 相似度分析
    print("\n步骤4: 进行相似度分析...")
    print("="*40)
    
    for new_user_id, new_user_embedding in new_user_embeddings.items():
        print(f"\n新用户 {new_user_id} 的最相似用户:")
        
        # 计算与已有用户的相似度
        similar_users = compute_user_similarity(new_user_embedding, existing_embeddings, top_k=5)
        
        for i, (similar_user_id, similarity) in enumerate(similar_users, 1):
            print(f"  {i}. 用户 {similar_user_id}: 相似度 {similarity:.4f}")
    
    # 5. 保存结果
    print(f"\n步骤5: 保存新用户向量...")
    output_path = os.path.join(Config.MODEL_SAVE_PATH, f'new_user_embeddings_{Config.MODEL_TYPE}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(new_user_embeddings, f)
    print(f"新用户向量已保存到: {output_path}")
    
    # 6. 显示统计信息
    print(f"\n📊 统计信息:")
    print(f"- 新用户数量: {len(new_user_embeddings)}")
    print(f"- 已有用户数量: {len(existing_embeddings)}")
    print(f"- 向量维度: {len(list(new_user_embeddings.values())[0])}")
    
    print("\n" + "="*60)
    print("✅ 新用户向量计算和相似度分析完成！")
    print("="*60)

if __name__ == "__main__":
    main() 