import pickle
import os

cache_dir = 'experiments/node2vec/processed_data'
cache_file = 'node2vec_walks_8d1da4de0ef21fa1cead39d96f4427af.pkl'
full_path = os.path.join(cache_dir, cache_file)

if os.path.exists(full_path):
    with open(full_path, 'rb') as f:
        walks = pickle.load(f)
    
    print(f"游走数量: {len(walks)}")
    print(f"总tokens: {sum(len(walk) for walk in walks)}")
    print(f"平均长度: {sum(len(walk) for walk in walks)/len(walks):.2f}")
    
    # 检查前几个游走
    print(f"前5个游走长度: {[len(walk) for walk in walks[:5]]}")
    
    # 计算训练样本数量估算
    total_tokens = sum(len(walk) for walk in walks)
    window_size = 5
    estimated_samples = total_tokens * window_size * 2  # 粗略估算
    print(f"预估训练样本数量: {estimated_samples:,}")
else:
    print(f"文件不存在: {full_path}") 