"""
模型评估模块
包括评估指标计算
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
from collections import defaultdict
import random
from config import Config

class Evaluator:
    """
    模型评估器
    """
    def __init__(self, model, user_sequences, url_mappings, config=Config):
        self.model = model
        self.user_sequences = user_sequences
        self.url_to_id = url_mappings['url_to_id']
        self.id_to_url = url_mappings['id_to_url']
        self.config = config
        
        # 获取嵌入向量
        self.item_embeddings = model.get_embeddings()
        
    def create_test_data(self, test_ratio=0.2):
        """
        创建测试数据集
        为每个用户随机选择一些访问记录作为测试集
        """
        train_sequences = {}
        test_data = []
        
        for user_id, sequence in self.user_sequences.items():
            if len(sequence) < 5:  # 序列太短，跳过
                continue
                
            # 随机选择测试项目
            test_size = max(1, int(len(sequence) * test_ratio))
            test_indices = random.sample(range(len(sequence)), test_size)
            
            train_seq = [item for i, item in enumerate(sequence) if i not in test_indices]
            test_items = [sequence[i] for i in test_indices]
            
            if len(train_seq) > 0:
                train_sequences[user_id] = train_seq
                for item in test_items:
                    test_data.append((user_id, item))
        
        return train_sequences, test_data
    
    def compute_user_embedding(self, user_sequence):
        """
        计算单个用户的嵌入向量
        """
        if not user_sequence:
            return np.zeros(self.item_embeddings.shape[1])
        
        unique_items = list(set(user_sequence))
        item_embeds = self.item_embeddings[unique_items]
        user_embed = np.mean(item_embeds, axis=0)
        
        return user_embed
    
    def recommend_items(self, user_embedding, top_k=10, exclude_items=None):
        """
        基于用户嵌入推荐物品
        """
        if exclude_items is None:
            exclude_items = set()
        
        # 计算用户嵌入与所有物品嵌入的相似度
        similarities = cosine_similarity([user_embedding], self.item_embeddings)[0]
        
        # 排序并排除已访问的物品
        item_scores = [(i, score) for i, score in enumerate(similarities) if i not in exclude_items]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k推荐
        recommendations = [item_id for item_id, score in item_scores[:top_k]]
        scores = [score for item_id, score in item_scores[:top_k]]
        
        return recommendations, scores
    
    def evaluate_recommendations(self, test_ratio=0.2):
        """
        评估推荐性能
        """
        print("开始评估推荐性能...")
        
        # 创建测试数据
        train_sequences, test_data = self.create_test_data(test_ratio)
        
        # 评估指标
        hit_counts = {k: 0 for k in [1, 5, 10, 20]}
        ndcg_scores = {k: [] for k in [1, 5, 10, 20]}
        total_users = 0
        
        for user_id, true_item in test_data:
            if user_id not in train_sequences:
                continue
            
            # 计算用户嵌入
            user_embed = self.compute_user_embedding(train_sequences[user_id])
            
            # 获取推荐列表
            exclude_items = set(train_sequences[user_id])
            recommendations, scores = self.recommend_items(
                user_embed, top_k=20, exclude_items=exclude_items
            )
            
            # 计算Hit Rate
            for k in hit_counts.keys():
                if true_item in recommendations[:k]:
                    hit_counts[k] += 1
            
            # 计算NDCG
            for k in ndcg_scores.keys():
                if true_item in recommendations[:k]:
                    rank = recommendations[:k].index(true_item) + 1
                    ndcg = 1.0 / np.log2(rank + 1)
                else:
                    ndcg = 0.0
                ndcg_scores[k].append(ndcg)
            
            total_users += 1
        
        # 计算最终指标
        results = {}
        for k in hit_counts.keys():
            hit_rate = hit_counts[k] / total_users if total_users > 0 else 0
            avg_ndcg = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0
            
            results[f'Hit@{k}'] = hit_rate
            results[f'NDCG@{k}'] = avg_ndcg
        
        return results
    
    def evaluate_item_similarity(self, sample_size=100):
        """
        评估物品相似度的质量
        """
        print("评估物品相似度...")
        
        # 随机选择一些物品进行评估
        item_ids = list(range(len(self.id_to_url)))
        sample_items = random.sample(item_ids, min(sample_size, len(item_ids)))
        
        similarity_scores = []
        
        for item_id in sample_items:
            # 获取相似物品
            similar_items, scores = self.model.get_similar_items(item_id, top_k=10)
            
            # 计算平均相似度分数
            avg_similarity = np.mean(scores) if len(scores) > 0 else 0
            similarity_scores.append(avg_similarity)
        
        results = {
            'avg_item_similarity': np.mean(similarity_scores),
            'std_item_similarity': np.std(similarity_scores),
            'min_item_similarity': np.min(similarity_scores),
            'max_item_similarity': np.max(similarity_scores)
        }
        
        return results
    
    def evaluate_user_clustering(self, sample_size=200):
        """
        评估用户聚类质量
        """
        print("评估用户聚类质量...")
        
        # 随机选择用户样本
        user_ids = list(self.user_sequences.keys())
        sample_users = random.sample(user_ids, min(sample_size, len(user_ids)))
        
        # 计算用户嵌入
        user_embeddings = {}
        for user_id in sample_users:
            user_embed = self.compute_user_embedding(self.user_sequences[user_id])
            user_embeddings[user_id] = user_embed
        
        # 计算用户间相似度分布
        similarities = []
        for i, user1 in enumerate(sample_users):
            for user2 in sample_users[i+1:]:
                embed1 = user_embeddings[user1]
                embed2 = user_embeddings[user2]
                
                similarity = cosine_similarity([embed1], [embed2])[0][0]
                similarities.append(similarity)
        
        results = {
            'avg_user_similarity': np.mean(similarities),
            'std_user_similarity': np.std(similarities),
            'min_user_similarity': np.min(similarities),
            'max_user_similarity': np.max(similarities)
        }
        
        return results
    
    def evaluate_embedding_quality(self):
        """
        评估嵌入向量质量
        """
        print("评估嵌入向量质量...")
        
        # 计算嵌入向量的统计信息
        embeddings = self.item_embeddings
        
        # 向量范数分布
        norms = np.linalg.norm(embeddings, axis=1)
        
        # 向量间平均距离
        sample_size = min(1000, len(embeddings))
        sample_indices = random.sample(range(len(embeddings)), sample_size)
        sample_embeddings = embeddings[sample_indices]
        
        distances = []
        for i in range(len(sample_embeddings)):
            for j in range(i+1, len(sample_embeddings)):
                dist = np.linalg.norm(sample_embeddings[i] - sample_embeddings[j])
                distances.append(dist)
        
        results = {
            'avg_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'embedding_dim': embeddings.shape[1],
            'vocab_size': embeddings.shape[0]
        }
        
        return results
    
    def comprehensive_evaluation(self):
        """
        综合评估
        """
        print("开始综合评估...")
        
        results = {}
        
        # 推荐性能评估
        try:
            rec_results = self.evaluate_recommendations()
            results['recommendation'] = rec_results
        except Exception as e:
            print(f"推荐评估失败: {e}")
            results['recommendation'] = {}
        
        # 物品相似度评估
        try:
            sim_results = self.evaluate_item_similarity()
            results['item_similarity'] = sim_results
        except Exception as e:
            print(f"物品相似度评估失败: {e}")
            results['item_similarity'] = {}
        
        # 用户聚类评估
        try:
            cluster_results = self.evaluate_user_clustering()
            results['user_clustering'] = cluster_results
        except Exception as e:
            print(f"用户聚类评估失败: {e}")
            results['user_clustering'] = {}
        
        # 嵌入质量评估
        try:
            embed_results = self.evaluate_embedding_quality()
            results['embedding_quality'] = embed_results
        except Exception as e:
            print(f"嵌入质量评估失败: {e}")
            results['embedding_quality'] = {}
        
        return results
    
    def print_evaluation_results(self, results):
        """
        打印评估结果
        """
        print("\n" + "="*50)
        print("模型评估结果")
        print("="*50)
        
        # 推荐性能
        if 'recommendation' in results and results['recommendation']:
            print("\n推荐性能:")
            for metric, value in results['recommendation'].items():
                print(f"  {metric}: {value:.4f}")
        
        # 物品相似度
        if 'item_similarity' in results and results['item_similarity']:
            print("\n物品相似度:")
            for metric, value in results['item_similarity'].items():
                print(f"  {metric}: {value:.4f}")
        
        # 用户聚类
        if 'user_clustering' in results and results['user_clustering']:
            print("\n用户聚类:")
            for metric, value in results['user_clustering'].items():
                print(f"  {metric}: {value:.4f}")
        
        # 嵌入质量
        if 'embedding_quality' in results and results['embedding_quality']:
            print("\n嵌入质量:")
            for metric, value in results['embedding_quality'].items():
                print(f"  {metric}: {value:.4f}")
        
        print("="*50)

if __name__ == "__main__":
    # 示例使用
    from data_preprocessing import DataPreprocessor
    from model import Item2Vec
    from trainer import Trainer
    
    # 加载数据和模型
    preprocessor = DataPreprocessor()
    user_sequences = preprocessor.load_processed_data()
    
    # 加载训练好的模型
    vocab_size = len(preprocessor.url_to_id)
    model = Item2Vec(vocab_size, Config.EMBEDDING_DIM)
    
    # 这里应该加载训练好的模型权重
    # model.load_state_dict(torch.load('models/item2vec_model.pth'))
    
    # 创建评估器并评估
    url_mappings = {
        'url_to_id': preprocessor.url_to_id,
        'id_to_url': preprocessor.id_to_url
    }
    
    evaluator = Evaluator(model, user_sequences, url_mappings)
    results = evaluator.comprehensive_evaluation()
    evaluator.print_evaluation_results(results) 