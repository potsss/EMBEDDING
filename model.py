"""
Item2Vec模型定义
基于PyTorch实现的Word2Vec模型，用于训练URL的向量表示
优化版本：改进负采样和初始化策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config

class Item2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, config=Config):
        super(Item2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.config = config
        
        # 输入嵌入层（中心词）
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 输出嵌入层（上下文词）- 使用更小的初始化范围
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """
        改进的权重初始化策略
        """
        # 使用Xavier初始化，更适合深度学习
        init_range = 0.5 / self.embedding_dim
        
        # 输入嵌入使用较小的方差
        nn.init.uniform_(self.in_embeddings.weight, -init_range, init_range)
        
        # 输出嵌入初始化为零（常见的word2vec技巧）
        nn.init.constant_(self.out_embeddings.weight, 0)
    
    def forward(self, center_words, context_words, negative_words):
        """
        优化的前向传播
        Args:
            center_words: 中心词ID [batch_size]
            context_words: 上下文词ID [batch_size]
            negative_words: 负采样词ID [batch_size, negative_samples]
        """
        batch_size = center_words.size(0)
        
        # 获取中心词嵌入
        center_embeds = self.in_embeddings(center_words)  # [batch_size, embedding_dim]
        
        # 获取正样本（上下文词）嵌入
        context_embeds = self.out_embeddings(context_words)  # [batch_size, embedding_dim]
        
        # 计算正样本得分 - 使用点积
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        pos_loss = F.logsigmoid(pos_score)
        
        # 获取负样本嵌入
        neg_embeds = self.out_embeddings(negative_words)  # [batch_size, negative_samples, embedding_dim]
        
        # 计算负样本得分 - 优化的批量计算
        center_embeds_expanded = center_embeds.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        neg_score = torch.sum(neg_embeds * center_embeds_expanded, dim=2)  # [batch_size, negative_samples]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [batch_size]
        
        # 总损失
        loss = -(pos_loss + neg_loss).mean()
        
        return loss
    
    def get_embeddings(self, normalize=True):
        """
        获取训练好的嵌入向量
        Args:
            normalize: 是否进行L2归一化
        """
        embeddings = self.in_embeddings.weight.data.cpu().numpy()
        
        if normalize:
            # L2归一化，提升相似度计算效果
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            embeddings = embeddings / norms
        
        return embeddings
    
    def get_similar_items(self, item_id, top_k=10, normalize=True):
        """
        获取与指定item最相似的items
        Args:
            item_id: 目标物品ID
            top_k: 返回的相似物品数量
            normalize: 是否使用归一化的嵌入向量
        """
        embeddings = self.get_embeddings(normalize=normalize)
        item_embed = embeddings[item_id]
        
        if normalize:
            # 如果已归一化，直接使用点积（等价于余弦相似度）
            similarities = np.dot(embeddings, item_embed)
        else:
            # 计算余弦相似度
            similarities = np.dot(embeddings, item_embed) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(item_embed)
            )
        
        # 获取最相似的items（排除自身）
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        similar_scores = similarities[similar_indices]
        
        return similar_indices, similar_scores
    
    def save_embeddings(self, filepath, format='numpy'):
        """
        保存嵌入向量到文件
        Args:
            filepath: 保存路径
            format: 保存格式 ('numpy', 'txt', 'word2vec')
        """
        embeddings = self.get_embeddings(normalize=True)
        
        if format == 'numpy':
            np.save(filepath, embeddings)
        elif format == 'txt':
            np.savetxt(filepath, embeddings)
        elif format == 'word2vec':
            # 保存为word2vec格式，便于与其他工具兼容
            with open(filepath, 'w') as f:
                f.write(f"{self.vocab_size} {self.embedding_dim}\n")
                for i, embedding in enumerate(embeddings):
                    embedding_str = ' '.join(map(str, embedding))
                    f.write(f"item_{i} {embedding_str}\n")
    
    def load_pretrained_embeddings(self, embeddings, freeze=False):
        """
        加载预训练的嵌入向量
        Args:
            embeddings: 预训练的嵌入矩阵
            freeze: 是否冻结嵌入层参数
        """
        if embeddings.shape != (self.vocab_size, self.embedding_dim):
            raise ValueError(f"嵌入维度不匹配: 期望 {(self.vocab_size, self.embedding_dim)}, 得到 {embeddings.shape}")
        
        self.in_embeddings.weight.data = torch.FloatTensor(embeddings)
        
        if freeze:
            self.in_embeddings.weight.requires_grad = False

class UserEmbedding:
    """
    用户嵌入类，用于计算用户的向量表示
    优化版本：支持多种聚合策略
    """
    def __init__(self, model, user_sequences, url_mappings):
        self.model = model
        self.user_sequences = user_sequences
        self.url_to_id = url_mappings['url_to_id']
        self.id_to_url = url_mappings['id_to_url']
        self.user_embeddings = {}
    
    def compute_user_embeddings(self, weight_dict=None, aggregation='mean'):
        """
        计算所有用户的嵌入向量
        Args:
            weight_dict: 用户对各个URL的权重字典 {user_id: {url: weight}}
            aggregation: 聚合策略 ('mean', 'weighted_mean', 'max', 'attention')
        """
        item_embeddings = self.model.get_embeddings(normalize=True)
        
        for user_id, sequence in self.user_sequences.items():
            user_embed = self._compute_single_user_embedding(
                sequence, item_embeddings, weight_dict, user_id, aggregation
            )
            self.user_embeddings[user_id] = user_embed
        
        return self.user_embeddings
    
    def _compute_single_user_embedding(self, sequence, item_embeddings, weight_dict, user_id, aggregation):
        """
        计算单个用户的嵌入向量
        """
        unique_items = list(set(sequence))
        item_embeds = item_embeddings[unique_items]
        
        if aggregation == 'mean':
            # 简单平均
            user_embed = np.mean(item_embeds, axis=0)
            
        elif aggregation == 'weighted_mean' and weight_dict and user_id in weight_dict:
            # 加权平均
            user_embed = np.zeros(item_embeddings.shape[1])
            total_weight = 0
            
            for item_id in unique_items:
                url = self.id_to_url[item_id]
                weight = weight_dict[user_id].get(url, 1.0)
                user_embed += item_embeddings[item_id] * weight
                total_weight += weight
            
            if total_weight > 0:
                user_embed /= total_weight
            else:
                user_embed = np.mean(item_embeds, axis=0)
                
        elif aggregation == 'max':
            # 最大池化
            user_embed = np.max(item_embeds, axis=0)
            
        elif aggregation == 'attention':
            # 简单的注意力机制（基于频次）
            item_counts = {item: sequence.count(item) for item in unique_items}
            total_count = sum(item_counts.values())
            
            user_embed = np.zeros(item_embeddings.shape[1])
            for item_id in unique_items:
                attention_weight = item_counts[item_id] / total_count
                user_embed += item_embeddings[item_id] * attention_weight
        
        else:
            # 默认使用简单平均
            user_embed = np.mean(item_embeds, axis=0)
        
        # L2归一化
        norm = np.linalg.norm(user_embed)
        if norm > 0:
            user_embed = user_embed / norm
        
        return user_embed
    
    def get_user_similarity(self, user1_id, user2_id):
        """
        计算两个用户之间的相似度
        """
        if user1_id not in self.user_embeddings or user2_id not in self.user_embeddings:
            return 0.0
        
        embed1 = self.user_embeddings[user1_id]
        embed2 = self.user_embeddings[user2_id]
        
        # 余弦相似度（由于已归一化，直接点积即可）
        similarity = np.dot(embed1, embed2)
        return similarity
    
    def get_similar_users(self, user_id, top_k=10):
        """
        获取与指定用户最相似的用户
        """
        if user_id not in self.user_embeddings:
            return [], []
        
        target_embed = self.user_embeddings[user_id]
        similarities = []
        user_ids = []
        
        for uid, embed in self.user_embeddings.items():
            if uid != user_id:
                # 由于已归一化，直接点积即可
                similarity = np.dot(target_embed, embed)
                similarities.append(similarity)
                user_ids.append(uid)
        
        # 排序并返回top_k
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        similar_users = [user_ids[i] for i in sorted_indices]
        similar_scores = [similarities[i] for i in sorted_indices]
        
        return similar_users, similar_scores 