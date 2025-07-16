"""
Item2Vec模型定义
基于PyTorch实现的Word2Vec模型，用于训练URL的向量表示
优化版本：改进负采样和初始化策略
包含用户属性嵌入和融合模型
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

class Node2Vec(nn.Module):
    """
    Node2Vec模型，与Item2Vec非常相似，因为它也使用Skip-gram架构。
    主要区别在于它在图上生成的随机游走序列上进行训练，而不是直接在用户行为序列上。
    """
    def __init__(self, vocab_size, embedding_dim, config=Config):
        super(Node2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.config = config
        
        # 输入嵌入层（中心词）
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 输出嵌入层（上下文词）
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """
        权重初始化策略
        """
        init_range = 0.5 / self.embedding_dim
        nn.init.uniform_(self.in_embeddings.weight, -init_range, init_range)
        nn.init.constant_(self.out_embeddings.weight, 0)
    
    def forward(self, center_words, context_words, negative_words):
        """
        前向传播
        Args:
            center_words: 中心词ID [batch_size]
            context_words: 上下文词ID [batch_size]
            negative_words: 负采样词ID [batch_size, negative_samples]
        """
        center_embeds = self.in_embeddings(center_words)  # [batch_size, embedding_dim]
        context_embeds = self.out_embeddings(context_words)  # [batch_size, embedding_dim]
        
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        pos_loss = F.logsigmoid(pos_score)
        
        neg_embeds = self.out_embeddings(negative_words)  # [batch_size, negative_samples, embedding_dim]
        center_embeds_expanded = center_embeds.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        neg_score = torch.sum(neg_embeds * center_embeds_expanded, dim=2)  # [batch_size, negative_samples]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [batch_size]
        
        loss = -(pos_loss + neg_loss).mean()
        return loss
    
    def get_embeddings(self, normalize=True):
        """
        获取训练好的节点嵌入向量
        """
        embeddings = self.in_embeddings.weight.data.cpu().numpy()
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
        return embeddings

    def get_similar_items(self, item_id, top_k=10, normalize=True):
        """
        获取与指定item最相似的items (与Item2Vec中的方法相同)
        """
        embeddings = self.get_embeddings(normalize=normalize)
        item_embed = embeddings[item_id]
        
        if normalize:
            similarities = np.dot(embeddings, item_embed)
        else:
            similarities = np.dot(embeddings, item_embed) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(item_embed)
            )
        
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        similar_scores = similarities[similar_indices]
        return similar_indices, similar_scores

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

class AttributeEmbeddingModel(nn.Module):
    """
    用户属性嵌入模型
    处理类别型和数值型属性，生成属性表示向量
    """
    def __init__(self, attribute_info, config=Config):
        super(AttributeEmbeddingModel, self).__init__()
        self.attribute_info = attribute_info
        self.config = config
        
        # 为每个类别型属性创建嵌入层
        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_attrs = []
        self.numerical_attrs = []
        
        for attr_name, attr_info in attribute_info.items():
            if attr_info['type'] == 'categorical':
                vocab_size = attr_info['vocab_size']
                embedding_layer = nn.Embedding(vocab_size, config.ATTRIBUTE_EMBEDDING_DIM)
                self.categorical_embeddings[attr_name] = embedding_layer
                self.categorical_attrs.append(attr_name)
            else:
                self.numerical_attrs.append(attr_name)
        
        # 计算输出维度
        self.categorical_output_dim = len(self.categorical_attrs) * config.ATTRIBUTE_EMBEDDING_DIM
        self.numerical_output_dim = len(self.numerical_attrs)
        self.total_output_dim = self.categorical_output_dim + self.numerical_output_dim
        
        # 属性融合层（可选）
        if self.total_output_dim > 0:
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.total_output_dim, config.FUSION_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.FUSION_HIDDEN_DIM, config.ATTRIBUTE_EMBEDDING_DIM)
            )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for embedding in self.categorical_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        
        if hasattr(self, 'fusion_layer'):
            for layer in self.fusion_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, categorical_inputs, numerical_inputs):
        """
        前向传播
        Args:
            categorical_inputs: dict {attr_name: tensor}
            numerical_inputs: tensor [batch_size, num_numerical_attrs]
        """
        embeddings = []
        
        # 处理类别型属性
        for attr_name in self.categorical_attrs:
            if attr_name in categorical_inputs:
                attr_input = categorical_inputs[attr_name]
                attr_embedding = self.categorical_embeddings[attr_name](attr_input)
                embeddings.append(attr_embedding)
        
        # 处理数值型属性
        if len(self.numerical_attrs) > 0 and numerical_inputs is not None:
            embeddings.append(numerical_inputs)
        
        if not embeddings:
            return None
        
        # 拼接所有属性嵌入
        combined_embedding = torch.cat(embeddings, dim=-1)
        
        # 通过融合层
        if hasattr(self, 'fusion_layer'):
            return self.fusion_layer(combined_embedding)
        else:
            return combined_embedding

class UserLocationEmbedding(nn.Module):
    """用户位置嵌入计算模块"""
    
    def __init__(self, config, base_station_embeddings, location_processor=None):
        super().__init__()
        self.config = config
        self.base_station_embeddings = base_station_embeddings
        self.location_processor = location_processor
        self.embedding_dim = config.LOCATION_EMBEDDING_DIM
        
        # 根据基站特征模式决定是否需要特征融合层
        if config.BASE_STATION_FEATURE_MODE == "text_embedding" and location_processor:
            # 创建特征融合层
            feature_dim = location_processor.get_feature_dimension()
            if feature_dim > 0:
                self.feature_fusion = nn.Linear(
                    self.embedding_dim + feature_dim, 
                    self.embedding_dim
                )
                self.use_features = True
            else:
                self.use_features = False
        else:
            self.use_features = False
        
        # 位置嵌入聚合层
        self.aggregation_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    def forward(self, user_base_stations, user_weights):
        """
        计算用户位置嵌入
        
        Args:
            user_base_stations: 用户连接的基站ID列表
            user_weights: 对应的权重列表
            
        Returns:
            用户位置嵌入向量
        """
        if not user_base_stations:
            # 如果用户没有位置数据，返回零向量
            return torch.zeros(self.embedding_dim)
        
        # 获取基站嵌入
        base_station_embeddings = []
        weights = []
        
        for bs_id, weight in zip(user_base_stations, user_weights):
            if bs_id in self.base_station_embeddings:
                base_embedding = self.base_station_embeddings[bs_id]
                
                # 如果启用了特征融合
                if self.use_features and self.location_processor:
                    feature = self.location_processor.get_base_station_feature(bs_id)
                    if feature is not None:
                        # 拼接基站嵌入和特征
                        feature_tensor = torch.tensor(feature, dtype=torch.float32)
                        combined = torch.cat([base_embedding, feature_tensor], dim=0)
                        base_embedding = self.feature_fusion(combined)
                
                base_station_embeddings.append(base_embedding)
                weights.append(weight)
        
        if not base_station_embeddings:
            return torch.zeros(self.embedding_dim)
        
        # 堆叠嵌入
        embeddings = torch.stack(base_station_embeddings)
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # 权重归一化
        weights = weights / weights.sum()
        
        # 加权平均
        weighted_embedding = torch.sum(embeddings * weights.unsqueeze(1), dim=0)
        
        # 通过聚合层
        return self.aggregation_layer(weighted_embedding)

class UserFusionModel(nn.Module):
    """
    用户多模态融合模型
    支持行为向量、属性向量和位置向量的融合
    """
    def __init__(self, behavior_dim, attribute_dim=None, location_dim=None, config=Config):
        super(UserFusionModel, self).__init__()
        self.behavior_dim = behavior_dim
        self.attribute_dim = attribute_dim
        self.location_dim = location_dim
        self.config = config
        
        # 计算输入维度
        input_dim = behavior_dim
        if attribute_dim is not None:
            input_dim += attribute_dim
        if location_dim is not None:
            input_dim += location_dim
        
        # 融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(input_dim, config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.FUSION_HIDDEN_DIM, config.FUSION_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.FUSION_HIDDEN_DIM // 2, config.FINAL_USER_EMBEDDING_DIM)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for layer in self.fusion_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, behavior_embeddings, attribute_embeddings=None, location_embeddings=None):
        """
        融合多模态嵌入
        Args:
            behavior_embeddings: [batch_size, behavior_dim]
            attribute_embeddings: [batch_size, attribute_dim] (可选)
            location_embeddings: [batch_size, location_dim] (可选)
        """
        # 拼接所有可用的嵌入
        embeddings_to_fuse = [behavior_embeddings]
        
        if attribute_embeddings is not None:
            embeddings_to_fuse.append(attribute_embeddings)
        if location_embeddings is not None:
            embeddings_to_fuse.append(location_embeddings)
        
        combined = torch.cat(embeddings_to_fuse, dim=-1)
        
        # 通过融合网络
        fused_embedding = self.fusion_layers(combined)
        
        # L2归一化
        fused_embedding = F.normalize(fused_embedding, p=2, dim=-1)
        
        return fused_embedding

class MaskedAttributePredictionModel(nn.Module):
    """
    掩码属性预测模型
    用于训练属性嵌入的自监督任务
    支持多模态融合（行为+属性+位置）
    """
    def __init__(self, attribute_embedding_model, user_fusion_model, attribute_info, config=Config):
        super(MaskedAttributePredictionModel, self).__init__()
        self.attribute_embedding_model = attribute_embedding_model
        self.user_fusion_model = user_fusion_model
        self.attribute_info = attribute_info
        self.config = config
        
        # 为每个属性创建预测头
        self.prediction_heads = nn.ModuleDict()
        
        for attr_name, attr_info in attribute_info.items():
            if attr_info['type'] == 'categorical':
                # 分类预测头
                self.prediction_heads[attr_name] = nn.Linear(
                    config.FINAL_USER_EMBEDDING_DIM, 
                    attr_info['vocab_size']
                )
            else:
                # 回归预测头
                self.prediction_heads[attr_name] = nn.Linear(
                    config.FINAL_USER_EMBEDDING_DIM, 
                    1
                )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for head in self.prediction_heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0)
    
    def forward(self, behavior_embeddings, categorical_inputs, numerical_inputs, masked_attrs, location_embeddings=None):
        """
        前向传播
        Args:
            behavior_embeddings: 用户行为嵌入
            categorical_inputs: 类别属性输入（可能包含掩码）
            numerical_inputs: 数值属性输入（可能包含掩码）
            masked_attrs: 被掩码的属性列表
            location_embeddings: 用户位置嵌入（可选）
        """
        # 获取属性嵌入（使用掩码后的输入）
        attribute_embeddings = self.attribute_embedding_model(categorical_inputs, numerical_inputs)
        
        # 融合行为、属性和位置嵌入
        fused_embeddings = self.user_fusion_model(
            behavior_embeddings, 
            attribute_embeddings, 
            location_embeddings
        )
        
        # 预测被掩码的属性
        predictions = {}
        for attr_name in masked_attrs:
            if attr_name in self.prediction_heads:
                predictions[attr_name] = self.prediction_heads[attr_name](fused_embeddings)
        
        return predictions

class EnhancedUserEmbedding:
    """
    增强的用户嵌入类
    结合行为向量、属性向量和位置向量
    """
    def __init__(self, behavior_model, attribute_model, fusion_model, user_sequences, 
                 user_attributes, url_mappings, attribute_info, location_model=None, 
                 user_location_sequences=None, base_station_mappings=None, location_weights=None,
                 location_processor=None):
        self.behavior_model = behavior_model
        self.attribute_model = attribute_model
        self.fusion_model = fusion_model
        self.user_sequences = user_sequences
        self.user_attributes = user_attributes
        self.url_mappings = url_mappings
        self.attribute_info = attribute_info
        self.location_model = location_model
        self.user_location_sequences = user_location_sequences
        self.base_station_mappings = base_station_mappings
        self.location_weights = location_weights
        self.location_processor = location_processor
        self.enhanced_user_embeddings = {}
    
    def compute_enhanced_user_embeddings(self):
        """
        计算增强的用户嵌入（行为+属性+位置）
        """
        # 获取行为嵌入
        basic_user_embedding = UserEmbedding(
            self.behavior_model, self.user_sequences, self.url_mappings
        )
        behavior_embeddings = basic_user_embedding.compute_user_embeddings()
        
        # 获取位置嵌入（如果可用）
        location_embeddings = {}
        if self.location_model and self.user_location_sequences:
            # 获取基站嵌入
            base_station_embeddings = {}
            for bs_id, idx in self.base_station_mappings['base_station_to_id'].items():
                if hasattr(self.location_model, 'in_embeddings'):
                    embedding = self.location_model.in_embeddings.weight[idx].detach()
                    base_station_embeddings[bs_id] = embedding
                elif hasattr(self.location_model, 'embeddings'):
                    embedding = self.location_model.embeddings.weight[idx].detach()
                    base_station_embeddings[bs_id] = embedding
            
            # 创建位置嵌入计算器
            from config import Config
            location_user_embedding = UserLocationEmbedding(
                Config, base_station_embeddings, self.location_processor
            )
            
            # 计算位置嵌入
            location_embeddings = {}
            for user_id, data in self.location_weights.items():
                if user_id in self.user_location_sequences:
                    base_stations = list(data.keys())
                    weights = list(data.values())
                    
                    # 计算位置嵌入
                    location_embedding = location_user_embedding(base_stations, weights)
                    location_embeddings[user_id] = location_embedding.detach().numpy()
        
        # 为所有用户计算增强嵌入
        self.enhanced_user_embeddings = {}
        
        for user_id in behavior_embeddings.keys():
            # 获取用户行为嵌入
            behavior_tensor = torch.tensor([behavior_embeddings[user_id]], dtype=torch.float32)
            
            # 获取用户属性嵌入
            attribute_embedding = None
            if user_id in self.user_attributes:
                user_attrs = self.user_attributes[user_id]
                
                # 准备属性输入
                categorical_inputs = {}
                numerical_values = []
                
                for attr_name, attr_value in user_attrs.items():
                    if self.attribute_info[attr_name]['type'] == 'categorical':
                        categorical_inputs[attr_name] = torch.tensor([attr_value], dtype=torch.long)
                    else:
                        numerical_values.append(attr_value)
                
                numerical_inputs = torch.tensor([numerical_values], dtype=torch.float32) if numerical_values else None
                
                # 计算属性嵌入
                with torch.no_grad():
                    attribute_embedding = self.attribute_model(categorical_inputs, numerical_inputs)
            
            # 获取用户位置嵌入
            location_embedding = None
            if user_id in location_embeddings:
                location_embedding = torch.tensor([location_embeddings[user_id]], dtype=torch.float32)
            
            # 融合所有可用的嵌入
            with torch.no_grad():
                fused_embedding = self.fusion_model(
                    behavior_tensor, 
                    attribute_embedding, 
                    location_embedding
                )
                self.enhanced_user_embeddings[user_id] = fused_embedding.squeeze().cpu().numpy()
        
        return self.enhanced_user_embeddings
    
    def get_similar_users(self, user_id, top_k=10):
        """
        获取相似用户（基于增强嵌入）
        """
        if user_id not in self.enhanced_user_embeddings:
            return [], []
        
        target_embed = self.enhanced_user_embeddings[user_id]
        similarities = []
        user_ids = []
        
        for uid, embed in self.enhanced_user_embeddings.items():
            if uid != user_id:
                similarity = np.dot(target_embed, embed)
                similarities.append(similarity)
                user_ids.append(uid)
        
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        similar_users = [user_ids[i] for i in sorted_indices]
        similar_scores = [similarities[i] for i in sorted_indices]
        
        return similar_users, similar_scores 