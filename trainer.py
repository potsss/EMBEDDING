"""
模型训练器
包括模型训练、早停策略、训练曲线、断点保存和加载等功能
包含属性训练器
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random
from collections import Counter
from config import Config
from model import Item2Vec, Node2Vec, AttributeEmbeddingModel, UserFusionModel, MaskedAttributePredictionModel

class SkipGramDataset(Dataset):
    """
    Skip-gram数据集
    优化版本：改进负采样和添加子采样
    """
    def __init__(self, sequences, window_size, negative_samples, vocab_size, subsample_threshold=1e-3):
        self.sequences = sequences
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.vocab_size = vocab_size
        self.subsample_threshold = subsample_threshold
        
        # 统计词频
        self.word_counts = self._count_words()
        self.total_words = sum(self.word_counts.values())
        
        # 计算子采样概率
        self.subsample_probs = self._compute_subsample_probs()
        
        # 构建训练样本
        self.samples = self._build_samples()
        
        # 构建高效的负采样表
        self.neg_table = self._build_negative_table()
    
    def _count_words(self):
        """统计词频"""
        word_counts = {}
        for sequence in self.sequences:
            for word in sequence:
                word_counts[word] = word_counts.get(word, 0) + 1
        return word_counts
    
    def _compute_subsample_probs(self):
        """
        计算子采样概率（用于处理高频词）
        """
        subsample_probs = {}
        for word, count in self.word_counts.items():
            freq = count / self.total_words
            # 子采样公式：P(w) = 1 - sqrt(t/f(w))
            if freq > self.subsample_threshold:
                prob = 1 - np.sqrt(self.subsample_threshold / freq)
                subsample_probs[word] = max(0, prob)
            else:
                subsample_probs[word] = 0
        return subsample_probs
    
    def _should_subsample(self, word):
        """判断是否应该子采样该词"""
        if word in self.subsample_probs:
            return np.random.random() < self.subsample_probs[word]
        return False
    
    def _build_samples(self):
        """
        构建训练样本（中心词-上下文词对）
        添加子采样功能
        """
        samples = []
        for sequence in self.sequences:
            # 应用子采样
            filtered_sequence = [word for word in sequence if not self._should_subsample(word)]
            
            for i, center_word in enumerate(filtered_sequence):
                # 动态窗口大小
                actual_window = np.random.randint(1, self.window_size + 1)
                
                # 获取上下文窗口
                start = max(0, i - actual_window)
                end = min(len(filtered_sequence), i + actual_window + 1)
                
                for j in range(start, end):
                    if i != j:  # 排除中心词自身
                        context_word = filtered_sequence[j]
                        samples.append((center_word, context_word))
        
        return samples
    
    def _build_negative_table(self):
        """
        构建高效的负采样表（基于词频的3/4次方）
        使用别名采样方法提高效率
        """
        # 计算采样概率
        probs = []
        words = []
        
        for word_id in range(self.vocab_size):
            count = self.word_counts.get(word_id, 1)  # 避免零计数
            prob = count ** 0.75  # 3/4次方
            probs.append(prob)
            words.append(word_id)
        
        # 归一化概率
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]
        
        # 构建别名表（Alias Method）
        self.alias_table = self._build_alias_table(probs, words)
        
        return words  # 保持兼容性
    
    def _build_alias_table(self, probs, words):
        """
        构建别名表用于O(1)时间复杂度的采样
        """
        n = len(probs)
        alias = [0] * n
        prob = [0.0] * n
        
        # 将概率缩放到n倍
        scaled_probs = [p * n for p in probs]
        
        # 分离大于1和小于1的概率
        small = []
        large = []
        
        for i, p in enumerate(scaled_probs):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        # 构建别名表
        while small and large:
            small_idx = small.pop()
            large_idx = large.pop()
            
            prob[small_idx] = scaled_probs[small_idx]
            alias[small_idx] = large_idx
            
            scaled_probs[large_idx] = scaled_probs[large_idx] + scaled_probs[small_idx] - 1.0
            
            if scaled_probs[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)
        
        # 处理剩余的概率
        while large:
            prob[large.pop()] = 1.0
        
        while small:
            prob[small.pop()] = 1.0
        
        return {'prob': prob, 'alias': alias, 'words': words}
    
    def _negative_sampling(self, positive_word):
        """
        高效的负采样（使用别名表）
        """
        negative_words = []
        alias_table = self.alias_table
        n = len(alias_table['words'])
        
        while len(negative_words) < self.negative_samples:
            # 使用别名表进行采样
            i = np.random.randint(0, n)
            if np.random.random() < alias_table['prob'][i]:
                word = alias_table['words'][i]
            else:
                word = alias_table['words'][alias_table['alias'][i]]
            
            if word != positive_word and word not in negative_words:
                negative_words.append(word)
        
        return negative_words
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        center_word, context_word = self.samples[idx]
        negative_words = self._negative_sampling(context_word)
        
        return (
            torch.tensor(center_word, dtype=torch.long),
            torch.tensor(context_word, dtype=torch.long),
            torch.tensor(negative_words, dtype=torch.long)
        )

class EarlyStopping:
    """
    早停策略
    """
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        if self.verbose:
            print(f"早停检查: {self.counter}/{self.patience}")

class AttributeDataset(Dataset):
    """
    属性训练数据集
    用于掩码属性预测任务
    支持多模态（行为+属性+位置）
    """
    def __init__(self, user_sequences, user_attributes, attribute_info, behavior_model, 
                 url_mappings, masking_ratio=0.15, config=Config, location_model=None,
                 user_location_sequences=None, base_station_mappings=None, location_weights=None):
        self.user_sequences = user_sequences
        self.user_attributes = user_attributes
        self.attribute_info = attribute_info
        self.behavior_model = behavior_model
        self.url_mappings = url_mappings
        self.masking_ratio = masking_ratio
        self.config = config
        
        # 位置相关参数
        self.location_model = location_model
        self.user_location_sequences = user_location_sequences
        self.base_station_mappings = base_station_mappings
        self.location_weights = location_weights
        
        # 预计算行为嵌入
        self.behavior_embeddings = self._precompute_behavior_embeddings()
        
        # 预计算位置嵌入（如果可用）
        self.location_embeddings = self._precompute_location_embeddings()
        
        # 筛选有行为和属性数据的用户
        self.valid_users = [
            user_id for user_id in self.behavior_embeddings.keys() 
            if user_id in self.user_attributes
        ]
        
        print(f"有效用户数量（同时有行为和属性数据）: {len(self.valid_users)}")
    
    def _precompute_behavior_embeddings(self):
        """预计算用户行为嵌入"""
        from model import UserEmbedding
        user_embedding_calculator = UserEmbedding(
            self.behavior_model, self.user_sequences, self.url_mappings
        )
        return user_embedding_calculator.compute_user_embeddings()
    
    def _precompute_location_embeddings(self):
        """预计算用户位置嵌入"""
        if self.location_model is None or self.user_location_sequences is None:
            return {}
        
        from model import UserLocationEmbedding
        
        # 首先获取基站嵌入
        base_station_embeddings = {}
        if self.base_station_mappings:
            for bs_id in self.base_station_mappings['base_station_to_id'].keys():
                idx = self.base_station_mappings['base_station_to_id'][bs_id]
                if hasattr(self.location_model, 'in_embeddings'):
                    embedding = self.location_model.in_embeddings.weight[idx].detach()
                    base_station_embeddings[bs_id] = embedding
        
        # 创建位置嵌入计算器
        user_location_embedding = UserLocationEmbedding(
            self.config, base_station_embeddings, self.location_weights
        )
        
        # 计算用户位置嵌入
        user_location_embeddings = {}
        if self.user_location_sequences:
            for user_id, data in self.user_location_sequences.items():
                if isinstance(data, dict) and 'base_stations' in data:
                    base_stations = data['base_stations']
                    weights = data['weights']
                    location_embedding = user_location_embedding(base_stations, weights)
                    user_location_embeddings[user_id] = location_embedding.detach().numpy()
        
        return user_location_embeddings
    
    def __len__(self):
        return len(self.valid_users)
    
    def __getitem__(self, idx):
        user_id = self.valid_users[idx]
        
        # 获取用户行为嵌入
        behavior_embedding = torch.tensor(self.behavior_embeddings[user_id], dtype=torch.float32)
        
        # 获取用户位置嵌入（如果可用）
        location_embedding = None
        if user_id in self.location_embeddings:
            location_embedding = torch.tensor(self.location_embeddings[user_id], dtype=torch.float32)
        
        # 获取用户属性
        user_attrs = self.user_attributes[user_id].copy()
        
        # 随机选择要掩码的属性
        all_attrs = list(user_attrs.keys())
        num_to_mask = max(1, int(len(all_attrs) * self.masking_ratio))
        masked_attrs = random.sample(all_attrs, num_to_mask)
        
        # 准备输入数据（掩码版本）
        categorical_inputs = {}
        numerical_values = []
        masked_categorical_inputs = {}
        masked_numerical_values = []
        
        # 真实标签
        categorical_targets = {}
        numerical_targets = {}
        
        categorical_attrs = [attr for attr in all_attrs 
                           if self.attribute_info[attr]['type'] == 'categorical']
        numerical_attrs = [attr for attr in all_attrs 
                         if self.attribute_info[attr]['type'] == 'numerical']
        
        # 处理类别型属性
        for attr_name in categorical_attrs:
            original_value = user_attrs[attr_name]
            if attr_name in masked_attrs:
                # 掩码该属性（用0代替，假设0是特殊的掩码token）
                masked_categorical_inputs[attr_name] = torch.tensor(0, dtype=torch.long)
                categorical_targets[attr_name] = torch.tensor(original_value, dtype=torch.long)
            else:
                categorical_inputs[attr_name] = torch.tensor(original_value, dtype=torch.long)
        
        # 处理数值型属性
        for attr_name in numerical_attrs:
            original_value = user_attrs[attr_name]
            if attr_name in masked_attrs:
                # 掩码该属性（用0代替）
                masked_numerical_values.append(0.0)
                numerical_targets[attr_name] = torch.tensor(original_value, dtype=torch.float32)
            else:
                numerical_values.append(original_value)
        
        # 合并掩码和非掩码输入
        all_categorical_inputs = {**categorical_inputs, **masked_categorical_inputs}
        all_numerical_values = numerical_values + masked_numerical_values
        
        return {
            'user_id': user_id,
            'behavior_embedding': behavior_embedding,
            'location_embedding': location_embedding,
            'categorical_inputs': all_categorical_inputs,
            'numerical_inputs': torch.tensor(all_numerical_values, dtype=torch.float32) if all_numerical_values else None,
            'masked_attrs': masked_attrs,
            'categorical_targets': categorical_targets,
            'numerical_targets': numerical_targets
        }

class AttributeTrainer:
    """
    属性训练器
    用于训练属性嵌入和融合模型
    支持多模态融合（行为+属性+位置）
    """
    def __init__(self, behavior_model, user_sequences, user_attributes, attribute_info, 
                 url_mappings, config=Config):
        self.behavior_model = behavior_model
        self.user_sequences = user_sequences
        self.user_attributes = user_attributes
        self.attribute_info = attribute_info
        self.url_mappings = url_mappings
        self.config = config
        self.device = config.DEVICE_OBJ
        
        # 位置相关参数（将由外部设置）
        self.location_model = None
        self.user_location_sequences = None
        self.base_station_mappings = None
        self.location_weights = None
        
        # 创建属性模型
        self.attribute_model = AttributeEmbeddingModel(attribute_info, config).to(self.device)
        
        # 融合模型将由外部设置（支持多模态）
        self.fusion_model = None
        
        # 掩码预测模型将在设置融合模型后创建
        self.prediction_model = None
        
        # 优化器、调度器等将在initialize_models中设置
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.writer = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def initialize_models(self):
        """
        初始化模型、优化器等（在设置融合模型后调用）
        """
        if self.fusion_model is None:
            raise ValueError("融合模型未设置，请先设置fusion_model")
        
        # 创建掩码预测模型
        self.prediction_model = MaskedAttributePredictionModel(
            self.attribute_model, self.fusion_model, self.attribute_info, self.config
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            list(self.attribute_model.parameters()) + 
            list(self.fusion_model.parameters()) + 
            list(self.prediction_model.parameters()),
            lr=self.config.ATTRIBUTE_LEARNING_RATE
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=self.config.ATTRIBUTE_EARLY_STOPPING_PATIENCE,
            min_delta=0.001,
            verbose=True
        )
        
        # TensorBoard
        tensorboard_dir = os.path.join(self.config.TENSORBOARD_DIR, 'attribute_training')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        
        # 训练状态
        self.train_losses = []
        self.val_losses = []
        
        print(f"属性训练器初始化完成:")
        print(f"  属性数量: {len(self.attribute_info)}")
        print(f"  类别型属性: {[attr for attr, info in self.attribute_info.items() if info['type'] == 'categorical']}")
        print(f"  数值型属性: {[attr for attr, info in self.attribute_info.items() if info['type'] == 'numerical']}")
    
    def create_dataloader(self):
        """创建数据加载器"""
        dataset = AttributeDataset(
            self.user_sequences, self.user_attributes, self.attribute_info,
            self.behavior_model, self.url_mappings, self.config.MASKING_RATIO, self.config,
            location_model=self.location_model,
            user_location_sequences=self.user_location_sequences,
            base_station_mappings=self.base_station_mappings,
            location_weights=self.location_weights
        )
        
        # 分割训练和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.ATTRIBUTE_BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,  # 属性训练通常数据量不会太大，设为0避免问题
            pin_memory=self.config.PIN_MEMORY
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.ATTRIBUTE_BATCH_SIZE,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=self.config.PIN_MEMORY
        )
        
        return train_loader, val_loader
    
    def _collate_fn(self, batch):
        """自定义批次整理函数"""
        batch_size = len(batch)
        
        # 收集行为嵌入
        behavior_embeddings = torch.stack([item['behavior_embedding'] for item in batch])
        
        # 收集位置嵌入（如果可用）
        location_embeddings = None
        if any(item['location_embedding'] is not None for item in batch):
            location_embeds = []
            for item in batch:
                if item['location_embedding'] is not None:
                    location_embeds.append(item['location_embedding'])
                else:
                    # 如果该用户没有位置嵌入，使用零向量
                    location_dim = self.config.LOCATION_EMBEDDING_DIM
                    location_embeds.append(torch.zeros(location_dim, dtype=torch.float32))
            location_embeddings = torch.stack(location_embeds)
        
        # 收集所有属性名
        all_categorical_attrs = set()
        all_numerical_attrs = []
        
        for item in batch:
            all_categorical_attrs.update(item['categorical_inputs'].keys())
            if item['numerical_inputs'] is not None:
                if not all_numerical_attrs:  # 初始化
                    all_numerical_attrs = list(range(len(item['numerical_inputs'])))
        
        # 整理类别型输入
        batch_categorical_inputs = {}
        for attr_name in all_categorical_attrs:
            attr_values = []
            for item in batch:
                if attr_name in item['categorical_inputs']:
                    attr_values.append(item['categorical_inputs'][attr_name])
                else:
                    attr_values.append(torch.tensor(0, dtype=torch.long))  # 默认掩码值
            batch_categorical_inputs[attr_name] = torch.stack(attr_values)
        
        # 整理数值型输入
        batch_numerical_inputs = None
        if all_numerical_attrs:
            numerical_matrix = []
            for item in batch:
                if item['numerical_inputs'] is not None:
                    numerical_matrix.append(item['numerical_inputs'])
                else:
                    numerical_matrix.append(torch.zeros(len(all_numerical_attrs), dtype=torch.float32))
            batch_numerical_inputs = torch.stack(numerical_matrix)
        
        # 收集目标和掩码信息
        masked_attrs_list = [item['masked_attrs'] for item in batch]
        categorical_targets = {}
        numerical_targets = {}
        
        # 整理目标
        all_target_attrs = set()
        for item in batch:
            all_target_attrs.update(item['categorical_targets'].keys())
            all_target_attrs.update(item['numerical_targets'].keys())
        
        for attr_name in all_target_attrs:
            if any(attr_name in item['categorical_targets'] for item in batch):
                targets = []
                for item in batch:
                    if attr_name in item['categorical_targets']:
                        targets.append(item['categorical_targets'][attr_name])
                    else:
                        targets.append(torch.tensor(-1, dtype=torch.long))  # 忽略标记
                categorical_targets[attr_name] = torch.stack(targets)
            
            if any(attr_name in item['numerical_targets'] for item in batch):
                targets = []
                for item in batch:
                    if attr_name in item['numerical_targets']:
                        targets.append(item['numerical_targets'][attr_name])
                    else:
                        targets.append(torch.tensor(0.0, dtype=torch.float32))  # 忽略标记
                numerical_targets[attr_name] = torch.stack(targets)
        
        return {
            'behavior_embeddings': behavior_embeddings,
            'location_embeddings': location_embeddings,
            'categorical_inputs': batch_categorical_inputs,
            'numerical_inputs': batch_numerical_inputs,
            'masked_attrs_list': masked_attrs_list,
            'categorical_targets': categorical_targets,
            'numerical_targets': numerical_targets
        }
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.attribute_model.train()
        self.fusion_model.train()
        self.prediction_model.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="属性训练中")
        for batch in progress_bar:
            # 移动数据到设备
            behavior_embeddings = batch['behavior_embeddings'].to(self.device)
            
            # 处理位置嵌入（如果可用）
            location_embeddings = None
            if 'location_embeddings' in batch and batch['location_embeddings'] is not None:
                location_embeddings = batch['location_embeddings'].to(self.device)
            
            categorical_inputs = {}
            for attr_name, attr_tensor in batch['categorical_inputs'].items():
                categorical_inputs[attr_name] = attr_tensor.to(self.device)
            
            numerical_inputs = batch['numerical_inputs'].to(self.device) if batch['numerical_inputs'] is not None else None
            
            # 收集所有被掩码的属性
            all_masked_attrs = set()
            for masked_attrs in batch['masked_attrs_list']:
                all_masked_attrs.update(masked_attrs)
            
            # 前向传播
            predictions = self.prediction_model(
                behavior_embeddings, categorical_inputs, numerical_inputs, list(all_masked_attrs), location_embeddings
            )
            
            # 计算损失
            total_batch_loss = 0
            loss_count = 0
            
            # 类别型属性损失
            for attr_name, targets in batch['categorical_targets'].items():
                if attr_name in predictions:
                    targets = targets.to(self.device)
                    pred = predictions[attr_name]
                    
                    # 只计算非忽略位置的损失
                    mask = targets != -1
                    if mask.sum() > 0:
                        loss = F.cross_entropy(pred[mask], targets[mask])
                        total_batch_loss += loss
                        loss_count += 1
            
            # 数值型属性损失
            for attr_name, targets in batch['numerical_targets'].items():
                if attr_name in predictions:
                    targets = targets.to(self.device).unsqueeze(-1)
                    pred = predictions[attr_name]
                    
                    # 只计算非零位置的损失（假设0是忽略标记）
                    mask = targets != 0
                    if mask.sum() > 0:
                        loss = F.mse_loss(pred[mask], targets[mask])
                        total_batch_loss += loss
                        loss_count += 1
            
            if loss_count > 0:
                avg_batch_loss = total_batch_loss / loss_count
            else:
                avg_batch_loss = torch.tensor(0.0, requires_grad=True)
            
            # 反向传播
            self.optimizer.zero_grad()
            avg_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += avg_batch_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': avg_batch_loss.item()})
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, val_loader):
        """验证模型"""
        self.attribute_model.eval()
        self.fusion_model.eval()
        self.prediction_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 移动数据到设备
                behavior_embeddings = batch['behavior_embeddings'].to(self.device)
                
                # 处理位置嵌入（如果可用）
                location_embeddings = None
                if 'location_embeddings' in batch and batch['location_embeddings'] is not None:
                    location_embeddings = batch['location_embeddings'].to(self.device)
                
                categorical_inputs = {}
                for attr_name, attr_tensor in batch['categorical_inputs'].items():
                    categorical_inputs[attr_name] = attr_tensor.to(self.device)
                
                numerical_inputs = batch['numerical_inputs'].to(self.device) if batch['numerical_inputs'] is not None else None
                
                # 收集所有被掩码的属性
                all_masked_attrs = set()
                for masked_attrs in batch['masked_attrs_list']:
                    all_masked_attrs.update(masked_attrs)
                
                # 前向传播
                predictions = self.prediction_model(
                    behavior_embeddings, categorical_inputs, numerical_inputs, list(all_masked_attrs), location_embeddings
                )
                
                # 计算损失
                total_batch_loss = 0
                loss_count = 0
                
                # 类别型属性损失
                for attr_name, targets in batch['categorical_targets'].items():
                    if attr_name in predictions:
                        targets = targets.to(self.device)
                        pred = predictions[attr_name]
                        
                        mask = targets != -1
                        if mask.sum() > 0:
                            loss = F.cross_entropy(pred[mask], targets[mask])
                            total_batch_loss += loss
                            loss_count += 1
                
                # 数值型属性损失
                for attr_name, targets in batch['numerical_targets'].items():
                    if attr_name in predictions:
                        targets = targets.to(self.device).unsqueeze(-1)
                        pred = predictions[attr_name]
                        
                        mask = targets != 0
                        if mask.sum() > 0:
                            loss = F.mse_loss(pred[mask], targets[mask])
                            total_batch_loss += loss
                            loss_count += 1
                
                if loss_count > 0:
                    avg_batch_loss = total_batch_loss / loss_count
                    total_loss += avg_batch_loss.item()
                    num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self):
        """完整的属性训练流程"""
        print("开始属性训练...")
        
        # 初始化模型
        self.initialize_models()
        
        # 创建数据加载器
        train_loader, val_loader = self.create_dataloader()
        
        best_loss = float('inf')
        best_epoch = -1
        
        for epoch in range(self.config.ATTRIBUTE_EPOCHS):
            print(f"\nAttribute Epoch {epoch+1}/{self.config.ATTRIBUTE_EPOCHS}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"属性训练损失: {train_loss:.4f}, 属性验证损失: {val_loss:.4f}")
            
            # TensorBoard记录
            self.writer.add_scalar('AttributeLoss/Train', train_loss, epoch)
            self.writer.add_scalar('AttributeLoss/Validation', val_loss, epoch)
            self.writer.add_scalar('AttributeLearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                self.save_models(is_best=True)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("属性训练早停触发")
                break
        
        # 训练完成后再次保存最佳模型，确保模型一定被保存
        print(f"属性训练完成！最佳验证损失: {best_loss:.4f} (epoch {best_epoch+1})")
        self.save_models(is_best=True)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 关闭TensorBoard
        self.writer.close()
    
    def save_models(self, is_best=False):
        """保存属性模型"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        
        models_dict = {
            'attribute_model_state_dict': self.attribute_model.state_dict(),
            'fusion_model_state_dict': self.fusion_model.state_dict(),
            'prediction_model_state_dict': self.prediction_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'attribute_info': self.attribute_info,
            'config_dict': {
                'ATTRIBUTE_EMBEDDING_DIM': self.config.ATTRIBUTE_EMBEDDING_DIM,
                'FUSION_HIDDEN_DIM': self.config.FUSION_HIDDEN_DIM,
                'FINAL_USER_EMBEDDING_DIM': self.config.FINAL_USER_EMBEDDING_DIM
            }
        }
        
        # 保存模型
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, 'attribute_models.pth')
        torch.save(models_dict, model_path)
        
        if is_best:
            best_path = os.path.join(self.config.MODEL_SAVE_PATH, 'best_attribute_models.pth')
            torch.save(models_dict, best_path)
        
        print(f"属性模型已保存到: {model_path}")
    
    def plot_training_curves(self):
        """绘制属性训练曲线"""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='属性训练损失')
        plt.plot(self.val_losses, label='属性验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('属性训练损失')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.config.LOG_DIR, 'attribute_training_curves.png')
        plt.savefig(save_path)
        plt.close()  # 关闭图片释放内存
        
        print(f"属性训练曲线已保存到: {save_path}")

def load_attribute_models(model_path, attribute_info, config=Config):
    """
    加载训练好的属性模型
    """
    checkpoint = torch.load(model_path, map_location=config.DEVICE_OBJ)
    
    # 重建模型
    attribute_model = AttributeEmbeddingModel(attribute_info, config)
    attribute_model.load_state_dict(checkpoint['attribute_model_state_dict'])
    
    behavior_dim = config.EMBEDDING_DIM
    attribute_dim = config.ATTRIBUTE_EMBEDDING_DIM
    fusion_model = UserFusionModel(behavior_dim, attribute_dim, config)
    fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
    
    return attribute_model, fusion_model

# 新增(lsx)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    模型训练器
    """
    def __init__(self, model, config=Config):
        """
        初始化训练器
        Args:
            model: 要训练的模型
            config: 配置类
        """
        self.model = model
        self.config = config
        self.device = config.DEVICE_OBJ
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            # verbose=True  # 这会导致错误，废弃（lsx）
        )
        logger.info("Scheduler initialized")
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=0.001,
            verbose=True
        )
        
        # TensorBoard
        self.writer = SummaryWriter(config.TENSORBOARD_DIR)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 创建必要目录
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
    
    def create_dataloader(self, sequences_input):
        """
        创建数据加载器
        Args:
            sequences_input: 可以是用户序列字典 {user_id: [item_ids]} 或游走序列列表 [[item_ids]]
        """
        # 检查输入类型并相应处理
        if isinstance(sequences_input, dict):
            # 传统的用户序列字典格式
            all_sequences = list(sequences_input.values())
        elif isinstance(sequences_input, list):
            # Node2Vec 游走序列列表格式
            all_sequences = sequences_input
        else:
            raise ValueError(f"不支持的序列输入类型: {type(sequences_input)}")
        
        dataset = SkipGramDataset(
            sequences=all_sequences,
            window_size=self.config.WINDOW_SIZE,
            negative_samples=self.config.NEGATIVE_SAMPLES,
            vocab_size=self.model.vocab_size
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,  # 使用Config中的设置
            pin_memory=self.config.PIN_MEMORY  # 使用Config中的设置
        )
        
        return dataloader
    
    def train_epoch(self, dataloader):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="训练中")
        for batch in progress_bar:
            center_words, context_words, negative_words = batch
            center_words = center_words.to(self.device)
            context_words = context_words.to(self.device)
            negative_words = negative_words.to(self.device)
            
            # 前向传播
            loss = self.model(center_words, context_words, negative_words)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader):
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                center_words, context_words, negative_words = batch
                center_words = center_words.to(self.device)
                context_words = context_words.to(self.device)
                negative_words = negative_words.to(self.device)
                
                loss = self.model(center_words, context_words, negative_words)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        保存检查点
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            print(f"已加载检查点: {checkpoint_path}")
            return checkpoint['epoch'], checkpoint['loss']
        else:
            print(f"检查点不存在: {checkpoint_path}")
            return 0, float('inf')
    
    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if len(self.train_losses) > 10:
            # 显示最近的损失趋势
            recent_train = self.train_losses[-50:]
            recent_val = self.val_losses[-50:]
            plt.plot(recent_train, label='训练损失（最近50轮）')
            plt.plot(recent_val, label='验证损失（最近50轮）')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('最近训练趋势')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.LOG_DIR, 'training_curves.png'))
        plt.close()  # 关闭图片释放内存
    
    def train(self, sequences_input, resume_from_checkpoint=False):
        """
        完整的训练流程
        Args:
            sequences_input: 可以是用户序列字典 {user_id: [item_ids]} 或游走序列列表 [[item_ids]]
            resume_from_checkpoint: 是否从检查点恢复训练
        """
        print("开始训练...")
        
        # 创建数据加载器
        dataloader = self.create_dataloader(sequences_input)
        
        # 分割训练和验证集
        train_size = int(0.8 * len(dataloader.dataset))
        val_size = len(dataloader.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataloader.dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.config.BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=self.config.NUM_WORKERS, # 使用Config中的设置
                                pin_memory=self.config.PIN_MEMORY # 使用Config中的设置
                                )
        val_loader = DataLoader(val_dataset, 
                              batch_size=self.config.BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=self.config.NUM_WORKERS, # 使用Config中的设置
                              pin_memory=self.config.PIN_MEMORY # 使用Config中的设置
                              )
        
        # 恢复训练
        start_epoch = 0
        best_loss = float('inf')
        
        if resume_from_checkpoint:
            checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'latest_checkpoint.pth')
            start_epoch, best_loss = self.load_checkpoint(checkpoint_path)
        
        # 训练循环
        for epoch in range(start_epoch, self.config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            if epoch % self.config.EVAL_INTERVAL == 0:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                
                # TensorBoard记录
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch) # 记录学习率
                
                # 保存检查点
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                
                self.save_checkpoint(epoch, val_loss, is_best)
                
                # 更新学习率
                self.scheduler.step(val_loss)
                
                # 早停检查
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print("早停触发，停止训练")
                    break
            else:
                # 即使不验证，也记录训练损失和当前学习率
                print(f"训练损失: {train_loss:.4f}")
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
                self.save_checkpoint(epoch, train_loss)
        
        # 保存最终模型
        self.save_model()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 关闭TensorBoard
        self.writer.close()
        
        print("训练完成！")
    
    def save_model(self):
        """
        保存最终模型
        根据模型类型保存到不同的文件名，例如 item2vec_model.pth 或 node2vec_model.pth
        """
        model_filename = "item2vec_model.pth"
        if isinstance(self.model, Node2Vec):
            model_filename = "node2vec_model.pth"
            
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, model_filename)
        
        # 保存与模型相关的信息，config可以帮助重建环境
        # vocab_size 和 embedding_dim 可以从 self.model 获取
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.vocab_size,
            'embedding_dim': self.model.embedding_dim,
            # 'config': self.config # 保存整个Config对象可能不是最佳实践，特别是如果它包含不可序列化的部分
                                     # 或者可以考虑只保存必要的配置项
            'model_type': self.model.__class__.__name__ # 保存模型类名，便于加载时识别
        }
        # 确保Config对象中的DEVICE_OBJ不会被序列化，因为它是一个torch.device对象
        # 如果需要保存config，最好将其转换为字典并排除DEVICE_OBJ

        torch.save(save_dict, model_path)
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载: {model_path}")

def train_location_model(config, location_processor):
    """训练位置嵌入模型
    
    Args:
        config: 配置对象
        location_processor: 位置数据处理器
        
    Returns:
        训练好的位置嵌入模型和基站映射
    """
    print("开始训练位置嵌入模型...")
    
    # 处理用户基站连接数据
    user_base_stations = location_processor.process_user_base_stations(config.LOCATION_DATA_PATH)
    
    if not user_base_stations:
        print("没有有效的用户基站连接数据")
        return None, None
    
    # 创建基站序列
    sequences = location_processor.create_base_station_sequences(user_base_stations)
    
    if not sequences:
        print("无法生成基站序列")
        return None, None
    
    # 创建基站映射
    all_base_stations = set()
    for seq in sequences:
        all_base_stations.update(seq)
    
    base_station_to_id = {bs: i for i, bs in enumerate(sorted(all_base_stations))}
    id_to_base_station = {i: bs for bs, i in base_station_to_id.items()}
    
    print(f"基站总数: {len(all_base_stations)}")
    print(f"序列总数: {len(sequences)}")
    
    # 转换序列为ID
    id_sequences = []
    for seq in sequences:
        id_seq = [base_station_to_id[bs] for bs in seq]
        id_sequences.append(id_seq)
    
    # 根据模型类型训练
    if config.LOCATION_MODEL_TYPE == "item2vec":
        # 创建Item2Vec模型
        from model import Item2Vec
        model = Item2Vec(len(all_base_stations), config.LOCATION_EMBEDDING_DIM)
        
        # 创建训练器
        trainer = Trainer(model, config)
        
        # 训练模型
        trainer.train(id_sequences)
        
    elif config.LOCATION_MODEL_TYPE == "node2vec":
        # 创建Node2Vec模型
        from model import Node2Vec
        model = Node2Vec(len(all_base_stations), config.LOCATION_EMBEDDING_DIM)
        
        # 创建训练器
        trainer = Trainer(model, config)
        
        # 对于Node2Vec，需要先构建图和生成随机游走
        from utils.node2vec_utils import build_graph_from_sequences, generate_node2vec_walks_with_cache
        
        # 构建图
        graph = build_graph_from_sequences(id_sequences, directed=False)
        
        # 生成随机游走
        walks = generate_node2vec_walks_with_cache(
            graph=graph,
            num_walks=config.NUM_WALKS,
            walk_length=config.WALK_LENGTH,
            p=config.P_PARAM,
            q=config.Q_PARAM,
            use_cache=config.USE_WALKS_CACHE,
            force_regenerate=config.FORCE_REGENERATE_WALKS
        )
        
        # 训练模型
        trainer.train(walks)
        
    else:
        raise ValueError(f"不支持的位置模型类型: {config.LOCATION_MODEL_TYPE}")
    
    base_station_mappings = {
        'base_station_to_id': base_station_to_id,
        'id_to_base_station': id_to_base_station
    }
    
    print("位置嵌入模型训练完成")
    return model, base_station_mappings

if __name__ == "__main__":
    # 示例使用
    from data_preprocessing import DataPreprocessor
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    user_sequences = preprocessor.load_processed_data()
    
    # 创建模型
    vocab_size = len(preprocessor.url_to_id)
    model = Item2Vec(vocab_size, Config.EMBEDDING_DIM)
    
    # 创建训练器并训练
    trainer = Trainer(model)
    trainer.train(user_sequences) 