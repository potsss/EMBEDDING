"""
模型训练器
包括模型训练、早停策略、训练曲线、断点保存和加载等功能
"""
import torch
import torch.optim as optim
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
from model import Item2Vec, Node2Vec

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
            verbose=True
        )
        
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
        plt.show()
    
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