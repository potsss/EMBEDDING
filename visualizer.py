"""
可视化模块
专注于用户和物品嵌入向量的 t-SNE 可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import os
from config import Config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    """
    可视化器
    专注于用户和物品嵌入向量的 t-SNE 可视化
    """
    def __init__(self, model, user_sequences, url_mappings, config=Config):
        self.model = model
        self.user_sequences = user_sequences
        self.url_to_id = url_mappings['url_to_id']
        self.id_to_url = url_mappings['id_to_url']
        self.config = config
        
        # 获取嵌入向量
        self.item_embeddings = model.get_embeddings()
        
        # 创建可视化目录
        os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    
    def compute_user_embeddings(self):
        """
        计算所有用户的嵌入向量
        """
        user_embeddings = {}
        user_ids = []
        embeddings = []
        
        for user_id, sequence in self.user_sequences.items():
            if len(sequence) > 0:
                unique_items = list(set(sequence))
                item_embeds = self.item_embeddings[unique_items]
                user_embed = np.mean(item_embeds, axis=0)
                
                user_embeddings[user_id] = user_embed
                user_ids.append(user_id)
                embeddings.append(user_embed)
        
        return user_embeddings, np.array(embeddings), user_ids
    
    def _create_tsne_visualization(self, embeddings, labels, title, sample_size=None, perplexity=30, n_iter=1000):
        """
        通用的 t-SNE 可视化函数
        Args:
            embeddings: 嵌入向量数组
            labels: 标签列表
            title: 图表标题
            sample_size: 采样大小
            perplexity: t-SNE 困惑度
            n_iter: t-SNE 迭代次数
        """
        # 采样
        if sample_size and len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = [labels[i] for i in indices]
        else:
            sample_embeddings = embeddings
            sample_labels = labels
        
        # t-SNE 降维
        print(f"正在进行 {title} 的 t-SNE 降维...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(sample_embeddings)-1),
            n_iter=n_iter,
            random_state=self.config.RANDOM_SEED,
            verbose=1
        )
        reduced_embeddings = tsne.fit_transform(sample_embeddings)
        
        # 创建可视化
        plt.figure(figsize=(15, 10))
        
        # 绘制散点图
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            alpha=0.6,
            s=50,
            c=np.random.rand(len(reduced_embeddings)),
            cmap='viridis'
        )
        
        # 添加标题和标签
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('t-SNE 维度 1', fontsize=12)
        plt.ylabel('t-SNE 维度 2', fontsize=12)
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, label='随机颜色映射')
        
        # 优化布局
        plt.tight_layout()
        
        # 保存图片
        save_name = title.replace(' ', '_').lower()
        save_path = os.path.join(self.config.VISUALIZATION_DIR, f'{save_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
        
        # 显示图片
        plt.show()
        
        return pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'label': sample_labels
        })
    
    def visualize_user_embeddings(self, sample_size=500, perplexity=30, n_iter=1000):
        """
        使用 t-SNE 可视化用户嵌入向量
        """
        print("开始用户嵌入向量可视化...")
        
        # 计算用户嵌入
        user_embeddings, embeddings_array, user_ids = self.compute_user_embeddings()
        
        # 使用通用可视化函数
        return self._create_tsne_visualization(
            embeddings=embeddings_array,
            labels=user_ids,
            title='用户嵌入向量 t-SNE 可视化',
            sample_size=sample_size,
            perplexity=perplexity,
            n_iter=n_iter
        )
    
    def visualize_item_embeddings(self, sample_size=1000, perplexity=30, n_iter=1000):
        """
        使用 t-SNE 可视化物品嵌入向量
        """
        print("开始物品嵌入向量可视化...")
        
        # 准备物品标签
        item_labels = [self.id_to_url[i] for i in range(len(self.item_embeddings))]
        
        # 使用通用可视化函数
        return self._create_tsne_visualization(
            embeddings=self.item_embeddings,
            labels=item_labels,
            title='物品嵌入向量 t-SNE 可视化',
            sample_size=sample_size,
            perplexity=perplexity,
            n_iter=n_iter
        )

if __name__ == "__main__":
    # 示例使用
    from data_preprocessing import DataPreprocessor
    from model import Item2Vec
    
    # 加载数据和模型
    preprocessor = DataPreprocessor()
    user_sequences = preprocessor.load_processed_data()
    
    # 加载训练好的模型
    vocab_size = len(preprocessor.url_to_id)
    model = Item2Vec(vocab_size, Config.EMBEDDING_DIM)
    
    # 创建可视化器
    url_mappings = {
        'url_to_id': preprocessor.url_to_id,
        'id_to_url': preprocessor.id_to_url
    }
    
    visualizer = Visualizer(model, user_sequences, url_mappings)
    
    # 可视化用户和物品嵌入
    visualizer.visualize_user_embeddings()
    visualizer.visualize_item_embeddings() 