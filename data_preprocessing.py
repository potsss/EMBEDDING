"""
数据预处理模块
包括数据清洗、数据处理、数据加载
"""
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import os
from collections import defaultdict
from config import Config
import pickle
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, config=Config):
        self.config = config
        self.url_to_id = {}
        self.id_to_url = {}
        self.user_sequences = defaultdict(list)
        
    def clean_data(self, df):
        """
        数据清洗
        """
        print("开始数据清洗...")
        
        # 删除缺失值
        df = df.dropna()
        
        # 删除重复行
        df = df.drop_duplicates()
        
        # 确保weight为正数
        df = df[df['weight'] > 0]
        
        # 按时间戳排序 (将字符串转换为datetime对象进行排序)
        df['timestamp_dt'] = pd.to_datetime(df['timestamp_str'])
        df = df.sort_values(['user_id', 'timestamp_dt'])
        df = df.drop(columns=['timestamp_dt']) # 删除辅助列
        
        print(f"清洗后数据量: {len(df)}")
        return df
    
    def extract_domain(self, url):
        """
        从URL中提取domain
        """
        try:
            # 确保url是字符串类型
            if not isinstance(url, str):
                return None
            
            # 如果没有协议头，添加默认的 http://
            if not url.startswith(('http://', 'https://')):
                url_to_parse = 'http://' + url
            else:
                url_to_parse = url
                
            parsed_url = urlparse(url_to_parse)
            domain = parsed_url.netloc
            # 移除www前缀
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return None
    
    def process_data(self, df):
        """
        数据处理
        """
        print("开始数据处理...")
        
        # 提取domain
        tqdm.pandas(desc="提取domain")
        df['domain'] = df['url'].progress_apply(self.extract_domain)
        
        # 删除无法解析domain的行
        df = df.dropna(subset=['domain'])
        
        # 创建domain到ID的映射
        unique_domains = df['domain'].unique()
        self.url_to_id = {domain: idx for idx, domain in enumerate(unique_domains)}
        self.id_to_url = {idx: domain for domain, idx in self.url_to_id.items()}
        
        # 将domain转换为ID
        df['domain_id'] = df['domain'].map(self.url_to_id)
        
        print(f"唯一domain数量: {len(unique_domains)}")
        return df
    
    def create_user_sequences(self, df):
        """
        创建用户访问序列
        """
        print("创建用户访问序列...")
        
        for user_id in tqdm(df['user_id'].unique(), desc="处理用户序列"):
            user_data = df[df['user_id'] == user_id].sort_values('timestamp_str') # 使用 timestamp_str 进行排序
            
            # 根据权重重复domain_id
            sequence = []
            for _, row in user_data.iterrows():
                # 根据权重决定重复次数（最少1次，最多10次）
                repeat_count = min(max(1, int(row['weight'])), 10)
                sequence.extend([row['domain_id']] * repeat_count)
            
            if len(sequence) >= self.config.MIN_COUNT:
                self.user_sequences[user_id] = sequence
        
        print(f"有效用户数量: {len(self.user_sequences)}")
        return self.user_sequences
    
    def save_processed_data(self):
        """
        保存处理后的数据
        """
        os.makedirs(self.config.PROCESSED_DATA_PATH, exist_ok=True)
        
        # 保存映射关系
        with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'url_mappings.pkl'), 'wb') as f:
            pickle.dump({
                'url_to_id': self.url_to_id,
                'id_to_url': self.id_to_url
            }, f)
        
        # 保存用户序列
        with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'user_sequences.pkl'), 'wb') as f:
            pickle.dump(dict(self.user_sequences), f)
        
        print("处理后的数据已保存")
    
    def load_processed_data(self):
        """
        加载处理后的数据
        """
        # 加载映射关系
        with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'url_mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            self.url_to_id = mappings['url_to_id']
            self.id_to_url = mappings['id_to_url']
        
        # 加载用户序列
        with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'user_sequences.pkl'), 'rb') as f:
            self.user_sequences = pickle.load(f)
        
        print("已加载处理后的数据")
        return self.user_sequences
    
    def preprocess(self, data_path=None):
        """
        完整的数据预处理流程
        """
        if data_path is None:
            data_path = self.config.DATA_PATH
        
        # 读取原始数据
        print(f"读取数据: {data_path}")
        df = pd.read_csv(data_path, sep='\t')
        print(f"原始数据量: {len(df)}")
        
        # 确保时间戳列存在且为字符串类型，以防后续转换出错
        if 'timestamp_str' not in df.columns:
            if 'timestamp' in df.columns:
                df.rename(columns={'timestamp': 'timestamp_str'}, inplace=True)
            else:
                raise ValueError("数据中缺少 'timestamp_str' 或 'timestamp' 列")
        df['timestamp_str'] = df['timestamp_str'].astype(str)

        # 数据清洗
        df = self.clean_data(df)
        
        # 数据处理
        df = self.process_data(df)
        
        # 创建用户序列
        user_sequences = self.create_user_sequences(df)
        
        # 保存处理后的数据
        self.save_processed_data()
        
        return user_sequences

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    user_sequences = preprocessor.preprocess() 