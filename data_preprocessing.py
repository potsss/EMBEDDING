"""
数据预处理模块
包括数据清洗、数据处理、数据加载、用户属性数据处理
"""
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import os
from collections import defaultdict
from config import Config
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder

class AttributeProcessor:
    """
    用户属性数据处理器
    """
    def __init__(self, config=Config):
        self.config = config
        self.categorical_encoders = {}  # 类别属性编码器
        self.numerical_scaler = StandardScaler()  # 数值属性标准化器
        self.attribute_info = {}  # 属性信息存储
        self.processed_attributes = {}  # 处理后的属性数据
        
    def load_attribute_data(self, file_path=None):
        """
        加载用户属性数据
        """
        if file_path is None:
            file_path = self.config.ATTRIBUTE_DATA_PATH
            
        if not os.path.exists(file_path):
            print(f"属性数据文件不存在: {file_path}")
            return None
            
        print(f"加载属性数据: {file_path}")
        try:
            df = pd.read_csv(file_path, sep='\t')
            print(f"属性数据加载成功，形状: {df.shape}")
            return df
        except Exception as e:
            print(f"加载属性数据时出错: {e}")
            return None
    
    def analyze_attributes(self, df):
        """
        分析属性类型（数值型 vs 类别型）
        """
        print("分析属性类型...")
        attribute_info = {}
        
        # 假设第一列是user_id
        user_id_col = df.columns[0]
        attribute_columns = df.columns[1:]
        
        for col in attribute_columns:
            # 判断是数值型还是类别型
            if df[col].dtype in ['int64', 'float64']:
                # 检查是否是离散的整数值（可能是编码后的类别）
                unique_values = df[col].nunique()
                if unique_values <= 20 and df[col].dtype == 'int64':
                    attribute_info[col] = {
                        'type': 'categorical',
                        'vocab_size': unique_values,
                        'unique_values': sorted(df[col].unique())
                    }
                else:
                    attribute_info[col] = {
                        'type': 'numerical',
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
            else:
                # 字符串类型，按类别处理
                unique_values = df[col].nunique()
                attribute_info[col] = {
                    'type': 'categorical',
                    'vocab_size': unique_values,
                    'unique_values': list(df[col].unique())
                }
        
        self.attribute_info = attribute_info
        print(f"属性分析完成:")
        for col, info in attribute_info.items():
            if info['type'] == 'categorical':
                print(f"  {col}: 类别型, 唯一值数量: {info['vocab_size']}")
            else:
                print(f"  {col}: 数值型, 范围: [{info['min']:.2f}, {info['max']:.2f}]")
        
        return attribute_info, user_id_col
    
    def preprocess_categorical_attributes(self, df, categorical_cols):
        """
        预处理类别型属性
        """
        print("预处理类别型属性...")
        processed_df = df.copy()
        
        for col in categorical_cols:
            # 处理缺失值
            processed_df[col] = processed_df[col].fillna('Unknown')
            
            # 处理低频类别
            value_counts = processed_df[col].value_counts()
            rare_values = value_counts[value_counts < self.config.CATEGORICAL_MIN_FREQ].index
            processed_df[col] = processed_df[col].replace(rare_values, 'Other')
            
            # 编码
            encoder = LabelEncoder()
            processed_df[col] = encoder.fit_transform(processed_df[col].astype(str))
            self.categorical_encoders[col] = encoder
            
            # 更新词汇表大小
            self.attribute_info[col]['vocab_size'] = len(encoder.classes_)
            self.attribute_info[col]['encoder_classes'] = encoder.classes_
            
        return processed_df
    
    def preprocess_numerical_attributes(self, df, numerical_cols):
        """
        预处理数值型属性
        """
        print("预处理数值型属性...")
        processed_df = df.copy()
        
        for col in numerical_cols:
            # 处理缺失值（用均值填充）
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
        
        if numerical_cols and self.config.NUMERICAL_STANDARDIZATION:
            # 标准化
            processed_df[numerical_cols] = self.numerical_scaler.fit_transform(
                processed_df[numerical_cols]
            )
            print(f"数值属性已标准化: {numerical_cols}")
        
        return processed_df
    
    def process_attributes(self, file_path=None):
        """
        完整的属性预处理流程
        """
        # 加载数据
        df = self.load_attribute_data(file_path)
        if df is None:
            return None, None
        
        # 分析属性类型
        attribute_info, user_id_col = self.analyze_attributes(df)
        
        # 分离类别型和数值型属性
        categorical_cols = [col for col, info in attribute_info.items() 
                          if info['type'] == 'categorical']
        numerical_cols = [col for col, info in attribute_info.items() 
                         if info['type'] == 'numerical']
        
        # 预处理类别型属性
        if categorical_cols:
            df = self.preprocess_categorical_attributes(df, categorical_cols)
        
        # 预处理数值型属性
        if numerical_cols:
            df = self.preprocess_numerical_attributes(df, numerical_cols)
        
        # 转换为字典格式 {user_id: {attribute: value}}
        processed_attributes = {}
        for _, row in df.iterrows():
            user_id = row[user_id_col]
            attributes = {}
            for col in df.columns:
                if col != user_id_col:
                    attributes[col] = row[col]
            processed_attributes[user_id] = attributes
        
        self.processed_attributes = processed_attributes
        print(f"属性预处理完成，用户数量: {len(processed_attributes)}")
        
        return processed_attributes, attribute_info
    
    def save_processed_attributes(self):
        """
        保存处理后的属性数据
        """
        os.makedirs(self.config.PROCESSED_DATA_PATH, exist_ok=True)
        
        # 保存处理后的属性数据
        with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'user_attributes.pkl'), 'wb') as f:
            pickle.dump(self.processed_attributes, f)
        
        # 保存属性信息
        with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'attribute_info.pkl'), 'wb') as f:
            pickle.dump(self.attribute_info, f)
        
        # 保存编码器
        with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'attribute_encoders.pkl'), 'wb') as f:
            pickle.dump({
                'categorical_encoders': self.categorical_encoders,
                'numerical_scaler': self.numerical_scaler
            }, f)
        
        print("处理后的属性数据已保存")
    
    def load_processed_attributes(self):
        """
        加载处理后的属性数据
        """
        try:
            # 加载处理后的属性数据
            with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'user_attributes.pkl'), 'rb') as f:
                self.processed_attributes = pickle.load(f)
            
            # 加载属性信息
            with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'attribute_info.pkl'), 'rb') as f:
                self.attribute_info = pickle.load(f)
            
            # 加载编码器
            with open(os.path.join(self.config.PROCESSED_DATA_PATH, 'attribute_encoders.pkl'), 'rb') as f:
                encoders = pickle.load(f)
                self.categorical_encoders = encoders['categorical_encoders']
                self.numerical_scaler = encoders['numerical_scaler']
            
            print("已加载处理后的属性数据")
            return self.processed_attributes, self.attribute_info
        except FileNotFoundError:
            print("未找到处理后的属性数据")
            return None, None

class DataPreprocessor:
    def __init__(self, config=Config):
        self.config = config
        self.url_to_id = {}
        self.id_to_url = {}
        self.user_sequences = defaultdict(list)
        
        # 添加属性处理器
        self.attribute_processor = AttributeProcessor(config) if config.ENABLE_ATTRIBUTES else None
        
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
        
        # 处理用户属性数据（如果启用）
        if self.config.ENABLE_ATTRIBUTES and self.attribute_processor:
            print("\n" + "="*50)
            print("开始处理用户属性数据")
            print("="*50)
            
            processed_attributes, attribute_info = self.attribute_processor.process_attributes()
            if processed_attributes is not None:
                self.attribute_processor.save_processed_attributes()
                print(f"属性数据处理完成，属性数量: {len(attribute_info)}")
            else:
                print("属性数据处理失败或跳过")
        
        return user_sequences

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    user_sequences = preprocessor.preprocess() 