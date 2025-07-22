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
import random

class LocationProcessor:
    """
    基站位置数据处理器
    """
    def __init__(self, config=Config):
        self.config = config
        self.base_station_to_id = {}
        self.id_to_base_station = {}
        self.user_location_sequences = {}
        self.base_station_features = {}
        self.processed_location_data = {}
        self.base_station_text_embeddings = {}
        self.text_model = None
        
    def load_location_data(self, file_path=None):
        """
        加载基站连接数据
        期望格式：user_id, base_station_id, timestamp_str, duration
        """
        if file_path is None:
            file_path = self.config.LOCATION_DATA_PATH
            
        if not os.path.exists(file_path):
            print(f"位置数据文件不存在: {file_path}")
            return None
            
        print(f"加载位置数据: {file_path}")
        try:
            df = pd.read_csv(file_path, sep='\t')
            print(f"位置数据加载成功，形状: {df.shape}")
            return df
        except Exception as e:
            print(f"加载位置数据时出错: {e}")
            return None
    
    def load_base_station_features(self, features_path):
        """加载基站特征数据
        
        Args:
            features_path: 基站特征文件路径
        """
        if not os.path.exists(features_path):
            print(f"基站特征文件 {features_path} 不存在，将跳过特征加载")
            return
        
        print(f"正在加载基站特征数据: {features_path}")
        
        try:
            # 读取基站特征数据：ID，名称
            df = pd.read_csv(features_path, sep='\t', header=None, names=['base_station_id', 'name'])
            
            # 根据配置选择处理模式
            if self.config.BASE_STATION_FEATURE_MODE == "none":
                # 模式1：完全不使用特征
                print("基站特征模式：不使用特征")
                return
                
            elif self.config.BASE_STATION_FEATURE_MODE == "text_embedding":
                # 模式2：使用预训练语言模型编码名称
                print("基站特征模式：使用预训练语言模型编码名称")
                self._load_text_embedding_model()
                self._process_text_embeddings(df)
                
            else:
                raise ValueError(f"不支持的基站特征模式: {self.config.BASE_STATION_FEATURE_MODE}")
                
        except Exception as e:
            print(f"加载基站特征时出错: {e}")
            print("将跳过特征处理")
    
    def _load_text_embedding_model(self):
        """加载预训练的文本嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"正在加载预训练模型: {self.config.TEXT_EMBEDDING_MODEL}")
            self.text_model = SentenceTransformer(self.config.TEXT_EMBEDDING_MODEL)
            print("预训练模型加载完成")
        except ImportError:
            raise ImportError("请安装sentence-transformers库: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"加载预训练模型失败: {e}")
    
    def _process_text_embeddings(self, df):
        """处理文本嵌入"""
        print("正在生成文本嵌入...")
        
        # 获取所有基站名称
        base_station_names = df['name'].fillna('').tolist()
        base_station_ids = df['base_station_id'].tolist()
        
        # 生成文本嵌入
        if base_station_names:
            embeddings = self.text_model.encode(base_station_names, batch_size=32, show_progress_bar=True)
            
            # 存储嵌入
            for bs_id, embedding in zip(base_station_ids, embeddings):
                self.base_station_text_embeddings[bs_id] = embedding
                
        print(f"成功生成 {len(self.base_station_text_embeddings)} 个基站的文本嵌入")
        
    def get_base_station_feature(self, base_station_id):
        """获取基站特征
        
        Args:
            base_station_id: 基站ID
            
        Returns:
            基站特征向量或None
        """
        if self.config.BASE_STATION_FEATURE_MODE == "none":
            return None
        elif self.config.BASE_STATION_FEATURE_MODE == "text_embedding":
            return self.base_station_text_embeddings.get(base_station_id)
        else:
            return None
    
    def get_feature_dimension(self):
        """获取特征维度"""
        if self.config.BASE_STATION_FEATURE_MODE == "none":
            return 0
        elif self.config.BASE_STATION_FEATURE_MODE == "text_embedding":
            return self.config.TEXT_EMBEDDING_DIM
        else:
            return 0
    
    def process_location_data(self, df):
        """
        处理基站数据，生成用户基站序列
        """
        print("处理基站位置数据...")
        
        # 数据清洗
        df = df.dropna(subset=['user_id', 'base_station_id'])
        df = df.drop_duplicates()
        
        # 按用户和时间排序
        df['timestamp'] = pd.to_datetime(df['timestamp_str']).dt.date
        df = df.sort_values(['user_id', 'timestamp'])
        
        # 创建基站ID映射
        unique_base_stations = df['base_station_id'].unique()
        self.base_station_to_id = {bs: i for i, bs in enumerate(unique_base_stations)}
        self.id_to_base_station = {i: bs for bs, i in self.base_station_to_id.items()}
        
        print(f"唯一基站数量: {len(unique_base_stations)}")
        
        # 生成用户基站序列
        user_sequences = {}
        for user_id, user_data in df.groupby('user_id'):
            # 按时间排序的基站序列
            base_station_sequence = user_data['base_station_id'].tolist()
            
            # 转换为ID序列
            id_sequence = [self.base_station_to_id[bs] for bs in base_station_sequence]
            
            # 过滤太短的序列
            if len(id_sequence) >= self.config.LOCATION_MIN_COUNT:
                user_sequences[user_id] = id_sequence
        
        self.user_location_sequences = user_sequences
        
        # 存储权重信息（连接时长）
        user_weights = {}
        for user_id, user_data in df.groupby('user_id'):
            if user_id in user_sequences:
                weights = {}
                for _, row in user_data.iterrows():
                    bs_id = row['base_station_id']
                    duration = row.get('duration', 1.0)  # 默认权重为1
                    weights[bs_id] = weights.get(bs_id, 0) + duration
                user_weights[user_id] = weights
        
        self.processed_location_data = {
            'user_sequences': user_sequences,
            'user_weights': user_weights,
            'base_station_mappings': {
                'base_station_to_id': self.base_station_to_id,
                'id_to_base_station': self.id_to_base_station
            }
        }
        
        print(f"处理完成，有效用户数量: {len(user_sequences)}")
        return self.processed_location_data
    
    def process_base_station_features(self, df):
        """
        处理基站特征数据
        """
        if df is None:
            return None
            
        print("处理基站特征数据...")
        
        # 基站特征处理
        feature_info = {}
        for _, row in df.iterrows():
            bs_id = row['base_station_id']
            features = {
                'name': row.get('name', 'unknown'),
                'area_type': row.get('area_type', 'unknown'),
                'coverage_type': row.get('coverage_type', 'unknown')
            }
            feature_info[bs_id] = features
        
        self.base_station_features = feature_info
        print(f"基站特征处理完成，基站数量: {len(feature_info)}")
        return feature_info
    
    def save_processed_data(self, save_dir):
        """
        保存处理后的位置数据
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存基站序列
        location_sequences_path = os.path.join(save_dir, 'location_sequences.pkl')
        with open(location_sequences_path, 'wb') as f:
            pickle.dump(self.user_location_sequences, f)
        print(f"位置序列已保存到: {location_sequences_path}")
        
        # 保存基站映射
        base_station_mappings_path = os.path.join(save_dir, 'base_station_mappings.pkl')
        with open(base_station_mappings_path, 'wb') as f:
            pickle.dump(self.processed_location_data['base_station_mappings'], f)
        print(f"基站映射已保存到: {base_station_mappings_path}")
        
        # 保存权重信息
        location_weights_path = os.path.join(save_dir, 'location_weights.pkl')
        with open(location_weights_path, 'wb') as f:
            pickle.dump(self.processed_location_data['user_weights'], f)
        print(f"位置权重已保存到: {location_weights_path}")
        
        # 保存基站特征（如果有）
        if self.base_station_features:
            base_station_features_path = os.path.join(save_dir, 'base_station_features.pkl')
            with open(base_station_features_path, 'wb') as f:
                pickle.dump(self.base_station_features, f)
            print(f"基站特征已保存到: {base_station_features_path}")
    
    def load_processed_data(self, save_dir):
        """
        加载处理后的位置数据
        """
        try:
            # 加载位置序列
            location_sequences_path = os.path.join(save_dir, 'location_sequences.pkl')
            if os.path.exists(location_sequences_path):
                with open(location_sequences_path, 'rb') as f:
                    self.user_location_sequences = pickle.load(f)
            
            # 加载基站映射
            base_station_mappings_path = os.path.join(save_dir, 'base_station_mappings.pkl')
            if os.path.exists(base_station_mappings_path):
                with open(base_station_mappings_path, 'rb') as f:
                    mappings = pickle.load(f)
                    self.base_station_to_id = mappings['base_station_to_id']
                    self.id_to_base_station = mappings['id_to_base_station']
            
            # 加载权重信息
            location_weights_path = os.path.join(save_dir, 'location_weights.pkl')
            if os.path.exists(location_weights_path):
                with open(location_weights_path, 'rb') as f:
                    location_weights = pickle.load(f)
            else:
                location_weights = {}
            
            # 加载基站特征
            base_station_features_path = os.path.join(save_dir, 'base_station_features.pkl')
            if os.path.exists(base_station_features_path):
                with open(base_station_features_path, 'rb') as f:
                    self.base_station_features = pickle.load(f)
            
            self.processed_location_data = {
                'user_sequences': self.user_location_sequences,
                'user_weights': location_weights,
                'base_station_mappings': {
                    'base_station_to_id': self.base_station_to_id,
                    'id_to_base_station': self.id_to_base_station
                }
            }
            
            print(f"位置数据加载成功，用户数量: {len(self.user_location_sequences)}")
            return self.processed_location_data
            
        except Exception as e:
            print(f"加载位置数据时出错: {e}")
            return None
    
    def process_user_base_stations(self, data_path):
        """处理用户基站连接数据
        
        Args:
            data_path: 基站连接数据文件路径
            
        Returns:
            处理后的数据字典
        """
        print(f"正在加载用户基站连接数据: {data_path}")
        
        try:
            # 读取用户基站连接数据
            df = pd.read_csv(data_path, sep='\t')
            
            # 验证必要的列
            required_columns = ['user_id', 'base_station_id', 'timestamp_str', 'duration']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp_str']).dt.date
            
            # 过滤有效数据
            df = df[df['duration'] > 0]  # 过滤持续时间为0的记录
            
            # 按用户分组处理
            user_base_stations = {}
            for user_id, group in df.groupby('user_id'):
                # 按时间排序
                group = group.sort_values('timestamp')
                
                # 计算每个基站的连接权重（基于持续时间）
                base_station_weights = group.groupby('base_station_id')['duration'].sum()
                
                # 过滤连接次数过少的用户
                if len(base_station_weights) >= self.config.LOCATION_MIN_CONNECTIONS:
                    user_base_stations[user_id] = {
                        'base_stations': base_station_weights.index.tolist(),
                        'weights': base_station_weights.values.tolist(),
                        'total_duration': base_station_weights.sum()
                    }
            
            print(f"成功处理 {len(user_base_stations)} 个用户的基站连接数据")
            return user_base_stations
            
        except Exception as e:
            print(f"处理用户基站连接数据时出错: {e}")
            return {}
    
    def create_base_station_sequences(self, user_base_stations):
        """创建基站序列用于训练嵌入模型
        
        Args:
            user_base_stations: 用户基站连接数据
            
        Returns:
            基站序列列表
        """
        sequences = []
        
        for user_id, data in user_base_stations.items():
            base_stations = data['base_stations']
            weights = data['weights']
            
            # 根据权重生成序列（权重越大，在序列中出现次数越多）
            weighted_sequence = []
            for bs, weight in zip(base_stations, weights):
                # 根据权重决定出现次数（至少出现1次）
                max_weight = max(weights) if weights else 1
                count = max(1, int(weight / max_weight * 10))
                weighted_sequence.extend([str(bs)] * count)
            
            # 随机打乱序列
            import random
            random.shuffle(weighted_sequence)
            sequences.append(weighted_sequence)
        
        print(f"生成了 {len(sequences)} 个基站序列")
        return sequences

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
        
        # 添加位置处理器
        self.location_processor = LocationProcessor(config) if config.ENABLE_LOCATION else None
        
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
        df['timestamp_dt'] = pd.to_datetime(df['timestamp_str']).dt.date
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
        
        # 处理用户位置数据（如果启用）
        if self.config.ENABLE_LOCATION and self.location_processor:
            print("\n" + "="*50)
            print("开始处理用户位置数据")
            print("="*50)
            
            # 加载和处理基站特征数据
            self.location_processor.load_base_station_features(self.config.LOCATION_FEATURES_PATH)
            
            # 处理用户基站连接数据
            user_base_stations = self.location_processor.process_user_base_stations(self.config.LOCATION_DATA_PATH)
            
            if user_base_stations:
                # 创建基站序列用于训练
                sequences = self.location_processor.create_base_station_sequences(user_base_stations)
                
                # 获取基站总数
                all_base_stations = set()
                for data in user_base_stations.values():
                    all_base_stations.update(data['base_stations'])
                
                # 创建基站ID映射
                base_station_list = list(all_base_stations)
                self.location_processor.base_station_to_id = {bs: i for i, bs in enumerate(base_station_list)}
                self.location_processor.id_to_base_station = {i: bs for bs, i in self.location_processor.base_station_to_id.items()}
                
                # 更新位置序列为ID序列
                user_location_sequences = {}
                for user_id, data in user_base_stations.items():
                    base_stations = data['base_stations']
                    id_sequence = [self.location_processor.base_station_to_id[bs] for bs in base_stations]
                    user_location_sequences[user_id] = id_sequence
                
                self.location_processor.user_location_sequences = user_location_sequences
                
                # 设置处理后的数据
                self.location_processor.processed_location_data = {
                    'user_sequences': user_location_sequences,
                    'user_weights': {user_id: dict(zip(data['base_stations'], data['weights'])) for user_id, data in user_base_stations.items()},
                    'base_station_mappings': {
                        'base_station_to_id': self.location_processor.base_station_to_id,
                        'id_to_base_station': self.location_processor.id_to_base_station
                    }
                }
                
                # 保存处理后的位置数据
                self.location_processor.save_processed_data(self.config.PROCESSED_DATA_PATH)
                
                print(f"位置数据处理完成，用户数量: {len(user_location_sequences)}")
                print(f"  位置用户数量: {len(user_location_sequences)}")
                print(f"  基站数量: {len(all_base_stations)}")
            else:
                print("位置数据处理失败或跳过")
        
        return user_sequences

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    user_sequences = preprocessor.preprocess() 