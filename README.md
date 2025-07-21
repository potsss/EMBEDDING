# 多模态用户表示向量训练项目

这是一个基于多模态学习的用户表示向量训练项目，使用PyTorch实现，支持从用户行为数据、位置数据和属性数据中学习用户的综合向量表示。**项目支持三种向量训练：行为向量、位置向量和属性向量，并通过多模态融合生成最终的用户表示。**

## 项目概述

本项目采用分阶段训练策略，实现多模态用户表示学习。主要特点：

### 三种向量训练
- **行为向量训练**：使用Item2Vec/Node2Vec模型从用户访问行为中学习兴趣表示
- **位置向量训练**：基于用户基站连接数据学习位置偏好表示
- **属性向量训练**：通过掩码属性预测任务学习用户属性嵌入

### 多模态融合
- **独立训练策略**：三种向量独立训练，避免模态间的相互干扰
- **多模态融合**：将三种向量融合成最终的用户表示向量
- **灵活配置**：支持单独使用任一模态或组合使用多个模态

### 技术特性
- 基于PyTorch实现，支持GPU加速
- 包含完整的数据预处理、模型训练、评估和可视化流程
- 支持无阻塞运行，图片自动保存
- 支持断点续训和早停策略

## 数据格式

### 行为数据
用户行为数据应为TSV格式，包含以下字段：

```
user_id	url	timestamp_str	weight
user1	example.com	2023-01-01	1.5
user2	github.com	2023-01-02	2.0
```

- `user_id`: 用户ID
- `url`: 用户访问的URL
- `timestamp_str`: 访问时间戳 (字符串格式，如 "YYYY-MM-DD")
- `weight`: 访问权重（权重越大表示用户对该URL的兴趣越大）

### 位置数据（新增）
用户基站连接数据应为TSV格式，包含以下字段：

```
user_id	base_station_id	timestamp_str	duration
user1	BS_001	2023-01-01 08:00:00	1800
user2	BS_002	2023-01-01 09:00:00	2400
```

- `user_id`: 用户ID，与行为数据中的用户ID对应
- `base_station_id`: 基站ID
- `timestamp_str`: 连接时间戳
- `duration`: 连接持续时间（秒）

### 基站特征数据（新增）
基站特征数据应为TSV格式，包含以下字段：

```
base_station_id	name
BS_001	市中心商业区基站
BS_002	居民区基站
```

- `base_station_id`: 基站ID
- `name`: 基站名称或描述

### 属性数据
用户属性数据应为TSV格式，每行代表一个用户，每列代表一个属性：

```
user_id	age	gender	occupation	city	income_level	education_level	device_type
user1	25	男	工程师	北京	中等	本科	Android
user2	30	女	教师	上海	中等	硕士	iOS
```

- 第一列必须是`user_id`，与行为数据中的用户ID对应
- 支持类别型属性（如性别、职业）和数值型属性（如年龄）
- 系统会自动识别属性类型并进行相应的预处理

## 项目结构

```
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── config.py                # 配置文件
├── main.py                  # 主程序入口
├── data_preprocessing.py    # 数据预处理模块（包含三种数据处理）
├── model.py                 # 模型定义（包含三种嵌入模型和融合模型）
├── trainer.py               # 训练器（包含三种向量训练器）
├── visualizer.py            # 可视化模块
├── data/                    # 原始数据目录
│   ├── test_user_behavior.csv           # 用户行为数据（示例）
│   ├── sample_user_attributes.tsv       # 用户属性数据（示例）
│   ├── sample_user_base_stations.tsv    # 用户基站连接数据（示例）
│   └── sample_base_station_features.tsv # 基站特征数据（示例）
├── utils/                   # 工具文件夹
│   ├── __init__.py
│   ├── utils.py             # 通用工具函数
│   └── node2vec_utils.py    # Node2Vec相关工具函数
└── experiments/             # 实验结果的根目录
    └── {EXPERIMENT_NAME}/    # 单次实验的目录
        ├── processed_data/   # 处理后的数据
        │   ├── url_mappings.pkl            # URL映射
        │   ├── user_sequences.pkl          # 用户序列
        │   ├── user_attributes.pkl         # 处理后的属性数据
        │   ├── attribute_info.pkl          # 属性信息
        │   ├── attribute_encoders.pkl      # 属性编码器
        │   ├── location_sequences.pkl      # 位置序列数据（新增）
        │   ├── base_station_mappings.pkl   # 基站映射（新增）
        │   ├── location_weights.pkl        # 位置权重（新增）
        │   └── base_station_features.pkl   # 基站特征（新增）
        ├── models/           # 保存的模型
        │   ├── node2vec_model.pth          # Node2Vec行为模型
        │   ├── item2vec_model.pth          # Item2Vec位置模型
        │   ├── attribute_models.pth        # 属性模型
        │   ├── best_attribute_models.pth   # 最佳属性模型
        │   ├── user_embeddings.pkl        # 基础用户嵌入
        │   └── enhanced_user_embeddings_*.pkl # 增强用户嵌入（三种向量融合）
        ├── checkpoints/      # 训练检查点
        ├── logs/             # 日志文件
        │   ├── training_curves.png         # 行为模型训练曲线
        │   └── attribute_training_curves.png # 属性训练曲线
        ├── runs/             # TensorBoard日志
        │   └── attribute_training/         # 属性训练日志
        ├── visualizations/   # 可视化结果
        │   ├── 用户嵌入向量_t-sne_可视化.png
        │   └── 物品嵌入向量_t-sne_可视化.png
        └── experiment_config.json # 本次实验的配置快照
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

将数据文件放在 `data/` 目录下：

- **行为数据**：`data/test_user_behavior.csv`
- **位置数据**：`data/sample_user_base_stations.tsv`
- **基站特征**：`data/sample_base_station_features.tsv`
- **属性数据**：`data/sample_user_attributes.tsv`

项目已提供示例数据文件，可以直接使用进行测试。

### 2. 运行流程

#### 完整多模态训练流程（推荐）

```bash
# 运行完整的三种向量训练流程
python main.py --mode all

# 这个命令会依次执行：
# 1. 数据预处理（行为、位置、属性数据）
# 2. 行为向量训练（Node2Vec）
# 3. 位置向量训练（Item2Vec）
# 4. 属性向量训练（掩码属性预测）
# 5. 多模态融合（生成最终用户表示）
# 6. 可视化结果
```

#### 分步骤运行

```bash
# 1. 数据预处理
python main.py --mode preprocess

# 2. 模型训练
python main.py --mode train

# 3. 可视化
python main.py --mode visualize

# 4. 计算最终嵌入
python main.py --mode compute_embeddings
```

#### 仅使用单一模态

```bash
# 仅使用行为向量（在config.py中设置）
# ENABLE_ATTRIBUTES = False
# ENABLE_LOCATION = False
python main.py --mode all

# 仅使用行为+属性向量
# ENABLE_ATTRIBUTES = True
# ENABLE_LOCATION = False
python main.py --mode all

# 仅使用行为+位置向量
# ENABLE_ATTRIBUTES = False  
# ENABLE_LOCATION = True
python main.py --mode all
```

### 3. 新用户向量计算（推理阶段）

训练完成后，可以为未参与训练的新用户计算向量表示：

#### 方法一：使用独立脚本（推荐）

```bash
# 使用独立的新用户向量计算脚本
python compute_new_users.py \
    --experiment_name three_vector_test \
    --new_user_behavior_path data/new_user_behavior.csv \
    --new_user_attribute_path data/new_user_attributes.tsv \
    --new_user_location_path data/new_user_base_stations.tsv \
    --output_path results/new_user_vectors.pkl
```

#### 方法二：使用主程序

```bash
# 使用主程序的新用户计算模式
python main.py --mode compute_new_users \
    --experiment_name three_vector_test \
    --new_user_behavior_path data/new_user_behavior.csv \
    --new_user_attribute_path data/new_user_attributes.tsv \
    --new_user_location_path data/new_user_base_stations.tsv
```

#### 新用户数据格式

**新用户行为数据** (`data/new_user_behavior.csv`)：
```csv
user_id,url,timestamp_str,weight
new_user_001,example.com,2023-06-01,1.5
new_user_001,github.com,2023-06-01,2.0
```

**新用户属性数据** (`data/new_user_attributes.tsv`)：
```tsv
user_id	age	gender	city	device_type	education_level
new_user_001	28	Male	Beijing	smartphone	Bachelor
```

**新用户位置数据** (`data/new_user_base_stations.tsv`)：
```tsv
user_id	base_station_id	timestamp_str	duration
new_user_001	BS_001	2023-06-01 08:00:00	1800
```

**重要说明**：
- 新用户访问的URL必须在训练数据中出现过
- 新用户的属性值必须在训练数据的取值范围内
- 新用户连接的基站必须在训练数据中出现过
- 系统会自动跳过无法识别的URL、属性值或基站

### 4. 命令行参数

- `--mode`: 运行模式
  - `preprocess`: 仅数据预处理
  - `train`: 仅模型训练
  - `visualize`: 仅结果可视化
  - `compute_embeddings`: 仅计算并保存用户嵌入向量
  - `all`: 完整流程（默认）

- `--data_path`: 原始数据文件路径
- `--model_path`: 指定已训练模型的路径
- `--resume`: 从最新的检查点恢复训练
- `--no_train`: 跳过训练，直接使用已有模型
- `--experiment_name`: 自定义实验名称
- `--no_cache`: 禁用随机游走缓存
- `--force_regenerate`: 强制重新生成随机游走

### 5. 配置参数

在 `config.py` 中可以调整各种参数：

```python
# 实验配置
EXPERIMENT_NAME = "three_vector_test"   # 实验名称

# 数据相关配置
DATA_PATH = "data/test_user_behavior.csv"
ATTRIBUTE_DATA_PATH = "data/sample_user_attributes.tsv"
LOCATION_DATA_PATH = "data/sample_user_base_stations.tsv"      # 位置数据路径
LOCATION_FEATURES_PATH = "data/sample_base_station_features.tsv"  # 基站特征路径

# 行为向量相关配置
MODEL_TYPE = "node2vec"        # 可选 "item2vec" 或 "node2vec"
EMBEDDING_DIM = 128           # 行为嵌入维度
WINDOW_SIZE = 5               # 上下文窗口大小
MIN_COUNT = 5                 # 最小计数阈值
NEGATIVE_SAMPLES = 5          # 负采样数量

# 位置向量相关配置（新增）
ENABLE_LOCATION = True        # 启用位置向量
LOCATION_EMBEDDING_DIM = 128  # 位置嵌入维度
LOCATION_MIN_CONNECTIONS = 2  # 用户最少需要连接的基站数量
BASE_STATION_FEATURE_MODE = "none"  # 基站特征模式："none" 或 "text_embedding"

# 位置模型训练参数
LOCATION_LEARNING_RATE = 0.001
LOCATION_EPOCHS = 10
LOCATION_MODEL_TYPE = "item2vec"
LOCATION_WINDOW_SIZE = 5
LOCATION_NEGATIVE_SAMPLES = 5
LOCATION_MIN_COUNT = 1
LOCATION_BATCH_SIZE = 64

# 属性向量相关配置
ENABLE_ATTRIBUTES = True      # 启用属性向量
ATTRIBUTE_EMBEDDING_DIM = 64  # 属性嵌入维度
FUSION_HIDDEN_DIM = 256      # 融合层隐藏维度
FINAL_USER_EMBEDDING_DIM = 256  # 最终用户嵌入维度

# 属性训练相关配置
ATTRIBUTE_LEARNING_RATE = 0.001
ATTRIBUTE_EPOCHS = 8
ATTRIBUTE_BATCH_SIZE = 512
MASKING_RATIO = 0.15         # 掩码比例
ATTRIBUTE_EARLY_STOPPING_PATIENCE = 10

# Node2Vec 特定参数
P_PARAM = 1.0                # 返回参数 p
Q_PARAM = 1.0                # 进出参数 q
WALK_LENGTH = 20             # 随机游走长度
NUM_WALKS = 4                # 每个节点的游走次数

# 训练参数
LEARNING_RATE = 0.001
EPOCHS = 10                  # 训练轮次（已减少用于快速测试）
BATCH_SIZE = 1024
EARLY_STOPPING_PATIENCE = 10

# 随机种子
RANDOM_SEED = 42
```

## 功能模块

### 数据预处理 (`data_preprocessing.py`)

#### DataPreprocessor
- 行为数据清洗：删除缺失值、重复行，处理无效权重
- URL处理：提取domain，创建ID映射
- 序列构建：根据权重和时间戳构建用户访问序列

#### LocationProcessor（新增）
- 基站连接数据处理：解析用户基站连接记录
- 基站特征处理：支持文本嵌入或简单ID映射
- 位置序列生成：为位置向量训练准备数据

#### AttributeProcessor
- 属性类型识别：自动识别类别型和数值型属性
- 属性编码：类别型属性编码，数值型属性标准化
- 属性预处理：为属性向量训练准备数据

### 模型定义 (`model.py`)

#### 行为向量模型
- `Item2Vec`: 基于Skip-gram的物品嵌入模型
- `Node2Vec`: 基于图随机游走的物品嵌入模型
- `UserEmbedding`: 用户行为嵌入计算类

#### 位置向量模型（新增）
- `UserLocationEmbedding`: 用户位置嵌入计算模块
- 支持基于基站连接的位置表示学习
- 支持基站特征融合（可选）

#### 属性向量模型
- `AttributeEmbeddingModel`: 用户属性嵌入模型
- `MaskedAttributePredictionModel`: 掩码属性预测模型
- 支持类别型和数值型属性处理

#### 多模态融合模型
- `UserFusionModel`: 用户多模态融合模型
- `EnhancedUserEmbedding`: 增强用户嵌入计算类
- 支持三种向量的灵活融合

### 训练器 (`trainer.py`)

#### 通用训练器
- `Trainer`: 基础训练器，支持Item2Vec和Node2Vec
- `EarlyStopping`: 早停策略
- `train_location_model`: 位置模型训练函数

#### 属性训练器
- `AttributeTrainer`: 属性模型训练器
- `AttributeDataset`: 属性训练数据集
- 支持多模态融合训练

#### 训练特性
- 支持断点续训和最佳模型保存
- TensorBoard日志记录
- 训练曲线自动保存（无阻塞）
- 学习率调度

### 可视化器 (`visualizer.py`)

- 用户嵌入向量 t-SNE 可视化
- 物品嵌入向量 t-SNE 可视化
- 自动保存可视化结果，无需手动关闭

### Node2Vec 工具 (`utils/node2vec_utils.py`)

- `build_graph_from_sequences`: 构建物品交互图
- `generate_node2vec_walks`: 生成随机游走序列
- 支持缓存机制，提高训练效率

## 训练流程

### 1. 独立训练阶段
```
行为数据 → 行为向量训练 → 固定行为嵌入
位置数据 → 位置向量训练 → 固定位置嵌入  
属性数据 → 属性向量训练 → 固定属性嵌入
```

### 2. 多模态融合阶段
```
行为嵌入 + 位置嵌入 + 属性嵌入 → 融合网络 → 最终用户表示
```

### 3. 输出结果
```
用户ID → 256维最终用户嵌入向量
```

## 输出结果

### 1. 模型文件
- `node2vec_model.pth`: 行为向量模型（约12KB）
- `item2vec_model.pth`: 位置向量模型（约10KB）
- `attribute_models.pth`: 属性向量模型（约3.4MB）
- `enhanced_user_embeddings_node2vec.pkl`: 最终用户嵌入

### 2. 数据文件
- `location_sequences.pkl`: 位置序列数据
- `base_station_mappings.pkl`: 基站映射
- `location_weights.pkl`: 位置权重
- `user_attributes.pkl`: 处理后的属性数据
- `attribute_info.pkl`: 属性信息

### 3. 可视化结果
- `用户嵌入向量_t-sne_可视化.png`: 用户嵌入t-SNE图
- `物品嵌入向量_t-sne_可视化.png`: 物品嵌入t-SNE图
- `training_curves.png`: 训练曲线图
- `attribute_training_curves.png`: 属性训练曲线图

### 4. 日志与配置
- `experiments/{EXPERIMENT_NAME}/runs/`: TensorBoard日志
- `experiment_config.json`: 实验配置快照

## 使用示例

### 基础使用

```bash
# 运行完整的多模态训练流程
python main.py --mode all
```

### 程序化使用

```python
from main import preprocess_data, train_node2vec_model, compute_enhanced_user_embeddings
from config import Config

# 1. 数据预处理
user_sequences, url_mappings, user_attributes, attribute_info, \
user_location_sequences, base_station_mappings, location_weights = preprocess_data()

# 2. 训练行为模型
behavior_model, trainer = train_node2vec_model(user_sequences, url_mappings)

# 3. 计算最终用户嵌入
enhanced_embeddings = compute_enhanced_user_embeddings(
    behavior_model, attribute_model, fusion_model, 
    user_sequences, user_attributes, url_mappings, attribute_info,
    location_model, user_location_sequences, base_station_mappings, location_weights
)
```

## 性能优化

1. **GPU加速**: 自动检测并使用GPU
2. **批处理**: 使用DataLoader进行批处理训练
3. **早停**: 基于验证集损失防止过拟合
4. **检查点**: 支持断点续训，保存最佳模型
5. **负采样**: Skip-gram模型使用负采样提高效率
6. **缓存机制**: Node2Vec随机游走支持缓存
7. **无阻塞运行**: 图片自动保存，无需手动关闭

## 实验结果示例

基于项目提供的示例数据：
- **数据规模**: 5个用户，10个域名，8个基站，7个属性
- **训练结果**: 
  - 行为向量损失: 4.16 → 4.14
  - 位置向量损失: 4.16 → 4.14  
  - 属性向量损失: 收敛到 0.0000
- **向量维度**: 128维行为 + 128维位置 + 64维属性 → 256维融合嵌入
- **训练时间**: 约1-2分钟（CPU）

## 注意事项

1. **数据格式**: 确保所有数据文件格式正确，用户ID在各文件中保持一致
2. **参数调整**: 根据数据规模调整batch_size、embedding_dim等参数
3. **模态选择**: 可根据需要选择性启用某些模态
4. **内存管理**: 大规模数据建议使用GPU并适当调整batch_size

## 故障排除

### 常见问题

1. **维度不匹配错误**: 检查config.py中各模态的嵌入维度设置
2. **数据加载失败**: 确保数据文件路径正确，格式符合要求
3. **训练中断**: 使用--resume参数从检查点恢复训练
4. **内存不足**: 减小batch_size或embedding_dim

### 调试技巧

1. **使用示例数据**: 先使用项目提供的示例数据验证流程
2. **检查日志**: 查看TensorBoard日志监控训练过程
3. **分步执行**: 使用分步模式逐步排查问题
4. **验证数据**: 确保数据预处理步骤正确完成

## 扩展功能

1. **新模态支持**: 可扩展支持更多模态（如文本、图像等）
2. **模型替换**: 支持替换为其他嵌入模型
3. **在线推理**: 可扩展为在线推荐服务
4. **增量训练**: 支持增量更新用户表示

## 许可证

MIT License 