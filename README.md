# 多模态用户表示向量训练项目

这是一个基于多模态学习的用户表示向量训练项目，使用PyTorch实现，支持从用户行为数据、位置数据和属性数据中学习用户的综合向量表示。**项目支持三种向量训练：行为向量、位置向量和属性向量，并通过多模态融合生成最终的用户表示。同时支持为新用户计算向量表示。**

## 🌟 项目特色

### 🚀 核心功能
- **三模态融合**：行为 + 位置 + 属性的综合用户表示
- **新用户推理**：无需重新训练即可为新用户计算向量
- **灵活配置**：支持单独或组合使用任意模态
- **高性能训练**：GPU加速、断点续训、早停策略

### 📊 支持的模态
1. **行为向量**：基于用户访问序列的兴趣表示（Item2Vec/Node2Vec）
2. **位置向量**：基于基站连接数据的位置偏好表示
3. **属性向量**：基于用户画像的属性嵌入表示

### 🔧 技术亮点
- **独立训练策略**：避免模态间相互干扰
- **多模态融合网络**：自适应权重融合
- **新用户冷启动**：支持新用户向量计算
- **完整工作流**：数据预处理→模型训练→向量计算→结果可视化

## 📁 项目结构

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

## 📋 数据格式

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

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd v_bushu

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

#### 1. 完整训练流程
```bash
# 运行完整的训练和评估流程
python main.py --mode all --experiment_name my_experiment

# 或者分步执行
python main.py --mode preprocess    # 数据预处理
python main.py --mode train        # 模型训练
python main.py --mode compute_embeddings  # 计算用户向量
python main.py --mode visualize    # 结果可视化
```

#### 2. 新用户向量计算
```bash
# 为新用户计算向量（使用已训练的模型）
python main.py --mode compute_new_users --experiment_name my_experiment

# 使用独立脚本
python compute_new_users.py --experiment_path experiments/my_experiment
```

#### 3. 新用户推理示例
```bash
# 运行新用户推理示例，查看相似度分析
python example_new_user_inference.py
```

### 配置说明

在 `config.py` 中可以配置：

```python
class Config:
    # 实验配置
    EXPERIMENT_NAME = "user_embedding_experiment"
    
    # 模态开关
    ENABLE_ATTRIBUTES = True    # 启用属性向量
    ENABLE_LOCATION = True      # 启用位置向量
    
    # 模型参数
    MODEL_TYPE = "node2vec"     # 行为模型类型：item2vec/node2vec
    EMBEDDING_DIM = 128         # 行为向量维度
    ATTRIBUTE_EMBEDDING_DIM = 64    # 属性向量维度
    LOCATION_EMBEDDING_DIM = 128    # 位置向量维度
    FINAL_USER_EMBEDDING_DIM = 256  # 最终融合向量维度
    
    # 数据路径
    DATA_PATH = "data/test_user_behavior.csv"
    ATTRIBUTE_DATA_PATH = "data/sample_user_attributes.tsv"
    LOCATION_DATA_PATH = "data/sample_user_base_stations.tsv"
    
    # 新用户数据路径
    NEW_USER_BEHAVIOR_PATH = "data/new_user_behavior.csv"
    NEW_USER_ATTRIBUTE_PATH = "data/new_user_attributes.tsv"
    NEW_USER_LOCATION_PATH = "data/new_user_base_stations.tsv"
```

## 🔧 高级功能

### 新用户向量计算

项目支持为新用户（未参与训练的用户）计算向量表示：

```python
# 使用训练好的模型为新用户计算向量
new_user_embeddings = compute_new_user_embeddings(
    behavior_model=trained_behavior_model,
    attribute_model=trained_attribute_model,
    fusion_model=trained_fusion_model,
    # ... 其他参数
)
```

### 多模态融合策略

支持灵活的模态组合：
- **仅行为向量**：传统的协同过滤方法
- **行为+属性**：增强的用户表示
- **行为+位置**：地理感知的推荐
- **三模态融合**：最完整的用户表示

### 相似度分析

```python
# 计算新用户与训练用户的相似度
from example_new_user_inference import analyze_user_similarity

similarities = analyze_user_similarity(
    new_user_embeddings, 
    training_user_embeddings
)
```

## 📊 实验结果示例

基于项目提供的示例数据：
- **数据规模**: 5个训练用户，3个新用户，10个域名，8个基站，7个属性
- **训练结果**: 
  - 行为向量损失: 4.16 → 4.14
  - 位置向量损失: 4.16 → 4.14  
  - 属性向量损失: 收敛到 0.0000
- **向量维度**: 128维行为 + 128维位置 + 64维属性 → 256维融合嵌入
- **训练时间**: 约1-2分钟（CPU）

## 🛠️ 性能优化

1. **GPU加速**: 自动检测并使用GPU
2. **批处理**: 使用DataLoader进行批处理训练
3. **早停**: 基于验证集损失防止过拟合
4. **检查点**: 支持断点续训，保存最佳模型
5. **负采样**: Skip-gram模型使用负采样提高效率
6. **缓存机制**: Node2Vec随机游走支持缓存
7. **无阻塞运行**: 图片自动保存，无需手动关闭

## 🔍 API 参考

### 主要函数

```python
# 数据预处理
user_sequences, url_mappings, user_attributes, attribute_info, \
user_location_sequences, base_station_mappings, location_weights = preprocess_data()

# 训练行为模型
behavior_model, trainer = train_node2vec_model(user_sequences, url_mappings)

# 训练位置模型
location_model, base_station_mappings = train_location_model(Config, location_processor)

# 计算最终用户嵌入
enhanced_embeddings = compute_enhanced_user_embeddings(
    behavior_model, attribute_model, fusion_model, 
    user_sequences, user_attributes, url_mappings, attribute_info,
    location_model, user_location_sequences, base_station_mappings, location_weights
)

# 计算新用户向量
new_user_embeddings = compute_new_user_embeddings(
    behavior_model, attribute_model, fusion_model,
    url_mappings, attribute_info, base_station_mappings,
    location_model, location_processor
)
```

## ⚠️ 注意事项

1. **数据格式**: 确保所有数据文件格式正确，用户ID在各文件中保持一致
2. **参数调整**: 根据数据规模调整batch_size、embedding_dim等参数
3. **模态选择**: 可根据需要选择性启用某些模态
4. **内存管理**: 大规模数据建议使用GPU并适当调整batch_size
5. **新用户数据**: 新用户访问的URL必须在训练数据中出现过

## 🐛 故障排除

### 常见问题

1. **维度不匹配错误**: 检查config.py中各模态的嵌入维度设置
2. **新用户URL未知**: 确保新用户访问的URL在训练数据中存在
3. **属性编码错误**: 确保新用户属性值在训练数据的取值范围内
4. **位置数据不足**: 检查用户的基站连接数是否满足最小连接数要求

### 解决方案

```bash
# 检查数据格式
python -c "import pandas as pd; print(pd.read_csv('data/test_user_behavior.csv').head())"

# 验证模型加载
python -c "from main import load_trained_models; models = load_trained_models('experiments/your_experiment')"

# 调试新用户计算
python compute_new_users.py --experiment_path experiments/your_experiment --debug
```

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 联系方式

如有问题或建议，请通过Issue与我们联系。 