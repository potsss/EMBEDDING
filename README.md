# Item2Vec用户表示向量训练项目

这是一个基于Item2Vec模型的用户表示向量训练项目，使用PyTorch实现，用于从用户行为数据中学习用户和物品的向量表示。**新增了用户属性向量训练功能，支持多模态用户表示学习。**

## 项目概述

本项目的目标是训练一个用户表示向量，用于表示用户的兴趣偏好。主要特点：

- 使用Item2Vec模型训练URL的向量表示
- （新增）支持Node2Vec模型，通过构建物品交互图并生成随机游走来学习物品的向量表示
- **（新增）支持用户属性向量训练，通过掩码属性预测任务学习属性嵌入**
- **（新增）支持行为向量和属性向量的融合，生成增强的用户表示**
- 用户的表示向量是用户访问的URL向量表示的加权平均（或其他聚合方式）
- 基于PyTorch实现，支持GPU加速
- 包含完整的数据预处理、模型训练、评估和可视化流程

## 数据格式

### 行为数据
输入数据应为TSV格式，包含以下字段：

```
user_id	url	timestamp_str	weight
```

- `user_id`: 用户ID
- `url`: 用户访问的URL
- `timestamp_str`: 访问时间戳 (字符串格式，如 "YYYY-MM-DD")
- `weight`: 访问权重（权重越大表示用户对该URL的兴趣越大）

### 属性数据（新增）
用户属性数据应为TSV格式，每行代表一个用户，每列代表一个属性：

```
user_id	age	gender	occupation	city	...
user1	25	男	工程师	北京	...
user2	30	女	教师	上海	...
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
├── data_preprocessing.py    # 数据预处理模块（包含属性数据处理）
├── model.py                 # 模型定义（包含属性嵌入和融合模型）
├── trainer.py               # 训练器（包含属性训练器）
├── evaluator.py             # 评估器
├── visualizer.py            # 可视化模块
├── utils.py                 # 工具函数 (例如: create_sample_data)
├── data/                    # 原始数据目录
│   ├── user_behavior.csv    # 用户行为数据
│   └── user_attributes.tsv  # 用户属性数据（新增）
├── node2vec_utils.py        # Node2Vec图构建和随机游走工具
├── utils/                   # 工具文件夹
│   ├── __init__.py
│   ├── utils.py             # 通用工具函数
│   └── node2vec_utils.py    # Node2Vec相关工具函数
└── experiments/             # 实验结果的根目录
    └── {EXPERIMENT_NAME}/    # 单次实验的目录 (例如: experiments/edu_20230101_120000)
        ├── processed_data/   # 处理后的数据
        │   ├── url_mappings.pkl        # URL映射
        │   ├── user_sequences.pkl      # 用户序列
        │   ├── user_attributes.pkl     # 处理后的属性数据（新增）
        │   ├── attribute_info.pkl      # 属性信息（新增）
        │   └── attribute_encoders.pkl  # 属性编码器（新增）
        ├── models/           # 保存的模型
        │   ├── item2vec_model.pth          # Item2Vec模型
        │   ├── node2vec_model.pth          # Node2Vec模型
        │   ├── attribute_models.pth        # 属性模型（新增）
        │   ├── best_attribute_models.pth   # 最佳属性模型（新增）
        │   ├── user_embeddings.pkl        # 基础用户嵌入
        │   └── enhanced_user_embeddings_*.pkl # 增强用户嵌入（新增）
        ├── checkpoints/      # 训练检查点
        ├── logs/             # 日志文件
        │   ├── training_curves.png         # 行为模型训练曲线
        │   └── attribute_training_curves.png # 属性训练曲线（新增）
        ├── runs/             # TensorBoard日志
        │   └── attribute_training/         # 属性训练日志（新增）
        ├── visualizations/   # 可视化结果
        └── experiment_config.json # 本次实验的配置快照
```
*注意: `{EXPERIMENT_NAME}` 会根据 `config.py` 中的 `EXPERIMENT_NAME` 以及是否已存在同名目录（可能添加时间戳）动态生成。*

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

将用户行为数据放在 `data/` 目录下，例如 `data/your_data.csv`。确保数据格式符合要求。

**如果要使用属性向量功能**，还需要准备用户属性数据文件，例如 `data/user_attributes.tsv`。

如果需要示例数据，可以运行 (假设 `utils.py` 中有 `create_sample_data` 函数):
```python
# (在Python解释器或脚本中)
# from utils.utils import create_sample_data # 修改导入路径
# create_sample_data("data/sample_user_behavior.csv")
```

### 2. 运行流程

#### 基础流程（仅行为向量）

```bash
# 运行完整流程（数据预处理 + 训练 + 评估 + 可视化 + 计算嵌入）
# 模型类型在 config.py 中设置 (Config.MODEL_TYPE)
python main.py --mode all --data_path data/your_data.csv

# 例如，如果要在 config.py 中使用 Node2Vec，请先修改该文件
# 然后运行:
# python main.py --mode all --data_path data/your_data.csv

# 指定实验名称 (可选, 否则使用config.py中的默认名称)
# python main.py --mode all --data_path data/your_data.csv --experiment_name my_custom_experiment
```

#### 增强流程（行为向量 + 属性向量）

```bash
# 启用属性向量训练的完整流程
python main.py --mode all --data_path data/your_data.csv \
    --enable_attributes --attribute_data_path data/user_attributes.tsv

# 或者在 config.py 中设置 ENABLE_ATTRIBUTES = True，然后运行：
python main.py --mode all --data_path data/your_data.csv

# 分步骤运行（属性训练会在行为模型训练完成后自动进行）
python main.py --mode preprocess --data_path data/your_data.csv --enable_attributes --attribute_data_path data/user_attributes.tsv
python main.py --mode train 
python main.py --mode visualize
python main.py --mode compute_embeddings
```

### 3. 命令行参数

- `--mode`: 运行模式
  - `preprocess`: 仅数据预处理
  - `train`: 仅模型训练
  - `visualize`: 仅结果可视化
  - `compute_embeddings`: 仅计算并保存用户嵌入向量
  - `all`: 完整流程（默认）

- `--data_path`: 原始数据文件路径 (主要用于 `preprocess` 模式，或 `all` 模式首次运行时)
- `--model_path`: 指定已训练模型的路径 (用于`visualize`, `compute_embeddings` 模式，如果不想使用默认实验路径下的模型)
- `--resume`: 从最新的检查点恢复训练 (用于 `train` 模式)
- `--no_train`: 在 `train` 或 `all` 模式中跳过训练，直接使用已有模型 (需要模型已存在或通过 `--model_path` 指定)
- `--experiment_name`: 自定义实验名称。如果提供，则结果会保存在 `experiments/YOUR_CUSTOM_NAME` 或 `experiments/YOUR_CUSTOM_NAME_TIMESTAMP` 下。
- `--no_cache`: 禁用随机游走缓存
- `--force_regenerate`: 强制重新生成随机游走（忽略缓存）
- **`--enable_attributes`**: 启用属性向量训练（新增）
- **`--attribute_data_path`**: 用户属性数据文件路径（新增）

### 4. 配置参数

在 `config.py` 中可以调整各种参数：

```python
# 实验配置
EXPERIMENT_NAME = "edu"   # 默认实验名称

# 数据相关配置
DATA_PATH = "data/edu.csv" # 默认原始数据路径 (会被命令行参数覆盖)
ATTRIBUTE_DATA_PATH = "data/user_attributes.tsv" # 用户属性数据路径（新增）

# 模型相关配置
MODEL_TYPE = "item2vec"   # 或 "node2vec"，在此处修改模型类型
EMBEDDING_DIM = 128        # 嵌入维度
WINDOW_SIZE = 5           # 上下文窗口大小 (Item2Vec 和 Node2Vec SkipGram)
MIN_COUNT = 5             # 用户序列最小长度 (用于data_preprocessing.py)
NEGATIVE_SAMPLES = 5      # 负采样数量 (Item2Vec 和 Node2Vec SkipGram)

# 属性相关配置（新增）
ENABLE_ATTRIBUTES = False  # 是否启用属性向量训练
ATTRIBUTE_EMBEDDING_DIM = 64  # 属性嵌入维度
FUSION_HIDDEN_DIM = 256  # 融合层隐藏维度
FINAL_USER_EMBEDDING_DIM = 256  # 最终用户嵌入维度

# 属性训练相关配置（新增）
ATTRIBUTE_LEARNING_RATE = 0.001  # 属性训练学习率
ATTRIBUTE_EPOCHS = 50  # 属性训练轮次
ATTRIBUTE_BATCH_SIZE = 512  # 属性训练批次大小
MASKING_RATIO = 0.15  # 掩码比例
ATTRIBUTE_EARLY_STOPPING_PATIENCE = 10  # 属性训练早停耐心值

# 数值属性处理配置（新增）
NUMERICAL_STANDARDIZATION = True  # 是否对数值属性进行标准化
CATEGORICAL_MIN_FREQ = 5  # 类别属性最小频次（低频类别会被归为'其他'）

# Node2Vec 特定参数 (config.py)
P_PARAM = 1.0             # Node2Vec 返回参数 p
Q_PARAM = 1.0             # Node2Vec 进出参数 q
WALK_LENGTH = 80          # Node2Vec 随机游走长度
NUM_WALKS = 10            # Node2Vec 每个节点的游走次数

# 训练参数
LEARNING_RATE = 0.001     # 学习率
EPOCHS = 100              # 训练轮次
BATCH_SIZE = 1024         # 批次大小
EARLY_STOPPING_PATIENCE = 50  # 早停耐心值 (trainer.py中实际是10, config.py中是50, 以config.py为准)

# 评估相关配置
EVAL_INTERVAL = 2         # 验证和保存模型的epoch间隔
TOP_K = 10                # 推荐评估时的Top K

# 随机种子
RANDOM_SEED = 42
```

## 功能模块

### 数据预处理 (`data_preprocessing.py`)

- 数据清洗：删除缺失值、重复行，处理无效权重
- URL处理：提取domain，创建ID映射
- 序列构建：根据权重和时间戳构建用户访问序列，并按最小长度过滤

### Node2Vec 工具 (`utils/node2vec_utils.py`)

- `build_graph_from_sequences`: 从用户行为序列构建物品-物品交互图。
- `generate_node2vec_walks`: 在构建的图上根据 p 和 q 参数生成带偏向的随机游走序列。
- 生成随机游走图会比较花时间，可以控制超参数 `NUM_WALKS` 和 `WALK_LENGTH` 来减少时间。

### 通用工具 (`utils/utils.py`)

- 包含日志设置、文件读写、示例数据生成等多种辅助函数。

### 模型定义 (`model.py`)

- `Item2Vec`: 基于Skip-gram的物品嵌入模型，实现负采样损失，直接在用户序列上训练。
- `Node2Vec`: 同样基于Skip-gram的物品嵌入模型，但在`utils.node2vec_utils.py`生成的随机游走序列上训练。
- `UserEmbedding`: 用户嵌入计算类，支持多种聚合策略（默认为均值聚合）。

### 训练器 (`trainer.py`)

- 支持早停策略 (`EarlyStopping`)
- 自动保存最新和最佳检查点 (`save_checkpoint`)
- TensorBoard日志记录训练/验证损失和学习率
- 训练和验证损失曲线可视化并保存 (`plot_training_curves`)
- 学习率调度 (`ReduceLROnPlateau`)



### 可视化器 (`visualizer.py`)

- 用户嵌入向量 t-SNE 可视化
- 物品嵌入向量 t-SNE 可视化
- 只是直观展示一下结果，可以采用其他的方式来评估，这个步骤同样可以不执行。

## 输出结果

所有输出默认保存在 `experiments/{EXPERIMENT_NAME}/` 目录下 (如果指定了实验名或使用了带时间戳的目录)。

### 1. 模型与数据文件
- `experiments/{EXPERIMENT_NAME}/models/item2vec_model.pth`: 训练好的Item2Vec模型参数。
- `experiments/{EXPERIMENT_NAME}/models/node2vec_model.pth`: 训练好的Node2Vec模型参数。
- `experiments/{EXPERIMENT_NAME}/models/attribute_models.pth`: 训练好的属性模型参数（新增）。
- `experiments/{EXPERIMENT_NAME}/models/best_attribute_models.pth`: 最佳属性模型参数（新增）。
- `experiments/{EXPERIMENT_NAME}/models/user_embeddings.pkl`: 基于Item2Vec计算出的用户嵌入向量。
- `experiments/{EXPERIMENT_NAME}/models/user_embeddings_node2vec.pkl`: 基于Node2Vec计算出的用户嵌入向量。
- `experiments/{EXPERIMENT_NAME}/models/enhanced_user_embeddings_*.pkl`: 增强用户嵌入向量（行为+属性）（新增）。
- `experiments/{EXPERIMENT_NAME}/processed_data/url_mappings.pkl`: URL到ID的映射。
- `experiments/{EXPERIMENT_NAME}/processed_data/user_sequences.pkl`: 处理后的用户行为序列。
- `experiments/{EXPERIMENT_NAME}/processed_data/user_attributes.pkl`: 处理后的用户属性数据（新增）。
- `experiments/{EXPERIMENT_NAME}/processed_data/attribute_info.pkl`: 属性信息（新增）。
- `experiments/{EXPERIMENT_NAME}/processed_data/attribute_encoders.pkl`: 属性编码器（新增）。
- `experiments/{EXPERIMENT_NAME}/checkpoints/latest_checkpoint.pth`: 最新训练检查点。
- `experiments/{EXPERIMENT_NAME}/checkpoints/best_model.pth`: 验证集上表现最佳的模型检查点。

### 2. 可视化结果
- `experiments/{EXPERIMENT_NAME}/visualizations/`:
  - `user_embedding_tsne_visualization.png`: 用户嵌入t-SNE图。
  - `item_embedding_tsne_visualization.png`: 物品嵌入t-SNE图。
- `experiments/{EXPERIMENT_NAME}/logs/training_curves.png`: 训练和验证损失曲线图。
- `experiments/{EXPERIMENT_NAME}/logs/attribute_training_curves.png`: 属性训练损失曲线图（新增）。

### 3. 日志与配置
- `experiments/{EXPERIMENT_NAME}/runs/`: TensorBoard日志，用于更详细的训练过程监控。
- `experiments/{EXPERIMENT_NAME}/runs/attribute_training/`: 属性训练TensorBoard日志（新增）。
- `experiments/{EXPERIMENT_NAME}/experiment_config.json`: 本次实验运行时的详细配置信息。

## 使用示例

### 训练模型并获取用户嵌入

#### 基础用户嵌入（仅行为数据）

```python
# 确保在项目根目录下，并且相关模块可以导入
# 以下代码片段假设在 main.py 的上下文中或已正确设置环境

# 假设已通过 main.py 的 'preprocess' 或 'all' 模式生成了数据
# from main import preprocess_data, train_model, compute_user_embeddings, initialize_experiment_paths, create_directories
# from config import Config
# import os

# # 0. 初始化路径 (实际项目中这由main函数在开始时处理)
# initialize_experiment_paths() # 使用默认实验名
# create_directories() # 创建目录

# # 1. 数据预处理 (如果尚未处理)
# # user_sequences, url_mappings = preprocess_data(Config.DATA_PATH)
# # print("数据预处理完成。")

# # 2. 加载已处理数据 (如果已处理过)
# from data_preprocessing import DataPreprocessor
# preprocessor = DataPreprocessor()
# user_sequences = preprocessor.load_processed_data()
# url_mappings = {'url_to_id': preprocessor.url_to_id, 'id_to_url': preprocessor.id_to_url}
# print("已加载处理数据。")

# # 3. 训练模型并计算用户嵌入
# user_embeddings_filename = 'user_embeddings.pkl' if Config.MODEL_TYPE == 'item2vec' else 'user_embeddings_node2vec.pkl'
# user_embeddings_path = os.path.join(Config.MODEL_SAVE_PATH, user_embeddings_filename)
# compute_user_embeddings(model, user_sequences, url_mappings, user_embeddings_path)
```

#### 增强用户嵌入（行为数据 + 属性数据）

```python
# 启用属性功能
# Config.ENABLE_ATTRIBUTES = True

# # 1. 处理属性数据
# user_sequences, url_mappings, user_attributes, attribute_info = preprocess_data(Config.DATA_PATH)

# # 2. 训练行为模型
# if Config.MODEL_TYPE == 'item2vec':
#     model, trainer = train_model(user_sequences, url_mappings)
# elif Config.MODEL_TYPE == 'node2vec':
#     model, trainer = train_node2vec_model(user_sequences, url_mappings)

# # 3. 训练属性模型
# attribute_model, fusion_model = train_attribute_models(
#     model, user_sequences, user_attributes, attribute_info, url_mappings
# )

# # 4. 计算增强用户嵌入
# enhanced_embeddings_filename = f'enhanced_user_embeddings_{Config.MODEL_TYPE}.pkl'
# enhanced_embeddings_path = os.path.join(Config.MODEL_SAVE_PATH, enhanced_embeddings_filename)
# enhanced_embeddings = compute_enhanced_user_embeddings(
#     model, attribute_model, fusion_model, user_sequences, user_attributes, 
#     url_mappings, attribute_info, enhanced_embeddings_path
# )
```

### 获取物品推荐 (相似物品)

```python
# 假设 `loaded_model` 是已加载的 Item2Vec 模型
# `url_mappings` 包含 'id_to_url'

# 假设物品ID 0 存在
target_item_id = 0
if target_item_id < loaded_model.vocab_size:
    similar_items_indices, scores = loaded_model.get_similar_items(item_id=target_item_id, top_k=10)

    print(f"与物品ID {target_item_id} ({url_mappings['id_to_url'].get(target_item_id, '未知物品')}) 最相似的物品:")
    for item_idx, score in zip(similar_items_indices, scores):
        domain = url_mappings['id_to_url'].get(item_idx, '未知物品')
        print(f"  物品ID {item_idx} ({domain}): {score:.4f}")
else:
    print(f"物品ID {target_item_id} 超出词汇表范围。")
```

### 可视化结果

```python
# 假设 `loaded_model`, `user_sequences`, `url_mappings` 已准备好
# from visualizer import Visualizer
# from config import Config # Visualizer 会使用Config中的路径

# visualizer = Visualizer(loaded_model, user_sequences, url_mappings, Config)

# # 可视化物品嵌入 (使用默认参数)
# visualizer.visualize_item_embeddings(sample_size=500) # 可以指定采样大小等参数

# # 可视化用户嵌入 (使用默认参数)
# visualizer.visualize_user_embeddings(sample_size=500)
```
*可视化通常在 `main.py` 的 `visualize` 或 `all` 模式下自动执行。*

## 性能优化

1. **GPU加速**: 自动检测并使用GPU (`Config.DEVICE`)。
2. **批处理**: 使用 `DataLoader` 进行批处理训练。
3. **早停**: 防止过拟合，基于验证集损失 (`EarlyStopping`)。
4. **检查点**: 支持断点续训，保存最佳模型。
5. **负采样**: Skip-gram模型训练时使用负采样提高效率。

## 注意事项

1. 确保输入数据格式正确（TSV格式，包含`user_id`, `url`, `timestamp_str`, `weight`字段）。
2. 根据数据规模和硬件资源调整 `config.py` 中的参数，如 `BATCH_SIZE`, `EMBEDDING_DIM`。
3. 对于大规模数据集，强烈建议使用GPU进行训练。
4. 可视化模块（尤其是t-SNE）在数据量大时可能消耗较多时间和内存，默认会进行采样。

## 故障排除

### 常见问题
1. **内存不足 (Out of Memory)**:
    - 减小 `config.py` 中的 `BATCH_SIZE`。
    - 如果是在可视化阶段，减小 `visualize_item_embeddings` 或 `visualize_user_embeddings` 函数调用时的 `sample_size`。
    - 减小 `EMBEDDING_DIM`。
2. **训练缓慢**:
    - 确认 `Config.DEVICE` 是否已正确设置为 `cuda` 并且PyTorch能够访问到GPU。

3. **`FileNotFoundError`**:
    - 检查 `config.py` 中的 `DATA_PATH` 或命令行传入的 `--data_path` 是否正确。
    - 确保在分步执行时，前一步骤已成功生成所需文件（例如，`train` 模式需要预处理后的数据）。
    - 检查 `EXPERIMENT_NAME` 和相关的实验目录是否正确。
4. **KeyError (例如，在映射中找不到ID)**:
    - 确保数据预处理流程完整运行且没有错误。
    - 检查 `url_mappings.pkl` 是否正确生成和加载。

### 调试技巧
1. **使用小数据集测试**: 截取原始数据的一小部分进行测试，以便快速定位问题。
2. **打印日志和变量**: 在关键步骤添加 `print` 语句，检查中间变量的状态。
3. **使用TensorBoard**: 运行 `tensorboard --logdir experiments/{EXPERIMENT_NAME}/runs` 来监控训练过程中的损失和学习率。
4. **逐步执行**: 使用 `main.py` 的分步模式 (`preprocess`, `train` 等) 来隔离问题。
5. **检查配置文件**: 仔细核对 `config.py` 中的各项参数设置。

## 扩展功能

项目设计考虑了一定的可扩展性：

1. **多种嵌入模型**: `model.py` 中的 `Item2Vec` 可以被替换或扩展为其他序列嵌入算法 (如GRU4Rec, BERT4Rec等，但这需要较大改动)。本项目已扩展支持 `Node2Vec`。
2. **自定义聚合策略**: `UserEmbedding` 类可以方便地添加新的用户序列聚合方法。
3. **自定义评估指标**: `evaluator.py` 可以添加更多针对特定业务场景的评估指标。
4. **在线推理**: 当前项目主要关注离线训练和评估。要部署为在线推荐服务，需要额外开发服务接口和模型加载、实时预测逻辑。
5. **增量训练**: 当前训练流程是批量的。实现增量训练需要修改 `Trainer` 以支持从现有模型状态更新，并处理新数据。


## 许可证

MIT License 