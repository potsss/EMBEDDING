# Item2Vec用户表示向量训练项目

这是一个基于Item2Vec模型的用户表示向量训练项目，使用PyTorch实现，用于从用户行为数据中学习用户和物品的向量表示。

## 项目概述

本项目的目标是训练一个用户表示向量，用于表示用户的兴趣偏好。主要特点：

- 使用Item2Vec模型训练URL的向量表示
- 用户的表示向量是用户访问的URL向量表示的加权平均（或其他聚合方式）
- 基于PyTorch实现，支持GPU加速
- 包含完整的数据预处理、模型训练、评估和可视化流程

## 数据格式

输入数据应为TSV格式，包含以下字段：

```
user_id	url	timestamp_str	weight
```

- `user_id`: 用户ID
- `url`: 用户访问的URL
- `timestamp_str`: 访问时间戳 (字符串格式，如 "YYYY-MM-DD")
- `weight`: 访问权重（权重越大表示用户对该URL的兴趣越大）

## 项目结构

```
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── config.py                # 配置文件
├── main.py                  # 主程序入口
├── data_preprocessing.py    # 数据预处理模块
├── model.py                 # 模型定义
├── trainer.py               # 训练器
├── evaluator.py             # 评估器
├── visualizer.py            # 可视化模块
├── utils.py                 # 工具函数 (例如: create_sample_data)
├── data/                    # 原始数据目录 (例如: data/user_behavior.csv)
└── experiments/             # 实验结果的根目录
    └── {EXPERIMENT_NAME}/    # 单次实验的目录 (例如: experiments/edu_20230101_120000)
        ├── processed_data/   # 处理后的数据 (url_mappings.pkl, user_sequences.pkl)
        ├── models/           # 保存的模型 (item2vec_model.pth, user_embeddings.pkl)
        ├── checkpoints/      # 训练检查点 (latest_checkpoint.pth, best_model.pth)
        ├── logs/             # 日志文件 (training_curves.png)
        ├── runs/             # TensorBoard日志
        ├── visualizations/   # 可视化结果 (tsne_plots.png)
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

如果需要示例数据，可以运行 (假设 `utils.py` 中有 `create_sample_data` 函数):
```python
# (在Python解释器中或脚本中)
# from utils import create_sample_data
# create_sample_data("data/sample_user_behavior.csv")
```

### 2. 运行流程

```bash
# 运行完整流程（数据预处理 + 训练 + 评估 + 可视化 + 计算嵌入）
# 需提供原始数据路径
python main.py --mode all --data_path data/your_data.csv

# 指定实验名称 (可选, 否则使用config.py中的默认名称)
python main.py --mode all --data_path data/your_data.csv --experiment_name my_custom_experiment

# 或者分步骤运行 (后续步骤会自动查找默认实验路径下的数据和模型)
python main.py --mode preprocess --data_path data/your_data.csv
python main.py --mode train
python main.py --mode evaluate
python main.py --mode visualize
python main.py --mode compute_embeddings
```

### 3. 命令行参数

- `--mode`: 运行模式
  - `preprocess`: 仅数据预处理
  - `train`: 仅模型训练
  - `evaluate`: 仅模型评估
  - `visualize`: 仅结果可视化
  - `compute_embeddings`: 仅计算并保存用户嵌入向量
  - `all`: 完整流程（默认）

- `--data_path`: 原始数据文件路径 (主要用于 `preprocess` 模式，或 `all` 模式首次运行时)
- `--model_path`: 指定已训练模型的路径 (用于 `evaluate`, `visualize`, `compute_embeddings` 模式，如果不想使用默认实验路径下的模型)
- `--resume`: 从最新的检查点恢复训练 (用于 `train` 模式)
- `--no_train`: 在 `train` 或 `all` 模式中跳过训练，直接使用已有模型 (需要模型已存在或通过 `--model_path` 指定)
- `--experiment_name`: 自定义实验名称。如果提供，则结果会保存在 `experiments/YOUR_CUSTOM_NAME` 或 `experiments/YOUR_CUSTOM_NAME_TIMESTAMP` 下。

### 4. 配置参数

在 `config.py` 中可以调整各种参数：

```python
# 实验配置
EXPERIMENT_NAME = "edu"   # 默认实验名称

# 数据相关配置
DATA_PATH = "data/edu.csv" # 默认原始数据路径 (会被命令行参数覆盖)

# 模型相关配置
EMBEDDING_DIM = 128        # 嵌入维度
WINDOW_SIZE = 5           # 上下文窗口大小
MIN_COUNT = 5             # 用户序列最小长度 (用于data_preprocessing.py)
NEGATIVE_SAMPLES = 5      # 负采样数量

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

### 模型定义 (`model.py`)

- `Item2Vec`: 基于Skip-gram的物品嵌入模型，实现负采样损失。
- `UserEmbedding`: 用户嵌入计算类，支持多种聚合策略（默认为均值聚合）。

### 训练器 (`trainer.py`)

- 支持早停策略 (`EarlyStopping`)
- 自动保存最新和最佳检查点 (`save_checkpoint`)
- TensorBoard日志记录训练/验证损失和学习率
- 训练和验证损失曲线可视化并保存 (`plot_training_curves`)
- 学习率调度 (`ReduceLROnPlateau`)

### 评估器 (`evaluator.py`)

- 推荐性能评估（Hit Rate@K, NDCG@K）
- 物品相似度评估（平均/标准差/最小/最大相似度）
- 用户间相似度评估（平均/标准差/最小/最大相似度）
- 嵌入向量质量评估（平均/标准差范数，平均/标准差向量间距离）

### 可视化器 (`visualizer.py`)

- 用户嵌入向量 t-SNE 可视化
- 物品嵌入向量 t-SNE 可视化

## 输出结果

所有输出默认保存在 `experiments/{EXPERIMENT_NAME}/` 目录下 (如果指定了实验名或使用了带时间戳的目录)。

### 1. 模型与数据文件
- `experiments/{EXPERIMENT_NAME}/models/item2vec_model.pth`: 训练好的Item2Vec模型参数。
- `experiments/{EXPERIMENT_NAME}/models/user_embeddings.pkl`: 计算出的用户嵌入向量。
- `experiments/{EXPERIMENT_NAME}/processed_data/url_mappings.pkl`: URL到ID的映射。
- `experiments/{EXPERIMENT_NAME}/processed_data/user_sequences.pkl`: 处理后的用户行为序列。
- `experiments/{EXPERIMENT_NAME}/checkpoints/latest_checkpoint.pth`: 最新训练检查点。
- `experiments/{EXPERIMENT_NAME}/checkpoints/best_model.pth`: 验证集上表现最佳的模型检查点。

### 2. 可视化结果
- `experiments/{EXPERIMENT_NAME}/visualizations/`:
  - `user_embedding_tsne_visualization.png`: 用户嵌入t-SNE图。
  - `item_embedding_tsne_visualization.png`: 物品嵌入t-SNE图。
- `experiments/{EXPERIMENT_NAME}/logs/training_curves.png`: 训练和验证损失曲线图。

### 3. 日志与配置
- `experiments/{EXPERIMENT_NAME}/runs/`: TensorBoard日志，用于更详细的训练过程监控。
- `experiments/{EXPERIMENT_NAME}/experiment_config.json`: 本次实验运行时的详细配置信息。

## 使用示例

### 训练模型并获取用户嵌入

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


# # 3. 训练模型 (如果尚未训练)
# # vocab_size = len(url_mappings['url_to_id'])
# # from model import Item2Vec
# # item2vec_model = Item2Vec(vocab_size, Config.EMBEDDING_DIM)
# # from trainer import Trainer
# # trainer_instance = Trainer(item2vec_model)
# # trainer_instance.train(user_sequences)
# # print("模型训练完成。")

# # 4. 加载已训练的模型
# from main import load_trained_model # main.py中的辅助函数
# vocab_size = len(url_mappings['url_to_id'])
# model_load_path = os.path.join(Config.MODEL_SAVE_PATH, 'item2vec_model.pth')
# loaded_model = load_trained_model(model_load_path, vocab_size)
# if loaded_model is None:
#     print("模型加载失败，请检查路径和预处理步骤。")
# else:
#     print("模型加载成功。")

#     # 5. 计算用户嵌入
#     # user_embeddings = compute_user_embeddings(loaded_model, user_sequences, url_mappings) # compute_user_embeddings 在 main.py
#     # print(f"计算了 {len(user_embeddings)} 个用户嵌入。")

#     # 或者直接使用 UserEmbedding 类获取相似用户
#     from model import UserEmbedding
#     user_emb_calculator = UserEmbedding(loaded_model, user_sequences, url_mappings)
#     all_user_embeddings = user_emb_calculator.compute_user_embeddings() # 计算并存储在实例中
    
#     # 假设我们想找 'some_user_id' 的相似用户 (确保该ID存在于user_sequences.keys())
#     target_user_id = list(user_sequences.keys())[0] if user_sequences else None
#     if target_user_id:
#         similar_users, scores = user_emb_calculator.get_similar_users(target_user_id, top_k=5)
#         print(f"与用户 {target_user_id} 最相似的用户: {similar_users}")
#         print(f"相似度分数: {scores}")
#     else:
#         print("没有用户序列数据，无法获取相似用户。")

```
*上述代码片段为演示目的，实际使用时建议通过 `main.py` 的命令行接口运行完整流程。*

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

## 评估指标

项目通过 `evaluator.py` 计算以下几类指标：

- **推荐性能**: Hit Rate@K, NDCG@K
- **物品相似度质量**: 物品嵌入间平均/标准差/最小/最大余弦相似度
- **用户聚类质量 (用户间相似度)**: 用户嵌入间平均/标准差/最小/最大余弦相似度
- **嵌入质量**: 物品嵌入向量的平均/标准差范数，平均/标准差向量间距离

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
    - 尝试调整 `LEARNING_RATE`。
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
4. **逐步执行**: 使用 `main.py` 的分步模式 (`preprocess`, `train`, `evaluate` 等) 来隔离问题。
5. **检查配置文件**: 仔细核对 `config.py` 中的各项参数设置。

## 扩展功能

项目设计考虑了一定的可扩展性：

1. **多种嵌入模型**: `model.py` 中的 `Item2Vec` 可以被替换或扩展为其他序列嵌入算法 (如GRU4Rec, BERT4Rec等，但这需要较大改动)。
2. **自定义聚合策略**: `UserEmbedding` 类可以方便地添加新的用户序列聚合方法。
3. **自定义评估指标**: `evaluator.py` 可以添加更多针对特定业务场景的评估指标。
4. **在线推理**: 当前项目主要关注离线训练和评估。要部署为在线推荐服务，需要额外开发服务接口和模型加载、实时预测逻辑。
5. **增量训练**: 当前训练流程是批量的。实现增量训练需要修改 `Trainer` 以支持从现有模型状态更新，并处理新数据。


## 许可证

MIT License 