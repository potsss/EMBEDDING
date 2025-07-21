# 多模态用户嵌入表示学习项目

本项目实现了一个基于多模态数据的用户嵌入表示学习系统，结合用户行为序列（基于Node2Vec/Item2Vec）、用户属性和位置信息，生成综合的用户向量表示。

## 🚀 项目特性

- **多模态融合**: 结合行为、属性、位置三种模态的用户表示
- **灵活的模型选择**: 支持Node2Vec和Item2Vec两种序列建模方法
- **智能数据过滤**: 自动过滤新用户数据中不存在于训练集的实体
- **兼容性检查**: 自动生成数据兼容性报告，评估数据质量
- **完整的训练流程**: 从数据预处理到模型训练的端到端解决方案
- **新用户推理**: 支持为新用户计算向量表示
- **实验管理**: 完整的实验配置和结果管理系统

## 📁 项目结构

```
embedding项目/
├── main.py                    # 主程序入口
├── config.py                  # 配置文件
├── data_preprocessing.py      # 数据预处理模块
├── model.py                   # 模型定义
├── trainer.py                 # 训练器
├── compute_new_users.py       # 新用户向量计算（独立脚本）
├── check_new_user_data.py     # 新用户数据兼容性检查工具
├── train_label_discriminator.py  # 标签判别器训练
├── requirements.txt           # 依赖包列表
├── README.md                  # 项目文档
├── data/                      # 数据目录
│   ├── user_behavior.csv      # 用户行为数据
│   ├── user_attributes.tsv    # 用户属性数据
│   ├── user_base_stations.tsv # 用户位置数据
│   ├── new_user_behavior.csv  # 新用户行为数据
│   ├── new_user_attributes.tsv # 新用户属性数据
│   └── new_user_base_stations.tsv # 新用户位置数据
└── experiments/               # 实验结果目录
    └── {experiment_name}/     # 具体实验目录
        ├── processed_data/    # 预处理后的数据
        │   ├── training_entities.pkl # 训练实体记录
        │   └── new_user_compatibility_report.json # 兼容性报告
        ├── models/           # 保存的模型
        ├── checkpoints/      # 训练检查点
        ├── logs/            # 训练日志
        └── visualizations/  # 可视化结果
```

## 🛠️ 安装和环境设置

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- pandas, numpy, scikit-learn
- sentence-transformers（用于文本嵌入）

### 安装依赖
```bash
pip install -r requirements.txt
```

## 📊 数据格式

### 用户行为数据 (user_behavior.csv)
```csv
user_id,timestamp,url,duration
user_001,2023-01-01 10:00:00,example.com,120
user_001,2023-01-01 10:05:00,news.com,300
```

### 用户属性数据 (user_attributes.tsv)
```tsv
user_id	age	gender	city	income_level	education	occupation	device_type
user_001	25	M	Beijing	high	bachelor	engineer	mobile
```

### 用户位置数据 (user_base_stations.tsv)
```tsv
user_id	timestamp	base_station_id	duration
user_001	2023-01-01 10:00:00	BS_001	1800
```

## 🚀 使用方法

### 1. 数据预处理
```bash
python main.py --mode preprocess --experiment_name my_experiment
```

### 2. 模型训练
```bash
# 完整训练（行为+属性+位置）
python main.py --mode train --experiment_name my_experiment

# 仅行为模型
python main.py --mode train --experiment_name my_experiment --disable_attributes --disable_location
```

### 3. 新用户向量计算

#### 方法1: 使用主程序（推荐）
```bash
python main.py --mode compute_new_users --experiment_name my_experiment
```

#### 方法2: 使用独立脚本
```bash
python compute_new_users.py --experiment_path experiments/my_experiment
```

**🆕 新增功能**: 在计算新用户向量时，系统会：
1. **自动过滤数据**: 基于训练时保存的实体记录，过滤掉新用户数据中不存在的URL和基站
2. **生成兼容性报告**: 自动生成详细的数据兼容性报告并保存为JSON文件
3. **提供过滤统计**: 显示过滤了多少记录，哪些实体是未知的

#### 兼容性报告内容
报告保存在 `experiments/{experiment_name}/processed_data/new_user_compatibility_report.json`，包含：
- **实验信息**: 数据路径、训练实体数量
- **处理结果统计**: 成功处理的用户数量、平均序列长度、数据覆盖率
- **行为数据分析**: URL过滤统计、未知URL列表、覆盖率
- **位置数据分析**: 基站过滤统计、未知基站列表、覆盖率
- **兼容性评估**: 总体兼容性评分、评级和改进建议

### 4. 数据兼容性检查工具

如需单独检查新用户数据的兼容性，可使用专用工具：

```bash
# 检查默认数据文件
python check_new_user_data.py --experiment_name my_experiment

# 检查自定义数据文件
python check_new_user_data.py --experiment_name my_experiment \
    --new_user_behavior_path data/custom_behavior.csv \
    --new_user_location_path data/custom_location.tsv
```

该工具会生成详细的兼容性报告，包括：
- 数据覆盖率分析
- 未知实体列表
- 用户级别统计
- 总体兼容性评分和评级
- 具体的改进建议

## ⚙️ 配置参数

主要配置参数在 `config.py` 中：

```python
# 模型配置
MODEL_TYPE = 'node2vec'  # 'node2vec' 或 'item2vec'
EMBEDDING_DIM = 128
FINAL_USER_EMBEDDING_DIM = 256

# 训练配置  
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 256

# 多模态配置
ENABLE_ATTRIBUTES = True
ENABLE_LOCATION = True

# Node2Vec参数
NUM_WALKS = 4
WALK_LENGTH = 20
P_PARAM = 1.0
Q_PARAM = 1.0
```

## 📈 模型架构

### 1. 行为嵌入模块
- **Node2Vec**: 基于随机游走的图嵌入，适合复杂用户行为图
- **Item2Vec**: 基于Skip-gram的序列嵌入，适合简单序列数据

### 2. 属性嵌入模块
- 分类特征：嵌入层编码
- 数值特征：标准化处理
- 文本特征：预训练语言模型编码

### 3. 位置嵌入模块
- 基于Item2Vec的基站序列建模
- 支持时间序列位置轨迹

### 4. 多模态融合网络
- 注意力机制融合不同模态
- 残差连接和层归一化
- 生成最终用户表示向量

## 🔧 高级功能

### 实验管理
每个实验都有独立的目录和配置：
```bash
# 查看实验配置
cat experiments/my_experiment/experiment_config.json

# 查看训练日志
cat experiments/my_experiment/logs/training.log

# 查看兼容性报告
cat experiments/my_experiment/processed_data/new_user_compatibility_report.json
```

### 模型检查点
训练过程中自动保存检查点，支持断点续训：
```bash
# 从检查点继续训练
python main.py --mode train --experiment_name my_experiment --resume_from_checkpoint
```

### 🆕 数据过滤机制
为了避免新用户数据中包含训练时未见过的实体导致的错误：

1. **训练阶段**: 自动保存所有训练数据中出现的URL和基站ID到 `training_entities.pkl`
2. **推理阶段**: 加载实体记录，过滤新用户数据中的未知实体
3. **报告生成**: 详细记录过滤统计和数据质量分析
4. **建议提供**: 根据兼容性分析提供具体的改进建议

#### 兼容性评级说明
- **优秀 (≥80%)**: 数据兼容性很好，可以直接使用
- **良好 (60-79%)**: 数据基本兼容，但建议检查未知实体
- **一般 (40-59%)**: 数据兼容性较差，建议补充训练数据
- **较差 (<40%)**: 数据兼容性很差，需要重新准备数据

## 🐛 常见问题

### Q: 新用户计算时出现 "KeyError" 或维度不匹配？
A: 这通常是因为新用户数据包含训练时未见过的URL或基站。现在系统会自动过滤这些数据并生成报告。

### Q: 如何提高数据兼容性？
A: 查看生成的兼容性报告，根据建议：
1. 将重要的未知实体添加到训练数据中重新训练
2. 将未知实体映射到相似的已知实体
3. 检查数据格式和命名规范的一致性

### Q: 训练过程中内存不足？
A: 减少批量大小或嵌入维度：
```python
BATCH_SIZE = 128  # 默认256
EMBEDDING_DIM = 64  # 默认128
```

### Q: 如何调试新用户向量计算？
A: 使用调试模式：
```bash
python compute_new_users.py --experiment_path experiments/your_experiment --debug
```

### Q: 兼容性报告显示评分很低怎么办？
A: 
1. 检查新用户数据的URL和基站ID是否与训练数据一致
2. 考虑扩充训练数据集，包含更多的URL和基站
3. 使用数据映射，将未知实体映射到已知的相似实体

## 📝 更新日志

### v2.1.0 (最新)
- ✅ 新增自动数据过滤机制，避免未知实体导致的错误
- ✅ 集成兼容性报告生成，自动保存为JSON格式
- ✅ 新增独立的数据兼容性检查工具 `check_new_user_data.py`
- ✅ 优化过滤统计和报告展示，提供详细的兼容性评估
- ✅ 完善README文档和使用说明
- ✅ 添加兼容性评级系统和改进建议

### v2.0.0
- ✅ 完整的多模态用户嵌入系统
- ✅ 支持Node2Vec和Item2Vec两种模型
- ✅ 新用户向量计算功能
- ✅ 实验管理系统

## 📄 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

---

如有任何问题，请查看兼容性报告或联系项目维护者。
