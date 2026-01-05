import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class MetadataFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.embedding_dims = {}
        self.feature_columns = None
        self.actual_categorical_columns = []  # 记录实际处理的分类列
        self.fixed_feature_order = [  # 固定特征顺序
            'habitat', 'countryCode', 'region', 'district',
            'substrate', 'metaSubstrate', 'landcover',
            'biogeographicalRegion', 'iucnRedListCategory',
            'kingdom', 'phylum', 'class', 'order', 'family',
            'genus', 'species', 'specificEpithet'
        ]

    def preprocess_metadata(self, df, is_training=True):
        """预处理元数据 - 确保特征维度一致"""
        df_processed = df.copy()

        # 定义特征列（不包含category_id，因为它是目标）
        categorical_columns = self.fixed_feature_order

        numeric_columns = [
            'year', 'month', 'day', 'latitude', 'longitude',
            'coorUncert', 'elevation', 'poisonous'
        ]

        self.feature_columns = {
            'categorical': categorical_columns,
            'numeric': numeric_columns
        }

        # 确保所有预定义分类列都存在，如果不存在则创建并填充默认值
        for col in categorical_columns:
            if col not in df_processed.columns:
                df_processed[col] = 'unknown'
            # 转换为字符串并处理NaN值
            df_processed[col] = df_processed[col].astype(str)
            # 将 'nan' 字符串替换为 'unknown'
            df_processed[col] = df_processed[col].replace('nan', 'unknown')
            df_processed[col] = df_processed[col].fillna('unknown')

        # 记录实际存在的分类列（现在应该是全部）
        self.actual_categorical_columns = categorical_columns

        # 确保所有数值列都存在，如果不存在则创建并填充0
        for col in numeric_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
            # 转换为数值类型，无法转换的设为NaN然后填充
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col] = df_processed[col].fillna(0)

        # 处理 category_id - 作为分类目标，确保存在且为整数
        if 'category_id' in df_processed.columns:
            # 转换为整数，无法转换的设为-1
            df_processed['category_id'] = pd.to_numeric(df_processed['category_id'], errors='coerce')
            df_processed['category_id'] = df_processed['category_id'].fillna(-1).astype(int)

            # 统计类别信息
            unique_categories = df_processed['category_id'].nunique()
            if is_training:
                print(f"训练集类别数量: {unique_categories}")
            else:
                print(f"{'验证' if 'val' in str(df_processed) else '测试'}集类别数量: {unique_categories}")
        else:
            print("警告: 数据中缺少 'category_id' 列")
            df_processed['category_id'] = -1

        if is_training:
            print(f"训练集特征: {len(self.actual_categorical_columns)} 个分类特征, {len(numeric_columns)} 个数值特征")
        else:
            print(
                f"{'验证' if 'val' in str(df_processed) else '测试'}集特征: {len(self.actual_categorical_columns)} 个分类特征, {len(numeric_columns)} 个数值特征")

        # 处理分类变量
        for col in self.actual_categorical_columns:
            if is_training:
                # 训练集：创建标签编码器
                self.label_encoders[col] = LabelEncoder()
                # 获取唯一值并确保包含'unknown'
                unique_vals = list(df_processed[col].unique())
                if 'unknown' not in unique_vals:
                    unique_vals.append('unknown')
                self.label_encoders[col].fit(unique_vals)
                self.embedding_dims[col] = len(unique_vals)

                # 转换训练集数据
                df_processed[col + '_encoded'] = self.label_encoders[col].transform(df_processed[col])
            else:
                # 验证/测试集：使用训练集的编码器
                if col in self.label_encoders:
                    # 处理未见过的值
                    known_classes = set(self.label_encoders[col].classes_)
                    current_values = set(df_processed[col].unique())
                    unknown_values = current_values - known_classes

                    if unknown_values:
                        print(f"  将 {len(unknown_values)} 个未见过的值映射为 'unknown': {list(unknown_values)[:3]}...")

                    df_processed[col] = df_processed[col].apply(
                        lambda x: x if x in known_classes else 'unknown'
                    )

                    # 转换数据
                    df_processed[col + '_encoded'] = self.label_encoders[col].transform(df_processed[col])
                else:
                    # 如果没有训练集的编码器，创建默认编码
                    print(f"警告: 列 {col} 没有训练集编码器，使用默认编码")
                    unique_vals = df_processed[col].unique()
                    temp_encoder = LabelEncoder()
                    temp_encoder.fit(unique_vals)
                    df_processed[col + '_encoded'] = temp_encoder.transform(df_processed[col])

        return df_processed

    def fit_scaler(self, df):
        """拟合数值特征的标准化器"""
        numeric_cols = [col for col in self.feature_columns['numeric'] if col in df.columns]
        if numeric_cols:
            numeric_features = df[numeric_cols]
            self.scaler.fit(numeric_features)

    def transform_features(self, df):
        """转换特征为模型输入格式 - 确保维度一致"""
        # 获取分类特征 - 按照固定顺序
        categorical_features = []
        for col in self.fixed_feature_order:
            if col + '_encoded' in df.columns:
                categorical_features.append(torch.tensor(df[col + '_encoded'].values, dtype=torch.long))
            else:
                # 如果列不存在，创建全零张量
                categorical_features.append(torch.zeros(len(df), dtype=torch.long))

        # 获取数值特征并标准化 - 按照固定顺序
        numeric_features = []
        numeric_cols = []
        for col in self.feature_columns['numeric']:
            if col in df.columns:
                # 确保是数值类型
                numeric_vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                numeric_features.append(numeric_vals.reshape(-1, 1))
                numeric_cols.append(col)
            else:
                # 如果列不存在，创建全零张量
                numeric_features.append(np.zeros((len(df), 1)))
                numeric_cols.append(col)

        if numeric_features:
            numeric_array = np.hstack(numeric_features)
            # 只有在训练时拟合了scaler才进行转换
            if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == len(numeric_cols):
                numeric_array = self.scaler.transform(numeric_array)
            numeric_features_tensor = torch.tensor(numeric_array, dtype=torch.float)
        else:
            numeric_features_tensor = torch.empty((len(df), 0), dtype=torch.float)

        # 堆叠分类特征
        if categorical_features:
            categorical_features_tensor = torch.stack(categorical_features, dim=1)
        else:
            categorical_features_tensor = torch.empty((len(df), 0), dtype=torch.long)

        return categorical_features_tensor, numeric_features_tensor


class MetadataProjectionNetwork(nn.Module):
    """元数据投影网络：将元数据特征映射到统一嵌入空间"""

    def __init__(self, embedding_dims, numeric_dim=8, output_dim=256):
        super(MetadataProjectionNetwork, self).__init__()

        # 为每个分类特征创建嵌入层，使用安全的默认值
        self.embeddings = nn.ModuleDict()

        # 环境特征嵌入层
        embedding_config = {
            'habitat': (embedding_dims.get('habitat', 50), 8),
            'countryCode': (embedding_dims.get('countryCode', 20), 4),
            'region': (embedding_dims.get('region', 30), 8),
            'district': (embedding_dims.get('district', 100), 10),
            'substrate': (embedding_dims.get('substrate', 20), 6),
            'metaSubstrate': (embedding_dims.get('metaSubstrate', 20), 6),
            'landcover': (embedding_dims.get('landcover', 30), 8),
            'biogeographicalRegion': (embedding_dims.get('biogeographicalRegion', 10), 4),
            'iucnRedListCategory': (embedding_dims.get('iucnRedListCategory', 10), 6),
            'kingdom': (embedding_dims.get('kingdom', 5), 4),
            'phylum': (embedding_dims.get('phylum', 20), 6),
            'class': (embedding_dims.get('class', 30), 8),
            'order': (embedding_dims.get('order', 50), 10),
            'family': (embedding_dims.get('family', 100), 12),
            'genus': (embedding_dims.get('genus', 200), 16),
            'species': (embedding_dims.get('species', 300), 16),
            'specificEpithet': (embedding_dims.get('specificEpithet', 300), 16)
        }

        for feature_name, (vocab_size, embed_dim) in embedding_config.items():
            self.embeddings[feature_name] = nn.Embedding(vocab_size, embed_dim)

        # 计算总嵌入维度
        total_embed_dim = sum(embed_dim for _, embed_dim in embedding_config.values())

        # 固定输入维度，确保训练和推理时一致
        self.fixed_input_dim = total_embed_dim + numeric_dim

        print(f"投影网络输入维度: {self.fixed_input_dim}")

        # MLP处理组合特征
        self.mlp = nn.Sequential(
            nn.Linear(self.fixed_input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.2),

            nn.Linear(384, output_dim),  # 最终元数据嵌入向量
            nn.LayerNorm(output_dim)
        )

    def forward(self, categorical_features, numeric_features):
        """
        categorical_features: [batch_size, 17] - 17个分类特征
        numeric_features: [batch_size, 8] - 8个数值特征
        """
        batch_size = categorical_features.size(0)

        # 定义特征顺序（与预处理时一致）
        feature_order = [
            'habitat', 'countryCode', 'region', 'district',
            'substrate', 'metaSubstrate', 'landcover',
            'biogeographicalRegion', 'iucnRedListCategory',
            'kingdom', 'phylum', 'class', 'order', 'family',
            'genus', 'species', 'specificEpithet'
        ]

        # 对每个分类特征应用嵌入
        embedded_features = []
        for i, feature_name in enumerate(feature_order):
            if feature_name in self.embeddings:
                # 确保索引在有效范围内
                feature_indices = categorical_features[:, i]
                # 处理超出嵌入层大小的索引（安全措施）
                vocab_size = self.embeddings[feature_name].num_embeddings
                feature_indices = torch.clamp(feature_indices, 0, vocab_size - 1)

                embedded = self.embeddings[feature_name](feature_indices)
                embedded_features.append(embedded)

        # 拼接所有嵌入
        if embedded_features:
            all_embeddings = torch.cat(embedded_features, dim=1)
        else:
            all_embeddings = torch.zeros(batch_size, 0, device=categorical_features.device)

        # 与数值特征结合并通过MLP
        combined = torch.cat([all_embeddings, numeric_features], dim=1)

        # 检查维度是否匹配
        if combined.size(1) != self.fixed_input_dim:
            print(f"警告: 输入维度不匹配! 期望: {self.fixed_input_dim}, 实际: {combined.size(1)}")
            # 如果维度不匹配，进行填充或截断
            if combined.size(1) < self.fixed_input_dim:
                # 填充零
                padding = torch.zeros(batch_size, self.fixed_input_dim - combined.size(1),
                                      device=combined.device)
                combined = torch.cat([combined, padding], dim=1)
            else:
                # 截断
                combined = combined[:, :self.fixed_input_dim]

        metadata_embedding = self.mlp(combined)

        return metadata_embedding


class MetadataDataset:
    """元数据数据集类"""

    def __init__(self, metadata_path, extractor, split='train'):
        self.metadata_path = metadata_path
        self.extractor = extractor
        self.split = split
        self.df = None
        self.categorical_features = None
        self.numeric_features = None
        self.filenames = None
        self.category_ids = None  # 这是分类目标

        self.load_and_process_data()

    def load_and_process_data(self):
        """加载和处理元数据"""
        print(f"加载{self.split}集元数据...")
        try:
            self.df = pd.read_csv(self.metadata_path)
            print(f"原始数据形状: {self.df.shape}")
        except Exception as e:
            print(f"读取文件失败: {e}")
            return

        # 检查必要的列
        if 'filename' not in self.df.columns:
            print("错误: 数据中缺少 'filename' 列")
            return

        # 预处理 - 区分训练集和验证/测试集
        is_training = (self.split == 'train')
        self.df = self.extractor.preprocess_metadata(self.df, is_training=is_training)

        # 如果是训练集，拟合标准化器
        if is_training:
            self.extractor.fit_scaler(self.df)

        # 转换特征（不包含category_id）
        self.categorical_features, self.numeric_features = self.extractor.transform_features(self.df)

        # 获取文件名和类别ID（分类目标）
        self.filenames = self.df['filename'].tolist()

        # 确保 category_id 存在
        if 'category_id' in self.df.columns:
            self.category_ids = torch.tensor(self.df['category_id'].values, dtype=torch.long)
            print(f"{self.split}集类别ID范围: {self.category_ids.min()} 到 {self.category_ids.max()}")
            print(f"{self.split}集唯一类别数量: {len(torch.unique(self.category_ids))}")
        else:
            print(f"警告: {self.split}集缺少 'category_id' 列")
            self.category_ids = torch.full((len(self.df),), -1, dtype=torch.long)

        print(f"{self.split}集加载完成: {len(self.df)} 个样本")
        print(f"分类特征维度: {self.categorical_features.shape if self.categorical_features is not None else 'None'}")
        print(f"数值特征维度: {self.numeric_features.shape if self.numeric_features is not None else 'None'}")

    def __len__(self):
        return len(self.df) if self.df is not None else 0

    def __getitem__(self, idx):
        """支持索引访问"""
        if self.categorical_features is not None:
            cat_features = self.categorical_features[idx]
        else:
            cat_features = torch.tensor([], dtype=torch.long)

        if self.numeric_features is not None:
            num_features = self.numeric_features[idx]
        else:
            num_features = torch.tensor([], dtype=torch.float)

        filename = self.filenames[idx]
        category_id = self.category_ids[idx] if self.category_ids is not None else -1

        return cat_features, num_features, filename, category_id

    def get_batch(self, indices):
        """获取批次数据"""
        if self.categorical_features is not None:
            cat_batch = self.categorical_features[indices]
        else:
            cat_batch = torch.empty((len(indices), 0), dtype=torch.long)

        if self.numeric_features is not None:
            num_batch = self.numeric_features[indices]
        else:
            num_batch = torch.empty((len(indices), 0), dtype=torch.float)

        filenames_batch = [self.filenames[i] for i in indices]

        # 获取对应的 category_id（分类目标）
        if self.category_ids is not None:
            category_ids_batch = self.category_ids[indices]
        else:
            category_ids_batch = torch.full((len(indices),), -1, dtype=torch.long)

        return cat_batch, num_batch, filenames_batch, category_ids_batch


class MetadataExtractionPipeline:
    """元数据特征提取管道"""

    def __init__(self, extractor, projection_net):
        self.extractor = extractor
        self.projection_net = projection_net
        self.projection_net.to(device)
        self.projection_net.eval()

    def extract_metadata_embeddings(self, dataset, batch_size=32):
        """提取元数据嵌入向量"""
        if len(dataset) == 0:
            print(f"警告: {dataset.split}集为空")
            return {}

        embeddings_dict = {}

        num_batches = (len(dataset) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc=f"处理{dataset.split}集"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(dataset))
                indices = list(range(start_idx, end_idx))

                # 获取批次数据
                cat_batch, num_batch, filenames_batch, category_ids_batch = dataset.get_batch(indices)

                # 移动到设备
                if cat_batch is not None and cat_batch.numel() > 0:
                    cat_batch = cat_batch.to(device)
                else:
                    # 创建空的分类特征张量
                    cat_batch = torch.empty((len(indices), 0), dtype=torch.long, device=device)

                if num_batch is not None and num_batch.numel() > 0:
                    num_batch = num_batch.to(device)
                else:
                    # 创建空的数值特征张量
                    num_batch = torch.empty((len(indices), 0), dtype=torch.float, device=device)

                # 前向传播
                metadata_embeddings = self.projection_net(cat_batch, num_batch)

                # 存储结果
                for j, filename in enumerate(filenames_batch):
                    embeddings_dict[filename] = {
                        'metadata_embedding': metadata_embeddings[j].cpu(),
                        'category_id': category_ids_batch[j].item()  # 分类目标
                    }

        return embeddings_dict


def save_metadata_embeddings(embeddings_dict, filepath):
    """保存元数据嵌入向量"""
    serializable_dict = {}
    for key, value in embeddings_dict.items():
        serializable_dict[key] = {
            'metadata_embedding': value['metadata_embedding'].numpy().tolist(),
            'category_id': value['category_id']  # 分类目标
        }

    with open(filepath, 'w') as f:
        json.dump(serializable_dict, f, indent=2)


def load_metadata_embeddings(filepath):
    """加载元数据嵌入向量"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    embeddings_dict = {}
    for key, value in data.items():
        embeddings_dict[key] = {
            'metadata_embedding': torch.tensor(value['metadata_embedding']),
            'category_id': value['category_id']  # 分类目标
        }

    return embeddings_dict


def main():
    """主函数：处理所有元数据"""
    # 初始化组件
    metadata_extractor = MetadataFeatureExtractor()

    # 元数据路径
    metadata_dir = r"D:\Work(paper2)\metadata\FungiTastic-Mini"
    train_metadata_path = os.path.join(metadata_dir, "FungiTastic-Mini-Train.csv")
    val_metadata_path = os.path.join(metadata_dir, "FungiTastic-Mini-Val.csv")
    test_metadata_path = os.path.join(metadata_dir, "FungiTastic-Mini-Test.csv")

    # 检查文件是否存在
    for path in [train_metadata_path, val_metadata_path, test_metadata_path]:
        if not os.path.exists(path):
            print(f"警告: 文件不存在 - {path}")

    # 输出目录
    output_dir = r"D:\Work(paper2)\metadatatransform"
    os.makedirs(output_dir, exist_ok=True)

    # 首先处理训练集以获取嵌入维度信息
    print("初始化训练集...")
    train_dataset = MetadataDataset(train_metadata_path, metadata_extractor, 'train')

    if len(train_dataset) == 0:
        print("错误: 训练集为空，无法继续")
        return

    # 基于训练数据初始化投影网络
    numeric_dim = train_dataset.numeric_features.shape[1] if train_dataset.numeric_features is not None else 0
    print(f"数值特征维度: {numeric_dim}")
    print(f"嵌入维度信息: {metadata_extractor.embedding_dims}")

    projection_net = MetadataProjectionNetwork(
        embedding_dims=metadata_extractor.embedding_dims,
        numeric_dim=numeric_dim,
        output_dim=256  # 与图像特征维度匹配
    )

    # 创建特征提取管道
    pipeline = MetadataExtractionPipeline(metadata_extractor, projection_net)

    # 提取训练集嵌入
    print("提取训练集元数据嵌入...")
    train_embeddings = pipeline.extract_metadata_embeddings(train_dataset, batch_size=64)
    if train_embeddings:
        save_metadata_embeddings(train_embeddings, os.path.join(output_dir, "train_metadata_embeddings.json"))
        print(f"训练集提取了 {len(train_embeddings)} 个元数据嵌入向量")

        # 统计类别信息
        category_ids = [emb['category_id'] for emb in train_embeddings.values()]
        unique_categories = set(category_ids)
        print(f"训练集包含 {len(unique_categories)} 个唯一类别")
    else:
        print("训练集嵌入提取失败")

    # 处理验证集
    print("提取验证集元数据嵌入...")
    val_dataset = MetadataDataset(val_metadata_path, metadata_extractor, 'val')
    if len(val_dataset) > 0:
        val_embeddings = pipeline.extract_metadata_embeddings(val_dataset, batch_size=64)
        if val_embeddings:
            save_metadata_embeddings(val_embeddings, os.path.join(output_dir, "val_metadata_embeddings.json"))
            print(f"验证集提取了 {len(val_embeddings)} 个元数据嵌入向量")
        else:
            print("验证集嵌入提取失败")
    else:
        print("验证集为空，跳过")

    # 处理测试集
    print("提取测试集元数据嵌入...")
    test_dataset = MetadataDataset(test_metadata_path, metadata_extractor, 'test')
    if len(test_dataset) > 0:
        test_embeddings = pipeline.extract_metadata_embeddings(test_dataset, batch_size=64)
        if test_embeddings:
            save_metadata_embeddings(test_embeddings, os.path.join(output_dir, "test_metadata_embeddings.json"))
            print(f"测试集提取了 {len(test_embeddings)} 个元数据嵌入向量")
        else:
            print("测试集嵌入提取失败")
    else:
        print("测试集为空，跳过")

    # 保存预处理信息
    preprocessing_info = {
        'embedding_dims': metadata_extractor.embedding_dims,
        'feature_columns': metadata_extractor.feature_columns,
        'actual_categorical_columns': metadata_extractor.actual_categorical_columns,
        'numeric_dim': numeric_dim,
        'output_dim': 256
    }

    with open(os.path.join(output_dir, "metadata_preprocessing_info.json"), 'w') as f:
        json.dump(preprocessing_info, f, indent=2)

    # 打印统计信息
    if train_embeddings:
        sample_embedding = next(iter(train_embeddings.values()))['metadata_embedding']
        print(f"元数据嵌入向量维度: {sample_embedding.shape}")

        # 类别分布统计
        category_ids = [emb['category_id'] for emb in train_embeddings.values()]
        unique_categories = set(category_ids)
        print(f"训练集类别数量: {len(unique_categories)}")

        # 显示类别分布
        from collections import Counter
        category_counts = Counter(category_ids)
        print(f"训练集类别分布示例: {dict(list(category_counts.items())[:10])}")

    print(f"所有元数据嵌入向量已保存到 {output_dir} 目录")


if __name__ == "__main__":
    main()