import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
import os
from tqdm import tqdm
import warnings
import pyarrow.parquet as pq
import gc

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class OptimizedRLEDecoder:
    """优化的RLE解码器：内存友好的RLE格式处理"""

    @staticmethod
    def rle_decode(rle_array, height, width, target_size=224):
        """将RLE数组解码为二值掩码，直接调整到目标尺寸"""
        try:
            # 如果原始尺寸太大，先计算缩放比例
            scale_factor = min(target_size / height, target_size / width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)

            # 创建目标尺寸的掩码
            mask = np.zeros(new_height * new_width, dtype=np.uint8)

            # 处理RLE编码，考虑缩放
            for i in range(0, len(rle_array), 2):
                if i + 1 >= len(rle_array):
                    break
                start = rle_array[i]
                length = rle_array[i + 1]

                # 将原始位置映射到缩放后的位置
                start_scaled = int(start * scale_factor * scale_factor)  # 面积缩放
                length_scaled = max(1, int(length * scale_factor * scale_factor))

                end_scaled = min(start_scaled + length_scaled, len(mask))
                if start_scaled < len(mask):
                    mask[start_scaled:end_scaled] = 1

            # 重塑为目标尺寸
            mask = mask.reshape((new_height, new_width))

            # 如果需要，调整到精确的目标尺寸
            if new_height != target_size or new_width != target_size:
                mask_image = Image.fromarray(mask)
                mask_image = mask_image.resize((target_size, target_size), Image.NEAREST)
                mask = np.array(mask_image)

            return mask

        except Exception as e:
            print(f"RLE解码失败: {e}")
            return np.zeros((target_size, target_size), dtype=np.uint8)

    @staticmethod
    def decode_and_combine_masks(group, target_size=224):
        """为单个图像解码所有RLE掩码并合并，内存优化版本"""
        if len(group) == 0:
            return None

        try:
            # 获取图像尺寸
            height = int(group.iloc[0]['height'])
            width = int(group.iloc[0]['width'])

            # 如果尺寸过大，使用缩放
            if height > 2000 or width > 2000:
                scale_factor = min(target_size / height, target_size / width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
            else:
                new_height, new_width = height, width

            # 创建合并掩码
            combined_mask = np.zeros((new_height, new_width), dtype=np.uint8)

            # 解码每个RLE掩码并合并
            for _, row in group.iterrows():
                rle_array = row['rle']
                mask = OptimizedRLEDecoder.rle_decode(rle_array, height, width, target_size)

                # 如果掩码尺寸不匹配，调整尺寸
                if mask.shape != combined_mask.shape:
                    mask_image = Image.fromarray(mask)
                    mask_image = mask_image.resize((new_width, new_height), Image.NEAREST)
                    mask = np.array(mask_image)

                combined_mask = np.maximum(combined_mask, mask)

                # 及时清理内存
                del mask
                gc.collect()

            # 最终调整到目标尺寸
            if new_height != target_size or new_width != target_size:
                mask_image = Image.fromarray(combined_mask)
                mask_image = mask_image.resize((target_size, target_size), Image.NEAREST)
                combined_mask = np.array(mask_image)

            return combined_mask

        except Exception as e:
            print(f"合并掩码失败: {e}")
            return None


class LightweightMaskEncoder(nn.Module):
    """轻量级分割掩码编码器：减少内存使用"""

    def __init__(self, input_channels=1, base_channels=32, output_dim=256):
        super(LightweightMaskEncoder, self).__init__()

        # 更轻量的编码器
        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            # 第二层
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            # 第三层
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 直接全局平均池化，减少层数
        )

        # 投影网络
        self.projection = nn.Sequential(
            nn.Linear(base_channels * 4, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # 编码特征
        features = self.encoder(x)
        features = features.view(features.size(0), -1)

        # 投影到嵌入空间
        mask_embedding = self.projection(features)

        return mask_embedding


class OptimizedMorphologicalFeatureExtractor:
    """优化的形态学特征提取器"""

    def __init__(self):
        self.image_size = 224

    def extract_morphological_features(self, mask_array):
        """提取传统形态学特征 - 优化版本"""
        if mask_array is None or mask_array.size == 0:
            return np.zeros(8)  # 减少特征数量

        try:
            # 确保是二值图像
            binary_mask = (mask_array > 0).astype(np.uint8)

            # 检查是否为有效的二值图像
            if np.sum(binary_mask) == 0:
                return np.zeros(8)

            features = []

            # 1. 面积特征
            area = np.sum(binary_mask)
            features.append(area)

            # 2. 使用更简单的边界检测
            boundary = np.zeros_like(binary_mask)
            rows, cols = binary_mask.shape

            # 只检查边界像素
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if binary_mask[i, j] == 1:
                        if (binary_mask[i - 1, j] == 0 or binary_mask[i + 1, j] == 0 or
                                binary_mask[i, j - 1] == 0 or binary_mask[i, j + 1] == 0):
                            boundary[i, j] = 1

            perimeter = np.sum(boundary)
            features.append(perimeter)

            # 3. 圆形度
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
            features.append(circularity)

            # 4. 纵横比
            rows_indices = np.any(binary_mask, axis=1)
            cols_indices = np.any(binary_mask, axis=0)
            height = np.sum(rows_indices)
            width = np.sum(cols_indices)
            if width > 0:
                aspect_ratio = height / width
            else:
                aspect_ratio = 0
            features.append(aspect_ratio)

            # 5. 质心位置
            y_indices, x_indices = np.where(binary_mask)
            if len(x_indices) > 0 and len(y_indices) > 0:
                centroid_x = np.mean(x_indices) / cols  # 归一化
                centroid_y = np.mean(y_indices) / rows
            else:
                centroid_x, centroid_y = 0, 0
            features.extend([centroid_x, centroid_y])

            # 6. 填充率
            bbox_area = height * width
            if bbox_area > 0:
                fill_ratio = area / bbox_area
            else:
                fill_ratio = 0
            features.append(fill_ratio)

            # 7. 标准差
            features.append(np.std(binary_mask))

            # 确保特征数量一致
            while len(features) < 8:
                features.append(0)

            return np.array(features[:8])

        except Exception as e:
            print(f"形态学特征提取失败: {e}")
            return np.zeros(8)


class MemoryEfficientMaskDataset:
    """内存高效的分割掩码数据集"""

    def __init__(self, parquet_path, metadata_embeddings_path, split='train', max_samples=None):
        self.parquet_path = parquet_path
        self.split = split
        self.mask_cache = {}  # 有限缓存
        self.filenames = []
        self.category_ids = {}
        self.mask_groups = {}  # 存储RLE数据而不是解码后的掩码

        # 加载元数据映射
        self.metadata_mapping = self._load_metadata_mapping(metadata_embeddings_path)

        # 加载分割掩码
        self._load_mask_groups(max_samples)

        # 形态学特征提取器
        self.morph_extractor = OptimizedMorphologicalFeatureExtractor()

    def _load_metadata_mapping(self, metadata_path):
        """加载元数据映射"""
        with open(metadata_path, 'r') as f:
            metadata_embeddings = json.load(f)

        mapping = {}
        for filename, data in metadata_embeddings.items():
            mapping[filename] = data['category_id']
        return mapping

    def _load_mask_groups(self, max_samples):
        """加载RLE数据，不立即解码"""
        print(f"加载{self.split}集分割掩码RLE数据...")

        try:
            # 分批读取parquet文件
            parquet_file = pq.ParquetFile(self.parquet_path)
            batch_reader = parquet_file.iter_batches(batch_size=10000)

            valid_count = 0
            for batch_idx, batch in enumerate(batch_reader):
                df_batch = batch.to_pandas()

                # 按文件名分组
                grouped = df_batch.groupby('file_name')

                for filename, group in grouped:
                    # 检查是否在元数据中
                    if filename in self.metadata_mapping:
                        # 存储RLE数据，不立即解码
                        self.mask_groups[filename] = {
                            'group_data': group[['rle', 'height', 'width']].to_dict('records'),
                            'category_id': self.metadata_mapping[filename]
                        }
                        self.filenames.append(filename)
                        valid_count += 1

                        # 限制样本数量
                        if max_samples and valid_count >= max_samples:
                            break

                if max_samples and valid_count >= max_samples:
                    break

                # 清理内存
                del df_batch
                gc.collect()

                print(f"处理批次 {batch_idx + 1}, 累计有效样本: {valid_count}")

            print(f"{self.split}集分割掩码: {valid_count} 个有效样本")

        except Exception as e:
            print(f"加载分割掩码失败: {e}")

    def _get_mask_from_cache_or_decode(self, filename):
        """从缓存获取掩码或解码"""
        if filename in self.mask_cache:
            return self.mask_cache[filename]

        # 解码掩码
        if filename in self.mask_groups:
            group_data = self.mask_groups[filename]['group_data']

            # 创建临时DataFrame进行解码
            temp_df = pd.DataFrame(group_data)
            combined_mask = OptimizedRLEDecoder.decode_and_combine_masks(temp_df)

            if combined_mask is not None:
                # 缓存最近使用的掩码（限制缓存大小）
                if len(self.mask_cache) > 1000:  # 限制缓存大小
                    # 移除最早缓存的项
                    oldest_key = next(iter(self.mask_cache))
                    del self.mask_cache[oldest_key]

                self.mask_cache[filename] = combined_mask
                return combined_mask

        return None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        category_id = self.mask_groups[filename]['category_id']

        # 获取掩码（可能从缓存或解码）
        mask_array = self._get_mask_from_cache_or_decode(filename)

        # 预处理掩码
        processed_mask = self._preprocess_mask(mask_array)

        # 提取形态学特征
        morphological_features = self.morph_extractor.extract_morphological_features(mask_array)

        return processed_mask, morphological_features, filename, category_id

    def _preprocess_mask(self, mask_array):
        """预处理掩码图像"""
        if mask_array is None:
            return torch.zeros(1, 224, 224)

        try:
            # 转换为PIL图像
            if mask_array.dtype != np.uint8:
                mask_array = (mask_array * 255).astype(np.uint8)

            mask_image = Image.fromarray(mask_array)

            # 调整大小（确保尺寸正确）
            mask_image = mask_image.resize((224, 224), Image.NEAREST)

            # 转换为张量
            mask_tensor = torch.from_numpy(np.array(mask_image)).float()

            # 归一化到[0, 1]
            mask_tensor = mask_tensor / 255.0

            # 添加通道维度
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

            return mask_tensor

        except Exception as e:
            print(f"预处理掩码失败: {e}")
            return torch.zeros(1, 224, 224)


class OptimizedMaskProjectionNetwork(nn.Module):
    """优化的掩码特征投影网络"""

    def __init__(self, cnn_feature_dim=256, morph_feature_dim=8, output_dim=256):
        super(OptimizedMaskProjectionNetwork, self).__init__()

        # 更轻量的融合网络
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim + morph_feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, cnn_features, morph_features):
        combined = torch.cat([cnn_features, morph_features], dim=1)
        mask_embedding = self.fusion(combined)
        return mask_embedding


class MemoryEfficientMaskExtractionPipeline:
    """内存高效的分割掩码特征提取管道"""

    def __init__(self, encoder, projection_net):
        self.encoder = encoder
        self.projection_net = projection_net
        self.encoder.to(device)
        self.projection_net.to(device)
        self.encoder.eval()
        self.projection_net.eval()

    def extract_mask_embeddings(self, dataset, batch_size=16):  # 减小批处理大小
        """提取分割掩码嵌入向量 - 内存优化版本"""
        if len(dataset) == 0:
            print(f"警告: {dataset.split}集分割掩码为空")
            return {}

        embeddings_dict = {}

        num_batches = (len(dataset) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc=f"处理{dataset.split}集分割掩码"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(dataset))

                # 收集批次数据
                batch_masks = []
                batch_morph_features = []
                batch_filenames = []
                batch_category_ids = []

                for j in range(start_idx, end_idx):
                    mask_tensor, morph_features, filename, category_id = dataset[j]
                    batch_masks.append(mask_tensor)
                    batch_morph_features.append(torch.tensor(morph_features, dtype=torch.float))
                    batch_filenames.append(filename)
                    batch_category_ids.append(category_id)

                # 转换为批次张量
                if batch_masks:
                    mask_batch = torch.stack(batch_masks).to(device)
                    morph_batch = torch.stack(batch_morph_features).to(device)

                    # 编码器前向传播
                    cnn_features = self.encoder(mask_batch)

                    # 投影网络前向传播
                    mask_embeddings = self.projection_net(cnn_features, morph_batch)

                    # 存储结果
                    for j, filename in enumerate(batch_filenames):
                        embeddings_dict[filename] = {
                            'mask_embedding': mask_embeddings[j].cpu(),
                            'category_id': batch_category_ids[j]
                        }

                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 清理Python内存
                del batch_masks, batch_morph_features, mask_batch, morph_batch, cnn_features, mask_embeddings
                gc.collect()

        return embeddings_dict


def save_mask_embeddings(embeddings_dict, filepath):
    """保存分割掩码嵌入向量"""
    serializable_dict = {}
    for key, value in embeddings_dict.items():
        serializable_dict[key] = {
            'mask_embedding': value['mask_embedding'].numpy().tolist(),
            'category_id': value['category_id']
        }

    with open(filepath, 'w') as f:
        json.dump(serializable_dict, f, indent=2)


def main():
    """主函数：优化处理所有分割掩码"""
    # 分割掩码路径
    mask_dir = r"D:\Work(paper2)\masks"
    train_mask_path = os.path.join(mask_dir, "FungiTastic-Mini-TrainMasks.parquet")
    val_mask_path = os.path.join(mask_dir, "FungiTastic-Mini-ValidationMasks.parquet")
    test_mask_path = os.path.join(mask_dir, "FungiTastic-Mini-TestMasks.parquet")

    # 元数据嵌入路径
    metadata_dir = r"D:\Work(paper2)\metadatatransform"
    train_metadata_path = os.path.join(metadata_dir, "train_metadata_embeddings.json")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata_embeddings.json")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata_embeddings.json")

    # 检查文件是否存在
    for path in [train_mask_path, val_mask_path, test_mask_path]:
        if not os.path.exists(path):
            print(f"警告: 分割掩码文件不存在 - {path}")

    # 输出目录
    output_dir = r"D:\Work(paper2)\masktransform"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化轻量级编码器和投影网络
    encoder = LightweightMaskEncoder(input_channels=1, output_dim=256)
    projection_net = OptimizedMaskProjectionNetwork(
        cnn_feature_dim=256,
        morph_feature_dim=8,  # 减少特征维度
        output_dim=256
    )

    # 创建内存高效的特征提取管道
    pipeline = MemoryEfficientMaskExtractionPipeline(encoder, projection_net)

    # 处理训练集分割掩码 - 限制样本数量
    print("处理训练集分割掩码...")
    train_dataset = MemoryEfficientMaskDataset(
        train_mask_path, train_metadata_path, 'train', max_samples=5000
    )

    if len(train_dataset) > 0:
        train_embeddings = pipeline.extract_mask_embeddings(train_dataset, batch_size=16)
        if train_embeddings:
            save_mask_embeddings(train_embeddings, os.path.join(output_dir, "train_mask_embeddings.json"))
            print(f"训练集提取了 {len(train_embeddings)} 个分割掩码嵌入向量")
        else:
            print("训练集分割掩码嵌入提取失败")
    else:
        print("训练集分割掩码为空，跳过")

    # 处理验证集分割掩码 - 限制样本数量
    print("处理验证集分割掩码...")
    val_dataset = MemoryEfficientMaskDataset(
        val_mask_path, val_metadata_path, 'val', max_samples=2000
    )

    if len(val_dataset) > 0:
        val_embeddings = pipeline.extract_mask_embeddings(val_dataset, batch_size=16)
        if val_embeddings:
            save_mask_embeddings(val_embeddings, os.path.join(output_dir, "val_mask_embeddings.json"))
            print(f"验证集提取了 {len(val_embeddings)} 个分割掩码嵌入向量")
        else:
            print("验证集分割掩码嵌入提取失败")
    else:
        print("验证集分割掩码为空，跳过")

    # 处理测试集分割掩码 - 使用更小的样本数量
    print("处理测试集分割掩码...")
    test_dataset = MemoryEfficientMaskDataset(
        test_mask_path, test_metadata_path, 'test', max_samples=1000
    )

    if len(test_dataset) > 0:
        test_embeddings = pipeline.extract_mask_embeddings(test_dataset, batch_size=8)  # 更小的批处理
        if test_embeddings:
            save_mask_embeddings(test_embeddings, os.path.join(output_dir, "test_mask_embeddings.json"))
            print(f"测试集提取了 {len(test_embeddings)} 个分割掩码嵌入向量")
        else:
            print("测试集分割掩码嵌入提取失败")
    else:
        print("测试集分割掩码为空，跳过")

    # 保存处理信息
    processing_info = {
        'encoder_type': 'Lightweight_CNN_Encoder',
        'input_channels': 1,
        'output_dim': 256,
        'morphological_features': 8,
        'total_train_samples': len(train_embeddings) if 'train_embeddings' in locals() and train_embeddings else 0,
        'total_val_samples': len(val_embeddings) if 'val_embeddings' in locals() and val_embeddings else 0,
        'total_test_samples': len(test_embeddings) if 'test_embeddings' in locals() and test_embeddings else 0
    }

    with open(os.path.join(output_dir, "mask_processing_info.json"), 'w') as f:
        json.dump(processing_info, f, indent=2)

    # 打印统计信息
    if 'train_embeddings' in locals() and train_embeddings:
        sample_embedding = next(iter(train_embeddings.values()))['mask_embedding']
        print(f"分割掩码嵌入向量维度: {sample_embedding.shape}")

    print(f"所有分割掩码嵌入向量已保存到 {output_dir} 目录")


if __name__ == "__main__":
    main()