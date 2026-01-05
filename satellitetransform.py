import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
import warnings
import re

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class SatelliteFeatureExtractor:
    def __init__(self, model_type='resnet50', use_nir=True):
        self.use_nir = use_nir
        self.model_type = model_type
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        """加载预训练的遥感图像模型"""
        print(f"加载{self.model_type}模型用于{'NIR' if self.use_nir else 'RGB'}卫星图像...")

        if self.model_type == 'resnet50':
            # 使用在ImageNet上预训练的ResNet50
            self.model = models.resnet50(pretrained=True)
            # 移除最后的分类层
            self.model = nn.Sequential(*list(self.model.children())[:-1])

        elif self.model_type == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])

        elif self.model_type == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.model.to(device)
        self.model.eval()

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"✓ {self.model_type} 模型加载成功")


class SatelliteProjectionNetwork(nn.Module):
    """卫星图像投影网络：将卫星特征映射到统一嵌入空间"""

    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=256):
        super(SatelliteProjectionNetwork, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SatelliteDataset:
    """卫星图像数据集类 - 修正映射逻辑"""

    def __init__(self, satellite_dir, metadata_embeddings_path, use_nir=True, split='train'):
        self.satellite_dir = satellite_dir
        self.use_nir = use_nir
        self.split = split
        self.fungus_to_satellite = {}  # 真菌编码到卫星图像路径的映射
        self.filename_to_fungus = {}  # 文件名到真菌编码的映射
        self.category_ids = {}  # 文件名到类别ID的映射

        # 加载元数据嵌入以获取文件名和类别ID映射
        self.metadata_mapping = self._load_metadata_mapping(metadata_embeddings_path)

        # 构建卫星图像路径映射
        self._build_satellite_mapping()

        # 构建文件名到真菌编码的映射
        self._build_filename_mapping()

    def _load_metadata_mapping(self, metadata_path):
        """加载元数据映射"""
        with open(metadata_path, 'r') as f:
            metadata_embeddings = json.load(f)

        mapping = {}
        for filename, data in metadata_embeddings.items():
            mapping[filename] = data['category_id']
        return mapping

    def _extract_fungus_code(self, filename):
        """从文件名中提取真菌编码"""
        # 处理如 "0-2237852116.JPG" 或 "0-2237852116" 格式的文件名
        match = re.match(r'^(\d+)-(\d+)(?:\.\w+)?$', filename)
        if match:
            fungus_code = match.group(2)  # 提取真菌编码
            return fungus_code
        else:
            # 如果格式不匹配，尝试其他可能的格式
            print(f"警告: 无法解析文件名格式: {filename}")
            return None

    def _find_satellite_image(self, fungus_code):
        """在卫星图像目录中查找对应的图像文件"""
        base_dir = os.path.join(self.satellite_dir, 'NIR' if self.use_nir else 'RGB')

        # 遍历所有子目录查找匹配的卫星图像
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # 检查文件名是否包含真菌编码
                if fungus_code in file and any(
                        file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']):
                    return os.path.join(root, file)

        return None

    def _build_satellite_mapping(self):
        """构建真菌编码到卫星图像的映射"""
        print(f"构建{self.split}集卫星图像映射...")

        # 收集所有唯一的真菌编码
        unique_fungus_codes = set()
        for filename in self.metadata_mapping.keys():
            fungus_code = self._extract_fungus_code(filename)
            if fungus_code:
                unique_fungus_codes.add(fungus_code)

        print(f"找到 {len(unique_fungus_codes)} 个唯一的真菌编码")

        # 为每个真菌编码查找卫星图像
        found_count = 0
        for fungus_code in tqdm(unique_fungus_codes, desc="查找卫星图像"):
            satellite_path = self._find_satellite_image(fungus_code)
            if satellite_path:
                self.fungus_to_satellite[fungus_code] = satellite_path
                found_count += 1
            else:
                print(f"  未找到真菌编码 {fungus_code} 的卫星图像")

        print(f"卫星图像匹配: {found_count}/{len(unique_fungus_codes)} 个真菌编码找到对应图像")

    def _build_filename_mapping(self):
        """构建文件名到真菌编码和类别ID的映射"""
        for filename, category_id in self.metadata_mapping.items():
            fungus_code = self._extract_fungus_code(filename)
            if fungus_code and fungus_code in self.fungus_to_satellite:
                self.filename_to_fungus[filename] = fungus_code
                self.category_ids[filename] = category_id

        print(
            f"{self.split}集最终匹配: {len(self.filename_to_fungus)}/{len(self.metadata_mapping)} 个文件有对应卫星图像")

    def __len__(self):
        return len(self.filename_to_fungus)

    def __getitem__(self, idx):
        filename = list(self.filename_to_fungus.keys())[idx]
        fungus_code = self.filename_to_fungus[filename]
        category_id = self.category_ids[filename]
        satellite_path = self.fungus_to_satellite[fungus_code]

        try:
            # 加载卫星图像
            image = Image.open(satellite_path).convert('RGB')
            return image, filename, category_id
        except Exception as e:
            print(f"加载卫星图像失败 {satellite_path}: {e}")
            # 返回黑色图像作为占位符
            placeholder = Image.new('RGB', (224, 224), color='black')
            return placeholder, filename, category_id


class SatelliteExtractionPipeline:
    """卫星图像特征提取管道"""

    def __init__(self, extractor, projection_net):
        self.extractor = extractor
        self.projection_net = projection_net
        self.projection_net.to(device)
        self.projection_net.eval()

    def extract_satellite_embeddings(self, dataset, batch_size=16):
        """提取卫星图像嵌入向量"""
        if len(dataset) == 0:
            print(f"警告: {dataset.split}集卫星图像为空")
            return {}

        embeddings_dict = {}

        # 逐个处理图像
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc=f"处理{dataset.split}集卫星图像"):
                image, filename, category_id = dataset[i]

                try:
                    # 预处理图像
                    input_tensor = self.extractor.preprocess(image).unsqueeze(0).to(device)

                    # 提取特征
                    features = self.extractor.model(input_tensor)
                    features = features.squeeze()

                    # 如果特征是4D（某些模型输出），进行全局平均池化
                    if features.dim() > 1:
                        features = torch.flatten(features, start_dim=1)
                        if features.size(0) > 1:  # 如果是批量，取第一个
                            features = features[0]

                    # 确保特征在GPU上
                    features = features.to(device)

                    # 投影到嵌入空间
                    satellite_embedding = self.projection_net(features.unsqueeze(0))

                    # 存储结果 - 移动到CPU
                    embeddings_dict[filename] = {
                        'satellite_embedding': satellite_embedding.squeeze().cpu(),
                        'category_id': category_id
                    }

                except Exception as e:
                    print(f"处理卫星图像 {filename} 失败: {e}")
                    continue

        return embeddings_dict


def save_satellite_embeddings(embeddings_dict, filepath):
    """保存卫星图像嵌入向量"""
    serializable_dict = {}
    for key, value in embeddings_dict.items():
        serializable_dict[key] = {
            'satellite_embedding': value['satellite_embedding'].numpy().tolist(),
            'category_id': value['category_id']
        }

    with open(filepath, 'w') as f:
        json.dump(serializable_dict, f, indent=2)


def load_satellite_embeddings(filepath):
    """加载卫星图像嵌入向量"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    embeddings_dict = {}
    for key, value in data.items():
        embeddings_dict[key] = {
            'satellite_embedding': torch.tensor(value['satellite_embedding']),
            'category_id': value['category_id']
        }

    return embeddings_dict


def main():
    """主函数：处理所有卫星图像"""
    # 使用NIR卫星图像
    use_nir = True
    satellite_dir = r"D:\Work(paper2)\satelliteImages"

    # 元数据嵌入路径（用于文件名映射）
    metadata_dir = r"D:\Work(paper2)\metadatatransform"
    train_metadata_path = os.path.join(metadata_dir, "train_metadata_embeddings.json")
    val_metadata_path = os.path.join(metadata_dir, "val_metadata_embeddings.json")
    test_metadata_path = os.path.join(metadata_dir, "test_metadata_embeddings.json")

    # 检查文件是否存在
    for path in [train_metadata_path, val_metadata_path, test_metadata_path]:
        if not os.path.exists(path):
            print(f"警告: 元数据文件不存在 - {path}")

    # 输出目录
    output_dir = r"D:\Work(paper2)\satellitetransform"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化特征提取器和投影网络
    extractor = SatelliteFeatureExtractor(model_type='resnet50', use_nir=use_nir)

    # ResNet50的特征维度是2048
    projection_net = SatelliteProjectionNetwork(
        input_dim=2048,  # ResNet50特征维度
        output_dim=256  # 与其他模态保持一致
    )
    projection_net.to(device)

    # 创建特征提取管道
    pipeline = SatelliteExtractionPipeline(extractor, projection_net)

    # 处理训练集卫星图像
    print("处理训练集卫星图像...")
    train_dataset = SatelliteDataset(
        satellite_dir, train_metadata_path,
        use_nir=use_nir, split='train'
    )

    if len(train_dataset) > 0:
        train_embeddings = pipeline.extract_satellite_embeddings(train_dataset, batch_size=1)
        if train_embeddings:
            save_satellite_embeddings(train_embeddings, os.path.join(output_dir, "train_satellite_embeddings.json"))
            print(f"训练集提取了 {len(train_embeddings)} 个卫星图像嵌入向量")
        else:
            print("训练集卫星图像嵌入提取失败")
    else:
        print("训练集卫星图像为空，跳过")

    # 处理验证集卫星图像
    print("处理验证集卫星图像...")
    val_dataset = SatelliteDataset(
        satellite_dir, val_metadata_path,
        use_nir=use_nir, split='val'
    )

    if len(val_dataset) > 0:
        val_embeddings = pipeline.extract_satellite_embeddings(val_dataset, batch_size=1)
        if val_embeddings:
            save_satellite_embeddings(val_embeddings, os.path.join(output_dir, "val_satellite_embeddings.json"))
            print(f"验证集提取了 {len(val_embeddings)} 个卫星图像嵌入向量")
        else:
            print("验证集卫星图像嵌入提取失败")
    else:
        print("验证集卫星图像为空，跳过")

    # 处理测试集卫星图像
    print("处理测试集卫星图像...")
    test_dataset = SatelliteDataset(
        satellite_dir, test_metadata_path,
        use_nir=use_nir, split='test'
    )

    if len(test_dataset) > 0:
        test_embeddings = pipeline.extract_satellite_embeddings(test_dataset, batch_size=1)
        if test_embeddings:
            save_satellite_embeddings(test_embeddings, os.path.join(output_dir, "test_satellite_embeddings.json"))
            print(f"测试集提取了 {len(test_embeddings)} 个卫星图像嵌入向量")
        else:
            print("测试集卫星图像嵌入提取失败")
    else:
        print("测试集卫星图像为空，跳过")

    # 保存处理信息
    processing_info = {
        'model_type': 'resnet50',
        'satellite_type': 'NIR' if use_nir else 'RGB',
        'input_dim': 2048,
        'output_dim': 256,
        'total_train_samples': len(train_embeddings) if 'train_embeddings' in locals() and train_embeddings else 0,
        'total_val_samples': len(val_embeddings) if 'val_embeddings' in locals() and val_embeddings else 0,
        'total_test_samples': len(test_embeddings) if 'test_embeddings' in locals() and test_embeddings else 0
    }

    with open(os.path.join(output_dir, "satellite_processing_info.json"), 'w') as f:
        json.dump(processing_info, f, indent=2)

    # 打印统计信息
    if 'train_embeddings' in locals() and train_embeddings:
        sample_embedding = next(iter(train_embeddings.values()))['satellite_embedding']
        print(f"卫星图像嵌入向量维度: {sample_embedding.shape}")

    print(f"所有卫星图像嵌入向量已保存到 {output_dir} 目录")


if __name__ == "__main__":
    main()