import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import timm
import torchvision.transforms.functional as F
import json

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class MultiModelFeatureExtractor:
    def __init__(self):
        self.models = {}
        self.preprocessors = {}
        self._load_models()

    def _load_models(self):
        """使用timm库加载所有预训练模型"""
        print("加载预训练模型...")

        # 1. DINOv2
        try:
            self.models['dinov2'] = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                                                      pretrained=True,
                                                      num_classes=0)
            self.models['dinov2'].to(device).eval()
            self.preprocessors['dinov2'] = transforms.Compose([
                transforms.Resize(518),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            print("✓ DINOv2 加载成功")
        except Exception as e:
            print(f"✗ DINOv2 加载失败: {e}")

        # 2. BEiT (使用timm中的BEiT)
        try:
            self.models['beit'] = timm.create_model('beit_base_patch16_224',
                                                    pretrained=True,
                                                    num_classes=0)
            self.models['beit'].to(device).eval()
            self.preprocessors['beit'] = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
            print("✓ BEiT 加载成功")
        except Exception as e:
            print(f"✗ BEiT 加载失败: {e}")

        # 3. ViT作为SAM骨干网络的替代
        try:
            self.models['vit'] = timm.create_model('vit_base_patch16_224',
                                                   pretrained=True,
                                                   num_classes=0)
            self.models['vit'].to(device).eval()
            self.preprocessors['vit'] = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            print("✓ ViT 加载成功")
        except Exception as e:
            print(f"✗ ViT 加载失败: {e}")


class GeometricAugmentations:
    """几何增强类"""

    def __init__(self):
        # 简化增强策略，只保留最重要的几种
        self.augmentations = [
            self.original,
            self.center_crop,
            self.horizontal_flip,
            self.rotate_90,
        ]

    def original(self, img):
        return img

    def center_crop(self, img):
        w, h = img.size
        crop_size = int(0.8 * min(w, h))
        return transforms.CenterCrop(crop_size)(img)

    def horizontal_flip(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def rotate_90(self, img):
        return img.rotate(90)

    def apply_all(self, img):
        """应用所有增强"""
        augmented_images = []
        for aug in self.augmentations:
            try:
                augmented_img = aug(img.copy())
                augmented_images.append(augmented_img)
            except Exception as e:
                print(f"增强应用失败: {e}")
                augmented_images.append(img)
        return augmented_images


class ProjectionNetwork(nn.Module):
    """投影网络：将多模型特征映射到统一空间"""

    def __init__(self, input_dim=768 + 768 + 768, hidden_dim=1024, output_dim=768):
        super(ProjectionNetwork, self).__init__()
        # 使用LayerNorm替代BatchNorm，避免批次大小问题
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class FungiDataset(Dataset):
    def __init__(self, root_dir, extractor, augmentations, mode='train'):
        self.root_dir = root_dir
        self.extractor = extractor
        self.augmentations = augmentations
        self.mode = mode

        self.image_files = []
        for file in os.listdir(root_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_files.append(file)

        print(f"找到 {len(self.image_files)} 个图像文件")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')

            if self.mode == 'train':
                augmented_images = self.augmentations.apply_all(image)
            else:
                augmented_images = [image]

            return augmented_images, img_name

        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            placeholder = Image.new('RGB', (224, 224), color='black')
            return [placeholder], img_name


def custom_collate_fn(batch):
    """自定义collate函数处理PIL图像列表"""
    batch_images = []
    batch_names = []

    for item in batch:
        images, name = item
        batch_images.append(images)
        batch_names.append(name)

    return batch_images, batch_names


class FeatureExtractionPipeline:
    def __init__(self, extractor, projection_net):
        self.extractor = extractor
        self.projection_net = projection_net
        self.projection_net.to(device)
        self.projection_net.eval()  # 设置为评估模式

    def extract_single_image_features(self, img):
        """从单张图像提取多模型特征"""
        model_features = []

        # DINOv2 特征
        if 'dinov2' in self.extractor.models:
            try:
                dinov2_input = self.extractor.preprocessors['dinov2'](img).unsqueeze(0).to(device)
                with torch.no_grad():
                    dinov2_features = self.extractor.models['dinov2'](dinov2_input)
                    model_features.append(dinov2_features.cpu().squeeze())
            except Exception as e:
                print(f"DINOv2特征提取失败: {e}")

        # BEiT 特征
        if 'beit' in self.extractor.models:
            try:
                beit_input = self.extractor.preprocessors['beit'](img).unsqueeze(0).to(device)
                with torch.no_grad():
                    beit_features = self.extractor.models['beit'](beit_input)
                    model_features.append(beit_features.cpu().squeeze())
            except Exception as e:
                print(f"BEiT特征提取失败: {e}")

        # ViT 特征
        if 'vit' in self.extractor.models:
            try:
                vit_input = self.extractor.preprocessors['vit'](img).unsqueeze(0).to(device)
                with torch.no_grad():
                    vit_features = self.extractor.models['vit'](vit_input)
                    model_features.append(vit_features.cpu().squeeze())
            except Exception as e:
                print(f"ViT特征提取失败: {e}")

        return model_features

    def process_batch(self, batch_images, batch_names):
        """处理一个批次的数据"""
        batch_embeddings = {}

        for images, img_name in zip(batch_images, batch_names):
            try:
                augmented_features = []
                for img in images:
                    features = self.extract_single_image_features(img)
                    if features:
                        concatenated = torch.cat(features, dim=0)
                        augmented_features.append(concatenated)

                if augmented_features:
                    avg_features = torch.stack(augmented_features).mean(dim=0)

                    with torch.no_grad():
                        # 确保输入维度正确
                        if avg_features.dim() == 1:
                            avg_features = avg_features.unsqueeze(0)
                        projected_embedding = self.projection_net(avg_features.to(device))

                    batch_embeddings[img_name] = {
                        'embedding': projected_embedding.cpu().squeeze(),
                        'original_features': avg_features.cpu()
                    }

            except Exception as e:
                print(f"处理图像 {img_name} 失败: {e}")
                continue

        return batch_embeddings


def save_embeddings(embeddings_dict, filepath):
    """保存嵌入向量"""
    serializable_dict = {}
    for key, value in embeddings_dict.items():
        serializable_dict[key] = {
            'embedding': value['embedding'].numpy().tolist(),
            'original_features': value['original_features'].numpy().tolist()
        }

    with open(filepath, 'w') as f:
        json.dump(serializable_dict, f)


def load_embeddings(filepath):
    """加载嵌入向量"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    embeddings_dict = {}
    for key, value in data.items():
        embeddings_dict[key] = {
            'embedding': torch.tensor(value['embedding']),
            'original_features': torch.tensor(value['original_features'])
        }

    return embeddings_dict


def main():
    # 初始化组件
    extractor = MultiModelFeatureExtractor()
    augmentations = GeometricAugmentations()
    projection_net = ProjectionNetwork()

    # 创建特征提取管道
    pipeline = FeatureExtractionPipeline(extractor, projection_net)

    # 小数据集路径
    train_path = r"D:\Work(paper2)\images\FungiTastic-Mini\train\500p"
    val_path = r"D:\Work(paper2)\images\FungiTastic-Mini\val\500p"
    test_path = r"D:\Work(paper2)\images\FungiTastic-Mini\test\500p"

    # 创建输出目录
    output_dir = r"D:\Work(paper2)\imagestransform"
    os.makedirs(output_dir, exist_ok=True)

    # 处理训练集
    print("处理训练集...")
    train_dataset = FungiDataset(train_path, extractor, augmentations, 'train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    train_embeddings = {}
    for batch in tqdm(train_loader, desc="训练集"):
        batch_images, batch_names = batch
        batch_embeds = pipeline.process_batch(batch_images, batch_names)
        train_embeddings.update(batch_embeds)

        # 每处理100个批次保存一次，防止程序中断丢失所有进度
        if len(train_embeddings) % 200 == 0:
            save_embeddings(train_embeddings, os.path.join(output_dir, "train_embeddings_temp.json"))
            print(f"已保存 {len(train_embeddings)} 个训练集嵌入向量")

    save_embeddings(train_embeddings, os.path.join(output_dir, "train_embeddings.json"))
    print(f"训练集提取了 {len(train_embeddings)} 个嵌入向量")

    # 处理验证集
    print("处理验证集...")
    val_dataset = FungiDataset(val_path, extractor, augmentations, 'val')
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    val_embeddings = {}
    for batch in tqdm(val_loader, desc="验证集"):
        batch_images, batch_names = batch
        batch_embeds = pipeline.process_batch(batch_images, batch_names)
        val_embeddings.update(batch_embeds)

    save_embeddings(val_embeddings, os.path.join(output_dir, "val_embeddings.json"))
    print(f"验证集提取了 {len(val_embeddings)} 个嵌入向量")

    # 处理测试集
    print("处理测试集...")
    test_dataset = FungiDataset(test_path, extractor, augmentations, 'test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    test_embeddings = {}
    for batch in tqdm(test_loader, desc="测试集"):
        batch_images, batch_names = batch
        batch_embeds = pipeline.process_batch(batch_images, batch_names)
        test_embeddings.update(batch_embeds)

    save_embeddings(test_embeddings, os.path.join(output_dir, "test_embeddings.json"))
    print(f"测试集提取了 {len(test_embeddings)} 个嵌入向量")

    # 打印统计信息
    if train_embeddings:
        sample_embedding = next(iter(train_embeddings.values()))['embedding']
        print(f"嵌入向量维度: {sample_embedding.shape}")

        sample_features = next(iter(train_embeddings.values()))['original_features']
        print(f"原始特征维度: {sample_features.shape}")

    print(f"所有嵌入向量已保存到 {output_dir} 目录")

    # 保存模型信息
    model_info = {
        'embedding_dim': 768,
        'models_used': list(extractor.models.keys()),
        'augmentations_count': len(augmentations.augmentations),
        'dataset': 'FungiTastic-Mini'
    }

    with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)

    # 删除临时文件
    temp_file = os.path.join(output_dir, "train_embeddings_temp.json")
    if os.path.exists(temp_file):
        os.remove(temp_file)


if __name__ == "__main__":
    main()