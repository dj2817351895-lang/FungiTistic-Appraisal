import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
import scipy.sparse as sp
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pandas as pd
import gc

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 核心模型组件
# ==========================================

class MemoryEfficientHypergraphBuilder:
    def __init__(self, n_clusters=300):
        self.n_clusters = n_clusters

    def build(self, features):
        if features.shape[0] == 0: return sp.csr_matrix((0, 0))
        features_np = features.cpu().numpy()
        if features_np.shape[1] > 256:
            features_np = SparseRandomProjection(n_components=256, random_state=42).fit_transform(features_np)
        n_c = min(self.n_clusters, features.shape[0])
        kmeans = MiniBatchKMeans(n_clusters=n_c, batch_size=2048, random_state=42, n_init=3).fit(features_np)
        rows, cols = np.arange(features.shape[0]), kmeans.labels_
        return sp.csr_matrix((np.ones(features.shape[0]), (rows, cols)), shape=(features.shape[0], n_c))


class SimpleHypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, H_sparse):
        if H_sparse.shape[1] == 0: return self.linear(X)
        X = self.linear(X)
        hyperedge_feat = torch.sparse.mm(H_sparse.t(), X)
        return torch.sparse.mm(H_sparse, hyperedge_feat)


class MultiScaleHypergraphAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, dropout=0.3):
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                       nn.Dropout(dropout))
        self.conv = SimpleHypergraphConv(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.res_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, H):
        X_t = self.transform(X)
        return F.relu(self.norm(self.conv(X_t, H) + self.res_proj(X_t)))


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x): return self.head(x)


# ==========================================
# 2. 增强的可视化工具类 - 标题已设为 "model"
# ==========================================

class Visualizer:
    @staticmethod
    def plot_top_k_accuracy(y_true, y_score, output_dir, k_max=20):
        """绘制 Top-1 到 Top-20 的准确率增长曲线"""
        print(f"正在计算 Top-{k_max} 准确率...")
        y_true_tensor = torch.tensor(y_true).to(device)
        y_score_tensor = torch.tensor(y_score).to(device)

        top_k_accs = []
        for k in range(1, k_max + 1):
            _, top_k_idx = y_score_tensor.topk(k, dim=1)
            correct = top_k_idx.eq(y_true_tensor.view(-1, 1).expand_as(top_k_idx))
            acc = correct.any(dim=1).float().mean().item()
            top_k_accs.append(acc)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, k_max + 1), top_k_accs, marker='o', linestyle='-', color='b', linewidth=2)
        plt.fill_between(range(1, k_max + 1), top_k_accs, alpha=0.1, color='b')
        plt.title('model', fontsize=16)  # 标题修改
        plt.xlabel('k Value', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(range(1, k_max + 1))
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        plt.savefig(os.path.join(output_dir, 'top_k_accuracy_curve.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_category_performance_heatmap(y_true, y_pred, output_dir):
        """绘制类别性能热力图"""
        print("正在绘制类别性能分布热力图...")
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=np.unique(y_true))

        df = pd.DataFrame({
            'Category': np.unique(y_true),
            'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Support': support
        }).sort_values(by='Support', ascending=False)

        perf_matrix = df[['Precision', 'Recall', 'F1-Score']].values.T

        plt.figure(figsize=(20, 5))
        sns.heatmap(perf_matrix, cmap='RdYlGn', xticklabels=False, yticklabels=['Precision', 'Recall', 'F1'])
        plt.title('model', fontsize=16)  # 标题修改
        plt.xlabel('Categories (Sorted by Support)', fontsize=12)

        plt.savefig(os.path.join(output_dir, 'category_performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ==========================================
# 3. 训练与评估系统
# ==========================================

class Trainer:
    def __init__(self, base_paths, output_dir, num_classes=215):
        # 兼容性导入：尝试从HGNNbuild加载，失败则使用本地定义的逻辑
        try:
            from HGNNbuild import MultiModalFeatureLoader
            self.loader = MultiModalFeatureLoader(base_paths)
        except ImportError:
            print("无法从HGNNbuild加载Loader，请确保该文件在同一目录下。")

        self.builder = MemoryEfficientHypergraphBuilder(n_clusters=300)
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.data = {}
        os.makedirs(output_dir, exist_ok=True)

    def prepare_data(self):
        for split in ['train', 'val', 'test']:
            feats_dict, cat_ids, fnames = self.loader.load_split_features(split)
            X, valid_fnames = self.loader.get_concatenated_features(feats_dict, fnames)
            if X.shape[0] == 0: continue

            y = torch.tensor([cat_ids[f] if 0 <= cat_ids[f] < self.num_classes else 0 for f in valid_fnames],
                             dtype=torch.long)
            self.data[split] = {'X': X, 'y': y, 'fnames': valid_fnames}
        self.input_dim = self.data['train']['X'].shape[1]

    def init_model(self):
        self.model = MultiScaleHypergraphAttention(self.input_dim, output_dim=256).to(device)
        self.classifier = ClassificationHead(256, self.num_classes).to(device)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.classifier.parameters()), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train_loop(self, epochs=50, batch_size=256):
        best_val_acc = 0
        n_train = self.data['train']['X'].shape[0]

        for epoch in range(epochs):
            self.model.train();
            self.classifier.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                batch_X, batch_y = self.data['train']['X'][idx].to(device), self.data['train']['y'][idx].to(device)

                h_np = self.builder.build(batch_X)
                h_coo = h_np.tocoo()
                b_idx = torch.LongTensor(np.vstack((h_coo.row, h_coo.col)))
                b_val = torch.FloatTensor(h_coo.data)
                batch_H = torch.sparse.FloatTensor(b_idx, b_val, torch.Size(h_coo.shape)).to(device)

                self.optimizer.zero_grad()
                logits = self.classifier(self.model(batch_X, batch_H))
                loss = self.criterion(logits, batch_y)
                loss.backward();
                self.optimizer.step()

            _, val_acc = self.evaluate('val', batch_size)
            print(f"Epoch {epoch + 1}/{epochs} | Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'model': self.model.state_dict(), 'classifier': self.classifier.state_dict()},
                           os.path.join(self.output_dir, "best_model.pth"))

    def evaluate(self, split, batch_size=256):
        self.model.eval();
        self.classifier.eval()
        X, y = self.data[split]['X'], self.data[split]['y'].to(device)
        correct, total_loss = 0, 0
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                bx, by = X[i:i + batch_size].to(device), y[i:i + batch_size]
                h_np = self.builder.build(bx)
                h_coo = h_np.tocoo()
                bh = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((h_coo.row, h_coo.col))),
                                              torch.FloatTensor(h_coo.data), torch.Size(h_coo.shape)).to(device)
                logits = self.classifier(self.model(bx, bh))
                total_loss += self.criterion(logits, by).item() * len(bx)
                correct += (logits.argmax(1) == by).sum().item()
        return total_loss / X.shape[0], correct / X.shape[0]

    def predict_test_full(self):
        checkpoint = torch.load(os.path.join(self.output_dir, "best_model.pth"))
        self.model.load_state_dict(checkpoint['model'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.model.eval();
        self.classifier.eval()

        X = self.data['test']['X']
        all_scores, all_preds = [], []
        with torch.no_grad():
            for i in range(0, X.shape[0], 256):
                bx = X[i:i + 256].to(device)
                h_np = self.builder.build(bx)
                h_coo = h_np.tocoo()
                bh = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((h_coo.row, h_coo.col))),
                                              torch.FloatTensor(h_coo.data), torch.Size(h_coo.shape)).to(device)
                logits = self.classifier(self.model(bx, bh))
                all_scores.append(F.softmax(logits, dim=1).cpu())
                all_preds.append(logits.argmax(1).cpu())
        return self.data['test']['y'].numpy(), torch.cat(all_preds).numpy(), torch.cat(all_scores).numpy()


# ==========================================
# 4. 主程序 - 路径已修改
# ==========================================

def main():
    base_paths = {
        'image': r"D:\Work(paper2)\imagestransform",
        'metadata': r"D:\Work(paper2)\metadatatransform",
        'satellite': r"D:\Work(paper2)\satellitetransform",
        'mask': r"D:\Work(paper2)\masktransform"
    }
    # 路径修改为指定的 12.22 文件夹
    output_dir = r"D:\Work(paper2)\result\model12.22"

    trainer = Trainer(base_paths, output_dir, num_classes=215)
    trainer.prepare_data()
    trainer.init_model()

    trainer.train_loop(epochs=100)

    y_true, y_pred, y_score = trainer.predict_test_full()

    viz = Visualizer()
    # 1. Top-20 曲线
    viz.plot_top_k_accuracy(y_true, y_score, output_dir, k_max=20)
    # 2. 性能热力图
    viz.plot_category_performance_heatmap(y_true, y_pred, output_dir)

    # 最终结果输出
    print(f"\n所有图表标题已设为 'model'。")
    print(f"结果已保存在: {output_dir}")


if __name__ == "__main__":
    main()