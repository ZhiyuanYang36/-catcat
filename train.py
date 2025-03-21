import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, GATConv, HeteroConv, SAGEConv
import networkx as nx
import time
from datetime import datetime


# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取数据
data = pd.read_excel('data.xlsx')

# 处理分类特征
categorical_cols = ['所属省', '所属市', '企业类型', '国标行业中类']

# 使用标签编码处理省份和城市
label_encoders = {}
for col in ['所属省', '所属市']:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# 使用目标编码处理企业类型和行业类型（这些特征可能与目标变量有更强的相关性）
target_encoder = TargetEncoder()
data[['企业类型', '国标行业中类']] = target_encoder.fit_transform(data[['企业类型', '国标行业中类']], data['企业状态'])

# 保存原始的省份和行业信息（用于后续构建图）
original_province = data['所属省'].copy()
original_industry = data['国标行业中类'].copy()

# 准备特征和标签
X = data.drop(['序号', '企业状态'], axis=1)
y = data['企业状态']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集划分
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X_scaled, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.176, stratify=y_train_temp, random_state=42)

# SVM模型
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 预测
svm_val_pred = svm_model.predict(X_val)
svm_test_pred = svm_model.predict(X_test)

# 评估SVM模型
svm_val_accuracy = accuracy_score(y_val, svm_val_pred)
svm_val_precision = precision_score(y_val, svm_val_pred)
svm_val_recall = recall_score(y_val, svm_val_pred)
svm_val_f1 = f1_score(y_val, svm_val_pred)

svm_test_accuracy = accuracy_score(y_test, svm_test_pred)
svm_test_precision = precision_score(y_test, svm_test_pred)
svm_test_recall = recall_score(y_test, svm_test_pred)
svm_test_f1 = f1_score(y_test, svm_test_pred)

print("\nSVM模型评估结果：")
print(f"验证集 - 准确率: {svm_val_accuracy:.4f}, 精确率: {svm_val_precision:.4f}, 召回率: {svm_val_recall:.4f}, F1 值: {svm_val_f1:.4f}")
print(f"测试集 - 准确率: {svm_test_accuracy:.4f}, 精确率: {svm_test_precision:.4f}, 召回率: {svm_test_recall:.4f}, F1 值: {svm_test_f1:.4f}")

# 1. 逻辑回归模型
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_val_pred = lr_model.predict(X_val)
lr_test_pred = lr_model.predict(X_test)

# 评估逻辑回归模型
lr_val_accuracy = accuracy_score(y_val, lr_val_pred)
lr_val_precision = precision_score(y_val, lr_val_pred)
lr_val_recall = recall_score(y_val, lr_val_pred)
lr_val_f1 = f1_score(y_val, lr_val_pred)

print("\n逻辑回归模型评估结果：")
print(f"验证集 - 准确率: {lr_val_accuracy:.4f}, 精确率: {lr_val_precision:.4f}, 召回率: {lr_val_recall:.4f}, F1值: {lr_val_f1:.4f}")

# 2. 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test)

# 评估随机森林模型
rf_val_accuracy = accuracy_score(y_val, rf_val_pred)
rf_val_precision = precision_score(y_val, rf_val_pred)
rf_val_recall = recall_score(y_val, rf_val_pred)
rf_val_f1 = f1_score(y_val, rf_val_pred)

print("\n随机森林模型评估结果：")
print(f"验证集 - 准确率: {rf_val_accuracy:.4f}, 精确率: {rf_val_precision:.4f}, 召回率: {rf_val_recall:.4f}, F1值: {rf_val_f1:.4f}")

# 3. GCN模型
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # 第一层
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # 最后一层
        self.convs.append(GCNConv(hidden_dim, 2))
        
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# 4. GAT模型
class GAT(torch.nn.Module):
    def __init__(self, input_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, 32, heads=4, dropout=0.4)
        self.conv2 = GATConv(32 * 4, 16, heads=2, dropout=0.4)
        self.conv3 = GATConv(16 * 2, 2, heads=1, concat=False, dropout=0.4)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# 5. HeteroGNN模型
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('company', 'located', 'province'): SAGEConv((-1, -1), hidden_channels),
                ('province', 'rev_located', 'company'): SAGEConv((-1, -1), hidden_channels),
                ('company', 'same_industry', 'company'): SAGEConv(-1, hidden_channels),
            })
            self.convs.append(conv)

        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return F.log_softmax(self.lin(x_dict['company']), dim=1)

# 构建同质图数据（用于GCN和GAT）
def build_homo_graph(data, features):
    print("开始构建图...")
    start_time = time.time()

    # 创建一个空的无向图
    G = nx.Graph()

    # 添加所有节点
    G.add_nodes_from(range(len(data)))

    # 设置每个组内最大连接数的阈值
    MAX_CONNECTIONS = 1000

    # 1. 基于省份分组构建边
    print("构建省份关联边...")
    province_groups = data.groupby('所属省').groups
    for _, indices in province_groups.items():
        indices = list(indices)
        if len(indices) > MAX_CONNECTIONS:
            # 如果组内节点数量过多，随机选择部分节点进行连接
            selected_indices = np.random.choice(indices, size=MAX_CONNECTIONS, replace=False)
            indices = list(selected_indices)
        # 在同一组内的节点两两连接
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                G.add_edge(indices[i], indices[j])

    # 2. 基于行业分组构建边
    print("构建行业关联边...")
    industry_groups = data.groupby('国标行业中类').groups
    for _, indices in industry_groups.items():
        indices = list(indices)
        if len(indices) > MAX_CONNECTIONS:
            # 如果组内节点数量过多，随机选择部分节点进行连接
            selected_indices = np.random.choice(indices, size=MAX_CONNECTIONS, replace=False)
            indices = list(selected_indices)
        # 在同一组内的节点两两连接
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                G.add_edge(indices[i], indices[j])

    # 转换为PyTorch Geometric的数据格式
    print("转换为PyTorch Geometric格式...")
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # 添加反向边
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # 创建PyTorch Geometric数据对象
    x = torch.FloatTensor(features)
    y = torch.LongTensor(data['企业状态'].values)

    total_time = time.time() - start_time
    print(f"图构建完成！用时: {total_time:.2f}秒")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")

    return Data(x=x, edge_index=edge_index, y=y)

# 构建异构图数据（用于HeteroGNN）
def build_hetero_graph(data):
    print("开始构建异构图...")
    start_time = time.time()

    hetero_data = HeteroData()

    # 添加公司节点特征
    company_features = torch.FloatTensor(X_scaled)
    hetero_data['company'].x = company_features
    hetero_data['company'].y = torch.LongTensor(data['企业状态'].values)

    # 获取唯一的省份列表并创建映射
    provinces = data['所属省'].unique()
    province_mapping = {province: idx for idx, province in enumerate(provinces)}

    # 添加省份节点
    hetero_data['province'].x = torch.eye(len(provinces))

    # 使用向量化操作构建公司-省份边
    print("构建公司-省份关联...")
    company_indices = torch.arange(len(data))
    province_indices = torch.tensor([province_mapping[p] for p in data['所属省']])
    company_to_province = torch.stack([company_indices, province_indices])

    hetero_data['company', 'located', 'province'].edge_index = company_to_province
    hetero_data['province', 'rev_located', 'company'].edge_index = company_to_province[[1, 0]]

    # 使用分组操作构建同行业边
    print("构建同行业关联...")
    industry_groups = data.groupby('国标行业中类').groups
    same_industry_edges = []

    for _, indices in industry_groups.items():
        indices = list(indices)
        if len(indices) > 1:  # 只处理有多个公司的行业
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    same_industry_edges.extend([[indices[i], indices[j]], [indices[j], indices[i]]])

    if same_industry_edges:
        hetero_data['company', 'same_industry', 'company'].edge_index = torch.tensor(same_industry_edges).t()

    total_time = time.time() - start_time
    print(f"异构图构建完成！用时: {total_time:.2f}秒")

    return hetero_data

# 构建图数据
graph_data = build_homo_graph(data, X_scaled)
print(f"\n构建的同质图数据信息：")
print(f"节点数量: {graph_data.num_nodes}")
print(f"边数量: {graph_data.num_edges}")
print(f"节点特征维度: {graph_data.num_features}")

# 构建异构图数据
hetero_data = build_hetero_graph(data)
print(f"\n构建的异构图数据信息：")
print(f"公司节点数量: {hetero_data['company'].num_nodes}")
print(f"省份节点数量: {hetero_data['province'].num_nodes}")
print(f"公司-省份边数量: {hetero_data['company', 'located', 'province'].num_edges}")
print(f"同行业边数量: {hetero_data['company', 'same_industry', 'company'].num_edges if ('company', 'same_industry', 'company') in hetero_data.edge_types else 0}")

# 训练函数
def train_gnn_model(model, data, optimizer, criterion, device, batch_size=2000):
    model.train()
    optimizer.zero_grad()

    # 如果数据量大于batch_size，进行批处理
    if data.num_nodes > batch_size:
        # 随机选择batch_size个节点
        perm = torch.randperm(data.num_nodes)[:batch_size]
        sub_x = data.x[perm]
        sub_y = data.y[perm]

        # 获取这些节点相关的边
        mask = (data.edge_index[0].unsqueeze(1) == perm).any(1)
        sub_edge_index = data.edge_index[:, mask]

        # 重新映射节点索引
        node_idx = torch.zeros(data.num_nodes, dtype=torch.long)
        node_idx[perm] = torch.arange(batch_size)
        sub_edge_index = node_idx[sub_edge_index]

        # 前向传播
        out = model(sub_x.to(device), sub_edge_index.to(device))
        loss = criterion(out, sub_y.to(device))
    else:
        # 如果数据量较小，使用全部数据
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out, data.y.to(device))

    loss.backward()
    optimizer.step()
    return loss

# 评估函数
def evaluate_gnn_model(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)
        return pred.cpu()

# 训练各种GNN模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# GCN
print("\n开始训练GCN模型...")
start_time = time.time()
gcn_model = GCN(input_dim=X_scaled.shape[1]).to(device)
gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
gcn_criterion = torch.nn.NLLLoss()

# GAT
print("\n开始训练GAT模型...")
gat_model = GAT(input_dim=X_scaled.shape[1]).to(device)
gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01)
gat_criterion = torch.nn.NLLLoss()

# HeteroGNN
print("\n开始训练HeteroGNN模型...")
hetero_model = HeteroGNN(hidden_channels=64, num_layers=2).to(device)
hetero_optimizer = torch.optim.Adam(hetero_model.parameters(), lr=0.01)
hetero_criterion = torch.nn.NLLLoss()

# 训练模型
num_epochs = 2
print(f"\n开始训练 {num_epochs} 轮...")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 记录每个模型的损失历史
loss_history = {
    'GCN': [],
    'GAT': [],
    'HeteroGNN': []
}

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # 训练GCN
    gcn_loss = train_gnn_model(gcn_model, graph_data, gcn_optimizer, gcn_criterion, device)
    loss_history['GCN'].append(gcn_loss.item())

    # 训练GAT
    gat_loss = train_gnn_model(gat_model, graph_data, gat_optimizer, gat_criterion, device)
    loss_history['GAT'].append(gat_loss.item())

    # 训练HeteroGNN
    hetero_model.train()
    hetero_optimizer.zero_grad()
    out = hetero_model(hetero_data.x_dict, hetero_data.edge_index_dict)
    hetero_loss = hetero_criterion(out, hetero_data['company'].y)
    hetero_loss.backward()
    hetero_optimizer.step()
    loss_history['HeteroGNN'].append(hetero_loss.item())

    # 每10轮打印一次详细信息
    if (epoch + 1) % 2 == 0:
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        print(f'\n轮次 {epoch+1}/{num_epochs} (用时: {epoch_time:.2f}秒, 总用时: {total_time:.2f}秒)')
        print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'GCN Loss: {gcn_loss:.4f} (平均: {sum(loss_history["GCN"][-10:]) / 10:.4f})')
        print(f'GAT Loss: {gat_loss:.4f} (平均: {sum(loss_history["GAT"][-10:]) / 10:.4f})')
        print(f'HeteroGNN Loss: {hetero_loss:.4f} (平均: {sum(loss_history["HeteroGNN"][-10:]) / 10:.4f})')

total_training_time = time.time() - start_time
print(f"\n训练完成！")
print(f"总用时: {total_training_time:.2f}秒")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 绘制损失曲线
plt.figure(figsize=(12, 6))
for model_name, losses in loss_history.items():
    plt.plot(losses, label=model_name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png', bbox_inches='tight', dpi=300)
plt.close()

# 评估所有模型
print("\n所有模型评估结果：")
models_results = {
    "Logistic Regression": (lr_val_accuracy, lr_val_precision, lr_val_recall, lr_val_f1),
    "SVM": (svm_val_accuracy, svm_val_precision, svm_val_recall, svm_val_f1),
    "Random Forest": (rf_val_accuracy, rf_val_precision, rf_val_recall, rf_val_f1)
}

for model_name, metrics in models_results.items():
    print(f"\n{model_name}:")
    print(f"Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, F1-score: {metrics[3]:.4f}")

# 评估图神经网络模型
for model_name, model in [("GCN", gcn_model), ("GAT", gat_model)]:
    pred = evaluate_gnn_model(model, graph_data, device)
    accuracy = accuracy_score(graph_data.y.cpu(), pred)
    precision = precision_score(graph_data.y.cpu(), pred)
    recall = recall_score(graph_data.y.cpu(), pred)
    f1 = f1_score(graph_data.y.cpu(), pred)

    print(f"\n{model_name}:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# 评估HeteroGNN
hetero_model.eval()
with torch.no_grad():
    out = hetero_model(hetero_data.x_dict, hetero_data.edge_index_dict)
    pred = out.argmax(dim=1)
    accuracy = accuracy_score(hetero_data['company'].y.cpu(), pred.cpu())
    precision = precision_score(hetero_data['company'].y.cpu(), pred.cpu())
    recall = recall_score(hetero_data['company'].y.cpu(), pred.cpu())
    f1 = f1_score(hetero_data['company'].y.cpu(), pred.cpu())

print("\nHeteroGNN:")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# 可视化结果
# 1. ROC曲线
plt.figure(figsize=(10, 8))
models = {
    'Random Forest': rf_model,
    'Logistic Regression': lr_model,
    'SVM': svm_model
}

for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.savefig('roc_curves.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. 特征重要性图（随机森林）
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']  # 按优先级尝试多个中文字体
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建新的图形
plt.figure(figsize=(20, 12))  # 进一步增加图形大小
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 创建条形图
ax = sns.barplot(x='importance', y='feature', data=feature_importance.head(10))

# 调整布局
plt.title('Random Forest - Top 10 Feature Importance', pad=20, fontsize=16)
plt.xlabel('Importance', labelpad=10, fontsize=14)
plt.ylabel('Feature', labelpad=10, fontsize=14)

# 调整y轴标签的显示
plt.tick_params(axis='y', labelsize=14)  # 增加y轴标签大小
plt.tight_layout()

# 调整y轴标签的位置，确保中文显示完整
ax.yaxis.set_tick_params(pad=20)  # 增加标签与轴的距离

# 确保所有标签都使用中文字体
for label in ax.get_yticklabels():
    label.set_fontproperties('Microsoft YaHei')

# 保存图片时增加边距
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300, pad_inches=1.5)
plt.close()

# 3. 预测效果散点图
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.5)
plt.scatter(range(len(rf_test_pred)), rf_test_pred, label='Predicted', alpha=0.5)
plt.title('Random Forest - Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.savefig('prediction_scatter.png', bbox_inches='tight', dpi=300)
plt.close()

# 注：图神经网络的具体实现需要根据实际数据构建图结构
# 这里省略了图构建和训练过程

# 消融实验
print("\n开始进行消融实验...")

# 1. 图结构消融实验
def ablation_graph_structure(data, features, ablation_configs):
    results = {}
    for config_name, config in ablation_configs.items():
        print(f"\n执行图结构消融实验: {config_name}")
        
        # 创建图
        G = nx.Graph()
        G.add_nodes_from(range(len(data)))
        
        # 添加自环边
        for i in range(len(data)):
            G.add_edge(i, i)
        
        # 根据配置添加边
        if config['use_province_edges']:
            print("添加省份关联边...")
            province_groups = data.groupby('所属省').groups
            for _, indices in province_groups.items():
                indices = list(indices)
                if len(indices) > 1000:  # 限制每个组内的最大连接数
                    indices = np.random.choice(indices, size=1000, replace=False)
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        G.add_edge(indices[i], indices[j])
        
        if config['use_industry_edges']:
            print("添加行业关联边...")
            industry_groups = data.groupby('国标行业中类').groups
            for _, indices in industry_groups.items():
                indices = list(indices)
                if len(indices) > 1000:
                    indices = np.random.choice(indices, size=1000, replace=False)
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        G.add_edge(indices[i], indices[j])
        
        # 转换为PyTorch Geometric格式
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # 准备节点特征
        if config['use_province_features']:
            # 将DataFrame转换为numpy数组，然后转换为tensor
            x = torch.FloatTensor(features.values)
        else:
            # 移除省份相关特征
            province_cols = [col for col in features.columns if '所属省' in col or '所属市' in col]
            features_subset = features.drop(columns=province_cols)
            x = torch.FloatTensor(features_subset.values)
        
        y = torch.LongTensor(data['企业状态'].values)
        
        # 创建数据对象
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        
        # 训练模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 使用正确的参数初始化 GCN 模型
        model = GCN(input_dim=x.shape[1], hidden_dim=64, num_layers=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()
        
        # 训练模型
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
            loss = criterion(out, graph_data.y.to(device))
            loss.backward()
            optimizer.step()
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
            pred = out.argmax(dim=1)
            
        results[config_name] = {
            'accuracy': accuracy_score(graph_data.y.cpu(), pred.cpu()),
            'precision': precision_score(graph_data.y.cpu(), pred.cpu()),
            'recall': recall_score(graph_data.y.cpu(), pred.cpu()),
            'f1': f1_score(graph_data.y.cpu(), pred.cpu()),
            'num_edges': G.number_of_edges(),
            'num_nodes': G.number_of_nodes()
        }
    
    return results

# 定义图结构消融实验配置
graph_ablation_configs = {
    '完整模型': {
        'use_province_edges': True,
        'use_industry_edges': True,
        'use_province_features': True
    },
    '无省份边': {
        'use_province_edges': False,
        'use_industry_edges': True,
        'use_province_features': True
    },
    '无行业边': {
        'use_province_edges': True,
        'use_industry_edges': False,
        'use_province_features': True
    },
    '无省份特征': {
        'use_province_edges': True,
        'use_industry_edges': True,
        'use_province_features': False
    },
    '仅企业特征': {
        'use_province_edges': False,
        'use_industry_edges': False,
        'use_province_features': False
    }
}

# 执行图结构消融实验
graph_ablation_results = ablation_graph_structure(data, X, graph_ablation_configs)

# 可视化图结构消融实验结果
def plot_graph_ablation_results(results):
    plt.figure(figsize=(15, 8))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[config][metric] for config in results.keys()]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Graph Structure Configuration')
    plt.ylabel('Score')
    plt.title('Graph Structure Ablation Study')
    plt.xticks(x + width*1.5, list(results.keys()), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graph_ablation.png', bbox_inches='tight', dpi=300)
    plt.close()

# 绘制图结构消融实验结果
plot_graph_ablation_results(graph_ablation_results)

# 打印图结构消融实验结果
print("\n图结构消融实验结果：")
for config, metrics in graph_ablation_results.items():
    print(f"\n{config}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Number of edges: {metrics['num_edges']}")
    print(f"Number of nodes: {metrics['num_nodes']}")

# 2. 模型架构消融实验
def ablation_model_architecture(data, features, ablation_configs):
    results = {}
    # 构建基础图结构
    G = nx.Graph()
    G.add_nodes_from(range(len(data)))
    
    # 添加自环边
    for i in range(len(data)):
        G.add_edge(i, i)
    
    # 添加行业边
    industry_groups = data.groupby('国标行业中类').groups
    for _, indices in industry_groups.items():
        indices = list(indices)
        if len(indices) > 1000:
            indices = np.random.choice(indices, size=1000, replace=False)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                G.add_edge(indices[i], indices[j])
    
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # 将DataFrame转换为numpy数组，然后转换为tensor
    x = torch.FloatTensor(features.values)
    y = torch.LongTensor(data['企业状态'].values)
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config_name, config in ablation_configs.items():
        print(f"\n执行模型架构消融实验: {config_name}")
        
        # 根据配置创建模型
        if config_name == 'GCN':
            model = GCN(input_dim=x.shape[1], hidden_dim=64, num_layers=3).to(device)
        elif config_name == 'GAT':
            model = GAT(input_dim=x.shape[1]).to(device)
        elif config_name == 'GCN_NoAttention':
            model = GCN(input_dim=x.shape[1], hidden_dim=64, num_layers=3).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()
        
        # 训练模型
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
            loss = criterion(out, graph_data.y.to(device))
            loss.backward()
            optimizer.step()
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
            pred = out.argmax(dim=1)
        
        results[config_name] = {
            'accuracy': accuracy_score(graph_data.y.cpu(), pred.cpu()),
            'precision': precision_score(graph_data.y.cpu(), pred.cpu()),
            'recall': recall_score(graph_data.y.cpu(), pred.cpu()),
            'f1': f1_score(graph_data.y.cpu(), pred.cpu())
        }
    
    return results

# 定义模型架构消融实验配置
model_ablation_configs = {
    'GCN': {},
    'GAT': {},
    'GCN_NoAttention': {}
}

# 执行模型架构消融实验
model_ablation_results = ablation_model_architecture(data, X, model_ablation_configs)

# 可视化模型架构消融实验结果
def plot_model_ablation_results(results):
    plt.figure(figsize=(15, 8))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[config][metric] for config in results.keys()]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Model Architecture')
    plt.ylabel('Score')
    plt.title('Model Architecture Ablation Study')
    plt.xticks(x + width*1.5, list(results.keys()), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_ablation.png', bbox_inches='tight', dpi=300)
    plt.close()

# 绘制模型架构消融实验结果
plot_model_ablation_results(model_ablation_results)

# 打印模型架构消融实验结果
print("\n模型架构消融实验结果：")
for config, metrics in model_ablation_results.items():
    print(f"\n{config}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
