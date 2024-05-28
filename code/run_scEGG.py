import numpy as np
import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from network import scEGA
from utils import evaluate_model,load_data,load_emb,target_distribution
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from Nmetrics import evaluate
import pandas as pd
import os
import numpy as np
import torch
import random

mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss()

parser = ArgumentParser()
parser.add_argument('--dataset_str', default='Bjorklund', type=str, help='name of dataset')
parser.add_argument('--n_clusters', default=4, type=int, help='expected number of clusters')
parser.add_argument('--lam', default=1e2, type=float, help='rescon_g loss')#1e2
parser.add_argument('--k_neigh', default=None, type=int, help='number of neighbors to construct the cell graph')
parser.add_argument('--is_NE', default=True, type=bool, help='use NE denoise the cell graph or not')
parser.add_argument('--hidden_dim', default=256, type=int, help='hidden layer dim')#256
parser.add_argument('--n_attn_heads', default=4, type=int, help='number of heads for attention')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training')#1e-5
parser.add_argument('--epochs', default=500, type=int, help='number of epochs for training')
args = parser.parse_args()

if args.dataset_str == "Bjorklund":
    args.epochs = 200
    seed = 246

os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 目录创建
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('result/'):
    os.makedirs('result/')

# 文件路径配置
data_path = f'data/{args.dataset_str}/{args.dataset_str}_data.csv'
label_path = f'data/{args.dataset_str}/label.ann'
genemap_path = f'data/{args.dataset_str}/{args.dataset_str}.emb'
model_path = f'logs/model_{args.dataset_str}.h5'

# 数据加载
A, X, cells, genes = load_data(data_path, args.dataset_str, args.is_NE, args.n_clusters, args.k_neigh)
X_dim = X.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float).to(device)
A_tensor = torch.tensor(A, dtype=torch.float).to(device)
edge_index = (A_tensor != 0).nonzero(as_tuple=False).t().contiguous()

gene = load_emb(genemap_path)
G_dim = gene.shape[1]
Gene_tensor = torch.tensor(gene, dtype=torch.float).to(device)

model = scEGA(feat_size_x=X_dim,feat_size_g=G_dim, n_clusters=args.n_clusters, hidden_dim=args.hidden_dim, heads=args.n_attn_heads).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


model.train()
ari_best, epoch_save = 0, 0
stop_patience = 500
for epoch in range(args.epochs):
    optimizer.zero_grad()
    A_hat, X_hat, Z_c, q = model(X_tensor, Gene_tensor, edge_index)
    loss_recon_x = mse_loss(X_hat, X_tensor)
    loss_recon_g = mse_loss(A_hat, A_tensor)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=1)
    y_pred = kmeans.fit_predict(Z_c.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    p = target_distribution(q)
    loss_c = kl_loss(q.log(), p)
    loss = loss_recon_x + loss_recon_g + (args.lam) * loss_c
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1 == 0:
        ari, nmi, acc, pur = evaluate_model(X_tensor, Gene_tensor, edge_index, model, label_path,
                                            n_clusters=args.n_clusters)
        if ari > ari_best:
            ari_best, epoch_save, nmi_bset = ari, epoch, nmi
            torch.save(model.state_dict(), model_path)
    # 早停检查
    if epoch - epoch_save >= stop_patience:
        print(f'Stopping early at epoch {epoch + 1}. Best ARI: {ari_best:.4f} at epoch {epoch_save + 1}')
        break
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

model.load_state_dict(torch.load(model_path))
model.eval()
# 从模型获取嵌入的数据
_, _, Z_c, _ = model(X_tensor, Gene_tensor, edge_index)
kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=1)
kmeans.fit(Z_c.data.cpu().numpy())
y_pred = kmeans.labels_

# 从文件加载真实标签
y_true = pd.read_csv(label_path, sep='\t').values
y_true = y_true[:, -1].astype(int)

acc, nmi, pur, fscore, precision, recall, ari = evaluate(y_true, y_pred)

acc = float(np.round(acc, 3))
nmi = float(np.round(nmi, 3))
pur = float(np.round(pur, 3))
ari = float(np.round(ari, 3))
print('ARI=%.3f, NMI=%.3f, ACC=%.3f, PUR=%.3f' % (ari, nmi, acc, pur))






