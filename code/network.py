import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Parameter

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
class GraphAttentionEncoder(nn.Module):


    def __init__(self, in_features, out_features, heads=1):
        super(GraphAttentionEncoder, self).__init__()
        self.gat1 = GATConv(in_features, out_features, heads=heads, concat=True)
        self.gat2 = GATConv(out_features * heads, out_features, heads=1, concat=False)
    def forward(self, x, edge_index):
        x = x.float()
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(in_features, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, in_features)
    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded


class scEGA(nn.Module):
    def __init__(self, feat_size_x,feat_size_g,n_clusters, hidden_dim, heads=1):
        super(scEGA, self).__init__()
        self.gat_encoder = GraphAttentionEncoder(feat_size_x, hidden_dim, heads)
        self.ae = AutoEncoder(feat_size_g, hidden_dim)
        self.ae.apply(weights_init)
        self.tau = 1
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, hidden_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def forward(self, x,g, edge_index):
        gat_encoded = self.gat_encoder(x, edge_index)
        ae_encoded, _ = self.ae(g)
        ZZ = torch.matmul(gat_encoded, gat_encoded.T)
        # 应用sigmoid函数
        A_hat = torch.sigmoid(ZZ)
        X_hat = torch.matmul(gat_encoded, ae_encoded.T)
        q = 1.0 / (1.0 + torch.sum(torch.pow(gat_encoded.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return A_hat,X_hat,gat_encoded,q

    def forward_wo_gene(self, x,edge_index):
        gat_encoded = self.gat_encoder(x, edge_index)
        ZZ = torch.matmul(gat_encoded, gat_encoded.T)
        A_hat = torch.sigmoid(ZZ)
        return A_hat,gat_encoded

    def forward_wo_gene_add_clustering(self, x,edge_index):
        gat_encoded = self.gat_encoder(x, edge_index)
        ZZ = torch.matmul(gat_encoded, gat_encoded.T)
        A_hat = torch.sigmoid(ZZ)
        q = 1.0 / (1.0 + torch.sum(torch.pow(gat_encoded.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return A_hat,gat_encoded,q

