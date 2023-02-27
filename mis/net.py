import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as gnn
from torch_geometric.data import Data as PyGData

# class EmbNet(nn.Module):
#     def __init__(self, depth, feats, units, act_fn='relu', agg_fn='mean'):
#         super().__init__()
#         self.depth = depth
#         self.feats = feats
#         self.units = units
        
#         self.act_fn = getattr(F, act_fn)
#         self.convs = nn.ModuleList(
#             [gnn.GraphConv(in_channels = feats, out_channels = units, aggr=agg_fn, improved=True)] +
#             [gnn.GraphConv(units, units, aggr=agg_fn, improved=True) for _ in range(depth-1)])
#         self.bns = nn.ModuleList([gnn.BatchNorm(units) for _ in range(depth)])
    
#     def forward(self, x, edge_index):
#         for conv, norm in zip(self.convs, self.bns):
#             x = self.act_fn(norm(conv(x, edge_index)))
#         return x

class ParNet(nn.Module):
    def __init__(self, depth, units, preds, act_fn='relu'):
        super().__init__()
        self.units = units
        self.preds = preds
        self.depth = depth
        self.units_list = [units] * depth + [preds]
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(i, j) for i, j in zip(self.units_list, self.units_list[1:])])

    def forward(self, x):
        for i in range(self.depth-1):
            x = self.act_fn(x + self.lins[i](x))
        # x = F.softmax(self.lins[-1](x), dim=0)
        x = torch.sigmoid(self.lins[-1](x))
        return x

    
class Net(nn.Module):
    def __init__(self, depth_emb = 12, depth_par = 5, feats = 1, units = 64):
        super().__init__()
        # self.emb = EmbNet(depth_emb, feats, units)
        self.emb = gnn.GCN(feats, units, depth_emb, units, act='silu', norm='graph_norm')
        self.par_phe = ParNet(depth_par, units, 1)
        # self.par_phe = gnn.MLP(in_channels = units, hidden_channels = units, out_channels = 1, act='silu', num_layers = depth_par, norm="batch_norm")
        self.par_heu = ParNet(depth_par, units, 1)
        # self.par_heu = gnn.MLP(in_channels = units, hidden_channels = units, out_channels = 1, act='silu', num_layers = depth_par, norm="batch_norm")
    
    def forward(self, g: PyGData):
        x, edge_index = g.x, g.edge_index
        emb = self.emb(x, edge_index)
        # phe = F.softmax(self.par_phe(emb).squeeze(-1),dim=0) 
        phe = self.par_phe(emb).squeeze(-1)
        # heu = F.softmax(self.par_heu(emb).squeeze(-1),dim=0)
        heu = self.par_heu(emb).squeeze(-1)
        return phe, heu
        
if __name__ == "__main__":
    import networkx as nx
    from utils import networkx_to_pyg
    graph = nx.erdos_renyi_graph(50, 0.2, seed = 0x12345678)
    net = Net()
    g = networkx_to_pyg(graph)
    print(g)
    phe, heu = net(g)
    print(phe.shape, heu.shape)
    print(phe)
    print(heu)

