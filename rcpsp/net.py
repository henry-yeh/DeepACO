import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
from rcpsp_inst import RCPSPInstance

# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=5, edge_feats = 2, units=32, act_fn='silu', agg_fn='mean'): # TODO feats=1
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.edge_feats = edge_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError
    
    def forward(self, x, edge_index, edge_attr):
        if x.shape[1] < self.feats:
            x = torch.vstack([x, torch.zeros(x.shape[0], self.feats-x.shape[1], dtype=x.dtype, device=x.device)])
        w = edge_attr
        x = self.act_fn(self.v_lin0(x))
        w = self.act_fn(self.e_lin0(w))
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w

# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device
    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x) # last layer
        return x

# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim = -1)
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_net = EmbNet()
        # self.par_net_phe = ParNet()
        self.par_net_heu = ParNet()

    def forward(self, pyg, require_phe=False, require_heu=False):
        '''
        Args:
            pyg: torch_geometric.data.Data instance with x, edge_index, and edge attr
        Returns:
            phe: pheromone vector, torch tensor [n_nodes * k_sparsification,]
            heu: heuristic vector [n_nodes * k_sparsification,]
        '''
        assert require_heu or require_phe
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        phe, heu = None, None
        # if require_phe:
        #     phe = self.par_net_phe(emb)
        if require_heu:
            heu = self.par_net_heu(emb)
        return phe, heu
    
    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False
            
    @staticmethod
    def reshape(pyg, vector):
        '''Turn phe/heu vector into matrix with zero padding 
        '''
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix
        
if __name__=="__main__":
    from rcpsp_inst import read_RCPfile
    instance = read_RCPfile("../data/rcpsp/j30rcp/J301_3.RCP")
    pyg = instance.to_pyg_data()
    model = Net()
    phe, heu = model(pyg, require_phe=True, require_heu=True)
    print(phe)
    print(heu)