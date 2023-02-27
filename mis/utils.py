import networkx as nx
import torch
from torch_geometric.data import Data as PyGData

def networkx_to_adjlist(g: nx.Graph) -> list[list[int]]:
    adjlist: list[list[int]] = []
    for n in g.nodes:
        adjlist.append(list(g.neighbors(n)))
    return adjlist

@torch.no_grad()
def networkx_to_pyg(g: nx.Graph) -> PyGData:
    x = torch.ones(len(g), dtype=torch.float)
    edge_index = torch.tensor(list(g.edges)).T
    data = PyGData(x.unsqueeze(-1), edge_index)
    return data