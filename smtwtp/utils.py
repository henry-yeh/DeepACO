import torch
from torch_geometric.data import Data
import pickle

def instance_gen(n, device):
    due_time_norm = torch.rand(size=(n,), device=device)  # [n,]
    due_time = due_time_norm * (n)
    weights = torch.rand(size=(n,), device=device)  # [n,]
    processing_time = torch.rand(size=(n,), device=device) # [n]
    
    x = torch.stack([due_time_norm, weights]).T # (n, 2)
    x_depot = torch.zeros(size=(1, 2), device=device)
    x = torch.cat([x_depot, x], dim=0)
    
    _edge_attr = torch.cat([torch.zeros(size=(1,), device=device), processing_time]) # (n+1,) 
    edge_attr = torch.repeat_interleave(_edge_attr, n+1).unsqueeze(-1) # attr of <i,j> is the processing time of j
    nodes = torch.arange(n+1, device=device)
    u = nodes.repeat(n+1)
    v = torch.repeat_interleave(nodes, n+1)
    edge_index = torch.stack([u,v])
    pyg_data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data, due_time, weights, processing_time

def load_test_dataset(n_node, device):
    with open(f"../data/smtwtp/test{n_node}.pkl", "rb") as f:
        loaded_list = pickle.load(f)
    for i in range(len(loaded_list)):
        for j in range(len(loaded_list[0])):
            loaded_list[i][j] = loaded_list[i][j].to(device)
    return loaded_list

if __name__ == '__main__':
    torch.manual_seed(123456)
    import pathlib
    pathlib.Path('../data/smtwtp').mkdir(parents=False, exist_ok=True) 
    problem_sizes = [50, 100, 500]
    dataset_size = 100
    for p_size in problem_sizes:
        dataset = []
        for _ in range(dataset_size):        
            pyg_data, due_time, weights, processing_time = instance_gen(p_size, 'cpu')
            dataset.append([pyg_data, due_time, weights, processing_time])
        with open(f"../data/smtwtp/test{p_size}.pkl", "wb") as f:
            pickle.dump(dataset, f)