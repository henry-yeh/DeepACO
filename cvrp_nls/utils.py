import os
import torch
from torch_geometric.data import Data

CAPACITY_LIST = [(1, 10), (20, 30), (50, 40), (100, 50), (400, 150), (1000, 200), (2000, 300)] # ( # of nodes, capacity)
DEMAND_LOW = 1
DEMAND_HIGH = 9

def get_capacity(n:int):
    return list(filter(lambda x: x[0]<=n, CAPACITY_LIST))[-1][-1]

def gen_instance(n, device, position = False):
    """
    Implements data-generation method as described by Kool et al. (2019) and Hou et al. (2023).
    
    * Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! (arXiv:1803.08475). arXiv. https://doi.org/10.48550/arXiv.1803.08475
    * Hou, Q., Yang, J., Su, Y., Wang, X., & Deng, Y. (2023, February 1). Generalize Learned Heuristics to Solve Large-scale Vehicle Routing Problems in Real-time. The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=6ZajpxqTlQ
    """
    locations = torch.rand(size=(n+1, 2), device=device, dtype=torch.double)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=(n,), device=device, dtype=torch.double)
    demands_normalized = demands / get_capacity(n)
    all_demands = torch.cat((torch.zeros((1,), device=device, dtype=torch.double), demands_normalized))
    distances = gen_distance_matrix(locations)
    if position:
        return all_demands, distances, locations
    return all_demands, distances # (n+1), (n+1, n+1)

def gen_distance_matrix(tsp_coordinates):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2, dtype=torch.double)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10 # note here
    return distances

def gen_pyg_data(demands, distances, device, k_sparse=5):
    n = demands.size(0)
    # sparsify
    # part 1:
    topk_values, topk_indices = torch.topk(distances[1:, 1:], k = k_sparse, dim=1, largest=False)
    edge_index_1 = torch.stack([
        torch.repeat_interleave(torch.arange(n-1).to(topk_indices.device), repeats=k_sparse),
        torch.flatten(topk_indices)
    ]) + 1
    edge_attr_1 = topk_values.reshape(-1, 1)
    # part 2: keep all edges connected to depot
    edge_index_2 = torch.stack([ 
        torch.zeros(n-1, device=device, dtype=torch.long), 
        torch.arange(1, n, device=device, dtype=torch.long),
    ])
    edge_attr_2 = distances[1:, 0].reshape(-1, 1)
    edge_index_3 = torch.stack([ 
        torch.arange(1, n, device=device, dtype=torch.long),
        torch.zeros(n-1, device=device, dtype=torch.long), 
    ])
    edge_index = torch.concat([edge_index_1, edge_index_2, edge_index_3], dim=1)
    edge_attr = torch.concat([edge_attr_1, edge_attr_2, edge_attr_2])

    x = demands
    pyg_data = Data(x=x.unsqueeze(1).float(), edge_attr=edge_attr.float(), edge_index=edge_index)
    return pyg_data

def load_test_dataset(n_node, k_sparse, device, start_node = None):
    dataset = torch.load(f'../data/cvrp_nls/testDataset-{n_node}.pt', map_location=device)
    val_list = []
    for i in range(len(dataset)):
        demands, position, distances = dataset[i, 0, :], dataset[i, 1:3, :], dataset[i, 3:, :]
        pyg_data = gen_pyg_data(demands, distances, device, k_sparse=max(n_node//5, 4))
        val_list.append((pyg_data, demands, distances, position.T))
    return val_list

def load_val_dataset(n_node, k_sparse, device, start_node = None):
    if not os.path.isfile(f'../data/cvrp_nls/valDataset-{n_node}.pt'):
        dataset = []
        for i in range(100):
            demand, dist, position = gen_instance(n_node, device, True)
            instance = torch.vstack([demand, position.T, dist])
            dataset.append(instance)
        dataset = torch.stack(dataset)
        torch.save(dataset, f'../data/cvrp_nls/valDataset-{n_node}.pt')
    else:
        dataset = torch.load(f'../data/cvrp_nls/valDataset-{n_node}.pt', map_location=device)

    val_list = []
    for i in range(len(dataset)):
        demands, position, distances = dataset[i, 0, :], dataset[i, 1:3, :], dataset[i, 3:, :]
        pyg_data = gen_pyg_data(demands, distances, device, k_sparse=max(n_node//5, 4))
        val_list.append((pyg_data, demands, distances, position.T))
    return val_list

if __name__ == '__main__':
    import pathlib
    pathlib.Path('../data/cvrp_nls').mkdir(exist_ok=True) 
    for n in [100, 500, 1000, 2000]: # problem scale
        torch.manual_seed(123456)
        inst_list = []
        for _ in range(100):
            demand, dist, position = gen_instance(n, 'cpu', True)
            instance = torch.vstack([demand, position.T, dist])
            inst_list.append(instance)
        testDataset = torch.stack(inst_list)
        torch.save(testDataset, f'../data/cvrp_nls/testDataset-{n}.pt')
