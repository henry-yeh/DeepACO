import torch
from torch import Tensor
from torch_geometric.data import Data

def gen_prizes(coordinates: Tensor):
    n = len(coordinates)
    depot_coor = coordinates[0]
    distances = (coordinates - depot_coor).norm(p=2, dim=-1)
    prizes = 1 + torch.floor(99 * distances / distances.max())
    prizes /= prizes.max()
    return prizes

def gen_distance_matrix(coordinates):
    '''
    Args:
        _coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(coordinates)
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances


def gen_pyg_data(tsp_coordinates, k_sparse):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    n_nodes = len(tsp_coordinates)
    dis_to_depot = (tsp_coordinates - tsp_coordinates[0]).norm(dim=-1)
    prizes = gen_prizes(tsp_coordinates)
    x = torch.stack((dis_to_depot, prizes)).T
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances, 
                                           k=k_sparse, 
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances, prizes

def load_val_dataset(n_node, k_sparse, device):
    '''Return [(pyg_data, distance matrix, prizes)]
    '''
    val_list = []
    val_tensor = torch.load(f'../data/op/valDataset-{n_node}.pt')
    for coor in val_tensor:
        coor = coor.to(device)
        data, distances, prizes = gen_pyg_data(coor, k_sparse=k_sparse)
        val_list.append((data, distances, prizes))
    return val_list

def load_test_dataset(n_node, k_sparse, device):
    '''Return [(pyg_data, distance matrix, prizes)]
    '''
    val_list = []
    val_tensor = torch.load(f'./data/op/testDataset-{n_node}.pt')
    for coor in val_tensor:
        coor = coor.to(device)
        data, distances, prizes = gen_pyg_data(coor, k_sparse=k_sparse)
        val_list.append((data, distances, prizes))
    return val_list

if __name__ == "__main__":
    import pathlib
    pathlib.Path('../data/op').mkdir(parents=False, exist_ok=True) 
    torch.manual_seed(12345)
    for problem_size in [100, 200, 300]:
        coor = torch.rand(size=(30, problem_size, 2))
        torch.save(coor, f"../data/op/valDataset-{problem_size}.pt")
    torch.manual_seed(123456)
    for problem_size in [100, 200, 300]:
        coor = torch.rand(size=(100, problem_size, 2))
        torch.save(coor, f"../data/op/testDataset-{problem_size}.pt")