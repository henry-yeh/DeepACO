import torch
from torch_geometric.data import Data

CAPACITY = 50
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_instance(n, device):
    locations = torch.rand(size=(n, 2), device=device)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=(n,), device=device)
    depot = torch.tensor([DEPOT_COOR], device=device)
    all_locations = torch.cat((depot, locations), dim=0)
    all_demands = torch.cat((torch.zeros((1,), device=device), demands))
    distances = gen_distance_matrix(all_locations)
    return all_demands, distances # (n+1), (n+1, n+1)

def gen_distance_matrix(tsp_coordinates):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10 # note here
    return distances

def gen_pyg_data(demands, distances, device):
    n = demands.size(0)
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_index = torch.stack((u, v))
    edge_attr = distances.reshape(((n)**2, 1))
    x = demands
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data

def load_test_dataset(problem_size, device):
    test_list = []
    dataset = torch.load(f'./data/cvrp/testDataset-{problem_size}.pt', map_location=device)
    for i in range(len(dataset)):
        test_list.append((dataset[i, 0, :], dataset[i, 1:, :]))
    return test_list

if __name__ == '__main__':
    import pathlib
    pathlib.Path('../data/cvrp').mkdir(parents=False, exist_ok=True) 
    torch.manual_seed(123456)
    for n in [20, 100, 500]:
        inst_list = []
        for _ in range(100):
            demands, distances = gen_instance(n, 'cpu')
            inst = torch.cat((demands.unsqueeze(0), distances), dim=0) # (n+2, n+1)
            inst_list.append(inst)
        testDataset = torch.stack(inst_list)
        torch.save(testDataset, f'../data/cvrp/testDataset-{n}.pt')
        