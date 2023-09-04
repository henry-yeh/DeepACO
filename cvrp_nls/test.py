import time
import torch

from net import Net
from aco import ACO, get_subroutes
from utils import load_test_dataset
from tqdm import tqdm

from typing import Tuple, Union, List

torch.manual_seed(1234)


EPS = 1e-10
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def validate_route(distance: torch.Tensor, demands: torch.Tensor, routes: List[torch.Tensor]) -> Tuple[bool, float]:
    length = 0.0
    valid = True
    visited = {0}
    for r in routes:
        d = demands[r].sum().item()
        if d>1.000001:
            valid = False
        length += distance[r[:-1], r[1:]].sum().item()
        for i in r:
            i = i.item()
            if i<0 or i>=distance.size(0):
                valid = False
            else:
                visited.add(i)
    if len(visited) != distance.size(0):
        valid = False
    return valid, length

@torch.no_grad()
def infer_instance(model, pyg_data, demands, distances, positions, n_ants, t_aco_diff, k_sparse=None):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat.cpu(),
        demand = demands.cpu(),
        distances=distances.cpu(),
        device='cpu',
        swapstar=True,
        positions=positions.cpu(),
        inference=True,
    )
    
    results = torch.zeros(size=(len(t_aco_diff),))
    for i, t in enumerate(t_aco_diff):
        aco.run(t, inference = True)
        path = get_subroutes(aco.shortest_path)
        valid, results[i] = validate_route(distances, demands, path)
        if valid is False:
            print("invalid solution.")
    return results
        
    
@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse=None):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),))
    start = time.time()
    for pyg_data, demands, distances, positions in tqdm(dataset):
        results = infer_instance(model, pyg_data, demands, distances, positions, n_ants, t_aco_diff, k_sparse)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start


def main(n_node, model_file, k_sparse = None, n_ants=20, t_aco = None):
    k_sparse = k_sparse or n_node//10
    t_aco = list(range(1, t_aco+1)) if t_aco else list(range(1,11))
    test_list = load_test_dataset(n_node, k_sparse, device)
    # test_list = load_val_dataset(n_node,  k_sparse, device)
    print("problem scale:", n_node)
    print("checkpoint:", model_file)
    print("number of instances:", len(test_list))
    print("device:", 'cpu' if device == 'cpu' else device+"+cpu" )
    print("n_ants:", n_ants)

    net_tsp = Net().to(device)
    net_tsp.load_state_dict(torch.load(model_file, map_location=device))
    avg_aco_best, duration = test(test_list, net_tsp, n_ants, t_aco, k_sparse)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average cost is {}.".format(t, avg_aco_best[i]))


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to checkpoint file.")
    parser.add_argument("-i", "--iterations", type=int, default=None, help="Iterations of ACO to run")
    opt = parser.parse_args()
    n_nodes = opt.nodes

    filepath = opt.model or f'../pretrained/cvrp_nls/cvrp{n_nodes}.pt'
    if not os.path.isfile(filepath):
        print(f"Checkpoint file '{filepath}' not found!")
        exit(1)
    
    main(n_nodes, filepath, t_aco=opt.iterations)
