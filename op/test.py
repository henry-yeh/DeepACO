
import time
import torch

from net import Net
from aco import ACO
from utils import *


EPS = 1e-10
device = 'cpu'

max_len = {
    100: 4,
    200: 5,
    300: 6,
}

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

def infer_instance(model, instance, k_sparse, n_ants, t_aco_diff):
    pyg_data, distances, prizes = instance
    if model:
        model.eval()
        heu_mat = model.reshape(pyg_data, model(pyg_data)) + EPS
        aco = ACO(distances, prizes, max_len[len(prizes)], n_ants, heuristic=heu_mat, device=device)
    else:
        aco = ACO(distances, prizes, max_len[len(prizes)], n_ants, device=device, k_sparse=k_sparse)
        
    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    for i, t in enumerate(t_aco_diff):
        best_cost, _ = aco.run(t)
        results[i] = best_cost
    return results
    
@torch.no_grad()
def test(dataset, model, n_ants, k_sparse, t_aco):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    start = time.time()
    for instance in dataset:
        results = infer_instance(model, instance, k_sparse, n_ants, t_aco_diff)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start



n_ants = 20
t_aco = [1, 10, 20, 30, 40, 50, 100]
sparse_table = {
    100: 20,
    200: 50,
    300: 50
}
for n_node in [100, 200, 300]:
    k_sparse = sparse_table[n_node]
    test_list = load_test_dataset(n_node, k_sparse, device)
    net = Net().to(device)
    net.load_state_dict(torch.load(f'./pretrained/op/op{n_node}.pt', map_location=device))
    avg_aco_best, duration = test(test_list, net, n_ants, k_sparse, t_aco)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average obj. is {}.".format(t, avg_aco_best[i]))    

    avg_aco_best, duration = test(test_list, None, n_ants, k_sparse, t_aco)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average obj. is {}.".format(t, avg_aco_best[i]))    