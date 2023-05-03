import time
import torch

from net import Net
from aco import ACO
from utils import load_test_dataset
from tqdm import tqdm

# torch.manual_seed(12345)

EPS = 1e-10
device = 'cpu'

@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse=None):
    if model:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    
        aco = ACO(
            n_ants=n_ants,
            heuristic=heu_mat.cpu(),
            distances=distances.cpu(),
            device='cpu',
            local_search='gls',
            # elitist=True
            # min_max = True
        )
    
    else:
        aco = ACO(
            n_ants=n_ants,
            distances=distances.cpu(),
            device='cpu',
            local_search='gls',
            # elitist=True
            # min_max = True
        )
        # if k_sparse:
        #     aco.sparsify(k_sparse)
        
    results = torch.zeros(size=(len(t_aco_diff),))
    for i, t in enumerate(t_aco_diff):
        best_cost = aco.run(t, inference = True)
        results[i] = best_cost
    return results
        
    
@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse=None):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),))
    start = time.time()
    for pyg_data, distances in tqdm(dataset):
        results = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start

def compare(n_node, k_sparse = None, n_ants=30, t_aco = None):
    global device
    k_sparse = k_sparse or n_node//10
    t_aco = None or [1, 2, 5, 10, 20, 30, 50]
    device = 'cpu' if n_node < 200 else 'cuda:0'
    test_list = load_test_dataset(n_node, k_sparse, device, start_node = 0)
    print("number of instances:", len(test_list))
    print("device:", 'cpu' if device == 'cpu' else device+"+cpu" )

    print("=== MetaACO ===")
    net_tsp = Net().to(device)
    net_tsp.load_state_dict(torch.load(f'../pretrained/tsp_2opt/tsp{n_node}-best.pt', map_location=device))
    avg_aco_best, duration = test(test_list, net_tsp, n_ants, t_aco, k_sparse)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average cost is {}.".format(t, avg_aco_best[i]))

    return
    print("=== ACO ===")
    avg_aco_best, duration = test(test_list, None, n_ants, t_aco, k_sparse)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average cost is {}.".format(t, avg_aco_best[i]))

compare(200)
