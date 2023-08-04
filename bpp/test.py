import time
import torch
from torch.distributions import Categorical, kl

from net import Net
from aco import ACO
from utils import *

torch.manual_seed(1234)

EPS = 1e-10
device = 'cpu'

def infer_instance(model, demands, n_ants, t_aco_diff):
    if model:
        model.eval()
        pyg_data = gen_pyg_data(demands, device)
        heu_vec = model(pyg_data)
        heu_mat = heu_vec.reshape((n_node+1, n_node+1)) + EPS
        aco = ACO(
            demand=demands,
            n_ants=n_ants,
            heuristic=heu_mat,
            device=device
        )
    else:
        aco = ACO(
            demand=demands,
            n_ants=n_ants,
            device=device
        )
        
    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    for i, t in enumerate(t_aco_diff):
        best_cost = aco.run(t)
        results[i] = best_cost
    return results
    
@torch.no_grad()
def test(dataset, model, n_ants, t_aco):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    start = time.time()
    for demands in dataset:
        results = infer_instance(model, demands, n_ants, t_aco_diff)
        sum_results += results
    end = time.time()
    
    return sum_results / len(dataset), end-start



n_ants = 20
t_aco = [1, 5, 10, 20]

n_node = 120
    
test_list = load_test_dataset(n_node, device)

net = Net().to(device)
net.load_state_dict(torch.load(f'../pretrained/bpp/bpp{n_node}.pt', map_location=device))
avg_aco_best, duration = test(test_list, net, n_ants, t_aco)
print('total duration: ', duration)
for i, t in enumerate(t_aco):
    print("T={}, average obj. is {}.".format(t, avg_aco_best[i])) 

avg_aco_best, duration = test(test_list, None, n_ants, t_aco)
print('total duration: ', duration)
for i, t in enumerate(t_aco):
    print("T={}, average obj. is {}.".format(t, avg_aco_best[i]))    
      