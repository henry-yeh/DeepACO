import time
import torch
from torch.distributions import Categorical, kl
# from d2l.torch import Animator

from net import Net
from aco import Dynamic_AS as ACO
from utils import gen_pyg_data, load_val_dataset

torch.manual_seed(1234)

lr = 3e-4
EPS = 1e-10
T=5
device = 'cuda:0'


def train_instance(models, optimizer, data, n_ants, k_sparse):
    sum_loss = 0.
    count = 0
    print_costs = 0
    for coor in data:
        aco = ACO(
            coor,
            nets=models,
            k_sparse=k_sparse,
            device=device,
        )
    
        costs, log_probs = aco.sample()
        baseline = costs.mean()
        reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
        sum_loss += reinforce_loss
        count += 1
        
        print_costs += costs.mean().item()

    sum_loss = sum_loss/count
    optimizer.zero_grad()
    sum_loss.backward()
    # for model_id, model in enumerate(models):
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             if param.grad is None:
    #                 print(f"No gradient computed for {name}", model_id)
    #             # elif param.grad_fn is None:
    #             #     print(f"No gradient function found for {name}", model_id)
    #             else:
    #                 print(f"Gradient has been backpropagated through {name}", model_id)
                    
    for model in models:
        torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 1.0, norm_type = 2)
    optimizer.step()
    
    print(print_costs / batch_size)

# def infer_instance(model, pyg_data, distances, n_ants):
#     model.eval()
#     heu_vec = model(pyg_data)
#     heu_mat = model.reshape(pyg_data, heu_vec) + EPS
#     aco = ACO(
#         n_ants=n_ants,
#         heuristic=heu_mat.cpu(),
#         distances=distances.cpu(),
#         device='cpu',
#         two_opt=True,
#         )
#     costs, log_probs = aco.sample(inference = True)
#     aco.run(n_iterations=T, inference = True)
#     baseline = costs.mean()
#     best_sample_cost = torch.min(costs)
#     best_aco_cost = aco.lowest_cost
#     return baseline.item(), best_sample_cost.item(), best_aco_cost.item()

def generate_traindata(count, n_node):
    for _ in range(count):
        instance = torch.rand(size=(n_node, 2), device=device)
        # yield gen_pyg_data(instance, k_sparse=k_sparse, start_node=0)
        yield instance

def train_epoch(n_node, n_ants, k_sparse, steps_per_epoch, nets, optimizer, batch_size):
    for net in nets:
        net.train()
    for _ in range(steps_per_epoch):
        train_instance(nets, optimizer, generate_traindata(batch_size, n_node), n_ants, k_sparse)


# @torch.no_grad()
# def validation(n_ants, epoch, net, val_dataset, animator=None):
#     sum_bl, sum_sample_best, sum_aco_best = 0, 0, 0
    
#     for data, distances in val_dataset:
#         bl, sample_best, aco_best = infer_instance(net, data, distances, n_ants)
#         sum_bl += bl; sum_sample_best += sample_best; sum_aco_best += aco_best
    
#     n_val = len(val_dataset)
#     avg_bl, avg_sample_best, avg_aco_best = sum_bl/n_val, sum_sample_best/n_val, sum_aco_best/n_val
#     if animator:
#         animator.add(epoch+1, (avg_bl, avg_sample_best, avg_aco_best))
    
#     return avg_bl, avg_sample_best, avg_aco_best

def train(n_node, k_sparse, n_ants, steps_per_epoch, epochs, batch_size, n_stages):
    nets = torch.nn.ModuleList([Net(feats=1).to(device) for _ in range(n_stages)])
    optimizer = torch.optim.AdamW(nets.parameters(), lr=lr)
    # val_list = load_val_dataset(n_node, k_sparse, device, start_node=0)
    # if test_size is not None:
    #     val_list = val_list[:test_size]
    # animator = Animator(xlabel='epoch', xlim=[0, epochs],
    #                     legend=["Avg. sample obj.", "Best sample obj.", "Best ACO obj."])
    
    # avg_bl, avg_best, avg_aco_best = validation(n_ants, -1, net, val_list, animator)
    # val_results = [(avg_bl, avg_best, avg_aco_best)]
    
    sum_time = 0
    for epoch in range(0, epochs):
        start = time.time()
        train_epoch(n_node, n_ants, k_sparse, steps_per_epoch, nets, optimizer, batch_size)
        sum_time += time.time() - start
        # avg_bl, avg_sample_best, avg_aco_best = validation(n_ants, epoch, net, val_list, animator)
        # val_results.append((avg_bl, avg_sample_best, avg_aco_best))
        
    print('total training duration:', sum_time)
    
    # for epoch in range(-1, epochs):
    #     print(f'epoch {epoch}:', val_results[epoch+1])
        
    # torch.save(net.state_dict(), f'../pretrained/tsp_2opt/tsp{n_node}.pt')
    
    
n_node = 200
n_ants = 10
k_sparse = 40
steps_per_epoch = 32
epochs = 50
batch_size = 3
n_stages = 5
train(n_node, k_sparse, n_ants, steps_per_epoch, epochs, batch_size, n_stages)