import time
import torch
import numpy as np

from net import Net
from aco import ACO
from utils import gen_pyg_data, load_val_dataset


EPS = 1e-10
T=5

def train_instance(model, optimizer, data, n_ants):
    model.train()
    sum_loss = 0.0
    count = 0
    for pyg_data, distances in data:
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        
        aco = ACO(
            n_ants=n_ants,
            heuristic=heu_mat,
            distances=distances,
            device=device,
            local_search='gls',
        )
    
        costs, log_probs, paths = aco.sample()
        baseline = costs.mean()
        reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
        costs_2opt, _ = aco.sample_2opt(paths)
        baseline_2opt = costs_2opt.mean()
        cost = (costs_2opt - baseline_2opt) * 0.9 + (costs - baseline) * 0.1
        reinforce_loss = torch.sum(cost.detach() * log_probs.sum(dim=0)) / aco.n_ants
        sum_loss += reinforce_loss
        count += 1

    sum_loss = sum_loss/count
    optimizer.zero_grad()
    sum_loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 3.0, norm_type = 2)
    optimizer.step()

def infer_instance(model, pyg_data, distances, n_ants):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    
    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat.cpu(),
        distances=distances.cpu(),
        device='cpu',
        local_search='gls',
        )
    costs = aco.sample(inference = True)[0]
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    best_aco_1 = aco.run(n_iterations=1, inference = True)
    best_aco_T = aco.run(n_iterations=T-1, inference = True)
    return np.array([baseline.item(), best_sample_cost.item(), best_aco_1, best_aco_T])

def generate_traindata(count, n_node, k_sparse):
    for _ in range(count):
        instance = torch.rand(size=(n_node, 2), device=device)
        yield gen_pyg_data(instance, k_sparse=k_sparse, start_node=0)

def train_epoch(n_node,
                n_ants, 
                k_sparse, 
                epoch, 
                steps_per_epoch, 
                net, 
                optimizer,
                batch_size = 1,
                ):
    for _ in range(steps_per_epoch):
        train_instance(net, optimizer, generate_traindata(batch_size, n_node, k_sparse), n_ants)


@torch.no_grad()
def validation(n_ants, epoch, net, val_dataset):
    stats = []
    for data, distances in val_dataset:
        stats.append(infer_instance(net, data, distances, n_ants))
    avg_stats = [i.item() for i in np.stack(stats).mean(0)]

    return avg_stats


def train(n_node, k_sparse, n_ants, steps_per_epoch, epochs, batch_size = 3, test_size = None, pretrained = None):
    net = Net().to(device)
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=device))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    val_list = load_val_dataset(n_node, k_sparse, device, start_node=0)
    if test_size is not None:
        val_list = val_list[:test_size]
    # animator = Animator(xlabel='epoch', xlim=[0, epochs], figsize=(6,3),
    #                     legend=["Avg. sample obj.", "Best sample obj.", "Best ACO obj. (T=1)", f"Best ACO obj. (T={T})"])
    
    stats = validation(n_ants, -1, net, val_list)
    # animator.add(0, stats)
    val_results = [stats]
    best_result = (stats[-1], stats[-2], stats[-3])
    print(f'epoch 0:', stats)

    sum_time = 0
    for epoch in range(1, epochs+1):
        start = time.time()
        train_epoch(n_node, n_ants, k_sparse, epoch, steps_per_epoch, net, optimizer, batch_size=batch_size)
        sum_time += time.time() - start
        stats = validation(n_ants, epoch, net, val_list)
        print(f'epoch {epoch}:', stats)
        val_results.append(stats)
        scheduler.step()
        curr_result = (stats[-1], stats[-2], stats[-3])
        if curr_result <= best_result:
            torch.save(net.state_dict(), f'../pretrained/tsp_2opt/tsp{n_node}-best.pt')
            best_result = curr_result
        torch.save(net.state_dict(), f'../pretrained/tsp_2opt/tsp{n_node}-last.pt')

    print('\ntotal training duration:', sum_time)
    
    # torch.save(net.state_dict(), f'../pretrained/tsp_2opt/tsp{n_node}.pt')
    return f'../pretrained/tsp_2opt/tsp{n_node}.pt'


if __name__ == "__main__":
    # torch.manual_seed(1234)
    lr = 2e-5
    device = 'cuda:0'
    # pretrained_path = '../pretrained/tsp_2opt/tsp500.pt'
    # pretrained_path = '../pretrained/tsp_2opt/good/tsp1000-best.pt'
    pretrained_path = None
    n_node = 200
    k_sparse = n_node//10
    epochs = 50

    if n_node == 1000:
        batch_size = 3
        n_node, n_ants = 1000, 30
        steps_per_epoch = 20
        test_size = 10
    elif n_node == 50:
        batch_size = 40
        n_ants = 100
        steps_per_epoch = 30
        test_size = 100
    elif n_node == 200:
        batch_size = 20
        n_ants = 30
        steps_per_epoch = 30
        test_size = 20
    
    train(n_node, k_sparse, n_ants, steps_per_epoch, epochs, batch_size=batch_size, test_size = test_size, pretrained = pretrained_path)