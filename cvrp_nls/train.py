import time
import torch
import numpy as np
import os

from net import Net
from aco import ACO
from utils import gen_pyg_data, load_val_dataset, gen_instance


EPS = 1e-5
T=5

def train_instance(model, optimizer, data, n_ants):
    model.train()
    sum_loss = 0.0
    sample_avg_cost = 0.0
    sample_min_cost = 0.0
    avg_cost = 0.0
    min_cost = 0.0
    count = 0
    start = time.time()
    for pyg_data, demands, distances, positions in data:
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        
        aco = ACO(
            n_ants=n_ants,
            distances=distances.to(device),
            demand=demands.to(device),
            heuristic=heu_mat.to(device),
            device=device,
            swapstar=True,
            positions=positions
        )
    
        costs_2opt, log_probs, costs_raw = aco.sample_nls()
        baseline_raw = costs_raw.mean()
        baseline_2opt = costs_2opt.mean()
        
        cost = (costs_2opt - baseline_2opt) #* 0.98 + (costs_raw - baseline_raw) * 0.02
        reinforce_loss = torch.sum(cost.detach() * log_probs.sum(dim=0)) / aco.n_ants
        sum_loss += reinforce_loss
        count += 1

        avg_cost += costs_2opt.detach().mean().item()
        min_cost += costs_2opt.detach().min().item()
        sample_avg_cost += costs_raw.detach().mean().item()
        sample_min_cost += costs_raw.detach().min().item()

    sum_loss = sum_loss/count
    optimizer.zero_grad()
    sum_loss.backward() # type: ignore
    torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 3.0, norm_type = 2)# type: ignore
    optimizer.step()
    duration = time.time()-start
    print(
        f"loss: {sum_loss.item():.5f}",
        f"s-a-cost: {sample_avg_cost/count:.5f}", 
        f"s-b-cost: {sample_min_cost/count:.5f}",
        f"f-a-cost: {avg_cost/count:.5f}",
        f"f-b-cost: {min_cost/count:.5f}",
        f"duration: {duration:.2f}s",
    )

def infer_instance(model, pyg_data, demands, distances, positions, n_ants):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    
    aco = ACO(
        n_ants=n_ants,
        demand=demands,
        heuristic=heu_mat,
        distances=distances,
        device=device,
        swapstar=True,
        positions=positions,
        inference = True,
    )
    costs = aco.sample(inference = True)[0]
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    best_aco_1 = aco.run(n_iterations=1, inference = True).cpu()
    best_aco_T = aco.run(n_iterations=T-1, inference = True).cpu()
    return np.array([baseline.cpu().item(), best_sample_cost.cpu().item(), best_aco_1, best_aco_T])

def generate_traindata(count, n_node, k_sparse):
    for _ in range(count):
        instance = demands, distance, pos = gen_instance(n_node, device, True)
        yield gen_pyg_data(demands, distance, device), *instance

def train_epoch(n_node,
                n_ants, 
                k_sparse, 
                epoch, 
                steps_per_epoch, 
                net, 
                optimizer,
                batch_size = 1,
                ):
    for i in range(steps_per_epoch):
        print(f'-- [batch{i+1}]', end=' ')
        train_instance(net, optimizer, generate_traindata(batch_size, n_node, k_sparse), n_ants)


@torch.no_grad()
def validation(n_ants, epoch, net, val_dataset):
    stats = []
    for data, demands, distances, positions in val_dataset:
        stats.append(infer_instance(net, data, demands, distances, positions, n_ants))
    avg_stats = [i.item() for i in np.stack(stats).mean(0)]

    return avg_stats


def train(n_node, n_ants, steps_per_epoch, epochs, k_sparse = None, batch_size = 3, test_size = None, pretrained = None, savepath = "../pretrained/cvrp_nls"):
    k_sparse = k_sparse or n_node//10
    net = Net().to(device)
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=device))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    val_list = load_val_dataset(n_node, k_sparse, device, start_node=0)
    if test_size is not None:
        val_list = val_list[:test_size]
    
    stats = validation(n_ants, -1, net, val_list)
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
            torch.save(net.state_dict(), os.path.join(savepath, f'cvrp{n_node}-best.pt'))
            best_result = curr_result
        torch.save(net.state_dict(), os.path.join(savepath, f'cvrp{n_node}-last.pt'))

    print('\ntotal training duration:', sum_time)
    
    return f'../pretrained/cvrp_nls/cvrp{n_node}.pt'


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar='N', type=int, help="Problem scale")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str, 
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-a", "--ants", type=int, default=30, help="Number of ants (in ACO algorithm)")
    parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Epochs to run")
    parser.add_argument("-t", "--test_size", type=int, default=10, help="Number of instances for testing")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/cvrp_nls",
                        help="The directory to store checkpoints")
    opt = parser.parse_args()

    lr = opt.lr
    device = opt.device
    n_node = opt.nodes
    
    train(
        opt.nodes, 
        opt.ants, 
        opt.steps, 
        opt.epochs, 
        batch_size = opt.batch_size, 
        test_size = opt.test_size, 
        pretrained = opt.pretrained,
        savepath = opt.output,
    )