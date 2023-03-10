import time
import torch
from torch.distributions import Categorical, kl

from net import Net
from aco import ACO
from utils import *

torch.manual_seed(1234)

lr = 3e-4
M = 5
device = 'cuda:0'

def train_instance(model, optimizer, prize, weight, n_ants):
    model.train()
    src = gen_pyg_data(prize, weight)
    heu_mat = model(src).reshape((prize.size(0), prize.size(0)))
    heu_mat = heu_mat / (heu_mat.min() + 1e-10) + 1e-10
    aco = ACO(
        prize=prize,
        weight=weight,
        n_ants=n_ants,
        heuristic=heu_mat,
        device=device
        )
    objs, log_probs = aco.sample()
    baseline = objs.mean()
    reinforce_loss = torch.sum((baseline - objs) * log_probs.sum(dim=0)) / aco.n_ants
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()

def infer_instance(model, prize, weight, n_ants):
    model.eval()
    src = gen_pyg_data(prize, weight)
    heu_mat = model(src).reshape((prize.size(0), prize.size(0)))
    heu_mat = heu_mat / (heu_mat.min() + 1e-10) + 1e-10
    aco = ACO(
        prize=prize,
        weight=weight,
        n_ants=n_ants,
        heuristic=heu_mat,
        device=device
        )
    objs, log_probs = aco.sample()
    baseline = objs.mean()
    best_sample_obj = objs.max()
    return baseline.item(), best_sample_obj.item()

def train_epoch(problem_size,
                n_ants, 
                epoch, 
                steps_per_epoch, 
                net, 
                optimizer
                ):
    for _ in range(steps_per_epoch):
        prize, weight = gen_instance(problem_size, m=M, device=device)
        train_instance(net, optimizer, prize, weight, n_ants)


@torch.no_grad()
def validation(n_ants, epoch, net, valDataset):
    sum_bl, sum_sample_best = 0, 0
    n_val = 100
    for prize, weight in valDataset:
        bl, sample_best = infer_instance(net, prize, weight, n_ants)
        sum_bl += bl
        sum_sample_best += sample_best
        
    avg_bl, avg_sample_best = sum_bl/n_val, sum_sample_best/n_val
    return avg_bl, avg_sample_best

def train(problem_size, n_ants, steps_per_epoch, epochs):
    net = Net().to(device)
    print('total params:', sum(p.numel() for p in net.parameters())) 
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)
    
    valDataset = load_val_dataset(problem_size, device)
    avg_bl, avg_best = validation(n_ants, -1, net, valDataset)
    val_results = [(avg_bl, avg_best)]
    
    all_time_best = 0
    sum_time = 0
    for epoch in range(0, epochs):
        start = time.time()
        train_epoch(problem_size, n_ants, epoch, steps_per_epoch, net, optimizer)
        sum_time += time.time() - start
        avg_bl, avg_sample_best = validation(n_ants, epoch, net, valDataset)
        val_results.append((avg_bl, avg_sample_best))
        if avg_sample_best > all_time_best:
            all_time_best = avg_sample_best
            print(f'------save ckpt {epoch}-------')
            torch.save(net.state_dict(), f'./pretrained/mkp/mkp{problem_size}.pt')
        
    print('total training duration:', sum_time)
    
    for epoch in range(-1, epochs):
        print(f'epoch {epoch}:', val_results[epoch+1])
        
    
    
if __name__ == '__main__':
    import sys
    n_node, n_ants = 50, 50
    steps_per_epoch = 64
    epochs = 10
    for n_node in sys.argv[1:]:
        n_node = int(n_node)
        train(n_node, n_ants, steps_per_epoch, epochs)