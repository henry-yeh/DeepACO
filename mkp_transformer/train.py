import time
import torch
from torch.distributions import Categorical, kl

from net import TransformerModel as Net
from aco import ACO
from utils import *

torch.manual_seed(1234)

lr = 3e-4
M = 5
device = 'cuda:0'

def train_instance(model, optimizer, price, weight, n_ants):
    model.train()
    src = reformat(price, weight)
    heu_vec = model(src) + 1e-10
    aco = ACO(
        price=price,
        weight=weight,
        n_ants=n_ants,
        heuristic=heu_vec,
        device=device
        )
    objs, log_probs = aco.sample()
    baseline = objs.mean()
    reinforce_loss = torch.sum((baseline - objs) * log_probs.sum(dim=0)) / aco.n_ants
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()

def infer_instance(model, price, weight, n_ants):
    model.eval()
    src = reformat(price, weight)
    heu_vec = model(src) + 1e-10
    aco = ACO(
        price=price,
        weight=weight,
        n_ants=n_ants,
        heuristic=heu_vec,
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
        price, weight = gen_instance(problem_size, m=M, device=device)
        train_instance(net, optimizer, price, weight, n_ants)


@torch.no_grad()
def validation(n_ants, epoch, net, val_dataset, animator=None):
    sum_bl, sum_sample_best = 0, 0
    for price, weight in val_dataset:
        bl, sample_best = infer_instance(net, price, weight, n_ants)
        sum_bl += bl
        sum_sample_best += sample_best

    n_val = len(val_dataset)
    avg_bl, avg_sample_best = sum_bl/n_val, sum_sample_best/n_val
    if animator:
        animator.add(epoch+1, (avg_bl, avg_sample_best))
    
    return avg_bl, avg_sample_best

def train(problem_size, n_ants, steps_per_epoch, epochs):
    net = Net().to(device)
    print('total params:', sum(p.numel() for p in net.parameters())) 
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    val_list = load_val_dataset(problem_size, device)
    
    avg_bl, avg_best = validation(n_ants, -1, net, val_list)
    val_results = [(avg_bl, avg_best)]
    
    sum_time = 0
    for epoch in range(0, epochs):
        start = time.time()
        train_epoch(problem_size, n_ants, epoch, steps_per_epoch, net, optimizer)
        sum_time += time.time() - start
        avg_bl, avg_sample_best = validation(n_ants, epoch, net, val_list)
        val_results.append((avg_bl, avg_sample_best))
        
    print('total training duration:', sum_time)
    
    for epoch in range(-1, epochs):
        print(f'epoch {epoch}:', val_results[epoch+1])
        
    torch.save(net.state_dict(), f'./pretrained/mkp_transformer/mkp{problem_size}.pt')
    
    
if __name__ == '__main__':
    n_node, n_ants = 50, 50
    steps_per_epoch = 256
    epochs = 5
    for n_node in [300, 500]:
        train(n_node, n_ants, steps_per_epoch, epochs)