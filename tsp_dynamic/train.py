import time
import torch
import argparse
import pprint as pp

from net import Net
from aco import Dynamic_AS as ACO
from utils import load_val_dataset

LR = 5e-4
TEST_SIZE = 50
K_SPARSE = {
    100: 20,
    200: 40,
    500: 50,
    1000: 100
}

def train_instance(models, optimizer, data, n_ants, k_sparse):
    sum_loss = 0.
    count = 0
    print_costs = 0
    for coor in data:
        aco = ACO(
            coor,
            nets=models,
            k_sparse=k_sparse,
            device=DEVICE,
            n_ants=n_ants
        )
        costs, log_probs = aco.ar_sample(require_prob=True)
        baseline = costs.mean()
        reinforce_loss = torch.sum((costs - baseline) * log_probs.sum(dim=0)) / aco.n_ants
        sum_loss += reinforce_loss
        count += 1
        print_costs += costs.mean().item()
    sum_loss = sum_loss/count
    optimizer.zero_grad()
    sum_loss.backward()
    for model in models:
        torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 1.0, norm_type = 2)
    optimizer.step()

def infer_instance(nets, coor, n_ants, k_sparse):
    nets.eval()
    aco = ACO(
        coor,
        nets=nets,
        k_sparse=k_sparse,
        device=DEVICE,
        n_ants=n_ants
    )
    costs, _ = aco.ar_sample(require_prob=False)
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    return baseline.item(), best_sample_cost.item()

def generate_traindata(count, n_node):
    for _ in range(count):
        instance = torch.rand(size=(n_node, 2), device=DEVICE)
        yield instance

def train_epoch(n_node, n_ants, k_sparse, steps_per_epoch, nets, optimizer, batch_size):
    nets.train()
    for _ in range(steps_per_epoch):
        train_instance(nets, optimizer, generate_traindata(batch_size, n_node), n_ants, k_sparse)


@torch.no_grad()
def validation(n_ants, epoch, nets, val_dataset, k_sparse):
    sum_bl, sum_sample_best = 0, 0
    for coor in val_dataset:
        coor = coor.to(DEVICE)
        bl, sample_best = infer_instance(nets, coor, n_ants, k_sparse)
        sum_bl += bl
        sum_sample_best += sample_best
    
    n_val = len(val_dataset)
    avg_bl, avg_sample_best = sum_bl/n_val, sum_sample_best/n_val
    print(f"epoch {epoch}", avg_bl, avg_sample_best)
    return avg_bl, avg_sample_best

def train(n_node, k_sparse, n_ants, steps_per_epoch, epochs, batch_size, n_stages):
    nets = torch.nn.ModuleList([Net(feats=1).to(DEVICE) for _ in range(n_stages)])
    optimizer = torch.optim.AdamW(nets.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    val_dataset = load_val_dataset(n_node)[:TEST_SIZE]
    at_avg_bl, at_avg_sample_best = validation(n_ants, -1, nets, val_dataset, k_sparse)
    sum_time = 0
    for epoch in range(0, epochs):
        start = time.time()
        train_epoch(n_node, n_ants, k_sparse, steps_per_epoch, nets, optimizer, batch_size)
        scheduler.step()
        sum_time += time.time() - start
        avg_bl, avg_sample_best = validation(n_ants, epoch, nets, val_dataset, k_sparse)
        if avg_sample_best < at_avg_sample_best:
            at_avg_sample_best = avg_sample_best
            torch.save(nets.state_dict(), f'../pretrained/tsp_dyna/tsp{n_node}-{n_stages}.pt')
            print(f'[*] Save checkpoint {epoch}!')
    print('total training duration:', sum_time)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int)
    parser.add_argument('--device_id', type=str, default='0', help='CUDA device id')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=96, help='Updates for each epoch')
    parser.add_argument('--n_stages', type=int, default=2, help='Number of stages for the dynamic neural heuristic measures')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_ants', type=int, default=20, help='Number of ants for parallel sampling')
    opts = parser.parse_args()
    
    pp.pprint(vars(opts))
    DEVICE = opts.device = 'cuda:' + opts.device_id if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(opts.seed)
    train(
        opts.problem_size, 
        K_SPARSE[opts.problem_size], 
        opts.n_ants,
        opts.steps_per_epoch,
        opts.epochs,
        opts.batch_size,
        opts.n_stages
        )