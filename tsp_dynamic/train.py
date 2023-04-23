import time
import torch
from torch.distributions import Categorical, kl
from net import Net
from aco import Dynamic_AS as ACO
from utils import gen_pyg_data, load_val_dataset

torch.manual_seed(1234)

lr = 3e-4
EPS = 1e-10
T=5
device = 'cuda:0'
TEST_SIZE = 50

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
            n_ants=n_ants
        )
        costs, log_probs = aco.sample(require_prob=True)
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
    
    # print(print_costs / batch_size)

def infer_instance(nets, coor, n_ants, k_sparse):
    nets.eval()
    aco = ACO(
        coor,
        nets=nets,
        k_sparse=k_sparse,
        device=device,
        n_ants=n_ants
    )
    costs, log_probs = aco.sample(require_prob=False)
    baseline = costs.mean()
    best_sample_cost = torch.min(costs)
    return baseline.item(), best_sample_cost.item()

def generate_traindata(count, n_node):
    for _ in range(count):
        instance = torch.rand(size=(n_node, 2), device=device)
        # yield gen_pyg_data(instance, k_sparse=k_sparse, start_node=0)
        yield instance

def train_epoch(n_node, n_ants, k_sparse, steps_per_epoch, nets, optimizer, batch_size):
    nets.train()
    for _ in range(steps_per_epoch):
        train_instance(nets, optimizer, generate_traindata(batch_size, n_node), n_ants, k_sparse)


@torch.no_grad()
def validation(n_ants, epoch, nets, val_dataset, k_sparse):
    sum_bl, sum_sample_best = 0, 0
    for coor in val_dataset:
        bl, sample_best = infer_instance(nets, coor, n_ants, k_sparse)
        sum_bl += bl
        sum_sample_best += sample_best
    
    n_val = len(val_dataset)
    avg_bl, avg_sample_best = sum_bl/n_val, sum_sample_best/n_val
    print(f"epoch {epoch}", avg_bl, avg_sample_best)
    return avg_bl, avg_sample_best

def train(n_node, k_sparse, n_ants, steps_per_epoch, epochs, batch_size, n_stages):
    nets = torch.nn.ModuleList([Net(feats=1).to(device) for _ in range(n_stages)])
    optimizer = torch.optim.AdamW(nets.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    val_dataset = load_val_dataset(n_node).to(device)[:TEST_SIZE]
    validation(n_ants, -1, nets, val_dataset, k_sparse)
    sum_time = 0
    for epoch in range(0, epochs):
        start = time.time()
        train_epoch(n_node, n_ants, k_sparse, steps_per_epoch, nets, optimizer, batch_size)
        scheduler.step()
        sum_time += time.time() - start
        validation(n_ants, epoch, nets, val_dataset, k_sparse)
    print('total training duration:', sum_time)
        
    # torch.save(net.state_dict(), f'../pretrained/tsp_2opt/tsp{n_node}.pt')
    
if __name__ == '__main__':
    n_node = 200
    n_ants = 10
    k_sparse = 20
    steps_per_epoch = 64
    epochs = 20
    batch_size = 5
    n_stages = 1
    train(n_node, k_sparse, n_ants, steps_per_epoch, epochs, batch_size, n_stages)