import os
import torch   

def load_val_dataset(n_node):
    if not os.path.isfile(f'../data/tsp/valDataset-{n_node}.pt'):
        val_tensor = torch.rand((50, n_node, 2))
        torch.save(val_tensor, f'../data/tsp/valDataset-{n_node}.pt')
    else:
        val_tensor = torch.load(f'../data/tsp/valDataset-{n_node}.pt')
    return val_tensor