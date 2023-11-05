import torch
from torch import Tensor
import numpy as np

def gen_instance(n: int, m=2, device='cpu'):
    '''
    Generate *well-stated* MKP instances
    Args:
        n: # of knapsacks
        m: # of constraints, a.k.a., the problem dimensionality 
    '''
    price = torch.rand(size=(n,), device=device)
    weight_matrix = torch.rand(size=(m, n), device=device)
    max_weight, _ = torch.max(weight_matrix, dim=1)
    sum_weight = torch.sum(weight_matrix, dim=1)
    constraints = []
    for idx in range(m):
        constraint = np.random.uniform(low=max_weight[idx].item(), high=sum_weight[idx].item())
        constraints.append(constraint)
    constraints = torch.tensor(constraints, device=device)
    weight_matrix /= constraints.unsqueeze(1) # after norm, constraints are all 1
    return price, weight_matrix
    
def reformat(price: Tensor, weight: Tensor):
    '''
    Concatenate price tensor and weight tensor into input features for Transformer
    '''
    src = torch.cat((price.T.unsqueeze(-1), weight.T), dim=-1)
    src.unsqueeze_(1)
    return src # [seq_len, batch_size=1, emb_size=m+1]

def load_val_dataset(problem_size, device):
    val_list = []
    dataset = torch.load(f'./data/mkp_transformer/valDataset-{problem_size}.pt', map_location=device)
    for i in range(len(dataset)):
        val_list.append((dataset[i, 0], dataset[i, 1:]))
    return val_list

def load_test_dataset(problem_size, device):
    val_list = []
    dataset = torch.load(f'./data/mkp_transformer/testDataset-{problem_size}.pt', map_location=device)
    for i in range(len(dataset)):
        val_list.append((dataset[i, 0], dataset[i, 1:]))
    return val_list

if __name__ == '__main__':
    # generate val and test dataset
    import pathlib
    pathlib.Path('../data/mkp_transformer').mkdir(parents=False, exist_ok=True) 
    
    torch.manual_seed(12345)
    for problem_size in [300, 500]:
        testDataset = []
        for _ in range(30):
            price, weight = gen_instance(problem_size, 5)
            testDataset.append(torch.cat((price.unsqueeze(0), weight), dim=0))
        testDataset = torch.stack(testDataset)
        torch.save(testDataset, f'../data/mkp_transformer/valDataset-{problem_size}.pt')
    
    torch.manual_seed(123456)
    for problem_size in [300, 500]:
        testDataset = []
        for _ in range(100):
            price, weight = gen_instance(problem_size, 5)
            testDataset.append(torch.cat((price.unsqueeze(0), weight), dim=0))
        testDataset = torch.stack(testDataset)
        torch.save(testDataset, f'../data/mkp_transformer/testDataset-{problem_size}.pt')
            
    
    