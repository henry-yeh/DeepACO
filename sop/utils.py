import torch
from torch_geometric.data import Data

def ordering_constraint_gen(n, rand=0.2):
    r = []
    for i in range(1, n):
        r.append((0, i))

    a = [i for i in range(1, n)]
    precede = [set() for i in range(1, n)]
    for i in range(n - 3, -1, -1):
        for j in range(i + 1, n - 1):
            if torch.rand(size=(1,)) > rand:
                continue
            precede[i].add(j)
            for k in precede[j]:
                precede[i].add(k)
                
        for j in precede[i]:
            r.append((a[i], a[j]))
    return r

def adjacency_mat_gen(n, r):
    c = torch.ones(size=(n, n))
    c[torch.arange(n), torch.arange(n)] = 0
    for i, j in r: # i precedes j
        c[j][i] = 0
    return c

def preceding_mat_gen(n, r):
    '''
    The preceding nodes of node i are marked with 1 in prec_mat[i, :]
    '''
    prec_mat = torch.zeros(size=(n, n))
    for i, j in r:
        prec_mat[j, i] = 1
    return prec_mat
    
def cost_mat_gen(n):
    distances = torch.rand(size=(n, n))
    job_processing_cost = distances[0, :]
    distances[1:, :] += job_processing_cost
    return distances

def training_instance_gen(n, device):
    distance = cost_mat_gen(n).to(device)
    r = ordering_constraint_gen(n)
    mask = preceding_mat_gen(n, r).to(device)
    adj_mat = adjacency_mat_gen(n, r).to(device)
    return distance, adj_mat, mask
    
def gen_pyg_data(distances, adj, device):
    edge_index = torch.nonzero(adj).T
    edge_attr = distances[adj.bool()].unsqueeze(-1)
    x = distances[0, :].unsqueeze(-1)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data

if __name__ == "__main__":
    distances, adj_mat, mask = training_instance_gen(5, 'cpu')
    pyg_data = gen_pyg_data(distances, adj_mat)
    from net import Net
    net = Net()
    net(pyg_data)
    
    