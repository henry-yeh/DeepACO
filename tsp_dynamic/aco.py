import torch
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
import numpy as np


class Dynamic_AS():

    def __init__(self, 
                 coor,
                 nets,
                 k_sparse,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu',
                 ):
        self.coor = coor.to(device)
        self.problem_size = coor.size(0)
        self.distances = torch.cdist(coor, coor, 2)
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.nets = nets
        self.n_stage = len(nets) 
        self.steps_per_stage = self.problem_size // self.n_stage
        self.k_sparse = k_sparse
        assert self.problem_size % self.n_stage == 0
        assert self.steps_per_stage > self.k_sparse
        self.pheromone = torch.ones_like(self.distances)*0.1
        self.index = torch.arange(self.n_ants)
        self.shortest_path = None
        self.lowest_cost = float('inf')
        self.device = device

    def ar_sample(self, require_prob=False):
        
        log_probs_list = []
        # sample starting nodes for all ants, append to paths_list
        cur_nodes = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        mask[self.index, cur_nodes] = 0
        paths_list = [cur_nodes]
        for stage in range(self.n_stage):
            # generate prob_mats
            prob_mats = self.infer_heuristic(self.nets[stage], paths_list)
            path, log_prob, mask = self.gen_path(stage, cur_nodes, prob_mats, mask, require_prob=require_prob)
            cur_nodes = path[-1]
            paths_list += path
            log_probs_list += log_prob
        if require_prob:
            paths, log_probs = torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            paths, log_probs = torch.stack(paths_list), []
        costs = self.gen_path_costs(paths)
        return costs, log_probs

    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
            Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        return torch.sum(self.distances[u, v], dim=1)
    
    @torch.no_grad()
    def gen_pyg_batch(self, path):
        visited_nodes = path[:, 1:-1] # (n_ants, x)
        current_nodes = path[:, -1]
        terminating_nodes = path[:, 0]
        
        coor_repeated = self.coor.repeat(self.n_ants, 1, 1)
        # set the coors of visited nodes to a far away place
        coor_repeated[self.index.unsqueeze(-1), visited_nodes] = 100
        # set the two starting/termining nodes of SHPP back
        coor_repeated[self.index, terminating_nodes] = self.coor[terminating_nodes]
        coor_repeated[self.index, current_nodes] = self.coor[current_nodes]
        
        dist_mats = torch.cdist(coor_repeated, coor_repeated, 2)
        
        # prevent any self-loops
        dist_mats[:, torch.arange(self.problem_size), torch.arange(self.problem_size)] = 100
        
        # sparsify and scale
        topk_values, topk_indices = torch.topk(dist_mats, k=self.k_sparse, dim=2, largest=False)
        ant_wise_max_dist, _ = (topk_values.reshape(self.n_ants, -1)*(topk_values.reshape(self.n_ants, -1)<2)).max(dim=1)
        topk_values = topk_values/ant_wise_max_dist.reshape(-1, 1, 1)
        
        # construct data batch
        edge_index_list = [torch.stack([
            torch.repeat_interleave(torch.arange(self.problem_size).to(topk_indices.device),
                                    repeats=self.k_sparse),
            torch.flatten(topk_indices[i])
            ]) for i in range(self.n_ants)]
        edge_attr = topk_values.reshape(self.n_ants, -1, 1)
        x = torch.zeros((self.n_ants, self.problem_size, 1), device=self.device)
        x[torch.arange(self.n_ants), current_nodes] = x[torch.arange(self.n_ants), terminating_nodes] = 1      
        pyg_data_list = [Data(x=x[i], edge_index=edge_index_list[i], edge_attr=edge_attr[i]) for i in range(self.n_ants)]
        pyg_batch = Batch.from_data_list(pyg_data_list)
        return pyg_batch
        
    def infer_heuristic(self, net, path_list):
        '''
        Args:
            mask: (n_ants, p_size), 0 for visited and 1 for unvisited
            starting/terminating nodes: (n_ants, )
        Returns:
            heuristic: updated heuristic measures
        '''
        path = torch.stack(path_list).transpose(1, 0)
        batched_pyg_data = self.gen_pyg_batch(path)
        heatmap_vector = net(batched_pyg_data)
        heatmaps = net.reshape_batch(batched_pyg_data, heatmap_vector, self.n_ants, self.problem_size, self.k_sparse)
        return heatmaps + 1e-10
    
    def gen_path(self, stage, cur_nodes, prob_mats, mask, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        actions = cur_nodes
        n_steps = self.steps_per_stage if stage !=0 else self.steps_per_stage-1
        for _ in range(n_steps):
            # dynamic version:
            # the pheromone matrix is of shape (p_size, p_size)
            # the heuristic tensor is of shape (n_ants, p_size, p_size)
            # broadcasting to obtain prob_mat of shape (n_ants, p_size, p_size)
            dist = prob_mats[self.index, actions] * mask # (n_ants, p_size)
            dist = Categorical(dist, validate_args=False)
            actions = dist.sample() # shape: (n_ants,)
            paths_list.append(actions)
            if require_prob:
                log_probs = dist.log_prob(actions) # shape: (n_ants,)
                log_probs_list.append(log_probs)
                mask = mask.clone()
            mask[self.index, actions] = 0
        if require_prob:
            return paths_list, log_probs_list, mask
        else:
            return paths_list, [], mask