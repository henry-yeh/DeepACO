import torch
import numpy as np
import numba as nb
from torch.distributions import Categorical
import random
import concurrent.futures
from torch_geometric.data import Data, Batch



class Dynamic_AS():

    def __init__(self, 
                 coor,
                 nets,
                 k_sparse,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 two_opt=False,
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
        self.pheromone = torch.ones_like(self.distances)*0.5
        self.two_opt = two_opt
        # heuristic for test
        # self.heuristic = (1 / distances).unsqueeze(0).repeat(self.n_ants, 1, 1) if heuristic is None else heuristic
        self.shortest_path = None
        self.lowest_cost = float('inf')
        self.device = device

    def sample(self, inference=False):
        if inference:
            probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
            paths = inference_batch_sample(probmat.cpu().numpy(), self.n_ants, 0)
            paths = torch.from_numpy(paths.T.astype(np.int64)).to(self.device)
            costs = self.gen_path_costs(paths)
            return costs, None
        else:
            paths, log_probs = self.ar_gen_path(require_prob=True)
            costs = self.gen_path_costs(paths)
            return costs, log_probs

    @torch.no_grad()
    def run(self, n_iterations, inference = False):
        for _ in range(n_iterations):
            if inference:
                probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
                paths = inference_batch_sample(probmat.cpu().numpy(), self.n_ants, 0)
                paths = torch.from_numpy(paths.T.astype(np.int64)).to(self.device)
            else:
                paths = self.ar_gen_path(require_prob=False)
            # if self.two_opt:
            #     paths = self.local_search(paths)
            costs = self.gen_path_costs(paths)
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
            self.update_pheronome(paths, costs)
        return self.lowest_cost
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
            self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost
    
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
        try: assert (self.distances[u, v] > 0).all()
        except: 
            for idx in range(self.n_ants):
                if not self.distances[u[idx], v[idx]].all():
                    print(u[idx])
                    print(self.distances[u[idx], v[idx]])
                    print(self.distances)
        return torch.sum(self.distances[u, v], dim=1)
    
    @torch.no_grad()
    def gen_pyg_batch(self, mask, starting_nodes, terminating_nodes, k_sparse):
        # set the visted nodes to be very distant,
        # so when k_sparse < steps_per_stage, all the visited nodes have no connection with the unvisited ones
        # note we mask the current nodes after gen_pyg_data
        visited_index = torch.nonzero(1-mask).transpose(1, 0)
        coor_repeated = self.coor.repeat(self.n_ants, 1, 1)
        coor_repeated[visited_index[0], visited_index[1]] = 5 # set the coors of visited nodes to (5, 5)
        coor_repeated[torch.arange(self.n_ants), terminating_nodes] = self.coor[terminating_nodes]
        dist_mats = torch.cdist(coor_repeated, coor_repeated, 2)
        topk_values, topk_indices = torch.topk(dist_mats, k=k_sparse, dim=2, largest=False)
        
        # scale the sub-graph
        max_valid_dist = topk_values[topk_values < 2].max()
        topk_values = topk_values/max_valid_dist
        
        # construct data batch
        edge_index_list = [torch.stack([
            torch.repeat_interleave(torch.arange(self.problem_size).to(topk_indices.device),
                                    repeats=k_sparse),
            torch.flatten(topk_indices[i])
            ]) for i in range(self.n_ants)]
        edge_attr = topk_values.reshape(self.n_ants, -1, 1)
        x = torch.zeros((self.n_ants, self.problem_size, 1), device=self.device)
        x[torch.arange(self.n_ants), starting_nodes] = x[torch.arange(self.n_ants), terminating_nodes] = 1
        pyg_data_list = [Data(x=x[i], edge_index=edge_index_list[i], edge_attr=edge_attr[i]) for i in range(self.n_ants)]
        pyg_batch = Batch.from_data_list(pyg_data_list)
        return pyg_batch
        
    def infer_heuristic(self, net, mask, starting_nodes, terminating_nodes, step):
        '''
        Args:
            mask: (n_ants, p_size), 0 for visited and 1 for unvisited
            starting/terminating nodes: (n_ants, )
        Returns:
            heuristic: updated heuristic measures
        '''
        # the sub-graph could be smaller than self.k_sparse
        k_sparse = min(self.k_sparse, self.problem_size - step + 1)
        batched_pyg_data = self.gen_pyg_batch(mask, starting_nodes, terminating_nodes, k_sparse)
        heatmaps = net(batched_pyg_data)
        heatmaps = net.reshape_batch(batched_pyg_data, heatmaps, self.n_ants, self.problem_size, k_sparse) + 1e-10
        return heatmaps
    
    def ar_gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        # start = torch.zeros((self.n_ants, ), dtype = torch.long, device=self.device)
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        index = torch.arange(self.n_ants, device=self.device)
        dynamic_heuristic = self.infer_heuristic(self.nets[0], mask, start, start, 1)
        # print('initial heuristic', dynamic_heuristic.shape)
        # print(dynamic_heuristic)
        mask[index, start] = 0 # mask after inferring heuristics
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        actions = start
        for step in range(2, self.problem_size + 1):
            # dynamic version:
            # the pheromone matrix is of shape (p_size, p_size)
            # the heuristic tensor is of shape (n_ants, p_size, p_size)
            # broadcasting to obtain prob_mat of shape (n_ants, p_size, p_size)
            prob_mat = (self.pheromone ** self.alpha) * (dynamic_heuristic ** self.beta)
            dist = prob_mat[index, actions] * mask # for dynamic version, modify the index here
            dist = Categorical(dist, validate_args=False)
            actions = dist.sample() # shape: (n_ants,)
            paths_list.append(actions)
            if require_prob:
                log_probs = dist.log_prob(actions) # shape: (n_ants,)
                log_probs_list.append(log_probs)
                mask = mask.clone()
            if step % self.steps_per_stage == 0 and step//self.steps_per_stage < self.n_stage:
                # print(paths_list)
                # print('Update heuristic,', 'using #{} net'.format(step//self.steps_per_stage))
                # print(step, self.steps_per_stage, step//self.steps_per_stage)
                dynamic_heuristic = self.infer_heuristic(self.nets[step//self.steps_per_stage], \
                    mask, actions, start, step)
                # print('2stage heuristic', dynamic_heuristic.shape)
                # print(dynamic_heuristic)
            mask[index, actions] = 0  # mask after inferring heuristics
        if require_prob:
            # self.check_feasibility(torch.stack(paths_list).transpose(1, 0))
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
    
    # def local_search(self, paths):
    #     paths = batched_two_opt_python(self.distances.cpu().numpy(), paths.T.cpu().numpy(), max_iterations=100)
    #     return torch.from_numpy(paths.T.astype(np.int64)).to(self.device)

    # def check_feasibility(self, paths):
    #     pass
    #     assert paths.shape == (self.n_ants, self.problem_size)
    #     n = paths.size(1)
    #     for i in range(n):
    #         for idx in range(paths.size(0)):
    #             try: assert i in paths[idx]
    #             except: print(paths[idx])

if __name__ == '__main__':
    torch.set_printoptions(precision=3,sci_mode=False)
    from net import Net
    n_node = 100
    n_ants = 10
    k_sparse = 20 
    n_stages = 10
    device = 'cpu'
    nets = torch.nn.ModuleList([Net(feats=1).to(device) for _ in range(n_stages)])
    
    coor = torch.rand((n_node, 2))
    ant_system = Dynamic_AS(coor, nets, k_sparse, n_ants=n_ants)
    costs, probs = ant_system.sample(inference=False)