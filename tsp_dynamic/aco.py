import torch
from torch.distributions import Categorical
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
        assert self.steps_per_stage > self.k_sparse
        self.pheromone = torch.ones_like(self.distances)*0.1
        self.two_opt = two_opt
        self.shortest_path = None
        self.lowest_cost = float('inf')
        self.device = device

    def sample(self, require_prob=False):
        paths, log_probs = self.ar_gen_path(require_prob=require_prob)
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
    def gen_pyg_batch(self, mask, starting_nodes, terminating_nodes, k_sparse):
        # set the visted nodes to be very distant,
        # so when k_sparse < steps_per_stage, all the visited nodes have no connection with the unvisited ones
        # note we mask the current nodes after gen_pyg_data
        visited_index = torch.nonzero(1-mask) # shape (x, 2)
        coor_repeated = self.coor.repeat(self.n_ants, 1, 1)
        coor_repeated[visited_index[:, 0], visited_index[:, 1]] = 5 # set the coors of visited nodes to (5, 5)
        coor_repeated[torch.arange(self.n_ants), terminating_nodes] = self.coor[terminating_nodes]
        dist_mats = torch.cdist(coor_repeated, coor_repeated, 2)
        
        # prevent any self-loops
        dist_mats[:, torch.arange(self.problem_size), torch.arange(self.problem_size)] = 10
        
        topk_values, topk_indices = torch.topk(dist_mats, k=k_sparse, dim=2, largest=False)
        
        # scale the sub-graph
        max_valid_dist = topk_values[topk_values < 2].max()
        topk_values = topk_values/max_valid_dist
        assert (topk_values <= 1).all(), 'Try a smaller k_sparse'
        
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
        
    def infer_heuristic(self, net, mask, starting_nodes, terminating_nodes):
        '''
        Args:
            mask: (n_ants, p_size), 0 for visited and 1 for unvisited
            starting/terminating nodes: (n_ants, )
        Returns:
            heuristic: updated heuristic measures
        '''
        batched_pyg_data = self.gen_pyg_batch(mask, starting_nodes, terminating_nodes, self.k_sparse)
        heatmaps = net(batched_pyg_data)
        heatmaps = net.reshape_batch(batched_pyg_data, heatmaps, self.n_ants, self.problem_size, self.k_sparse) + 1e-10
        return heatmaps
    
    def ar_gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        index = torch.arange(self.n_ants, device=self.device)
        dynamic_heuristic = self.infer_heuristic(self.nets[0], mask, start, start)
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
                dynamic_heuristic = self.infer_heuristic(self.nets[step//self.steps_per_stage], \
                    mask, actions, start)
            mask[index, actions] = 0  # mask after inferring heuristics
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list), None
        
if __name__ == '__main__':
    torch.set_printoptions(precision=3,sci_mode=False)
    from net import Net
    n_node = 200
    n_ants = 10
    k_sparse = 40
    n_stages = 4
    device = 'cpu'
    nets = torch.nn.ModuleList([Net(feats=1).to(device) for _ in range(n_stages)])
    
    coor = torch.rand((n_node, 2))
    ant_system = Dynamic_AS(coor, nets, k_sparse, n_ants=n_ants)
    costs, probs = ant_system.sample(inference=False)