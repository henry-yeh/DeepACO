import torch
from torch.distributions import Categorical

class ACO():

    def __init__(self, 
                 distances,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                 device='cpu'
                 ):
        
        self.problem_size = len(distances)
        self.distances  = distances
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max
        
        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = None
        
        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

        self.heuristic_list = heuristic
        self.n_heads = len(heuristic)
        self.n_ants_group = self.n_ants // self.n_heads

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device
    
    def sample(self):
        paths_groups = []
        log_probs_groups = []
        costs_groups = []
        for i in range(self.n_heads):
            paths, log_probs = self.gen_path(i, require_prob=True)
            costs = self.gen_path_costs(paths)
            paths_groups.append(paths)
            log_probs_groups.append(log_probs)
            costs_groups.append(costs)
        return costs_groups, log_probs_groups

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths_groups = []
            costs_groups = []
            for i in range(self.n_heads):
                _paths = self.gen_path(i, require_prob=False)
                _costs = self.gen_path_costs(_paths)
                paths_groups.append(_paths)
                costs_groups.append(_costs)
            paths = torch.cat(paths_groups, dim=1)
            costs = torch.cat(costs_groups)
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
                if self.min_max:
                    max = self.problem_size / self.lowest_cost
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max
            
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
        
        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += 1.0/best_cost
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += 1.0/best_cost
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
                self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost
                
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path(self, i, require_prob=False):
        '''
        Tour contruction for ants of group i
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants_group,), device=self.device)
        mask = torch.ones(size=(self.n_ants_group, self.problem_size), device=self.device)
        mask[torch.arange(self.n_ants_group, device=self.device), start] = 0
        
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)
        
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        prev = start
        for _ in range(self.problem_size-1):
            actions, log_probs = self.pick_move(prev, mask, i, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[torch.arange(self.n_ants_group, device=self.device), actions] = 0
        
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, mask, i, require_prob):
        '''
        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
        '''
        pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
        heuristic = self.heuristic_list[i][prev] # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask) # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample() # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None # shape: (n_ants,)
        return actions, log_probs
        


if __name__ == '__main__':
    torch.set_printoptions(precision=3,sci_mode=False)
    input = torch.rand(size=(5, 2))
    distances = torch.norm(input[:, None] - input, dim=2, p=2)
    distances[torch.arange(len(distances)), torch.arange(len(distances))] = 1e10
    aco = ACO(distances)
    aco.sparsify(k_sparse=3)
    print(aco.run(20))