import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np
import numba
from functools import cached_property

CAPACITY = 150


@numba.njit()
def count_last_zero(x: np.ndarray):
    ret = np.zeros(len(x),)
    n, m = x.shape
    for i in range(n):
        count = 0
        for j in range(m-1, -1, -1):
            if x[i][j] == 0:
                count += 1
            else:
                ret[i] = count
                break
    return ret

@numba.njit()
def cal_fitness(s: np.ndarray, demand: np.ndarray, n_bins: np.ndarray):
    ret = np.zeros(len(s),)
    n, m =  s.shape
    for i in range(n):
        f = 0
        sub_f = 0
        for j in range(1, m):
            if s[i, j] != 0: # not dummy node
                sub_f += demand[s[i, j]]
            else:
                f += (sub_f / CAPACITY)**2
                sub_f = 0
        ret[i] = f / n_bins[i]
    return ret
            


class ACO():

    # Levine, J., & Ducatelle, F. (2004). Ant colony optimization and local search for bin packing and cutting stock problems. 
    # Journal of the Operational Research society, 55(7), 705-716.

    def __init__(self,  # 0: depot
                 demand,   # (n, )
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 pheromone=None,
                 heuristic=None,
                 device='cpu',
                 capacity=CAPACITY
                 ):
        
        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = demand
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        
        self.pheromone = torch.ones(self.problem_size, self.problem_size) if pheromone is None else pheromone
        self.heuristic = self.demand.unsqueeze(0).repeat(len(demand), 1) if heuristic is None else heuristic
        self.heuristic[:, 0] = 1e-5

        self.shortest_path = None
        self.best_fitness = 0

        self.device = device
    
    def sample(self):
        paths, log_probs = self.gen_path(require_prob=True)
        costs = self.gen_path_costs(paths)
        return costs, log_probs
        

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.gen_path(require_prob=False)
            costs = self.gen_path_costs(paths)
            best_cost, best_idx = costs.min(dim=0)
            if  - best_cost > self.best_fitness:
                self.shortest_path = paths[:, best_idx]
                self.best_fitness = - best_cost
            self.update_pheronome(paths, -costs)
        return self.best_fitness
       
    @torch.no_grad()
    def update_pheronome(self, paths, fits):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        
        if self.elitist:
            best_fit, best_idx = fits.max(dim=0)
            best_tour = paths[:, best_idx]
            self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += best_fit
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                fit = fits[i]
                self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += fit / self.n_ants
        
        self.pheromone[self.pheromone < 1e-10] = 1e-10
    
    @torch.no_grad()
    def gen_path_costs(self, paths:torch.Tensor):
        u = paths.permute(1, 0).numpy() # shape: (n_ants, max_seq_len)
        last_zeros = count_last_zero(u)
        n_bins = u.shape[1] - last_zeros - self.problem_size + 1 # number of bins
        fit = cal_fitness(u, self.demand_numpy, n_bins)
        return -torch.tensor(fit)


    def gen_path(self, require_prob=False):
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)
        
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        
        paths_list = [actions] # paths_list[i] is the ith move (tensor) for all ants
        
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        done = self.check_done(visit_mask, actions)
        while not done:
            actions, log_probs = self.pick_move(actions, visit_mask, capacity_mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            done = self.check_done(visit_mask, actions)
            
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, visit_mask, capacity_mask, require_prob):
        pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask) # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample() # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None # shape: (n_ants,)
        return actions, log_probs
    
    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1) # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        
        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
    
    @cached_property
    def demand_numpy(self):
        return self.demand.cpu().numpy()


if __name__=="__main__":
    from utils import gen_instance
    n=120
    demands = gen_instance(n, 'cpu')
    aco = ACO(
        demand=demands,
        n_ants=20
    )
    aco.run(100)
