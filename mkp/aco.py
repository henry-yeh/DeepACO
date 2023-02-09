import torch
from torch.distributions import Categorical
import numpy as np

class ACO():

    def __init__(self,                # note that constraints are set to 1 after normalize weight 
                 price: torch.tensor, # shape [n,]
                 weight: torch.tensor, # shape [m, n]
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
        
        self.price = price
        self.weight = weight
        self.n = len(price)
        self.m = len(weight)
        assert self.weight.shape == (self.m, self.n)
        
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
            self.pheromone = torch.ones_like(self.price)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

        # Fidanova S. Hybrid ant colony optimization algorithm for multiple knapsack problem
        self.heuristic = price / weight.sum(dim=0) if heuristic is None else heuristic
        
        # Leguizamon G, Michalewicz Z. A New Version of Ant System for Subset Problems
        self.Q = 1/self.price.sum()
        
        self.alltime_best_sol = None
        self.alltime_best_obj = 0

        self.device = device
    
    def sample(self):
        sols, log_probs = self.gen_sol(require_prob=True)
        objs = self.gen_sol_obj(sols)
        return objs, log_probs

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            sols = self.gen_sol(require_prob=False)
            objs = self.gen_sol_obj(sols)
            best_obj, best_idx = torch.stack(objs).max(dim=0)
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx.item()]
                if self.min_max:
                    max = self.alltime_best_obj * self.n * self.Q
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max
            self.update_pheronome(sols, objs, best_obj.item(), best_idx.item())

        return self.alltime_best_obj, self.alltime_best_sol
       
    
    @torch.no_grad()
    def update_pheronome(self, sols: list, objs: list, best_obj: float, best_idx: int):
        self.pheromone = self.pheromone * self.decay 
        if self.elitist:
            best_sol= sols[best_idx]
            self.pheromone[[best_sol]] += self.Q * best_obj
        
        else:
            for i in range(self.n_ants):
                sol = sols[i]
                obj = objs[i]
                self.pheromone[[sol]] += self.Q * obj
                
        if self.min_max:
            self.pheromone[(self.pheromone>1e-9) * (self.pheromone)<self.min] = self.min
            self.pheromone[self.pheromone>self.max] = self.max
    
    @torch.no_grad()
    def gen_sol_obj(self, solutions: list):
        objs = []
        for sol in solutions:
            obj = 0
            for item in sol:
                obj += self.price[item]
            objs.append(obj)
        return objs

    def gen_sol(self, require_prob=False):
        '''
        Solution contruction for all ants
        '''
        solutions = []
        log_probs_list = []
        for ant in range(self.n_ants):
            # init
            log_probs = []
            solution = []
            knapsack = torch.zeros(size=(self.m,), device=self.device)  # used capacity
            mask = torch.ones(size=(self.n,), device=self.device)
            mask, knapsack = self.update_knapsack(mask, knapsack, new_item=None)
            done = False
            # construction
            while not done:
                item, log_prob = self.pick_item(mask, require_prob)
                solution.append(item)
                log_probs.append(log_prob)
                mask = mask.clone()
                mask, knapsack = self.update_knapsack(mask, knapsack, item)
                done = self.check_done(mask) 
            solutions.append(solution)
            log_probs_list.append(log_probs)
            
        if require_prob:
            return solutions, log_probs_list  # shape: [n_ant, variable]
        else:
            return solutions
    
    def pick_item(self, mask, require_prob):
        dist = ((self.pheromone ** self.alpha) * (self.heuristic ** self.beta) * mask) # shape [self.n,]
        dist = Categorical(dist)
        item = dist.sample()
        log_prob = dist.log_prob(item) if require_prob else None
        # print(log_prob)
        return item.item(), log_prob
    
    def check_done(self, mask):
        # is mask all zero ?
        return (mask == 0).all()
    
    def update_knapsack(self, mask, knapsack, new_item):
        if new_item is not None:
            mask[new_item] = 0
            knapsack += self.weight[:, new_item]
        for test_idx in torch.nonzero(mask):
            new_knapsack = knapsack + self.weight[:, test_idx.item()]
            if not (new_knapsack <= 1).all():
                mask[test_idx] = 0
        return mask, knapsack

if __name__ == '__main__':
    torch.set_printoptions(precision=3,sci_mode=False)
    from utils import gen_instance
    price, weight = gen_instance(5)
    print('price:')
    print(price)
    print('weight:')
    print(weight)
    
    aco = ACO(price=price, weight=weight, n_ants=2)
    solutions, log_probs_list = aco.gen_sol(require_prob=True)
    
    print(solutions)
    print(log_probs_list)
    
    objs = aco.gen_sol_obj(solutions)
    print('objs', objs)
    print(torch.stack(objs))