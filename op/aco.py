import torch
from torch.distributions import Categorical
import numpy as np

class ACO():

    def __init__(self,
                 distances,
                 prizes,
                 max_len,
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
        
        self.n = len(prizes)
        self.distances = distances
        self.prizes = prizes
        self.max_len = max_len
        
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

        self.heuristic = self.prizes.unsqueeze(0) / self.distances if heuristic is None else heuristic
        
        self.Q = 1 / prizes.sum()
        
        self.alltime_best_sol = None
        self.alltime_best_obj = 0

        self.device = device

    @torch.no_grad()
    def sparsify(self, k_sparse):
        '''
        Sparsify the OP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(self.distances, 
                                        k=k_sparse, 
                                        dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(self.distances), device=self.device),
            repeats=k_sparse
            )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(self.distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = self.distances[edge_index_u, edge_index_v]
        self.heuristic = self.prizes.unsqueeze(0) / sparse_distances  
    
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
            u = torch.tensor(sol, device=self.device)
            obj = self.prizes[u].sum()
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
            solution = [0]
            mask = torch.ones(size=(self.n,), device=self.device)
            done = False
            travel_dis = 0
            cur_node = 0
            mask = self.update_mask(travel_dis, cur_node, mask)
            done = self.check_done(mask)
            # construction
            while not done:
                nxt_node, log_prob = self.pick_node(mask, cur_node, require_prob) # pick action
                # update solution and log_probs
                solution.append(nxt_node) 
                log_probs.append(log_prob)
                # update travel_dis, cur_node and mask
                travel_dis += self.distances[cur_node, nxt_node]
                cur_node = nxt_node
                mask = mask.clone()
                mask = self.update_mask(travel_dis, cur_node, mask)
                # check done
                done = self.check_done(mask)
            # record
            solutions.append(solution)
            log_probs_list.append(log_probs)
        if require_prob:
            return solutions, log_probs_list  # shape: [n_ant, variable]
        else:
            return solutions
    
    def pick_node(self, mask, cur_node, require_prob):
        pheromone, heuristic = self.pheromone[cur_node], self.heuristic[cur_node]
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask) # shape [self.n,]
        dist = Categorical(dist)
        item = dist.sample()
        log_prob = dist.log_prob(item) if require_prob else None
        return item.item(), log_prob
    
    def update_mask(self, travel_dis, cur_node, mask):
        mask[cur_node] = 0
        candidates = torch.nonzero(mask).squeeze()
        # after going to candidate node from cur_node, can it return to depot?
        trails = travel_dis + self.distances[cur_node, candidates] + self.distances[candidates, 0]
        fail_idx = candidates[trails > self.max_len]
        mask[fail_idx] = 0
        return mask
    
    def check_done(self, mask):
        # is all masked ?
        return (mask == 0).all()
        

if __name__ == '__main__':
    torch.set_printoptions(precision=10,sci_mode=False)
    from utils import *
    coor = torch.rand(size=(50, 2))
    prizes, max_len = gen_prizes(coor), 4
    distances = gen_distance_matrix(coor)
    aco = ACO(distances=distances, prizes=prizes, max_len=max_len, n_ants=20)
    print(aco.run(1))
    print(aco.run(5))
    print(aco.run(10))
    print(aco.run(100))
    