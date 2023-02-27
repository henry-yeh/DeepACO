import torch
from torch.distributions import Categorical
from typing import Optional


def avg_adjacent_deg_heuristic(adjlist: list[list[int]]) -> torch.Tensor:
    # Li, Youmei, and Zongben Xu. An Ant Colony Optimization Heuristic for Solving Maximum Independent Set Problems.
    degrees = list(map(len, adjlist))
    heuristic = []
    for i, adjacent in enumerate(adjlist):
        adjdeg = sum(degrees[j] for j in adjacent)
        heu = adjdeg / degrees[i]
        heuristic.append(heu)
    heuristic = torch.tensor(heuristic)
    heuristic /= heuristic.max()
    return heuristic


class ACO_MIS:

    @torch.no_grad()
    def __init__(self,
                 adjlist: list[list[int]],
                 n_ants = 20, 
                 decay = 0.95,
                 alpha = 1.0,
                 beta = 2.0,
                 Q: Optional[float] = None,
                 min = 0.05,
                 pheromone: Optional[torch.Tensor] = None,
                 heuristic: Optional[torch.Tensor] = None,
                 elitist = False,
                 min_max = False,
                 device='cpu',
                 ):
        self.adjlist = [torch.tensor(i, dtype=torch.long, device=device) for i in adjlist]
        self.n = len(self.adjlist)

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.Q = Q or 1 / self.n

        self.elitist = elitist
        self.min_max = min_max
        self.min = min
        self.max = float('inf')
        
        self.device = device
        self.epoch = 1
        self.solutions: list[torch.Tensor] = []
        self.costs = torch.zeros(self.n_ants, device=device, dtype=torch.int16)

        if pheromone is not None:
            self.pheromone = pheromone
        else:
            self.pheromone = torch.ones(self.n, dtype=torch.float, device=device)
            if self.min_max:
                self.pheromone = self.pheromone * self.min
            
        if heuristic is not None:
            self.heuristic = heuristic
        else:
            self.heuristic = avg_adjacent_deg_heuristic(adjlist).float().to(device)

        self.best_solution = set()
    
    @property
    def best_value(self):
        return len(self.best_solution)

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            self.construct_solutions()
            self.update_cost()
            self.update_pheromone()
            self.epoch += 1
        return self.best_solution

    def sample(self):
        log_probs = self.construct_solutions(True)
        self.update_cost()
        return self.costs.float(), log_probs

    def construct_solutions(self, require_prob = False):
        probability = self.pheromone.pow(self.alpha) * self.heuristic.pow(self.beta)
        log_prob = []
        del self.solutions
        self.solutions = []
        for i in range(self.n_ants):
            mask = torch.ones_like(probability, dtype=torch.bool)
            selected = []
            this_log_prob = []
            for k in range(self.n):
                # sample
                prob = probability * mask.clone()
                dist = Categorical(prob)
                result = dist.sample()
                selected.append(result.item()) # type: ignore 
                # update mask
                mask[result] = False
                mask[self.adjlist[result]] = False
                if require_prob:
                    this_log_prob.append(dist.log_prob(result))
                # none is valid
                if not mask.any():
                    break
            solution = torch.tensor(selected, dtype=torch.long, device=self.device)
            self.solutions.append(solution)
            if require_prob:
                log_prob.append(sum(this_log_prob)/len(this_log_prob))
        if require_prob:
            return torch.stack(log_prob)
    

    @torch.no_grad()
    def update_cost(self):
        for i, solution in enumerate(self.solutions):
            self.costs[i] = len(solution)
        bestindex = self.costs.argmax()
        if self.costs[bestindex] > len(self.best_solution):
            self.best_solution = {i.item() for i in self.solutions[bestindex]}
            self.max = len(self.best_solution) * self.n * self.Q

    
    @torch.no_grad()
    def update_pheromone(self):
        self.pheromone = self.pheromone * self.decay
        if self.elitist:
            bestindex = self.costs.argmax()
            solution = self.solutions[bestindex]
            cost = len(solution)
            self.pheromone[solution] += self.Q * cost
        else:
            for solution in self.solutions:
                cost = len(solution)
                self.pheromone[solution] += self.Q * cost
            
        if self.min_max:
            self.pheromone[self.pheromone > self.max] = self.max
            self.pheromone[self.pheromone < self.min] = self.min


if __name__ == "__main__":
    import networkx as nx
    from utils import networkx_to_adjlist
    g = nx.erdos_renyi_graph(50, 0.1, seed = 0x12345678)
    adjlist = networkx_to_adjlist(g)
    aco = ACO_MIS(adjlist, 30, device='cpu')
    result = aco.run(100)
    print(aco.pheromone)
    print(result, len(result))
