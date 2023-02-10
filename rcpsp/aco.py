import torch
from torch.distributions import Categorical
from typing import Optional
from rcpsp_inst import RCPSPInstance, Resource
import numpy as np
from typing import NamedTuple

def SSGS(rcpsp: RCPSPInstance, sequence: list[int]) -> list[int]:
    """serial schedule generation scheme"""
    n = rcpsp.n
    valid = [True for _ in range(n)]
    indegrees = np.array(rcpsp.indegrees, dtype=np.int8)
    adjlist = [np.array(arr, dtype=np.uint16) for arr in rcpsp.adjlist]
    start_time = [0 for _ in range(n)]
    end_time = [0 for _ in range(n)]
    resources = [Resource(i) for i in rcpsp.capacity]
    for g in range(n):
        # fetch an activity to arrange time
        for j in sequence:
            if valid[j] and indegrees[j]<=0:
                break
        else:
            raise Exception("The precendence graph may contain a loop.")
        node = rcpsp.activities[j]
        requirement = node.resources
        
        # get earlist feasible start time
        earlist_start = max((end_time[p.index] for p in node.pred), default = node.earlist_start)
        arrange = max((r.available_timestamp(v) for r, v in zip(resources, requirement) if v>0), default=0)
        arrange = min(max(arrange, earlist_start), node.latest_start)

        # update states
        for r, v in zip(resources, requirement):
            if v>0:
                r.request(arrange, v, node.duration)
        start_time[j] = arrange
        end_time[j] = arrange + node.duration
        valid[j] = False
        indegrees[adjlist[j]] -= 1
    return start_time

def SSGS_ordered(rcpsp: RCPSPInstance, sequence: list[int]) -> list[int]:
    """serial schedule generation scheme (when the input sequence is in topological order)"""
    n = rcpsp.n
    start_time = [0 for _ in range(n)]
    end_time = [0 for _ in range(n)]
    resources = [Resource(i) for i in rcpsp.capacity]
    for j in sequence:
        node = rcpsp.activities[j]
        requirement = node.resources

        # get earlist feasible start time
        earlist_start = max((end_time[p.index] for p in node.pred), default = node.earlist_start)
        arrange = max((r.available_timestamp(v) for r, v in zip(resources, requirement) if v>0), default=0)
        arrange = min(max(arrange, earlist_start), node.latest_start)

        # update states
        for r, v in zip(resources, requirement):
            if v>0:
                r.request(arrange, v, node.duration)
        start_time[j] = arrange
        end_time[j] = arrange + node.duration
    return start_time

@torch.no_grad()
def nLFT_heuristic(rcpsp: RCPSPInstance):
    n = rcpsp.n
    column = torch.tensor([act.latest_finish for act in rcpsp.activities])
    last_finish = column.max()
    column = last_finish - column + 1
    return column.expand(n, n)

@torch.no_grad()
def nGRPWA_heuristic(rcpsp: RCPSPInstance):
    n = rcpsp.n
    column = torch.tensor([len(act.succ_closure) for act in rcpsp.activities])
    column = column - column.min() + 1
    return column.expand(n, n)


@torch.no_grad()
def nWRUP_heuristic(rcpsp: RCPSPInstance, omega = 0.5):
    n = rcpsp.n
    column = []
    for act in rcpsp.activities:
        value = omega * act.outdegree
        value += (1-omega) * sum(req/cap for req,cap in zip(act.resources, rcpsp.capacity))
        column.append(value)
    column = torch.tensor(column)
    column = column - column.min() + 1
    return column.expand(n, n)


class Solution(NamedTuple):
    route: np.ndarray
    schedule: np.ndarray
    cost: int

class ACO_RCPSP:

    @torch.no_grad()
    def __init__(self, 
                 rcpsp: RCPSPInstance,
                 n_ants = 5, 
                 decay = 0.975,
                 alpha = 1.0,
                 beta = 2.0,
                 gamma = 0.0,
                 c = 0.6,
                 Q = 1.0,
                 min = 0.1,
                 elitist=False,
                 min_max=False,
                 pheromone: Optional[torch.Tensor] = None,
                 heuristic: Optional[torch.Tensor] = None,
                 device='cpu',
                 train = False,
                 ):
        """Implementing ACO-RCPSP algorithm as stated in [1]. Only a partial of the features is implemented.
        
        [1] Merkle, D., M. Middendorf, and H. Schmeck. “Ant Colony Optimization for Resource-Constrained Project Scheduling.” 2002. https://doi.org/10.1109/TEVC.2002.802450.
        """
        
        self.rcpsp = rcpsp
        self.n = rcpsp.n
        self.device = device
        self.adjlist = [np.array(i) for i in rcpsp.adjlist]

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.c = c
        self.elitist = elitist
        self.min_max = min_max
        self.min = min
        self.max = np.Infinity
        self.gamma = torch.tensor(gamma).to(device)
        self.train = train

        self.epoch = 1

        if pheromone is not None:
            assert pheromone.shape == (rcpsp.n, rcpsp.n)
            self.pheromone = pheromone
        else:
            self.pheromone = torch.ones(rcpsp.n, rcpsp.n, dtype=torch.float32, device=device)
            if self.min_max:
                self.pheromone *= self.min
        
        if heuristic is not None:
            assert heuristic.shape == (rcpsp.n, rcpsp.n)
            self.heuristic = heuristic
        else:
            heuristic = nWRUP_heuristic(self.rcpsp, omega = 0.3)
            heuristic = heuristic / heuristic.max() * nGRPWA_heuristic(self.rcpsp)
            self.heuristic = heuristic.to(device)
        
        self.routes = torch.zeros(self.n_ants, self.n, dtype=torch.long, device=device)
        self.costs = torch.zeros(self.n_ants, dtype = torch.long, device = device)
        self.range_pop = torch.arange(self.n_ants, device=self.device)

        self.best_solution = Solution(np.array([]), np.array([]), 0xffffffff)

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            self.construct_solutions()
            self.update_cost()
            self.update_pheromone()
            self.epoch += 1

        return self.best_solution
    
    def construct_solutions(self):
        probmat = self.pheromone.pow(self.alpha) * self.heuristic.pow(self.beta)
        not_visited = torch.ones(self.n_ants, self.n, dtype=torch.bool, device=self.device)
        indegrees = torch.tensor(self.rcpsp.indegrees, dtype=torch.int16).unsqueeze(0).expand(self.n_ants, self.n).to(self.device).contiguous()
        self.routes[:, 0] = prev = torch.tensor([0]).expand(self.n_ants)
        log_probs = []
        for k in range(self.n-1):
            # update status
            not_visited[self.range_pop, prev] = False
            for i, p in enumerate(prev):
                indegrees[i, self.adjlist[p]] -= 1
            # sample in topological order
            mask = not_visited * (indegrees == 0)

            if self.gamma < 0.05 or self.c == 1:
                # direct evaluation
                prob = probmat[prev] * mask
            else:
                # summation evaluation
                pheromone = self.pheromone[self.routes[:, :k+1]].reshape(self.n_ants, k+1, -1)
                if self.gamma != 1:
                    gamma = self.gamma.pow(torch.arange(k, -1, -1, device=self.device)).view(1,k+1,1)
                    pheromone = pheromone * gamma
                pheromone = pheromone.sum(dim=1) * mask
                summation_prob = pheromone.pow(self.alpha) * self.heuristic[prev].pow(self.beta)
                if self.c == 0:
                    prob = summation_prob
                else:
                    # balanced
                    direct_prob = probmat[prev] * mask
                    prob = self.c * direct_prob + (1-self.c) * summation_prob
            dist = Categorical(prob)
            self.routes[:, k+1] = prev = dist.sample()
            if self.train:
                log_prob = dist.log_prob(prev)
                log_probs.append(log_prob)
        if self.train:
            return torch.stack(log_probs)
    
    def sample(self):
        self.train = True
        log_probs = self.construct_solutions()
        self.update_cost()
        return self.costs.float(), log_probs
    
    @torch.no_grad()
    def update_cost(self):
        schedules = []
        for i, route in enumerate(self.routes):
            schedule = SSGS_ordered(self.rcpsp, route.cpu().numpy())
            schedules.append(schedule)
            self.costs[i] = schedule[-1]
        bestindex = self.costs.argmin()
        if self.costs[bestindex] < self.best_solution.cost:
            best_schedule = schedules[bestindex]
            self.best_solution = Solution(
                route = self.routes[bestindex].numpy(),
                schedule = np.array(best_schedule),
                cost = best_schedule[-1]
            )
            self.max = self.Q * self.n / best_schedule[-1]
    
    @torch.no_grad()
    def update_pheromone(self):
        self.pheromone = self.pheromone * self.decay

        best_route = self.best_solution.route
        self.pheromone[best_route[:-1], best_route[1:]] += self.Q / self.best_solution.cost

        if self.elitist:
            bestindex = self.costs.argmin()
            route = self.routes[bestindex]
            cost = self.costs[bestindex]
            self.pheromone[route[:-1], route[1:]] += self.Q / cost
        else:
            for route, cost in zip(self.routes, self.costs):
                self.pheromone[route[:-1], route[1:]] += self.Q / cost
        
        if self.min_max:
            self.pheromone[self.pheromone > self.max] = self.max
            self.pheromone[self.pheromone < self.min] = self.min


if __name__ == "__main__":
    from rcpsp_inst import read_RCPfile
    from matplotlib import pyplot as plt
    # instance = read_RCPfile("../data/rcpsp/j120rcp/X1_1.RCP")
    instance = read_RCPfile("../data/rcpsp/j60rcp/J601_1.RCP")
    # instance = read_RCPfile("../data/rcpsp/j30rcp/J301_3.RCP")
    schedule = SSGS(instance, list(range(len(instance))))
    print(schedule)
    assert instance.check_schedule(schedule)
    aco = ACO_RCPSP(instance, alpha=1.0, beta=2.0, gamma=1, elitist=True, min_max=True)
    result = aco.run(1000)
    assert instance.check_schedule(list(result.schedule))
    print(result.schedule)
    print(result.route)
    print(aco.pheromone.max())
    print(aco.max)
    plt.imshow(aco.pheromone)
    plt.show()