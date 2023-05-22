import torch
import numpy as np
import numba as nb
from torch.distributions import Categorical
from two_opt import batched_two_opt_python
import random
import concurrent.futures
from functools import cached_property

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
                 two_opt = False, # for compatibility
                 device='cpu',
                 local_search = 'nls',
                 ):
        
        self.problem_size = len(distances)
        self.distances = distances.to(device)
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
            self.pheromone = pheromone.to(device)
        
        assert local_search in [None, "2opt", "nls"]
        self.local_search_type = '2opt' if two_opt else local_search

        self.heuristic = 1 / distances if heuristic is None else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    @torch.no_grad()
    def sparsify(self, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
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
        self.heuristic = 1 / sparse_distances
    
    def sample(self, inference = False):
        if inference:
            probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
            paths = inference_batch_sample(probmat.cpu().numpy(), self.n_ants, 0)
            paths = torch.from_numpy(paths.T.astype(np.int64)).to(self.device)
            costs = self.gen_path_costs(paths)
            return costs, None, paths
        else:
            paths, log_probs = self.gen_path(require_prob=True)
            costs = self.gen_path_costs(paths)
            return costs, log_probs, paths
    
    def sample_2opt(self, paths):
        paths = self.local_search(paths)
        costs = self.gen_path_costs(paths)
        return costs, paths
    
    def local_search(self, paths, inference = False):
        if self.local_search_type == "2opt":
            paths = self.two_opt(paths, inference)
        elif self.local_search_type == "nls":
            paths = self.nls(paths, inference)
        return paths

    @torch.no_grad()
    def run(self, n_iterations, inference = False):
        for _ in range(n_iterations):
            if inference:
                probmat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
                paths = inference_batch_sample(probmat.cpu().numpy(), self.n_ants, 0)
                paths = torch.from_numpy(paths.T.astype(np.int64)).to(self.device)
            else:
                paths = self.gen_path(require_prob=False)

            paths = self.local_search(paths, inference)
            costs = self.gen_path_costs(paths)
            
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost.item()
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
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)
    
    def gen_numpy_path_costs(self, paths, numpy_distances):
        '''
        Args:
            paths: numpy ndarray with shape (n_ants, problem_size), note the shape
        Returns:
            Lengths of paths: numpy ndarray with shape (n_ants,)
        '''
        assert paths.shape == (self.n_ants, self.problem_size)
        u = paths
        v = np.roll(u, shift=1, axis=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return np.sum(numpy_distances[u, v], axis=1)
    
    def gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.zeros((self.n_ants, ), dtype = torch.long, device=self.device)
        # start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        index = torch.arange(self.n_ants, device=self.device)
        prob_mat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)

        mask[index, start] = 0
        
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)
        
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        prev = start
        for _ in range(self.problem_size-1):
            dist = prob_mat[prev] * mask
            dist = dist / dist.sum(axis=-1, keepdims=True)
            dist = Categorical(dist, validate_args=False)
            actions = dist.sample() # shape: (n_ants,)
            paths_list.append(actions)
            if require_prob:
                log_probs = dist.log_prob(actions) # shape: (n_ants,)
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[index, actions] = 0
        
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
    
    @cached_property
    def distances_numpy(self):
        return self.distances.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def heuristic_numpy(self):
        return self.heuristic.detach().cpu().numpy().astype(np.float32)
    
    @cached_property
    def heuristic_dist(self):
        return 1 / (self.heuristic_numpy/self.heuristic_numpy.max(-1, keepdims=True) + 1e-5)
    
    def two_opt(self, paths, inference = False):
        maxt = 10000 if inference else self.problem_size//4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)

        return best_paths
    
    def nls(self, paths, inference = False, T_nls = 10, T_p = 20):
        maxt = 10000 if inference else self.problem_size//4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_costs = self.gen_numpy_path_costs(best_paths, self.distances_numpy)
        new_paths = best_paths
        
        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances_numpy, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_numpy_path_costs(new_paths, self.distances_numpy)

            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]
        
        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)

        return best_paths

@nb.jit(nb.uint16[:](nb.float32[:,:],nb.int64), nopython=True, nogil=True)
def _inference_sample(probmat: np.ndarray, startnode = 0):
    n = probmat.shape[0]
    route = np.zeros(n, dtype=np.uint16)
    mask = np.ones(n, dtype=np.uint8)
    route[0] = lastnode = startnode   # fixed starting node
    for j in range(1, n):
        mask[lastnode] = 0
        prob = probmat[lastnode] * mask
        rand = random.random() * prob.sum()
        for k in range(n):
            rand -= prob[k]
            if rand <= 0:
                break
        lastnode = route[j] = k
    return route


def inference_batch_sample(probmat: np.ndarray, count=1, startnode = None):
    n = probmat.shape[0]
    routes = np.zeros((count, n), dtype=np.uint16)
    probmat = probmat.astype(np.float32)
    if startnode is None:
        startnode = np.random.randint(0, n, size=count)
    else:
        startnode = np.ones(count) * startnode
    if count <= 4 and n < 500:
        for i in range(count):
            routes[i] = _inference_sample(probmat, startnode[i])
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(count):
                future = executor.submit(_inference_sample, probmat, startnode[i])
                futures.append(future)
            for i, future in enumerate(futures):
                routes[i] = future.result()
    return routes

    
