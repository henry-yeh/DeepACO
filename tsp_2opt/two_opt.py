import numpy as np
import numba as nb
import concurrent.futures
from functools import partial

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True)
def two_opt_once(distmat, tour, fixed_i = 0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i==0 else range(fixed_i, fixed_i+1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i-1], tour[(j+1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (  distmat[node_prev, node_j] 
                        + distmat[node_i, node_next]
                        - distmat[node_prev, node_i] 
                        - distmat[node_j, node_next])                    
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q+1] = np.flip(tour[p: q+1])
        return delta
    else:
        return 0.0


@nb.njit(nb.uint16[:](nb.float32[:,:], nb.uint16[:], nb.int64), nogil=True)
def _two_opt_python(distmat, tour, max_iterations=1000):
    iterations = 0
    tour = tour.copy()
    min_change = -1.0
    while min_change < -1e-6 and iterations < max_iterations:
        min_change = two_opt_once(distmat, tour, 0)
        iterations += 1
    return tour

def batched_two_opt_python(dist: np.ndarray, tours: np.ndarray, max_iterations=1000):
    dist = dist.astype(np.float32)
    tours = tours.astype(np.uint16)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for tour in tours:
            future = executor.submit(partial(_two_opt_python, distmat=dist, max_iterations=max_iterations), tour = tour)
            futures.append(future)
        return np.stack([f.result() for f in futures])
