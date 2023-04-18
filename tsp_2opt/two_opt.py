import torch
import numpy as np
import numba as nb
import concurrent.futures
from functools import partial

def batched_two_opt_torch(distances, tour, max_iterations=1000, device="cpu"):
    iterations = 0
    returnnumpy = False
    if isinstance(tour, np.ndarray):
        tour = torch.from_numpy(tour.copy())
        returnnumpy = True
    else:
        tour = tour.clone()
    if isinstance(distances, np.ndarray):
        distances = torch.from_numpy(distances)

    with torch.inference_mode():
        cuda_dist = distances #.to(device)
        cuda_tour = tour      #.to(device)
        batch_size = cuda_tour.shape[0]
        n = distances.shape[0]
        min_change = -1.0
        while min_change < 0.0:
            points_i = cuda_tour[:, :-1].unsqueeze(-1)
            points_j = cuda_tour[:, :-1].unsqueeze(1)
            points_i_plus_1 = cuda_tour[:, 1:].unsqueeze(-1)
            points_j_plus_1 = cuda_tour[:, 1:].unsqueeze(1)

            A_ij = cuda_dist[points_i, points_j]
            A_i_plus_1_j_plus_1 = cuda_dist[points_i_plus_1, points_j_plus_1]
            A_i_i_plus_1 = cuda_dist[points_i, points_i_plus_1]
            A_j_j_plus_1 = cuda_dist[points_j, points_j_plus_1]

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, n, rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, n)

            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                iterations += 1
            else:  
                break

            if iterations >= max_iterations:
                break
        if returnnumpy:
            # tour = cuda_tour.cpu().numpy()
            tour = cuda_tour.numpy()
        else:
            tour = cuda_tour
            # tour = cuda_tour.cpu()
    return tour, iterations

def batched_two_opt_torch_org(points, tour, max_iterations=1000, device="cpu"):
  ''' source: https://github.com/Edward-Sun/DIFUSCO/blob/main/difusco/utils/tsp_utils.py '''
  iterator = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_points = torch.from_numpy(points).to(device)
    cuda_tour = torch.from_numpy(tour).to(device)
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, len(points))

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break
    tour = cuda_tour.cpu().numpy()
  return tour, iterator

# @jit(nopython = True, parallel = True)
def batched_two_opt_numpy(dist: np.ndarray, tours: np.ndarray, max_iterations=1000):
    iterations = 0
    tours = tours.copy()
    batch_size = tours.shape[0]
    n = dist.shape[0]
    min_change = -1.0

    while min_change < 0.0:
        index1 = tours[:, :-1]
        index2 = tours[:, 1:]
        points_i = np.expand_dims(index1, -1)
        points_j = np.expand_dims(index1, -2)
        points_i_plus_1 = np.expand_dims(index2, -1)
        points_j_plus_1 = np.expand_dims(index2, -2)

        A_ij = dist[points_i, points_j]
        A_i_plus_1_j_plus_1 = dist[points_i_plus_1, points_j_plus_1]
        A_i_i_plus_1 = dist[points_i, points_i_plus_1]
        A_j_j_plus_1 = dist[points_j, points_j_plus_1]

        change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
        valid_change = np.triu(change, k=2)

        min_change = np.min(valid_change)
        flatten_argmin_index = np.argmin(valid_change.reshape(batch_size, -1), axis=-1)
        min_i = np.floor_divide(flatten_argmin_index, n)
        min_j = np.remainder(flatten_argmin_index, n)

        if min_change < -1e-6:
            for i in range(batch_size):
                tours[i, min_i[i] + 1:min_j[i] + 1] = np.flip(tours[i, min_i[i] + 1:min_j[i] + 1], axis=0)
            iterations += 1
        else:  
            break

        if iterations >= max_iterations:
            break
    return tours, iterations

@nb.njit(nb.uint16[:](nb.float32[:,:], nb.uint16[:], nb.int64), nogil=True)
def _two_opt_python(distmat, tour, max_iterations=1000):
    iterations = 0
    tour = tour.copy()
    n = tour.shape[0]
    min_change = -1.0
    while min_change < 0 and iterations < max_iterations:
        min_change = 0
        p = q = 0
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                node_i, node_j = tour[i], tour[j]
                node_prev, node_next = tour[i-1], tour[(j+1) % n]
                if node_prev == node_j or node_next == node_i:
                    continue
                change = (  distmat[node_prev, node_j] 
                          + distmat[node_i, node_next]
                          - distmat[node_prev, node_i] 
                          - distmat[node_j, node_next])                    
                if change < min_change:
                    min_change = change
                    p, q = i, j
        if min_change < -1e-6:
            tour[p: q+1] = np.flip(tour[p: q+1])
            iterations += 1
        else:  
            break
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


if __name__ == "__main__":
    import timeit
    n = 100
    bs = 100
    repeat = 10
    print(f"n = {n} batch_size = {bs} repeat = {repeat}")
    points = np.random.randn(n, 2)
    distances = np.sqrt(((points.reshape(n, 1, 2) - points)**2).sum(-1))
    route = np.arange(n)
    routes = []
    for _ in range(bs):
        np.random.shuffle(route)
        routes.append(route.copy())
    routes = np.stack(routes)

    torch_points = torch.from_numpy(points)
    torch_routes = torch.from_numpy(routes)
    torch_dist = torch.from_numpy(distances)
    cuda_points = torch_points.clone().cuda()
    cuda_routes = torch_routes.clone().cuda()
    cuda_dist = torch_dist.clone().cuda()

    # t = timeit.timeit(lambda: batched_two_opt_torch_org(points, routes), number=repeat)
    # print("torch original cpu:", t)
    # t = timeit.timeit(lambda: batched_two_opt_torch(torch_dist, torch_routes), number=repeat)
    # print("torch modified cpu:", t)
    # t = timeit.timeit(lambda: batched_two_opt_torch_org(points, routes, device='cuda:0'), number=repeat)
    # print("torch original cuda:", t)
    t = timeit.timeit(lambda: batched_two_opt_torch(cuda_dist, cuda_routes, device='cuda:0'), number=repeat)
    print("torch modified cuda:", t)
    # t = timeit.timeit(lambda: batched_two_opt_numpy_org(points, routes), number=repeat)
    # print("numpy original (with jit):", t)
    # t = timeit.timeit(lambda: batched_two_opt_numpy(distances, routes), number=repeat)
    # print("numpy modified:", t)
    t = timeit.timeit(lambda: batched_two_opt_python(distances, routes), number=repeat)
    print("python cpu (with jit, parallel):", t)
