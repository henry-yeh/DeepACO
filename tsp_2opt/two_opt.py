import torch
import numpy as np


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

if __name__ == "__main__":
    n = 100
    points = torch.randn((n, 2))
    distances = torch.sqrt(((points.unsqueeze(-2) - points)**2).sum(-1))
    route = torch.arange(n)
    routes = torch.stack([route, torch.roll(route, 5)])
    for _ in range(100):
        result = batched_two_opt_torch(distances.numpy(), routes.numpy())
        print(*result)
