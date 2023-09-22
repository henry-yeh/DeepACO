# [NeurIPS 2023] DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization

Welcome! This repository contains the code implementation of paper [*DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization*](). DeepACO is a generic framework that leverages deep reinforcement learning to automate heuristic designs. It serves to strengthen the heuristic measures of existing ACO algorithms and dispense with laborious manual design in future ACO applications.

![diagram](./diagram.png)


---

### Dependencies

- CUDA 11.0
- PyTorch 1.7.0
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) 2.0.7
- [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.9
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 2.0.4
- d2l
- [networkx](https://networkx.org/) 2.8.4
- [numpy](https://numpy.org/) 1.23.3
- [numba](https://numba.pydata.org/) 0.56.4

---

### Usage
- **Traveling Salesman Problem (TSP).** Please refer to `tsp/` for vanilla DeepACO and `tsp_nls/` for DeepACO with NLS on TSP.
- **Capacitated Vehicle Routing Problem (CVRP).** Please refer to `cvrp/` for vanilla DeepACO and `cvrp_nls/` for DeepACO with NLS on CVRP.
- **Orienteering Problem (OP).** Please refer to `op/`.
- **Prize Collecting Travelling Salesman Problem (PCTSP).** Please refer to `pctsp/`.
- **Sequential Ordering Problem (SOP).** Please refer to `sop/`.
- **Single Machine Total Weighted Tardiness Problem (SMTWTP).** Please refer to `smtwtp/`.
- **Resource-Constrained Project Scheduling Problem (RCPSP).** Please refer to `rcpsp/`.
- **Multiple Knapsack Problem (MKP).** Please refer to `mkp/` for the implementation of pheromone model $PH_{suc}$ and `mkp_transformer/` for that of $PH_{items}$.
- **Bin Packing Problem (BPP).** Please refer to `bpp/`.

----


🤩 If you encounter any difficulty using our code, please do not hesitate to submit an issue or contact us! Any advice on improving the code would be greatly appreciated!

😍 If you do find our code helpful, please consider kindly giving a star, and citing our paper.

```bibtex
@article{ye2023deepaco,
  title={DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization},
  author={Ye, Haoran and Wang, Jiarui and Cao, Zhiguang and Liang, Helan and Li, Yong},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
