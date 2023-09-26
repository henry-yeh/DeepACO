# [NeurIPS 2023] DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization

**Welcome!** This repository contains the code implementation of paper [*DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization*](https://arxiv.org/abs/2309.14032). DeepACO is a generic framework that leverages deep reinforcement learning to automate heuristic designs. It serves to strengthen the heuristic measures of existing ACO algorithms and dispense with laborious manual design in future ACO applications.

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
- **Traveling Salesman Problem (TSP).** Please refer to [`tsp/`](/henry-yeh/DeepACO/tree/main/tsp) for vanilla DeepACO and [`tsp_nls/`](/henry-yeh/DeepACO/tree/main/tsp_nls) for DeepACO with NLS on TSP.
- **Capacitated Vehicle Routing Problem (CVRP).** Please refer to [`cvrp/`](/henry-yeh/DeepACO/tree/main/cvrp) for vanilla DeepACO and [`cvrp_nls/`](/henry-yeh/DeepACO/tree/main/cvrp_nls) for DeepACO with NLS on CVRP.
- **Orienteering Problem (OP).** Please refer to [`op/`](/henry-yeh/DeepACO/tree/main/op).
- **Prize Collecting Travelling Salesman Problem (PCTSP).** Please refer to [`pctsp/`](/henry-yeh/DeepACO/tree/main/pctsp).
- **Sequential Ordering Problem (SOP).** Please refer to [`sop/`](/henry-yeh/DeepACO/tree/main/sop).
- **Single Machine Total Weighted Tardiness Problem (SMTWTP).** Please refer to [`smtwtp/`](/henry-yeh/DeepACO/tree/main/smtwtp).
- **Resource-Constrained Project Scheduling Problem (RCPSP).** Please refer to [`rcpsp/`](/henry-yeh/DeepACO/tree/main/rcpsp).
- **Multiple Knapsack Problem (MKP).** Please refer to [`mkp/`](/henry-yeh/DeepACO/tree/main/mkp) for the implementation of pheromone model $PH_{suc}$ and [`mkp_transformer/`](/henry-yeh/DeepACO/tree/main/mkp_transformer) for that of $PH_{items}$.
- **Bin Packing Problem (BPP).** Please refer to [`bpp/`](/henry-yeh/DeepACO/tree/main/bpp).

----


🤩 If you encounter any difficulty using our code, please do not hesitate to submit an issue or directly contact us! Clearly, we still have a long way to go to write beautiful code, any advice on improving our code would be greatly appreciated!

😍 If you do find our code helpful (or if you would be so kind as to offer us some encouragement), please consider kindly giving a star, and citing our paper.

```bibtex
@inproceedings{ye2023deepaco,
  title={DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization},
  author={Ye, Haoran and Wang, Jiarui and Cao, Zhiguang and Liang, Helan and Li, Yong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
