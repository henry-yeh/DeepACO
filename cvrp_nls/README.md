# DeepACO with NLS for CVRP

We adopted the C++ implementation of the SWAP* algorithm provided by Vidal T. et al [1,2] ([source repo](https://github.com/vidalt/HGS-CVRP)) and added a C++ interface to directly utilize the local search algorithm in the program. We also utilized and modified the Python wrapper provided by [PyHygese](https://github.com/chkwon/PyHygese). We would like to express our gratitude for the contributions made by the aforementioned paper and project. 

To run our program, please follow the steps below to compile the shared library (if the file `./HGS-CVRP-main/build/libhgscvrp.so` we provide isn't compatiable with your setup)
```bash
$ cd HGS-CVRP-main/build
$ cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
$ make bin
```
Generate the test datasets with
```bash
$ cd ../.. # return to this dir
# generate datasets with fixed seed, which may take up to 4GiB of space
# configure problem scale and number of instances in this file
$ python3 ./utils.py
```

Then use the following code to execute our program.
```bash
# for training
$ python3 train.py <problem_scale> # use -h for detailed usage
# for cvrp100
$ python3 test.py 100
# for cvrp500
$ python3 test.py 500
# for cvrp1000
$ python3 test.py 1000 -m ../pretrained/cvrp_nls/cvrp500.pt
# for cvrp2000
$ python3 test.py 2000 -m ../pretrained/cvrp_nls/cvrp500.pt
```

### References

[1] Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., Rei, W. (2012). 
A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. Operations Research, 60(3), 611-624. 
https://doi.org/10.1287/opre.1120.1048 (Available [HERE](https://w1.cirrelt.ca/~vidalt/papers/HGS-CIRRELT-2011.pdf) in technical report form).

[2] Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. Computers & Operations Research, 140, 105643.
https://doi.org/10.1016/j.cor.2021.105643 (Available [HERE](https://arxiv.org/abs/2012.10384) in technical report form).