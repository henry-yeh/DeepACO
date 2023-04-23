'''
Convert Concorde outputs to pytorch files compatible with our codes.
'''

import sys
import os
import torch

filepath = sys.argv[1]
if not os.path.isfile(filepath):
    raise FileNotFoundError(f"'{filepath}' is not a file")

destpath = filepath[:-4] + ".pt"


with open(filepath, 'r') as f, torch.no_grad():
    instances = []
    while 1:
        row = f.readline()
        if not row:
            break
        positions, optimal = row.split("output")
        positions = list(map(float, positions.strip().split()))
        # optimal = optimal.strip().split()
        positions = torch.tensor(positions).reshape((-1, 2))
        instances.append(positions)
    instances = torch.stack(instances)
    torch.save(instances, destpath)
print("done")


