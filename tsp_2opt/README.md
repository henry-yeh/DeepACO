## DeepACO with NLS

### Training

The checkpoints will be saved in [`../pretrained/tsp_2opt`](../pretrained/tsp_2opt) with suffix `-best.pt` and `-last.pt` by default.

TSP200:
```raw
$ python3 train.py 200
```

TSP500:
```raw
$ python3 train.py 500
```

TSP1000:
```raw
$ python3 train.py 1000
```

### Testing

TSP200:
```
$ python3 test.py 200
```

TSP500:
```
$ python3 test.py 500
```

TSP1000:
```
$ python3 test.py 1000
```
