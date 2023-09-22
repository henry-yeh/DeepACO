## DeepACO with NLS for TSP

### Training

The checkpoints will be saved in [`../pretrained/tsp_nls`](../pretrained/tsp_nls) with suffix `-best.pt` and `-last.pt` by default.

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
