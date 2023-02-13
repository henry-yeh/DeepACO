from typing import List, Optional, Union, Tuple
from queue import LifoQueue
import numpy as np
from functools import cached_property
import glob
import os
from torch_geometric.data import Data as PyGData
import torch

class Activity:
    def __init__(self, index: int, duration: int = 0, resources: Optional[List[int]] = None) -> None:
        self.index = index
        self.duration = duration
        self.pred = []
        self.succ = []
        self.resources = resources or []
        self.latest_finish = 0xfffffff # just a large integer
        self.earlist_start = 0

    def add_successor(self, other):
        self.succ.append(other)
        other.pred.append(self)
    
    @cached_property
    def latest_start(self):
        return self.latest_finish - self.duration

    @cached_property
    def earlist_finish(self):
        return self.earlist_start + self.duration
    
    @cached_property
    def succ_closure(self) -> set[int]:
        closure = set()
        for act in self.succ:
            closure.add(act.index)
            closure.update(act.succ_closure)
        return closure

    @cached_property
    def pred_closure(self) -> set[int]:
        closure = set()
        for act in self.pred:
            closure.add(act.index)
            closure.update(act.pred_closure)
        return closure
    
    @cached_property
    def indegree(self):
        return len(self.pred)
    
    @cached_property
    def outdegree(self):
        return len(self.succ)


class Resource:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.available = capacity
        self.last_event_time = 0
        self.queue = []
    
    def available_timestamp(self, amount: int) -> int:
        """Returns the earlist timestamp at which the requested amount is available."""
        assert amount <= self.capacity
        if amount == 0:
            return 0
        amount -= self.available
        if amount<=0:
            return self.last_event_time
        for release_time, release_amount in self.queue:
            amount -= release_amount
            if amount <= 0:
                return release_time
        raise Exception()
    
    def request(self, timestamp, amount, duration):
        assert timestamp >= self.last_event_time
        self.last_event_time = timestamp
        newqueue = []
        for release in self.queue:
            if release[0] <= timestamp:
                self.available += release[1]
            else:
                newqueue.append(release)
        newqueue.append((timestamp + duration, amount))
        self.queue = sorted(newqueue)
        self.available -= amount
        assert self.available >= 0, "Unable to fulfill this request"



class RCPSPInstance:
    def __init__(self, activities: List[Activity], resource_capacity: List[int], max_total_time: Optional[int] = None):
        self.activities = activities
        self.capacity = resource_capacity
        self._calc_earlist_start_time()
        self._calc_latest_finish_time(max_total_time)
    
    @property
    def activity_zero(self):
        return self.activities[0]
    
    @property
    def n(self):
        return len(self.activities)

    def __len__(self):
        return len(self.activities)
    
    def _calc_latest_finish_time(self, max_total_time: Optional[int] = None):
        if max_total_time is None:
            max_total_time = sum((act.duration for act in self.activities))
        self.activities[-1].latest_finish = max_total_time
        stack = LifoQueue()
        stack.put(self.activities[-1])
        while not stack.empty():
            node = stack.get()
            lf = node.latest_finish - node.duration
            for n in node.pred:
                if n.latest_finish > lf:
                    n.latest_finish = lf
                stack.put(n)
    
    def _calc_earlist_start_time(self):
        stack = LifoQueue()
        stack.put(self.activities[0])
        while not stack.empty():
            node = stack.get()
            es = node.earlist_start + node.duration
            for n in node.succ:
                if n.earlist_start < es:
                    n.earlist_start = es
                stack.put(n)
    
    @property
    def indegrees(self):
        return [act.indegree for act in self.activities]
    
    @property
    def outdegrees(self):
        return [act.outdegree for act in self.activities]

    @cached_property
    def adjlist(self):
        adjlist = []
        for act in self.activities:
            adjlist.append([i.index for i in act.succ])
        return adjlist

    @cached_property
    def adjmatrix(self):
        mat = np.zeros((len(self), len(self)), dtype=np.uint8)
        for index, row in enumerate(self.adjlist):
            mat[index, row] = 1
        return mat
    
    def get_duration(self):
        return [i.duration for i in self.activities]

    def get_resource_matrix(self):
        mat = []
        for act in self.activities:
            mat.append(act.resources)
        return np.array(mat, dtype=np.uint16)
    
    def check_schedule(self, start_time: List[int]) -> bool:
        schedule = sorted(enumerate(start_time), key=lambda x:x[1])
        resources = [Resource(i) for i in self.capacity]
        finished_at: List[Union[None, int]] = [None for _ in range(len(self))]
        for index, st in schedule:
            node = self.activities[index]

            # precedence constraint
            for act in node.pred:
                t = finished_at[act.index]

                if t is None or t > st:
                    # Does not satisfy precedence constraint.
                    return False
                
            # resource constraint
            for r, v in zip(resources, node.resources):
                try:
                    r.request(st, v, node.duration)
                except AssertionError:
                    # Does not satisfy resource constraint
                    return False
            finished_at[index] = st + node.duration
        return True

    def get_extended_adjlist(self):
        allindex = set(range(self.n))
        extended_adjlist = []
        for i, act in enumerate(self.activities):
            no_relation = allindex - act.succ_closure - act.pred_closure
            no_relation.remove(i)
            extended_adjlist.append(list(no_relation))
        return extended_adjlist
    
    def to_pyg_data(self, device = "cpu"):
        # node feature
        r = self.get_resource_matrix()
        r = r.astype(np.float32) / np.array(self.capacity)
        t = np.array(self.get_duration(), dtype = np.float32)
        t = t / t.max()
        x = np.hstack([t.reshape(self.n, 1), r])
        # precedence constraint edges
        norm_edge_index = adjlist_to_edge_index(self.adjlist)
        norm_edge_attr = torch.tensor([[1,0]]).float().expand(norm_edge_index.shape[1],2)
        # extended edges
        ext_edge_index = adjlist_to_edge_index(self.get_extended_adjlist())
        ext_edge_attr = torch.tensor([[0,1]]).float().expand(ext_edge_index.shape[1],2)
        # This extra edge is necessary for the Pooling Layers of PyG to function normally
        add_edge_index = torch.tensor([[self.n-1], [self.n-1]])
        add_edge_attr = torch.tensor([[0,0]])

        x = torch.from_numpy(x).float().to(device)
        edge_index = torch.hstack([norm_edge_index, ext_edge_index, add_edge_index]).to(device)
        edge_attr = torch.vstack([norm_edge_attr, ext_edge_attr, add_edge_attr]).to(device)
        return PyGData(x, edge_index, edge_attr)

        
def adjlist_to_edge_index(adjlist):
    sources = []
    targets = []
    for src, tgts in enumerate(adjlist):
        sources.append(torch.tensor(src, dtype = torch.long).expand(len(tgts)))
        targets.append(torch.tensor(tgts, dtype = torch.long))
    sources = torch.concat(sources)
    targets = torch.concat(targets)
    edge_index = torch.stack([sources, targets])
    return edge_index

def readints(f) -> List[int]:
    return list(map(int, f.readline().strip().split()))

def read_RCPfile(filepath):
    with open(filepath) as f:
        n_jobs, n_resources = readints(f)
        resource_capacity = readints(f)
        assert len(resource_capacity) == n_resources
        nodes = [Activity(i) for i in range(n_jobs)]
        for act in nodes:
            line = iter(readints(f))
            act.duration = next(line)
            act.resources = [next(line) for _ in range(n_resources)]
            n_successors = next(line)
            successors = list(line)   # consume all items remaining in `line`
            assert len(successors) == n_successors
            for succ_index in successors:  # connect the nodes
                # The index in RCP file starts from 1, 
                # but it's easier for programming with it starting from 0.
                successor = nodes[succ_index - 1]
                act.add_successor(successor)
        assert f.read().strip() == "", "Make sure that nothing is left in the file."
    assert len(nodes[0].pred) == 0, "The first node should have no predecessor."
    assert len(nodes[-1].succ) == 0, "The last node should have no successor."

    return RCPSPInstance(nodes, resource_capacity)

def load_dataset(directory: str, test_size = 100) -> Tuple[list[RCPSPInstance], list[RCPSPInstance]]:
    """Load a set of RCP files from a folder.
    Only the first {test_size} files (in lexicographic order) are included in the testset.

    Args:
        directory (str)
        test_size (int, optional): Size of testset. Defaults to 100.
    Returns:
        trainset (list[RCPSPInstance])
        testset (list[RCPSPInstance])
    """
    files = glob.glob(os.path.join(directory, "*.RCP"))
    files.sort()
    data = []
    for path in files:
        instance = read_RCPfile(path)
        data.append(instance)
    return data[test_size:], data[:test_size]

if __name__ == "__main__":
    inst = read_RCPfile("../data/rcpsp/j30rcp/J301_1.RCP")
    inst.to_pyg_data()