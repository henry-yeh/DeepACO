from typing import List, Optional, Union
from queue import LifoQueue
import numpy as np


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
    
    @property
    def latest_start(self):
        return self.latest_finish - self.duration

    @property
    def earlist_finish(self):
        return self.earlist_start + self.duration


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
    
    def get_indegrees(self):
        return [len(act.pred) for act in self.activities]
    
    def get_outdegrees(self):
        return [len(act.succ) for act in self.activities]

    def get_adjlist(self):
        adjlist = []
        for act in self.activities:
            adjlist.append([i.index for i in act.succ])
        return adjlist

    def get_adjmatrix(self):
        mat = np.zeros((len(self), len(self)), dtype=np.uint8)
        for index, row in enumerate(self.get_adjlist()):
            mat[index, row] = 1
        return mat

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

if __name__ == "__main__":
    inst = read_RCPfile("../data/rcpsp/j30rcp/J301_1.RCP")
    inst.get_adjlist()
    inst.get_adjmatrix()