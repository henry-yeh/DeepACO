from rcpsp_inst import RCPSPInstance, Resource
import numpy as np

def SSGS(rcpsp: RCPSPInstance, sequence: list[int]) -> list[int]:
    """serial schedule generation scheme"""
    n = len(rcpsp)
    valid = [True for _ in range(n)]
    indegrees = np.array(rcpsp.get_indegrees(), dtype=np.int8)
    adjlist = [np.array(arr, dtype=np.uint16) for arr in rcpsp.get_adjlist()]
    start_time = [0 for _ in range(n)]
    end_time = [0 for _ in range(n)]
    resources = [Resource(i) for i in rcpsp.capacity]
    for g in range(n):
        # fetch an activity to arrange time
        for j in sequence:
            if valid[j] and indegrees[j]<=0:
                break
        else:
            raise Exception("The precendence graph may contain a loop.")
        node = rcpsp.activities[j]
        requirement = node.resources
        
        # get earlist feasible start time
        earlist_start = max((end_time[p.index] for p in node.pred), default = node.earlist_start)
        arrange = max((r.available_timestamp(v) for r, v in zip(resources, requirement) if v>0), default=earlist_start)
        arrange = min(max(arrange, earlist_start), node.latest_start)

        # update states
        for r, v in zip(resources, requirement):
            if v>0:
                r.request(arrange, v, node.duration)
        start_time[j] = arrange
        end_time[j] = arrange + node.duration
        valid[j] = False
        indegrees[adjlist[j]] -= 1
    return start_time


class ACO_RCPSP:
    pass


if __name__ == "__main__":
    from rcpsp_inst import read_RCPfile
    instance = read_RCPfile("../data/rcpsp/j120rcp/X1_1.RCP")
    schedule = SSGS(instance, list(range(len(instance))))
    print(schedule)
    assert instance.check_schedule(schedule)