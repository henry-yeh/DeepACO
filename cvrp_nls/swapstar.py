# Adapted from https://github.com/chkwon/PyHygese/blob/master/hygese/hygese.py

import os
import platform
from ctypes import (
    Structure,
    CDLL,
    POINTER,
    c_int,
    c_double,
    c_char,
    sizeof,
    cast,
    byref,
    c_char_p,
)
from dataclasses import dataclass
from typing import List
import numpy as np
import sys
import torch
import random
import time

def write_routes(routes: List[List[int]], filepath: str):
    with open(filepath, "w") as f:
        for i, r in enumerate(routes):
            f.write(f"Route #{i+1}: "+' '.join([str(x.item()) for x in r if x>0])+"\n")
    return

def read_routes(filepath):
    routes = []
    with open(filepath, "r") as f:
        while 1:
            line = f.readline().strip()
            if line.startswith("Route"):
                routes.append(torch.tensor([0, *map(int, line.split(":")[1].split()), 0]))
            else:
                break
    return routes

def get_lib_filename():
    path = "HGS-CVRP-main/build/libhgscvrp.so"
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(f"Shared library file `{path}` not found")


# basedir = os.path.abspath(os.path.dirname(__file__))
basedir = os.path.dirname(os.path.realpath(__file__))
# os.add_dll_directory(basedir)
HGS_LIBRARY_FILEPATH = os.path.join(basedir, get_lib_filename())

c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)
C_INT_MAX = 2 ** (sizeof(c_int) * 8 - 1) - 1
C_DBL_MAX = sys.float_info.max


# Must match with AlgorithmParameters.h in HGS-CVRP: https://github.com/vidalt/HGS-CVRP
class CAlgorithmParameters(Structure):
    _fields_ = [
        ("nbGranular", c_int),
        ("mu", c_int),
        ("lambda", c_int),
        ("nbElite", c_int),
        ("nbClose", c_int),
        ("targetFeasible", c_double),
        ("seed", c_int),
        ("nbIter", c_int),
        ("timeLimit", c_double),
        ("useSwapStar", c_int),
    ]


@dataclass
class AlgorithmParameters:
    nbGranular: int = 20
    mu: int = 25
    lambda_: int = 40
    nbElite: int = 4
    nbClose: int = 5
    targetFeasible: float = 0.2
    seed: int = 0
    nbIter: int = 20000
    timeLimit: float = 0.0
    useSwapStar: bool = True

    @property
    def ctypes(self) -> CAlgorithmParameters:
        return CAlgorithmParameters(
            self.nbGranular,
            self.mu,
            self.lambda_,
            self.nbElite,
            self.nbClose,
            self.targetFeasible,
            self.seed,
            self.nbIter,
            self.timeLimit,
            int(self.useSwapStar),
        )


class _SolutionRoute(Structure):
    _fields_ = [("length", c_int), ("path", c_int_p)]


class _Solution(Structure):
    _fields_ = [
        ("cost", c_double),
        ("time", c_double),
        ("n_routes", c_int),
        ("routes", POINTER(_SolutionRoute)),
    ]


class RoutingSolution:
    def __init__(self, sol_ptr):
        if not sol_ptr:
            raise TypeError("The solution pointer is null.")

        self.cost = sol_ptr[0].cost
        self.time = sol_ptr[0].time
        self.n_routes = sol_ptr[0].n_routes
        self.routes = []
        for i in range(self.n_routes):
            r = sol_ptr[0].routes[i]
            path = r.path[0 : r.length]
            self.routes.append(path)


class Solver:
    def __init__(self, parameters=AlgorithmParameters(), verbose=False):
        if platform.system() == "Windows":
            hgs_library = CDLL(HGS_LIBRARY_FILEPATH, winmode=0)
        else:
            hgs_library = CDLL(HGS_LIBRARY_FILEPATH)

        self.algorithm_parameters = parameters
        self.verbose = verbose

        # solve_cvrp
        self._c_api_solve_cvrp = hgs_library.solve_cvrp
        self._c_api_solve_cvrp.argtypes = [
            c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double,
            c_double,
            c_char,
            c_char,
            c_int,
            POINTER(CAlgorithmParameters),
            c_char,
        ]
        self._c_api_solve_cvrp.restype = POINTER(_Solution)

        # solve_cvrp_dist_mtx
        self._c_api_local_search = hgs_library.local_search
        self._c_api_local_search.argtypes = [
            c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double,
            c_double,
            c_char,
            c_int,
            POINTER(CAlgorithmParameters),
            c_char,
            c_int,
            c_int,
        ]
        self._c_api_local_search.restype = c_int

        # delete_solution
        self._c_api_delete_sol = hgs_library.delete_solution
        self._c_api_delete_sol.restype = None
        self._c_api_delete_sol.argtypes = [POINTER(_Solution)]
    
    def local_search(self, data, routes: List[List[int]], count:int = 1,rounding=True,):
        # required data
        demand = np.asarray(data["demands"])
        vehicle_capacity = data["vehicle_capacity"]
        n_nodes = len(demand)

        # optional depot
        depot = data.get("depot", 0)
        if depot != 0:
            raise ValueError("In HGS, the depot location must be 0.")

        # optional num_vehicles
        maximum_number_of_vehicles = data.get("num_vehicles", C_INT_MAX)

        # optional service_times
        service_times = data.get("service_times")
        if service_times is None:
            service_times = np.zeros(n_nodes)
        else:
            service_times = np.asarray(service_times)

        # optional duration_limit
        duration_limit = data.get("duration_limit")
        if duration_limit is None:
            is_duration_constraint = False
            duration_limit = C_DBL_MAX
        else:
            is_duration_constraint = True

        is_rounding_integer = rounding

        x_coords = data.get("x_coordinates")
        y_coords = data.get("y_coordinates")
        dist_mtx = data.get("distance_matrix")

        if x_coords is None or y_coords is None:
            assert dist_mtx is not None
            x_coords = np.zeros(n_nodes)
            y_coords = np.zeros(n_nodes)
        else:
            x_coords = np.asarray(x_coords)
            y_coords = np.asarray(y_coords)

        assert len(x_coords) == len(y_coords) == len(service_times) == len(demand)
        assert (x_coords >= 0.0).all()
        assert (y_coords >= 0.0).all()
        assert (service_times >= 0.0).all()
        assert (demand >= 0.0).all()

        dist_mtx = np.asarray(dist_mtx)
        assert dist_mtx.shape[0] == dist_mtx.shape[1]
        assert (dist_mtx >= 0.0).all()

        callid = (time.time_ns()*10000+random.randint(0,10000))%C_INT_MAX

        tmppath = "/tmp/route-{}".format(callid)
        resultpath = "/tmp/swapstar-result-{}".format(callid)
        write_routes(routes, tmppath)
        try:
            self._local_search(
                x_coords,
                y_coords,
                dist_mtx,
                service_times,
                demand,
                vehicle_capacity,
                duration_limit,
                is_duration_constraint,
                maximum_number_of_vehicles,
                self.algorithm_parameters,
                self.verbose,
                callid,
                count,
            )

            result = read_routes(resultpath)
        except Exception as e:
            print(routes)
            print([demand[r].sum() for r in routes])
        else:
            os.remove(resultpath)
        finally:
            os.remove(tmppath)
        
        return result

    def _local_search(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        dist_mtx: np.ndarray,
        service_times: np.ndarray,
        demand: np.ndarray,
        vehicle_capacity: int,
        duration_limit: float,
        is_duration_constraint: bool,
        maximum_number_of_vehicles: int,
        algorithm_parameters: AlgorithmParameters,
        verbose: bool,
        callid: int,
        count:int,
    ):
        n_nodes = x_coords.size

        x_ct = x_coords.astype(c_double).ctypes
        y_ct = y_coords.astype(c_double).ctypes
        s_ct = service_times.astype(c_double).ctypes
        d_ct = demand.astype(c_double).ctypes

        m_ct = dist_mtx.reshape(n_nodes * n_nodes).astype(c_double).ctypes
        ap_ct = algorithm_parameters.ctypes


        # struct Solution *solve_cvrp_dist_mtx(
        # 	int n, double* x, double* y, double *dist_mtx, double *serv_time, double *dem,
        # 	double vehicleCapacity, double durationLimit, char isDurationConstraint,
        # 	int max_nbVeh, const struct AlgorithmParameters *ap, char verbose);
        sol_p = self._c_api_local_search(
            n_nodes,
            cast(x_ct, c_double_p),
            cast(y_ct, c_double_p),
            cast(m_ct, c_double_p),
            cast(s_ct, c_double_p),
            cast(d_ct, c_double_p),
            vehicle_capacity,
            duration_limit,
            is_duration_constraint,
            maximum_number_of_vehicles,
            byref(ap_ct),
            verbose,
            callid,
            count,
        )

        result = sol_p
        return result

def swapstar(demands, matrix, positions, routes, count=1):
    ap = AlgorithmParameters()
    hgs_solver = Solver(parameters=ap, verbose=False)

    data = dict()
    x = positions[:, 0]
    y = positions[:, 1]
    data['x_coordinates'] = x
    data['y_coordinates'] = y

    data['depot'] = 0
    data['demands'] = demands*1000
    data["num_vehicles"] = len(routes)
    data['vehicle_capacity'] = 1000.001  # to avoid floating-point error

    # Solve with calculated distances
    data['distance_matrix'] = matrix
    try:
        result = hgs_solver.local_search(data, routes, count)
    except Exception as e:
        print(e)
        return routes
    return result

