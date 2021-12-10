import tempfile

import pytest
import numpy as np

from paminco import load_sioux, Network
from paminco._base import ParametricSolution
from paminco.algo.mca import MCA


def sioux():
    sioux = load_sioux()
    sioux.set_demand(("20", "3", 100000))
    sioux.cost.integrate(inplace=True)
    mca_ue = MCA(sioux)
    mca_ue.run()
    return mca_ue.param_solution


def simple_electrical():
    edge_data = np.array([[ "s", "v1"],
                          [ "s", "v2"],
                          ["v1", "v2"],
                          ["v1",  "t"],
                          ["v2",  "t"]])
    poly_cost = np.array([[0, 0, 1],    # F_0(x) = 0 * x^0 + 0 * x^1 + 1 * x^2
                          [0, 3, 0.5],  # F_0(x) = 0 * x^0 + 3 * x^1 + 0.5 * x^2
                          [0, 0, 0.5],
                          [0, 3, 0.5],
                          [0, 0, 1]])
    demand_data = (("s", "t", 1))
    d = {"s": 0, "v1": 1, "v2": 2, "t": 3}  # determines how labels are mapped to indices
    net = Network(edge_data,
                  cost_data=poly_cost,
                  demand_data=demand_data,
                  kw_edge={"map_labels_to_indices": d})
    
    mca = MCA(net, lambda_max=8)
    mca.run()
    return mca.param_solution


def compare_param_sols(sol1, sol2):
    def equal(arr1, arr2, exact: bool = True):
        if exact is True:
            comp_func = np.array_equal
        else:
            comp_func = np.allclose
        
        if arr1 is None and arr2 is None:
            return True
        if (isinstance(arr1, np.ndarray)
                and isinstance(arr2, np.ndarray)
                and comp_func(arr1, arr2)):
            return True
        return False
    
    assert len(sol1) == len(sol2)
    
    idx = np.random.randint(low=0, high=len(sol1), size=30)
    for i in idx:
        assert sol1[i].param == sol2[i].param, f"Param not equal: {sol1[i].param} != {sol2[i].param}"
        assert equal(sol1[i].flow, sol2[i].flow, False), f"Flow not equal: {sol1[i].flow} != {sol2[i].flow}"
        assert equal(sol1[i].potential, sol2[i].potential, False), f"Potential not equal: {sol1[i].potential} != {sol2[i].potential}"
        assert sol1[i].cost == sol2[i].cost, f"Cost not equal: {sol1[i].cost} != {sol2[i].cost}"
        
    assert equal(sol1.dflow, sol2.dflow, exact=False)
    assert equal(sol1.dpi, sol2.dpi, exact=False)


@pytest.mark.parametrize("sol1", [sioux(), simple_electrical()])
def test_csv(sol1):
    f = tempfile.mkstemp(suffix=".csv")[1]
    sol1.to_csv(f)
    sol2 = ParametricSolution.from_csv(f)
    compare_param_sols(sol1, sol2)


@pytest.mark.parametrize("sol1", [sioux(), simple_electrical()])
def test_npz(sol1):
    f = tempfile.mkstemp(suffix=".npz")[1]
    sol1.save_to_numpy(f)
    sol2 = ParametricSolution.from_npz(f)
    compare_param_sols(sol1, sol2)
