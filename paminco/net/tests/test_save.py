import tempfile
import warnings

import pytest
import numpy as np
from paminco.algo.mca import MCAInterpolationRule

from paminco.net import load_sioux
from paminco.net.network import Network
from paminco.net.shared import Shared, Nodes, Edges
from paminco.net.demand import LinearDemandFunction
from paminco.net.cost import NetworkCost


@pytest.mark.parametrize("cost_type", ["polynomial", "symbolic"])
@pytest.mark.parametrize("make_aff_demand", [True, False])
def test_stack_sioux(cost_type, make_aff_demand):
    """Test if from/to xml/npz works correctly."""
    net = load_sioux()
    if make_aff_demand is True:
        net.set_demand((net.demand(0.3), net.demand(0.7)), mode="affine", is_label=False)
    
    f = tempfile.mkstemp(suffix=".xml")[1]
    net.to_xml(f, prettify=False)
    net2 = Network.from_xml(f)
    
    f2 = tempfile.mkstemp(suffix=".npz")[1]
    net2.save_to_numpy(f2, prettify=True)
    net3 = Network.from_npz(f2)
    
    assert np.array_equal(net.cost(2000), net3.cost(2000))
    assert (net.demand(4.123) != net3.demand(4.123)).nnz == 0


class TestSaveNP:
    def _test_save(self):
        net = load_sioux
        f = tempfile.mkstemp(suffix=".npz")[1]

        # save nodes
        net.nodes.save_to_numpy(f)
        s2 = Nodes.from_npz(f)
        assert s2 == net.nodes

        # save edges
        net.edges.save_to_numpy(f)
        s2 = Edges.from_npz(f)
        assert s2 == net.edges

        # save shared
        net.shared.save_to_numpy(f)
        s2 = Shared.from_npz(f)
        assert s2 == net.shared

        # save all
        net.save_to_numpy(f)
        net2 = Network.from_npz(f)
        assert net == net2

    def test_save_pwqc(self):
        net = load_sioux()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.clean(remove_zones=True,
                      remove_parallel_edges=True,
                      remove_zero_cost_edges=True,
                      remove_isolated_nodes=True,
                      remove_unreachable_nodes=True,
                      remove_commodities=True)
        i_rule = MCAInterpolationRule(1.01, 1, net.m, 1000)
        net._c = net.cost.interpolate(i_rule)
        f = tempfile.mkstemp(suffix=".npz")[1]
        net.save_to_numpy(f)
        net2 = Network.from_npz(f)
        assert net == net2, "saving network fails for pwqcost"
