import tempfile

import pytest
import numpy as np

from paminco.net import load_sioux
from paminco.net.network import Network
from paminco.net._data_gas import temporary_gas_files

from paminco.optim import NetworkFW
from paminco.optim.subproblem import SubproblemMethod


class TestNetworkFW:
    
    @pytest.mark.slow
    def test_load_save_net(self):
        net_sioux = load_sioux()
        # should be irrelevant if we use newly created network or loaded one
        f = tempfile.mkstemp(suffix=".npz")[1]
        net_sioux.save_to_numpy(f)
        net2 = Network.from_npz(f)
        fw = NetworkFW(net_sioux, max_iter=3)
        fw2 = NetworkFW(net2, max_iter=3)
        fw.run()
        fw2.run()
        
        for att in ["cost", "lb", "blb"]:
            assert getattr(fw, att) == getattr(fw2, att)
        assert np.array_equal(fw.flow, fw2.flow)

    def test_braess_paradox(self):
        edge_data = np.array([["s", "v2"],
                              ["s", "v1"],
                              ["v2", "t"],
                              ["v1", "t"],
                              ["v1", "v2"]])
        marginal_cost = np.array([[1, 0],
                                  [0, 1],
                                  [0, 1],
                                  [1, 0],
                                  [0, 0]])
        demand_data = ("s", "t", 50)
        braess = Network(edge_data=edge_data, cost_data=marginal_cost, demand_data=demand_data)
        braess.integrate_cost()  # -> compute user equilibrium
        fw = NetworkFW(braess)
        fw.run(max_iter=100)
        
        braess.delete_edges(4)
        fw2 = NetworkFW(braess)
        fw2.run(max_iter=100)
        
        assert fw.cost > fw2.cost

    @pytest.mark.parametrize(
        "instancename",
        ["gas11", "gas24", "gas40"]
    )
    @pytest.mark.parametrize(
        "subproblem_method",
        [SubproblemMethod.LP_HIGHS, SubproblemMethod.LP_SIMPLEX, SubproblemMethod.AUTO]
    )
    def test_gas_small(self, instancename, subproblem_method):
        with temporary_gas_files(instancename) as tmpfiles:
            net = Network.from_gaslib(*tmpfiles)
            net.integrate_cost()
        fw = NetworkFW(net, subproblem_method=subproblem_method)
        precision = 0.001
        fw.run(epsilon=precision)
        # Test if solution satisfies demands
        # TODO: Optimality check?
        b = net.demand.b.sparse().toarray().flatten()
        assert np.allclose(net.gamma_times(fw.x), b)

    @pytest.mark.parametrize(
        "instancename",
        ["gas135", "gas582"]
    )
    @pytest.mark.very_slow
    def test_gas_medium(self, instancename):
        with temporary_gas_files(instancename) as tmpfiles:
            net = Network.from_gaslib(*tmpfiles)
            net.integrate_cost()
        fw = NetworkFW(net)
        precision = 0.01
        fw.run(epsilon=precision)
        # Test if solution satisfies demands
        # TODO: Optimality check?
        b = net.demand.b.sparse().toarray().flatten()
        assert np.allclose(net.gamma_times(fw.x), b)