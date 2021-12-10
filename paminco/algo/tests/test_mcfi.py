from numpy.lib.npyio import load
import pytest
import numpy as np

from paminco.callback import CallBackFlag
from paminco.net.network import Network
from paminco.net import load_sioux
from paminco.net._data_gas import temporary_gas_files

from paminco.optim import NetworkFW 
from paminco.algo.mca import MCA
from paminco.algo.mcfi import MCFI


@pytest.fixture
def net_sioux():
    return load_sioux()
        

class TestMCFI:
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.parametrize("rng", [1, 123, 42])
    def test_given_breakpoints(self, rng):
        # Load network
        net = load_sioux()
        
        # Setup random demand params
        rng = np.random.default_rng(rng)
        params = rng.random(3) * 10
        
        # Run singles
        fws = []
        for p in params:
            fw = NetworkFW(net, max_iter=10)
            fw.run(param=p)
            fws.append(fw)
        cost = np.array([fw.cost for fw in fws])
        
        # Run and compare multiple
        mcfi = MCFI(net, warmstart=False, kw_optim={"max_iter": 10})
        mcfi.run(params)
        assert np.isclose(cost, mcfi.cost_at(params)).all()
        for i, _ in enumerate(params):
            assert np.isclose(mcfi.flow_at(params[i]), fws[i].flow).all()
            
        assert len(mcfi.param_solution) == len(params)
        
    @pytest.mark.e2e
    @pytest.mark.very_slow
    @pytest.mark.slow
    @pytest.mark.parametrize("rng", [123, 42])
    def test_run_adaptive(self, rng):
        # setup random generator
        rng = np.random.default_rng(rng)

        # load network and set random commodity
        net = load_sioux()
        s, t = rng.choice(np.arange(net.n), 2, replace=False)
        net.set_demand((s, t, 30000), mode="linear", is_label=False)

        # run solvers
        mca = MCA(net, lambda_max=1)
        mca.run(print=False)

        mcfi_smpl = MCFI(net, 
                         adaptive_method="basic", 
                         lambda_min=0,
                         lambda_max=1,
                         )
        mcfi_smpl.run(print=False)

        mcfi_ext = MCFI(net, 
                        adaptive_method="constant_support", 
                        lambda_min=0,
                        lambda_max=1,
                        )
        mcfi_ext.run(print=False)

        thres_cost = 1e-3
        thres_flow = 500
        param = np.random.random(10) + 0.0001
        solvers = [mca, mcfi_smpl, mcfi_ext]
        for p in param:
            p = max(p, 1)
            costs = [s.cost_at(p) for s in solvers]
            for (i, j) in [(0, 1), (0, 2), (1, 2)]:
                # compare costs
                abs_diff = abs(costs[i] - costs[j])
                rel_diff = abs_diff / (min(costs[i], costs[j]))
                assert rel_diff < thres_cost
                
                # compare flows
                flow1 = solvers[i].flow_at(p)
                flow2 = solvers[j].flow_at(p)
                assert ((flow1 - flow2) < thres_flow).all()        
        
    @pytest.mark.e2e     
    @pytest.mark.slow
    @pytest.mark.parametrize("rng", [1, 123, 42])
    def test_warmstart(self, rng):
        # load network
        net = load_sioux()
        
        # setup random demand params
        rng = np.random.default_rng(rng)
        params = rng.random(3) * 10
        params = np.sort(params)
        
        # run and compare multiple
        mcfi_cold = MCFI(net, warmstart=False, kw_optim={"max_iter": 10})
        mcfi_cold.run(params)
        mcfi_warm = MCFI(net, warmstart=True, kw_optim={"max_iter": 10})
        mcfi_warm.run(params)
        assert np.isclose(mcfi_warm.cost_at(params[0]), mcfi_cold.cost_at(params[0]))
        assert (mcfi_warm.cost_at(params) <= mcfi_cold.cost_at(params)).all()

    @pytest.mark.parametrize("instance", ["gas11", "gas24", "gas40"])
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_small_gas_mcfi_mca(self, instance):
        self.gas_mcfi_against_mca(instance)

    def gas_mcfi_against_mca(self, instance):
        # load network
        with temporary_gas_files(instance) as tmpfiles:
            net = Network.from_gaslib(*tmpfiles)
            net.integrate_cost()

        # setup random demand params
        rng = np.random.default_rng()

        # setup random affine linear demand

        # -> Find all sources and sinks
        nodes = net.nodes.to_df()
        source_lbls = nodes[nodes['label'].str.contains('entry|source')]
        source_lbls = list(source_lbls['label'])
        # --> if no nodes have names with entry/source use all nodes 
        if len(source_lbls) == 0:
            source_lbls = list(nodes['label'])
        sink_lbls = nodes[nodes['label'].str.contains('exit|sink')]
        sink_lbls = list(sink_lbls['label'])
        # --> if no nodes have names with exit/sink use all nodes 
        if len(sink_lbls) == 0:
            sink_lbls = list(nodes['label'])
        # -> Pick a random source and sink
        s = rng.choice(source_lbls)
        t = None
        # ensure t != s with while
        while t is None or t == s:
            t = rng.choice(sink_lbls)
        r = sum(net.demand.b.total_rate) / 2

        # init demand
        demand_direction = (s, t, r)
        net.set_demand((net.demand.b, demand_direction), mode='affine')

        # parameters
        alpha = 1.01
        beta = 1

        # run MCFI
        mcfi = MCFI(net,
                    alpha=alpha,
                    beta=beta,
                    lambda_max=1,
                    adaptive_method="constant_support")
        mcfi.run()

        # run MCA to compare flows
        mca = MCA(net, alpha=alpha, beta=beta, lambda_max=1)
        mca.run()

        # choose random parameters to compare flows
        params = rng.random(50)
        params = np.unique(params)
        for p in params:
            x_mcfi = mcfi.flow_at(p)
            x_mca = mca.flow_at(p)
            # Check if total costs are similar
            assert np.isclose(sum(net.cost(x_mcfi)), 
                              sum(net.cost(x_mca)), 
                              rtol=alpha-1, 
                              atol=beta)
            # Check if flows are similar [with much more tolerance]
            # (not guaranteed by alpha-beta-property, but holds in these examples)
            assert np.allclose(x_mcfi, x_mca, rtol=(alpha-1)*10, atol=beta*10)
