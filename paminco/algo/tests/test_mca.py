import pytest
import numpy as np

from paminco.net import load_sioux
from paminco.net.network import Network

from paminco.algo.mca import MCA
from paminco.algo.mcfi import MCFI


@pytest.mark.parametrize("affine_data",
    [
        (("1", "20", 10000), ("13", "2", 20000)),
        (("8", "13", 10000), ("11", "23", 20000))
    ]
    )
def test_affine_demands(affine_data, lambda_max=1):
    net = load_sioux()
    net.clean(True, True, True, True, True)
    net.set_demand(affine_data, mode="affine")
    mca = MCA(net, lambda_max=lambda_max)
    mca.run()


@pytest.mark.e2e
@pytest.mark.very_slow
@pytest.mark.slow
@pytest.mark.parametrize("max_flow, thres",
    [
        (100000, 0.015),
    ])
@pytest.mark.parametrize("rng", [42, 99])
# @pytest.mark.parametrize("inverse_method", ["inverse", "cholesky"])
@pytest.mark.parametrize("inverse_method", ["cholesky"])
# @pytest.mark.parametrize("recomp_interval", [1, 5])
@pytest.mark.parametrize("recomp_interval", [1])
@pytest.mark.parametrize("pivot_mode", ["lex", "random"])
def test_compare_fw(max_flow, thres, rng, inverse_method, recomp_interval, pivot_mode):
    rng = np.random.default_rng(rng)
    
    # load network
    net = load_sioux()
    net.cost.integrate(inplace=True)
    net.clean(True, True, True, True, True)
    s, t = rng.choice(np.arange(net.n), 2, replace=False)
    net.set_demand((s, t, max_flow), mode="linear", is_label=False)
    
    # run mca
    mca = MCA(net,
              lambda_max=1,
              print=False,
              inverse_method=inverse_method,
              recomp_interval=recomp_interval,
              pivot_mode=pivot_mode,
              max_iter=20)
    mca.run()
    
    # randomly select 5 breakpoints from mca
    params = rng.choice(mca.all_params(), 5, replace=False)
    params = np.sort(params)
    
    # run fw with params
    mcfi = MCFI(net, epsilon=0.1)
    mcfi.run(params)
    
    # check that solutions are valid and close to mcfi solutions
    for p in params:
        p = max(0.001, p)
        abs_diff = mca.cost_at(p) - mcfi.cost_at(p)
        rel_diff = abs_diff / mcfi.cost_at(p)
        assert rel_diff < thres
    
    assert len(mcfi.param_solution.arr_param) == len(params)


def test_laplace_recomputation_vs_update():
    net = load_sioux()
    net.set_demand(('1', '20', 10000), mode='linear')

    # Compute solution with recomputation in every step
    mca_re = MCA(net, recomp_interval=1)
    mca_re.run()

    # Compute solution with Laplace updates only
    mca_up = MCA(net, recomp_interval=0)
    mca_up.run()

    # Ensure solutions are the same
    assert np.allclose(mca_re.flow_at(1), mca_up.flow_at(1))