import pytest

from paminco import NetworkFW
from paminco.net import load_sioux


@pytest.mark.e2e
@pytest.mark.slow
def test_user_eq_sioux():
    # Read optimal flow and cost of user equilibrium (:= UE)
    optimal_cost = 4231335.287107440
    
    # run 200 iterations of our implementation
    net = load_sioux()
    net.integrate_cost()  # integrate cost to calc UE
    fw = NetworkFW(net, epsilon=1e-20, max_iter=200)
    fw.run()
    
    calculated_cost = fw.cost
    divergence_abs = calculated_cost - optimal_cost
    divergence_rel = divergence_abs / optimal_cost
    assert divergence_rel < 0.001
