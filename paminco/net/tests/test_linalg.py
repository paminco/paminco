import numpy as np
import scipy.sparse as sps

from paminco.net import load_sioux


def test_laplacian():
    def laplacian2(net, weight, reduced: bool = False):
        N = -net.Gamma()
        W = sps.diags(weight)
        lap = N * W * N.T
        if reduced is True:
            return lap[1:, 1:]
        return lap
    net = load_sioux()
    weight = np.random.random(net.m) * 1000 - 500
    for reduced in [True, False]:
        diff = (net.laplacian(weight, reduced=reduced)
                - laplacian2(net, weight, reduced=reduced))
        assert (diff > 1e-8).nnz == 0, \
            "laplacian calc does not work."
