import pytest
import numpy as np
import scipy.sparse as sps
from paminco.net import load_sioux

from paminco.net import load_sioux
from paminco.net.network import Network
from paminco.linalg import star_inv, InverseMethod, star_update_by_edge


@pytest.fixture
def net_sioux():
    return load_sioux()


def rnd_possemdef_mat(rng=None, size: int = 20):
    rng = np.random.default_rng(rng)
    A = rng.random((size, size))
    return A.dot(A.T)
    

@pytest.mark.parametrize("rng", [42, 123, 99])
@pytest.mark.parametrize("reduced", [True, False])
@pytest.mark.parametrize("return_reduced", [True, False])
def test_compare_lstar(rng, reduced, return_reduced):
    mat = rnd_possemdef_mat(rng)
    star_inv_arr = star_inv(mat, InverseMethod.INVERSE, reduced, return_reduced)
    star_inv_sparse = star_inv(sps.csc_matrix(mat), InverseMethod.INVERSE, reduced, return_reduced, sparse=True).toarray()
    star_inv_cho = star_inv(mat, InverseMethod.CHOLESKY, reduced, return_reduced).toarray()
    assert np.isclose(star_inv_arr, star_inv_cho).all()
    assert np.isclose(star_inv_arr, star_inv_sparse).all()
    assert np.isclose(star_inv_cho, star_inv_sparse).all()
    
    
@pytest.mark.parametrize("rng", [42, 99])
@pytest.mark.parametrize("edge", [2, 0, 17])
# @pytest.mark.parametrize("delta_c", [0.2, 800, -25])
# TODO-PW: chol update fails for negative delta_c
@pytest.mark.parametrize("delta_c", [0.2, 800])
def test_update_lstar(net_sioux, rng, edge, delta_c):
    mat = rnd_possemdef_mat(rng)
    
    star_inv_arr = star_inv(mat, InverseMethod.INVERSE)
    star_inv_arr_up = star_update_by_edge(net_sioux, star_inv_arr, edge, delta_c)
    star_inv_sparse = star_inv(sps.csc_matrix(mat), InverseMethod.INVERSE, sparse=True)
    star_inv_sparse_up = star_update_by_edge(net_sioux, star_inv_sparse, edge, delta_c)
    star_inv_cho = star_inv(mat, InverseMethod.CHOLESKY)
    star_inv_cho_up = star_update_by_edge(net_sioux, star_inv_cho, edge, delta_c).toarray()
    
    assert star_inv_arr.shape == star_inv_cho.shape
    assert star_inv_arr.shape == star_inv_sparse.shape
    assert star_inv_cho.shape == star_inv_sparse.shape
    assert np.isclose(star_inv_arr_up, star_inv_cho_up).all()
    assert np.isclose(star_inv_arr_up, star_inv_sparse_up).all()
    assert np.isclose(star_inv_cho_up, star_inv_sparse_up).all()
    