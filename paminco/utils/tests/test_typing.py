import numpy as np
import scipy.sparse as sps

from paminco.utils.typing import IntEnum2, is_int, is_iterable, sparse_format
from paminco.utils.testing import assert_raises


def test_intenum2():
    class Enummer(IntEnum2):
        FROG = 12
        LION = 444
        ELEFANT = 33
        
    assert Enummer._members_str() == "['FROG', 'LION', 'ELEFANT']"
    assert Enummer.make("frog") == Enummer.FROG
    assert Enummer.make("Frog") == Enummer.FROG
    assert Enummer.make(12) == Enummer.FROG
    assert Enummer.make(Enummer.FROG) == Enummer.FROG
    assert_raises(ValueError, Enummer.make, "froggo")


def test_is_iterable():
    assert is_iterable(3) is False
    assert is_iterable((3, 8)) is True
    assert is_iterable([3, 8]) is True


def test_sparse_format():
    mat = np.random.random((8, 8))
    mat = sps.coo_matrix(mat)
    
    assert isinstance(sparse_format(mat, "array"), np.ndarray)
    assert (sparse_format(mat, "csc") - mat).nnz == 0
    assert (sparse_format(mat, "coo") - mat).nnz == 0
    assert (sparse_format(mat, "csr") - mat).nnz == 0
    assert_raises(ValueError, sparse_format, mat, "cia")