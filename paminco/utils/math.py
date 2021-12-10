"""Module providing some math utils."""
import numpy as np


def np_divide_a_by_b(
        a,
        b,
        default_value: float = 0.0,
        cast_to_numpy: bool = False
        ) -> np.ndarray:
    """Safely divide a by b.
    
    Parameters
    ----------
    a : ndarray
        Numerator
    b : ndarray
        Denominator.
    default_value : float, default=0.0
        Fill value where b equals 0.
    
    Returns
    -------
    res : ndarray
        a / b filled with ``default_value`` where ``b == 0``.
    
    See Also
    --------
    numpy.divide
    """
    if cast_to_numpy is True:
        a = np.array(a)
        b = np.array(b)
    
    shape = a.shape
    if a.size < b.size:
        shape = b.shape
    out = np.full(shape, default_value)
    return np.divide(a, b, out=out, where=(b != 0), casting='unsafe')


def find_min_col_lex(M) -> int:
    """Find the minimal column of a matrix ``M`` lexiographically.
    
    Parameters
    ----------
    M : ndarray
        Matrix for which to find minimum column.
    
    Returns
    -------
    mincol : int
        Index of minimum column in M.
    """
    number_of_rows = M.shape[0]
    col_min = M.min(axis=1).reshape(number_of_rows, 1)
    min_cols_idx = (M == col_min)
    c = min_cols_idx.sum(1).reshape(number_of_rows, 1)
    minrow = -min((x, -i) for i, x in enumerate(c))[1]
    d = M[minrow, ]
    mincol = -min((x, -i) for i, x in enumerate(d))[1]
    return mincol
