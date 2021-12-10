import copy

import numpy as np
from scipy import sparse
from scipy import linalg as splinalg

from paminco.utils.typing import IntEnum2


class SingularLaplaceError(Exception):
    """Error that occured while computing the Laplace matrix."""

    def __init__(
            self,
            msg=None,
            network=None,
            weight=None,
            n_cc=None,
            cc=None
            ) -> None:
        if msg is None:
            msg = "Submatrix of Laplace matrix is singular."
        self.msg = msg
        self.net = network
        self.weight = weight
        self.n_cc = n_cc
        self.cc = cc

        if self.n_cc is None or self.cc is None:
            self._info_from_network()

    def _info_from_network(self):
        if self.net is None or self.weight is None:
            return
        non_zero_edges = np.where(self.weight != 0)[0]
        n_cc, cc = self.net.connected_components(edges=non_zero_edges)
        self.non_zero_edges = non_zero_edges
        self.n_cc = n_cc
        self.cc = cc

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        out = self.msg
        if (hasattr(self, 'cc')):
            out += '\n'
            out += f"Graph decomposes into {self.n_cc} components"
        return out


class InverseMethod(IntEnum2):
    """Enum defining the method to invert a matrix."""

    INVERSE = 0
    """Calculate the inverse via :func:`numpy.linalg.inv`."""

    CHOLESKY = 1
    """Calculate the inverse via Cholesky decomposition.
    
    See Also
    --------
    CholeskyInverse
    """


class CholeskyInverse:
    """A class representing the inverse of some matrix based on the cho decomp.

    Parameters
    ----------
    cho_fac : tuple
        The cholesky decomposition as returned by scipy.linalg.cho_factor
    return_reduced : bool, default=False
        If true, this represents the reduced matrix (w/o zero row/column)
        
    Attributes
    ----------
    cho: TODO
    return_redcued: TODO
    array: TODO
    
    Notes
    -----
    A cholesky inverse works in part like a ndarray. It implements the
    dotproduct (dot), returns the shape of the inverse (shape), supports
    slicing and can be converted in an ndarray using toarray().
    
    Using ``toarray()`` or ``slicing`` will result in an explicit computation
    of the inverse. The actual inverse will then be cached and make future
    calls of toarray, dot, and slicing faster.
    """

    def __init__(self, cho_fac, return_reduced: bool = False):  # noqa D107
        self._cho = cho_fac
        self._return_reduced = return_reduced
        self._array = None

    def __getitem__(self, *args, **kwargs):
        """
        Magic method for slicing.

        Maps to the corresponding function of the ndarray representation of
        this object.
        """
        arr = self.toarray()
        return arr.__getitem__(*args, **kwargs)
    
    def dot(self, other, reduced: bool = False):
        """Compute the dot product of self and other.

        Parameters
        ----------
        other : array-like or sparse.spmatrix
            Vector or matrix to multiply with.
        reduced : bool
            Flag, if other is already in reduced form (i.e., has n-1 rows)

        Returns
        -------
        ndarray
            The result of self<dot>other
        """
        if self._array is not None and not reduced:
            return self._array.dot(other)

        # Cast to numpy array if necessary
        try:
            other = other.toarray()
        except AttributeError:
            other = np.array(other)

        if not reduced:
            other = other[1:]
        if len(other.shape) == 1:
            red = splinalg.cho_solve(self._cho, other)
            if self._return_reduced:
                return red
            else:
                return np.insert(red, 0, 0)
        cols = [splinalg.cho_solve(self._cho, other[:, c])
                for c in range(other.shape[1])]
        red = np.stack(cols, axis=1)
        if self._return_reduced:
            return red
        else:
            out = np.vstack([np.zeros(red.shape[1]), red])
            return out

    def toarray(self, caching: bool = True) -> np.ndarray:
        """Return the inverse as ndarray.

        Parameters
        ----------
        caching : bool
            If true (default) the ndarray is stored internally, this object
            will use the ndarray for all subsequent computations (dotproduct
            or slicing)

        Returns
        -------
        ndarray
            The inverse represented by this object as array.
        """
        if self._array is not None:
            return self._array
        out = self.dot(np.identity(self._cho[0].shape[0]), reduced=True)
        if not self._return_reduced:
            out = np.hstack([np.zeros((out.shape[0], 1)), out])
        if caching:
            self._array = copy.deepcopy(out)
        return out

    def update_by_edge(self, net, edge, delta_c):
        """Update inverse accoding to edgeweight change delta_c on edge."""
        if delta_c < 0:
            downdate = True
            delta_c = - delta_c
        else:
            downdate = False
        n = len(self._cho[0])
        x = np.zeros((n,), dtype=np.float64)

        v, w = net.edges.indices[edge]
        if v >= 1:
            x[v - 1] = - delta_c ** 0.5
        if w >= 1:
            x[w - 1] = delta_c ** 0.5
        new_cho = _cholupdate(self._cho, x, downdate)
        return CholeskyInverse(new_cho, self._return_reduced)

    @property
    def shape(self):
        """Return the shape of the matrix represented by this object."""
        if self._return_reduced:
            return self._cho[0].shape
        return (self._cho[0].shape[0] + 1, self._cho[0].shape[1] + 1)

    def __str__(self):
        """Return a string representation."""
        out = "Cholesky Inverse Wrapper with shape " + str(self.shape)
        return out


def _cholupdate(cho_fac, x, downdate: bool = False):
    """
    Perform a rank-1-update on the cholesky decomposition cho_fac.

    If L in chol_fac is the cholesky decomposition of A, this returns the
    cholesky decomposition of
     .. math::
        \begin{align*}
            & A + x x^T \quad \text{(update) or }  \\
            & A - x x^T \quad \text{(downdate)}
        \end{align*}

    Parameters
    ----------
    cho_fac : tuple
        Tuple (L, lower) where L is the cholesky decomposition and
        lower a boolean indicating whether L is a lower or upper
        triangular matrix.
    x : array_like
        The array used for the update formula.
    downdate : bool, default=False
        Indicates if downdate (-) or an update (+) formula is used.

    See Also
    --------
    scipy.linalg.cho_factor
    """
    sgn = -1 if downdate else +1
    L, lower = cho_fac
    # Create a copy of the cholesky factorization -> L will be overwritten
    L = np.array(L)
    # ensure lower triangular matrix
    if not lower:
        L = L.T
    n = len(x)
    # Update formula
    for k in range(n):
        if x[k] != 0:
            r = (L[k, k] ** 2 + sgn * x[k]**2) ** (0.5)
            c = r / L[k, k]
            s = x[k] / L[k, k]
            if r == 0 or not np.isfinite(r):
                raise ValueError(
                    "Cholesky update failed: new matrix is not positive definite."
                )
        else:
            r = L[k, k]
            c = 1
            s = 0
        L[k, k] = r
        if k < n - 1:
            L[(k + 1):n, k] = (L[(k + 1):n, k] + sgn * s * x[(k + 1):n]) / c
            x[(k + 1):n] = c * x[(k + 1):n] - s * L[(k + 1):n, k]
    return (L, True)


def star_inv(
        matrix,
        method: InverseMethod = InverseMethod.INVERSE,
        reduced: bool = False,
        return_reduced: bool = False,
        sparse: bool = False
        ):
    """Compute the pseudo inverse of ``matrix`` mapping into subspace with v_1=0.

    Parameters
    ----------
    matrix : ndarray or sparse matrix
        The matrix.
    method : InverseMethod = InverseMethod.INVERSE
        Method to use to compute inverse.

        InverseMethod.INVERSE  - Use scipy.linalg.inv, return ndarray.

        InverseMethod.CHOLESKY  - Use scipy.linalg.cho_factor, return
        :class:`~paminco.linalg.CholeskyInverse`.

    reduced : bool, default=False
        Whether ``matrix`` is in reduced form, i.e., w/o first row and
        column
    return_reduced : bool, default=False
        Whether to return reduced inverse (i.e. w/o first row/column)
        only.
    sparse : bool, default=False
        Flag to use scipy.sparse.linalg.inv method, if applicable. Has
        effect only if ``method == InverseMethod.INVERSE`` and ``matrix``
        is sparse. Maybe considerably slower than normal use, even if
        matrix is sparse.

    Returns
    -------
    ndarray or CholeskyInverse
        The pseudo inverse of matrix.

    """
    if (sparse and not type(matrix) == np.ndarray and
            method == InverseMethod.INVERSE):
        return _star_inv_sparse(matrix,
                                reduced=reduced,
                                return_reduced=return_reduced)
    if type(matrix) != np.ndarray:
        matrix = matrix.toarray()

    if method == InverseMethod.INVERSE:
        return _star_inv_arr(matrix,
                             reduced=reduced,
                             return_reduced=return_reduced)
    if method == InverseMethod.CHOLESKY:
        return _star_cho_arr(matrix,
                             reduced=reduced,
                             return_reduced=return_reduced)

    raise ValueError(f"{method} is not a valid inverse method.")


def _star_inv_arr(matrix, reduced: bool = False, return_reduced: bool = False):
    sub_mat = matrix if reduced else matrix[1:, 1:]
    sub_inv = splinalg.inv(sub_mat)
    if return_reduced:
        return sub_inv
    if not reduced:
        out = np.zeros(matrix.shape)
    else:
        out = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1))
    out[1:, 1:] = sub_inv
    return out


def _star_cho_arr(matrix, reduced: bool = False, return_reduced: bool = False):
    sub_mat = matrix if reduced else matrix[1:, 1:]
    chof = splinalg.cho_factor(sub_mat)
    return CholeskyInverse(chof, return_reduced)


def _star_inv_sparse(matrix, reduced: bool = False, return_reduced: bool = False):
    sub_mat = matrix if reduced else matrix[1:, 1:]
    sub_mat = sub_mat.tocsc()
    sub_inv = sparse.linalg.inv(sub_mat)
    if return_reduced:
        return sub_inv
    add_row = sparse.csr_matrix(([], ([], [])), shape=(1, sub_inv.shape[1]))
    out = sparse.vstack([add_row, sub_inv])
    add_col = sparse.csc_matrix(([], ([], [])), shape=(out.shape[0], 1))
    out = sparse.hstack([add_col, out])
    return out


def star_update_by_edge(net, lstar, edge: int, delta_c: float):
    r"""Update the Laplace pseudoinverse ``lstar`` when edge weight changes.

    .. math::
        \mathbf{L}^{\ast}_{t^{i+1}} =
        (
            \mathbf{I}_n - \frac
                {\Delta c_e}
                {1 + \Delta c_e \gamma_e^T \mathbf{L}^{\ast}_{t^{i}} \gamma_e}
            \mathbf{L}^{\ast}_{t^{i}} \gamma_e \gamma_e^T
        )
        \mathbf{L}^{\ast}_{t^{i}}
    
    Parameters
    ----------
    net : Network
        The network to which ``lstar`` belongs.
    lstar : ndarray, scipy.sparse.spmatrix, or CholeskyInverse
        The current lstar inverse.
    edge : int
        The index of the edge.
    delta_c : float
        The amount of change in the edge weight.

    Returns
    -------
    ndarray or CholeskyInverse
        The updated inverse.

    """
    if isinstance(lstar, sparse.spmatrix):
        lstar = lstar.toarray()
    if type(lstar) == np.ndarray:
        return _star_update_arr(net, lstar, edge, delta_c)
    if type(lstar) == CholeskyInverse:
        return lstar.update_by_edge(net, edge, delta_c)


def _star_update_arr(net, lstar, edge: int, delta_c: float):
    lstar_gamma = net.times_gamma(lstar, edge=edge)
    denominator = 1 + delta_c * net.times_gamma(lstar_gamma, edge=edge)
    if denominator == 0:
        raise np.linalg.LinAlgError(
            "Discriminant of the Sherman-Morrison-"
            "update formula is 0. Updated matrix is "
            "singular!"
        )
    fac = delta_c / denominator
    outer_product = np.outer(lstar_gamma, lstar_gamma)
    return lstar - fac * outer_product
