"""Module that handles some types and defines an extension to IntEnum."""
from enum import IntEnum

import numpy as np
import scipy.sparse as sps


class IntEnum2(IntEnum):
    """Class that extends enum.IntEnum with alternative constructor."""

    @classmethod
    def from_str(cls, name: str):
        """Construct based on ``name``.
        
        ``name`` will be capitalized before passing into default class
        constructor. Spaces are replaced by underscores.
        """
        return cls[name.upper().replace(" ", "_")]

    @classmethod
    def make(cls, val):
        """Construct based on a variety of inputs.
        
        Parameters
        ----------
        val: str, int or IntEnum2
        
        Raises
        ------
        ValueError
            If ``val`` is str, but is no valid member when capitalized.
        TypeError
            If ``val`` is neither str, int nor IntEnum2.
        
        See Also
        --------
        IntEnum2.from_str
        """
        if isinstance(val, (int, cls)):
            return cls(val)
        elif isinstance(val, str):
            try:
                return cls.from_str(val)
            except KeyError:
                raise ValueError(
                    f"'{val}' is not a valid member of "
                    f"'{cls.__name__}'.\n The following members "
                    f"are available (lower case possible):\n\t"
                    f"{cls._members_str()}."
                )
        raise TypeError(f"Type of 'val' = {type(val)} not valid")

    @classmethod
    def _members_str(cls):
        return str([d.name for d in cls.__members__.values()])


def is_int(obj) -> bool:
    """Check whether object is int, numpy.int32, or numpy.int64."""
    if isinstance(obj, (int, np.int32, np.int64)):
        return True
    return False


def is_iterable(obj) -> bool:
    """Check whether object has an iterator."""
    try:
        iter(obj)
    except Exception:
        return False
    return True


def sparse_format(
        M: sps.spmatrix,
        return_as: str
        ) -> sps.spmatrix:
    """Change format of sparse matrix ``M``.
    
    Parameters
    ----------
    M : spmatrix
        Sparse scipy matrix.
    return_as : str
        Return type.
    
    Returns
    -------
    spmatrix
        Sparse matrix in ``return_as`` format.
    """
    if return_as == 'array':
        return M.toarray()
    if return_as == 'csc':
        return M.tocsc()
    if return_as == 'coo':
        return M.tocoo()
    if return_as == 'csr':
        return M.tocsr()
    raise ValueError(f"{return_as} is invalid return format")
