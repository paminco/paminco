"""Module with miscellaneous util funcs."""
from collections import UserDict


class Cache(UserDict):
    """Class that acts as a cache.
    
    Parameters
    ----------
    counter : bool, default=True
        Whether to cound read and write accesses.
    
    Attributes
    ----------
    valids : dict
        Whether dict element is valid and can be pulled from cache.
    accesses : dict, optional
        Counter of reads.
    writes : dict, optional
        Counter of writes.
    """
    
    def __init__(self, counter: bool = True):
        super().__init__()
        
        # Dict storing if cache element is valid
        self.valids = {}
        
        if counter:
            self.accesses = {}
            self.writes = {}

    def __setitem__(self, key: str, value) -> None:
        if hasattr(self, "writes"):
            try:
                self.writes[key] += 1
            except KeyError:
                self.writes[key] = 1
        super().__setitem__(key, value)
        self.valids[key] = True

    def __getitem__(self, key: str):
        if hasattr(self, "accesses"):
            try:
                self.accesses[key] += 1
            except KeyError:
                self.accesses[key] = 1
        return super().__getitem__(key)

    def reset(self, hard: bool = False) -> None:
        for k in self:
            self.valids[k] = False
        if hard is True:
            for key in self.keys():
                super().__setitem__(key, None)

    def is_valid(self, key: str) -> bool:
        """Check if element can be used from cache.
        
        Parameters
        ----------
        key : str
            Name of cache element.
        
        Returns
        -------
        bool
            Whether cache element ``key`` is valid.
        """
        return self.valids.get(key, False)

    def set_invalid(self, *keys) -> None:
        """Set cache elements to invalid.
        
        Parameters
        ----------
        keys : tuple of str
            Name(s) of cache element(s).
        """
        for key in keys:
            if key in self:
                self.valids[key] = False


def callback_to_list(callback):
    """Cast callback to list.
    
    Parameters
    ----------
    callback : callable or list of callables
        Callable object(s).
    
    Returns
    -------
    list
        List of callable objects.
    """
    check_callable = True
    
    # convert callback to list
    if not isinstance(callback, list):
        if callback is None:
            callback = []
            check_callable = False
        elif callable(callback):
            callback = [callback]
        else:
            raise TypeError("'callback' must be callables or list of "
                            "callables.")
    
    # check if all callbacks are callable
    if check_callable is True:
        for c in callback:
            if callable(c) is False:
                raise TypeError("'callback' is not callable.")
    
    return callback
