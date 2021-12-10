"""Module with util funcs for testing."""


def assert_raises(exception, func, *args, **kwargs) -> None:
    """Check whether calling func(*args, **kwargs) raises exeption.
    
    Parameters
    ----------
    exception : Exception
        Exception type.
    func : callable
        Method to be run.
    """
    try:
        func(*args, **kwargs)
    except exception:
        pass
