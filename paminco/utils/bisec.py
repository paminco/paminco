"""TODO-doc TODO-PW"""
import warnings
import numpy as np


def bisec_fast(func, tol=1e-02, lo=0, up=1, exclude_lo=False,
               exclude_up=False, val_constr=None, debug=False):
    #TODO-doc TODO-PW
    """Use the bisection method to find a root of func between lo and up.

    Parameters
    ----------
    func : callable
        Objective function.
    tol : float, default=1e-02
        The tolerance for a solution, i.e., x is a solution if
        ``|func(x)| < tol``.
    lo : float, default=0
        The lower end of the interval.
    up : float, default=1
        The upper end of the interval.
        if func(lo) and func(up) do NOT have different signs, the method
        tries to increase up until this holds. If this is not possible
        a RuntimeException is raised.
    exclude_lo : bool, default=False
        Flag to exclude the lower bound from the solution space.
    exclude_up : bool, default=False
        Flag to exclude the upper bound from the solution space.
    val_constr : str, optional
        Constraints for the value of the solution. Possible values:
          - ``None`` : every solution x with | func(x) | < tol is a solution
          - ``'npos'`` : only x with func(x) <= 0 are solutions
          - ``'nneg'`` : only x with func(x) >= 0 are solutions
    debug : bool, default=False
        If True, the function prints debug information to stdout.
    
    Returns
    -------
    type
        TODO
    
    Notes
    -----
    Assumes that func is a continuous function.
    Throws an exception if no up can be found s.t.
    func(lo) and func(up) do not have different signs
    
    """
    def is_sol(x, val):
        """Check if x and val satisfy all conditions of a solution."""
        if abs(val) >= tol:
            return False
        if (exclude_lo and lo == x) or (exclude_up and up == x):
            return False
        if val_constr is None:
            return True
        if val_constr == 'nneg' and val < 0:
            return False
        if val_constr == 'npos' and val > 0:
            return False
        return True

    vlo = func(lo)
    if is_sol(lo, vlo):
        # print("Lowerbound", vlo, "is already a root") if debug else None
        return lo
    slo = -1 if vlo < 0 else 1
    # print("LB value", vlo, "@ x =", lo, "with ", slo) if debug else None
    vup = func(up)
    sup = -1 if vup < 0 else 1
    # print("Search for upper bound") if debug else None
    # print("UB value", vup, "@ x =", up, "with ", sup) if debug else None
    while slo * sup > 0 or (exclude_up and slo * sup == 0):
        up *= 2
        vup = func(up)
        sup = -1 if vup < 0 else 1
        # print("UB value", vup, "@ x =", up, "with ", sup) if debug else None
        if up > 1e30:
            raise RuntimeError("Bisec does not converge.")

    if is_sol(up, vup):
        return up

    i = 0

    mi = None
    while i < 100:
        last_mi = mi
        mi = (lo + up)/2
        # If middle point of the interval does not change because of maximum precision
        # return an approximate solution
        if (mi == last_mi):
            warnings.warn("Maximum float precision was exceeded in bisec method!", RuntimeWarning)
            if (val_constr == 'nneg'):
                return mi if mi > 0 else up if sup > 0 else lo
            if (val_constr == 'npos'):
                return mi if mi < 0 else up if sup < 0 else lo
            return mi
        vmi = func(mi)
        smi = -1 if vmi < 0 else 1
        # print("i {:5d} F value".format(i), vmi,
        #       "@ x =", mi, "with ", smi) if debug else None
        if is_sol(mi, vmi):
            return mi
        if slo * smi <= 0:
            up = mi
            vup = vmi
            sup = smi
        elif sup * smi <= 0:
            lo = mi
            vlo = vmi
            slo = smi
        else:
            raise RuntimeError("Bisec does not converge.")
        i += 1

    raise RuntimeError("Bisec does not converge.")


def bisec_method(func, tol=0.01, lo=None, up=None, debug=False, flex_up=False,
                 increasing=None):
    # TODO-doc TODO-PW
    """
    Execute the bisection method to find a root of func.

    Parameters
    ----------
    func: function-handle
        the function
    tol: float or tuple
        the error tolerance. Can be given as float, then return a solution x
        with -tol <= f(x) <= tol.
        Can also be specified as tuple tol = (tol[0], tol[1]). In this case,
        return x with tol[0] <= f(x) <= tol[1]
    lo: float
        lower bound on the root
    up: float
        upper bound on the root
    debug: bool
        If true, print debug information
    flex_up: bool
        Flexible upper bound flag: If True, the function checks if func(up) > 0
        and increases up if this is not the case.
        If False, func(up) > 0 is assumed implicitely

    If lo and up are not given, lo is assumed to be 0 and up is determined
    automatically.
    """
    if isinstance(tol, float):
        tol = (-tol, tol)
    if debug:
        print("(Recursive) Call of bisec_method with (lo,up) =", (lo, up))
    # Find initial upper bound
    if lo is None:
        lo = 0
    if up is None:
        up = 1
        flex_up = True
    if flex_up:
        sign_at_lo = np.sign(func(lo))
        if sign_at_lo == 0:
            print("The lower bound is already the solution")
            return lo
        if debug:
            print("Find initial upper bound")
            print("Sign at lower bound is", sign_at_lo,
                  "; looking for upper bound with sign", -sign_at_lo)
        k = 2
        while func(up) * sign_at_lo >= 0:
            if debug:
                print(" -> up =", up, "with function value", func(up))
            j = int(k/2) * ((-1) ** k)
            up = 2 ** j
            if up > 1e10:
                raise ValueError("No solution can be found in given interval.")
            k += 1
        if debug:
            print(" -> Found up =", up, "with function value", func(up))
    # Determine if function is increasing or decreasing if necesseray
    if increasing is None:
        sign_at_lo = np.sign(func(lo))
        sign_at_up = np.sign(func(up))
        # Make a sanity check of the sign at lower and upper bound
        if sign_at_lo * sign_at_up > 0:
            raise ValueError("No solution can be found in given interval.")
        # Increasing function iff sign at right side is positive
        increasing = (sign_at_up > 0)
        if debug:
            print("Function is "+("increasing" if increasing else "decreasing"))
    x = (lo + up) / 2
    val = func(x)
    if debug:
        print(val, "@", x)
    if tol[0] <= val <= tol[1]:
        return x
    if (up - lo) < 1e-16:   # max(abs(tol[0]), abs(tol[1])):
        err_msg = "Error in bisection method.\n No solution can be found in " \
                  "given interval. Last Value {:f} @ x = {:f}".format(x, val)
        raise ValueError(err_msg)
    if (val <= tol[0] and increasing) or (val >= tol[1] and not increasing):
        return bisec_method(func, tol, lo=x, up=up, debug=debug,
                            increasing=increasing)
    return bisec_method(func, tol, lo=lo, up=x, debug=debug,
                        increasing=increasing)
