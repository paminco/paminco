import abc
import warnings
from distutils.version import StrictVersion

import scipy as sp
import scipy.sparse as sps
import numpy as np

from paminco.net.shared import FlowDirection
from paminco.utils.misc import Cache

from paminco.utils.typing import IntEnum2


class SubproblemMethod(IntEnum2):
    """Enum defining the subproblem method used in Frank-Wolfe."""

    AUTO = 0
    """Autoselect approriate subproblem method."""

    SHORTEST_PATH = 1
    """
    Find min cost flow by calculating shortest path.
    
    Works for networks with single commodity pairs such as
    [(1, 2, 40), ... (17, 22, 200)] only.
    """

    LP = 2
    """
    Find min cost flow via linear programming.
    
    See Also
    --------
    scipy.optimize.linprog
    """

    LP_INTERIOR_POINT = 3
    """Use 'interior-point' as method for linear programming."""

    LP_SIMPLEX = 4
    """Use 'revised simplex’' as method for linear programming."""

    LP_HIGHS = 5
    """Use 'highs' as method for linear programming."""

    def autodetect(self, network):
        """Find the best method for the given `network` if method is not set."""
        if self != SubproblemMethod.AUTO:
            return self
        
        return SubproblemMethod.from_network(network)

    @classmethod
    def from_network(cls, network):
        if min(network.edges.lb) != 0:
            # lowerbounds are non-zero -> force LP method
            return cls(SubproblemMethod.LP)
        elif network.demand.all_single_pairs is False:
            # exists commodity with more than one source or sink,
            # -> force LP method
            return cls(SubproblemMethod.LP)
        else:
            # lower bounds == 0 and single-source-single-sink
            return cls(SubproblemMethod.SHORTEST_PATH)

    def is_LP_method(self) -> bool:
        return self in [SubproblemMethod.LP,
                        SubproblemMethod.LP_SIMPLEX,
                        SubproblemMethod.LP_INTERIOR_POINT,
                        SubproblemMethod.LP_HIGHS]


def subproblem_solver(method, network, **kwargs):
    method = SubproblemMethod.make(method)
    method = method.autodetect(network)
    if method.is_LP_method():
        return LPSubproblemSolver(network, method, **kwargs)
    return SPSubproblemSolver(network, method, **kwargs)


class SubproblemSolver(abc.ABC):

    def __init__(self, network, method):
        self.network = network
        self.method = SubproblemMethod.make(method)
        self._check_valid()

    def __call__(self, *args, **kw):
        return self.solve(*args, **kw)

    @abc.abstractmethod
    def _check_valid(self): ...
    @abc.abstractmethod
    def solve(self): ...


class SPSubproblemSolver(SubproblemSolver):

    def __init__(
            self,
            network,
            method,
            param: float = 1,
            multiprocessing: bool = False,
            ):
        super().__init__(network, method)
        self.param = param
        self.multiprocessing = multiprocessing
        self.cache = Cache()

    def _check_valid(self) -> None:
        if (min(self.network.edges.lb) != 0 or
                self.network.demand.all_single_pairs is False):
            raise ValueError(
                f"{self.__class__} is only usable for single commodity and "
                "non-negative lower bounds."
            )

    def reset_cache(self) -> None:
        self.cache.reset()

    def solve(self, weight: np.ndarray) -> np.ndarray:
        demand = self.network.demand
        source_sink_rate = demand.get_source_sink_rate(self.param)
        f = self.network.flow_on_shortest(
            weight=weight,
            demand_triples=list(zip(*source_sink_rate)),
            unique_sources=demand.unique_sources,
            multiprocessing=self.multiprocessing,
            commodity_wise=False,
        )
        return f


class LPSubproblemSolver(SubproblemSolver):

    def __init__(
            self,
            network,
            method,
            param: float = 1,
            simplex_warmstart: bool = True,
            overwrite_method: bool = False,
            use_bounded_lp: bool = True
            ):
        super().__init__(network, method)
        self.param = param
        self.simplex_warmstart = simplex_warmstart
        self.overwrite_method = overwrite_method
        self.use_bounded_lp = use_bounded_lp
        # Factor for the LP Bounds
        # (in theory a factor of 1 is sufficient,
        #  a slightly higher factor avoids infeasibility
        #  due to numerical problems)
        self.bound_factor = 1.5
        self.cache = Cache()

    def _check_valid(self): pass

    def solve(self, weight: np.ndarray) -> np.ndarray:
        """Find min cost flow w.r.t. ``weight``.
        
        minimize::
        
            weight @ x
        
        such that::
        
            Gamma @ x == demand(param)
            lb <= x <= ub
            
        Parameters
        ----------
        weight : ndarray
        
        Returns
        -------
        x : ndarray
            Min cost flow.
        """
        if self.network.is_single_commodity is False:
            raise NotImplementedError(
                "Multicommodity LP subproblem solver not yet implemented."
            )
        
        # Get LHS, RHS
        A_eq = self.network.Gamma()[:-1]  # network caches gamma
        b_eq = self.network.demand(self.param).toarray().flatten()[:-1]
        
        # Invalidate last simplex solution (old param)
        self.cache.set_invalid("last_simplex_sol")
        
        result = None
        bounds = self.network.edges.bounds

        # If use_bounded_lp flag is set, cap all edge flow bounds
        # at an upper bound for the flow equal to the total inflow
        if self.use_bounded_lp:
            x_max = sum(abs(b_eq)) / 2
            bounds = np.maximum(bounds, -x_max) * self.bound_factor
            bounds = np.minimum(bounds, x_max) * self.bound_factor

        # Handle method setting
        if (self.method in [SubproblemMethod.LP, SubproblemMethod.LP_HIGHS]
                and StrictVersion(sp.__version__) < StrictVersion('1.6.0')):
            if self.method == SubproblemMethod.LP_HIGHS:
                warnings.warn(
                    "Method 'highs' requires scipy version >= '1.6.0', system "
                    f"version: {sp.__version__}, using simplex."
                )
            self.method = SubproblemMethod.LP_SIMPLEX

        if self.method == SubproblemMethod.LP_INTERIOR_POINT:
            # Interior point
            result = self._solve_lp_interior(weight,
                                             A_eq,
                                             b_eq,
                                             bounds)
        elif self.method == SubproblemMethod.LP_SIMPLEX:
            # Simplex
            result = self._solve_lp_simplex(weight,
                                            A_eq,
                                            b_eq,
                                            bounds)
        else:
            result = self._solve_lp_highs(weight,
                                          A_eq,
                                          b_eq,
                                          bounds)

        if result.status != 0:
            raise RuntimeError(
                "Problem when solving FW subproblem with LP."
                f"\nScipy Linprog returned status {result.status} with message"
                f"\n{result.message}."
            )
        return result.x

    def reset_cache(self) -> None:
        """Reset all cache variables of all methods."""
        self.cache.reset()

    def _solve_lp_highs(
            self,
            weight,
            A_eq,
            b_eq,
            bounds,
            ):
        # Create explicit (non-sparse) matrix if necessary
        if sps.issparse(A_eq):
            A_eq = A_eq.toarray()
            self._A_eq = A_eq
        
        result = sp.optimize.linprog(weight,
                                     A_eq=A_eq,
                                     b_eq=b_eq,
                                     bounds=bounds,
                                     method='highs')
        # If highs failed, retry with interior point method
        if result.status == 4 or result.status == 1:
            warnings.warn(f"HIGHS failed (Status: {result.status}), trying interior point!")
            if self.overwrite_method is True:
                self.method = SubproblemMethod.LP_INTERIOR_POINT
            return self._solve_lp_interior(weight, A_eq, b_eq, bounds)
        
        return result

    def _solve_lp_interior(
            self,
            weight,
            A_eq,
            b_eq,
            bounds,
            ):
        # Use the Interior point method (fails experimentally in many cases)
        opt = {'sparse': True, 'tol': 1e-6}
        result = sp.optimize.linprog(weight,
                                     A_eq=A_eq,
                                     b_eq=b_eq,
                                     bounds=bounds,
                                     options=opt,
                                     method='interior-point')
        # If Interior points failed, retry with simplex method
        if result.status == 4 or result.status == 1:
            warnings.warn(f"Interior point failed (Status: {result.status}), trying simplex!")
            if self.overwrite_method is True:
                self.method = SubproblemMethod.LP_SIMPLEX
            return self._solve_lp_simplex(weight, A_eq, b_eq, bounds)
        
        return result

    def _solve_lp_simplex(
            self,
            weight,
            A_eq,
            b_eq,
            bounds,
            ):
        # Create explicit (non-sparse) matrix if necessary
        if sps.issparse(A_eq):
            A_eq = A_eq.toarray()
            self._A_eq = A_eq
            
        # Reuse last solution if warmstart flag is set
        if (self.simplex_warmstart and
                self.cache.is_valid("last_simplex_solution") is True):
            x0 = self.cache["last_simplex_solution"]
        else:
            x0 = None
            
        # Temporarily deactivate scipy singular warning
        olderr = sp.special.seterr(singular='ignore')
        
        # try-block is needed because linprog may throw an exception
        # if solving fails with warmstart solution
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sp.optimize.linprog(weight,
                                             A_eq=A_eq,
                                             b_eq=b_eq,
                                             bounds=bounds,
                                             x0=x0,
                                             method='revised simplex')
        except Exception as e:
            # Solving fails with exception when using starting solution
            # => Try without starting solution and deactivate warmstart
            if x0 is not None:
                result = sp.optimize.linprog(weight,
                                             A_eq=A_eq,
                                             b_eq=b_eq,
                                             bounds=bounds,
                                             method='revised simplex')
                self.simplex_warmstart = False
            else:
                raise e
            
        # Reset Scipy warning settings
        sp.special.seterr(**olderr)
        
        # Case Warmstart solution is no basis solution
        if x0 is not None and (result.status == 6 or result.status == 4):
            # remove cached solution and retry
            self.cache.set_invalid("last_simplex_solution")
            result = self._solve_lp_simplex(weight, A_eq, b_eq, bounds)
            
        # If warmstart is set and a solution could be computed, store this
        # solution
        if self.simplex_warmstart and result.status == 0:
            self.cache["last_simplex_solution"] = result.x
        else:
            self.cache.set_invalid("last_simplex_solution")
        return result
