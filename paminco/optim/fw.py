from __future__ import annotations

import copy

import numpy as np
import scipy.optimize as spopt

from paminco._base import Config
from paminco.utils.typing import IntEnum2


def linesearch(
        fun,
        y: np.ndarray,
        z: np.ndarray,
        x0: float = 0.5,
        tol: float = 1e-6,
        **kwargs
        ) -> float:
    """Minimize ``fun`` on a line between ``y`` and ``z``.
    
    Parameters
    ----------
    fun : callable f(x)
        Objective function.
    y : ndarray, shape (n,)
        Starting point.
    z : ndarray, shapef (n,)
        Starting point.
    x0 : ndarray, shape (n,), default=0.5
        Initial guess.
    tol : float, default=1e-6
        Tolerance for termination.
    kwargs : keyword arguments
        Further keyword arguments passed onto scipy.optimize.minimize.
    
    Returns
    -------
    s : float
        Optimal step size so that `fun` is minimized at x = (1-s)*y + s*z.
    
    See Also
    --------
    scipy.optimize.minimize
    """
    def fun_ls(lambd_, y, z):
        return fun((1 - lambd_) * y + lambd_ * z)

    res = spopt.minimize(fun_ls,
                         x0=x0,
                         args=(y, z, ),
                         tol=tol,
                         bounds=([0., 1], ),
                         **kwargs)
    s = res.x[0]
    return s


class FWBreakFlag(IntEnum2):
    """Enum defining the status of the FW optimizer."""
    
    COST_INVALID = -1
    """Function value is smaller than specified lower bound."""

    NOT_SET = 0
    """Undefined breakflag."""

    CONVERGED = 1
    """Frank-Wolfe coverged successfuly for specified tolerance."""

    MAX_ITER = 2
    """Reached maximum number of iterations."""

    USER_INTERRUPT = 99
    """Interrupt by user."""

    def __str__(self) -> str:
        return self.__doc__

    def to_status(self) -> int:
        if self in [FWBreakFlag.CONVERGED, FWBreakFlag.MAX_ITER]:
            return 0
        if self == FWBreakFlag.NOT_SET:
            return 1
        if self == FWBreakFlag.COST_INVALID:
            return 2
        if self == FWBreakFlag.USER_INTERRUPT:
            return 3


class FWMode(IntEnum2):
    """Enum defining the update mode of the FW solver.
    
    References
    ----------
    .. [1] Florian, Michael, J Guálat, and H Spiess. "An efficient
        implementation of the “Partan” variant of the linear approximation
        method for the network equilibrium problem." Networks 17.3 (1987):
        319-339.
    """
    
    STEP_SIZE_DETERIATION = 0
    r"""Solution is updated as
    :math:`\mathbf{x}_{i+1}
    = (1-\eta)*\mathbf{x}_{i} + \eta * \mathbf{s}_i` where
    :math:`\eta = \frac{2}{2+i}`.
    """
    
    LINESEARCH = 1
    r"""Solution is updated as
    :math:`\mathbf{x}_{i+1} = (1-\eta)*\mathbf{x}_{i} + \eta * \mathbf{s}_i`
    where :math:`\eta` is found by minimizing :math:`f` on a line between
    :math:`\mathbf{x}_i` and :math:`\mathbf{s}_i`.
    """
    
    PARTAN = 2
    """TODO-PJ"""


class FWX:
    """Class that keep tracks of (intermediate) best x in Frank-Wolfe."""

    def __init__(self):
        self.x = None
        self.s = None
        self.x_eta = None
        self.x_pmax = None
        self.x_bef = None


class FWConfig(Config):
    """Options for Frank-Wolfe optimizer.

    Parameters
    ----------
    print : bool, default=False
        Whether to print iteration summary at the end of each iteration.
    epsilon : float, default=1e-3
        Convergence threshold. FW converges if
        ``(self.funval - self.blb) / self.funval < epsilon``.
    mode : FWMode, int, or str, default=FWMode.PARTAN
        Update mode of solution, see FWMode.
    max_iter : int, default=2000
        Maximum number of iterations.
    lb : float, default=0
        Lower bound of objective function. If funval equals this bound,
        optimal solution is assumed to be found.
    
    See Also
    --------
    FWMode
    """

    options = ["print", "epsilon", "mode", "max_iter", "lb"]
    """Options for FW."""

    def __init__(self, **kwargs):
        self.print = False
        self.epsilon = 1e-3
        self._mode = FWMode.PARTAN
        self.max_iter = 2000
        self.lb = 0
        
        self.map_kwargs(**kwargs)

    def map_kwargs(self, **kw):
        """Map `kw` to config."""
        for (k, v) in kw.items():
            if k in self.options:
                setattr(self, k, v)
            else:
                raise ValueError(f"{k} is not a valid FW setting.")

    @property
    def mode(self) -> FWMode:
        return self._mode
    
    @mode.setter
    def mode(self, value) -> None:
        self._mode = FWMode.make(value)
    
    mode.__doc__ = FWMode.__doc__


class FW:
    r"""Frank-Wolfe algorithm for constrained convex optimization.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
        
        ``fun(x) -> float``
        
        where ``x`` is an 1-D array with shape (n,).
    
    fprime : callable
        Function that returns the gradient vector:.
        
        ``fprime(x) -> array_like``
        
        where ``x`` and the return value are both 1-D arrays with shape
        (n,).
    
    subproblem_solver : callable
        Minimize the linear approximation of the problem:
        
        .. math::
            \underset{\mathbf{s} \in \mathcal{D}}{\mathrm{argmin}}
            \quad \mathbf{s}^T\nabla f(\mathbf{x}_i)
        
        Is called with the signature:
        
        ``subproblem_solver(f'(x)) -> array_like``.
    
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,), where n
        is the number of independent variables.
    kwargs : keyword arguments
        Further options, see FWConfig.
    
    Attributes
    ----------
    xes : FWX
        Object keeping track of intermediate solutions.
    x
    config
    
    See Also
    --------
    FWConfig : Options accepted by the solver.
    paminco.optim.fw_net.NetworkFW : Use FW to find a min cost flow.
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
    
    .. [2] Florian, Michael, J Guálat, and H Spiess. "An efficient
           implementation of the “Partan” variant of the linear approximation
           method for the network equilibrium problem." Networks 17.3 (1987):
           319-339.
    """

    def __init__(
            self,
            fun,
            fprime,
            subproblem_solver,
            x0,
            **kwargs
            ) -> None:
        self._c = FWConfig(**kwargs)
        self.fun = fun
        self.fprime = fprime
        self.subproblem_solver = subproblem_solver
        self.x0 = x0

    def __str__(self) -> str:
        return (f"Iteration {self.i:4d} | funval: {self.funval:,.2f}")

    def __call__(self, *args, **kw):
        return self.run(*args, **kw)

    def _init_run(self) -> None:
        self.xes = FWX()
        self.i = 0
        self.funval = None
        self.blb = -np.inf
        self.lb = -np.inf
        self.gap = np.inf
        self.eta_bef = 0
        self.eta = 0
        self.pmax = 0
        self.partan = 0
        self.breakflag = FWBreakFlag.NOT_SET

    def run(self, callback=None, **kw) -> None:
        """Run the Frank-Wolfe algorithm.
        
        Parameters
        ----------
        callback : callable, optional
            Called after each iteration with the signature:
            
                ``callback(res)``
            
            where ``res`` is a OptimizeResult.
        
        kw : keyword arguments
            Further options, see FWConfig.
        
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult``
            object. TODO-PJ: which attributes.
        
        See Also
        --------
        FWConfig : Options accepted by the solver.
        scipy.optimize.OptimizeResult : Details on the optimization result.
        """
        self._init_run()
        self._c.map_kwargs(**kw)
        self.x = self.x0
        self.funval = self.fun(self.x)
        
        # Loop until some break condition is reached
        while self.breakflag == FWBreakFlag.NOT_SET:
            self.i += 1
            if (self.config.lb is not None) and (self.funval < self.config.lb):
                self.breakflag = FWBreakFlag.COST_INVALID
                break
            
            self.xes.s = self.subproblem_solver(self.fprime(self.x))
            self._check_convergence()
            self._perform_eta_step()
            self._perform_partan_step()
            self.funval = self.fun(self.x)
            
            if self.i == self.config.max_iter:
                self.breakflag = FWBreakFlag.MAX_ITER
            
            if callback is not None:
                r = spopt.OptimizeResult(fun=self.funval,
                                         x=self.x,
                                         nit=self.i,
                                         status=self.breakflag.to_status(),
                                         success=(self.breakflag.to_status() == 0))
                callback(r)

            if self.config.print is True:
                print(self)
        
        return spopt.OptimizeResult(fun=self.funval,
                                    x=self.x,
                                    nit=self.i,
                                    status=self.breakflag.to_status(),
                                    success=(self.breakflag.to_status() == 0))

    def _get_gradient(self) -> np.ndarray:
        return self.fprime(self.x)

    def _check_convergence(self) -> None:
        # Update (best) lower bound
        self.lb = (self.funval + sum((self.xes.s - self.xes.x) * self._get_gradient()))
        self.blb = max(self.blb, self.lb)
        
        if (self.config.lb is not None) and (np.isclose(self.funval, self.config.lb)):
            # If the cost of the solution equals lower bound (if given),
            # optimal solution is found
            self.gap = 0
            self.breakflag = FWBreakFlag.CONVERGED
        else:
            self.gap = (self.funval - self.blb) / self.funval
            if self.gap < self.config.epsilon:
                self.breakflag = FWBreakFlag.CONVERGED
    
    def _perform_eta_step(self) -> None:
        """Perform eta step of fw iteration.

        Optimal solution of line between current flow and flow found by
        shortest path routine for current edge weights.
        """
        # Save step size eta of previous iteration
        self.eta_bef = self.eta
        
        # Find optimal eta step size: intermediate x is best x between
        # currently best x and s
        if self.config.mode == FWMode.STEP_SIZE_DETERIATION:
            self.eta = 2 / (2 + self.i)
        else:
            self.eta = linesearch(self.fun,
                                  self.xes.x,
                                  self.xes.s)

        self.xes.x_eta = (1 - self.eta) * self.xes.x + self.eta * self.xes.s

    def _perform_partan_step(self) -> None:
        """Perform partan step of fw iteration."""
        # Save x of previous iteration
        self.xes.x_bef = copy.deepcopy(self.xes.x)
        
        # Partan step is not performed in first step or if mode is not partan
        # -> x = x_eta
        if self.i == 1 or self.config.mode != FWMode.PARTAN:
            self.xes.x = self.xes.x_eta
            return
        
        # Find pmax flow
        p_val = (1 / (1 - (1 - self.eta_bef) * (1 - self.eta) * (1 - self.partan)))
        self.pmax = max(1, p_val)
        self.xes.x_pmax = (self.xes.x_bef
                           + self.pmax * (self.xes.x_eta - self.xes.x_bef))
        
        # x is best linear combination between x_eta and x_pmax
        self.partan = linesearch(self.fun,
                                 self.xes.x_eta,
                                 self.xes.x_pmax)
        self.xes.x = ((1 - self.partan) * self.xes.x_eta
                      + self.partan * self.xes.x_pmax)

    @property
    def x(self):
        """Get current best x."""
        return self.xes.x
    
    @x.setter
    def x(self, val):
        self.xes.x = val

    @property
    def config(self):
        """Settings of the FW optimizer.
        
        See Also
        --------
        FWConfig
        """
        return self._c
