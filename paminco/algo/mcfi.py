from __future__ import annotations

import copy

import numpy as np
import scipy.optimize as scopt

from paminco._base import AlphaBetaApproximativeSolver, Config, ParametricSolution
from paminco.callback import CallBackFlag
from paminco.net.demand import AffineDemandFunction
from paminco.net.network import Support, Network
from paminco.utils.misc import callback_to_list
from paminco.utils.bisec import bisec_method
from paminco.utils.typing import is_iterable, IntEnum2
from paminco.optim import LinearWarmstart, NetworkFW


class AdaptiveMethod(IntEnum2):
    """Enum defining the step rule used in the MCFI algorithm."""
    BASIC = 0
    """Basic step size rule.
    
    Assumes that the objective function is non-decreasing."""

    CONSTANT_SUPPORT = 1
    """Extended step size rule.
    
    Assumes that all marginal cost functions are differentiable
    with f'(x) > 0, all marginal cost functions are convex
    for x > 0 and concave for x < 0, and that the support
    does not change between any two breakpoints."""


class MCFIConfig(Config):
    """Settings for the Marginal Cost Flow Interpolator.
    
    Attributes
    ----------
    lambda_min : float
        Minimum demand factor to find mincost flow for.
    lambda_max : float
        Maximum demand factor to find mincost flow for.
    adaptive_method : AdaptiveMethod
        Enum defining he step rule used in the MCFI algorithm.
    alpha : float
        Relative tolerance for alpha-beta-approximation.
    beta : float
        Absolute tolerance for alpha-beta-approximation.
    epsilon : float
        Convergence threshold for optimizer.
    warmstart : bool
        If True and feasible, optimizer is warmstarted.
    print : bool
        if True, an interation summary is printed after each iteration.
        
    See also
    --------
    AdaptiveMethod
    """
    
    paras = [
        "lambda_min",
        "lambda_max",
        "adaptive_method",
        "alpha",
        "beta",
        "epsilon",
        "warmstart",
        "print",
    ]
    
    def __init__(self, **kw):
        self.lambda_min = 0
        self.lambda_max = 1
        self._adaptive_method = AdaptiveMethod.BASIC
        self.alpha = 1 + 1e-2
        self.beta = 1
        self.epsilon = 1e-3
        self.warmstart = True
        self.print = False
        
        self.map_kwargs(**kw)

    def map_kwargs(self, **kw):
        for (k, v) in kw.items():
            if k in self.paras:
                setattr(self, k, v)
            else:
                raise ValueError(f"{k} is not a valid MCFI setting.")

    @property
    def adaptive_method(self):
        return self._adaptive_method
    
    @adaptive_method.setter
    def adaptive_method(self, value) -> None:
        self._adaptive_method = AdaptiveMethod.make(value)
    
    adaptive_method.__doc__ = AdaptiveMethod.__doc__


class MCFI(AlphaBetaApproximativeSolver):
    """Minimum Cost Flow Interpolation.
    
    Computes min costs flows for breakpoints. For parameters
    in-between the breakpoints, a min cost flow is then interpolated.
    
    Parameters
    ----------
    net : Network
        Network to find min cost flows for. A min cost flow ``f`` for
        some ``param`` is found by::
        
            minimize net.cost(f)
            s.t. net.Gamma.dot(f) = net.demand(param).
    
    name : str, optional
        Name of solver.
    callback : callable, or list of callable, optional
        Callbacks called during initialization and run method. Can be
        used for debugging and timing.
    use_simple_timer : bool, default=True
        Whether to time MCFI. If ``True``, timestamps for intializtion
        and every iteration will be saved to attribute ``timer``.
    kw_optim : keyword arguments, optional
        Further arguments passed to min cost flow optimizer.
    kwargs : keyword arguments, optional
        Further parameters for MCFI, see MCFIConfig.
    
    Attributes
    ----------
    config : MCFIConfig
        Settings for MCFI.
    network : Network
        Network associated with MCFI.
    name : str
        Name of solver.
    param_solution : ParametricSolution
        List that stores min cost flows found for some parameters.
        Handles interpolation of flows.
    
    See Also
    --------
    MCFIConfig : All settings for MCFI.
    paminco.optim.fw_net.NetworkFW :
        Optimizer to find min cost flows for a given parameter.
    
    Examples
    --------
    Find user equilibrium (UE) for various specified demand factors:
    
        >>> import paminco
        >>> net = paminco.net.load_sioux()
        >>> net.integrate_cost()  # integrate to calc UE
        >>> mcfi = MCFI(net)
        >>> demand_fac = [0.2, 0.5, 0.6]
        >>> mcfi.run(param=demand_fac)
        >>> mcfi.cost_at(demand_fac).round(0)
        array([ 638723., 1673515., 2062773.])
    """

    def __init__(
            self,
            net: Network,
            name=None,
            callback=None,
            use_simple_timer: bool = True,
            optim=None,
            kw_optim=None,
            **kwargs
            ) -> None:
        super().__init__(net,
                         name=name,
                         callback=callback,
                         use_simple_timer=use_simple_timer)
        # Map kwargs to config object
        self._c = MCFIConfig(**kwargs)
        
        # Initialize optimizer -> find MCF for some param
        if kw_optim is None:
            kw_optim = {}
        
        # Pass epsilon to optimizer (if not specifically given in kw_optim)
        if 'epsilon' not in kw_optim:
            kw_optim['epsilon'] = self.config.epsilon
        
        if optim is None:
            optim = NetworkFW
        self.optim = optim(net, **kw_optim)
        
        # Storing breakpoint solutions
        self._param_solution = ParametricSolution()
        
        # Flag end of iteration to listeners
        self.callback(CallBackFlag.INIT_END)

    def _preprocess(self) -> None:
        pass

    def run(
            self,
            param=None,
            callback=None,
            kw_optim=None,
            **kwargs
            ) -> None:
        """Run MCFI.
        
        Parameters
        ----------
        param: iterable, optional
            Parameters for which to find *minimum cost flow*. If None,
            adaptive method is used to determine parameters.
        callback : callable / list of callables, optional
            Callables are called with signature ``cb(self, CallBackFlag)``
            Full callables consist of those defined in initialization plus
            those passed here::
            
                Start of run       -> cb(self, CallBackFlag.RUN_START)
            
                Start of iteration -> cb(self, CallBackFlag.ITER_START)
            
                End of iteration   -> cb(self, CallBackFlag.ITER_END)
            
                End of run         -> cb(self, CallBackFlag.End_RUN)
        
        kwargs : keyword arguments
            Further options for solver, see :class:`MCFIConfig`.
        
        See Also
        --------
        AdaptiveMethod
        MCFIConfig
        paminco.callback
        """
        if kw_optim is None:
            kw_optim = {}
        self.config.map_kwargs(**kwargs)
        self.optim.config.map_kwargs(**kw_optim)
        
        # disable warmstarting if affine problem
        if isinstance(self.network.demand, AffineDemandFunction):
            self.config.warmstart = False
        
        self.i = 0
        if param is None:
            self._check_adaptive_conditions()
            self._run_adaptive(callback)
        elif is_iterable(param):
            self._run_with_params(param, callback)
        else:
            raise ValueError("'param' must be None or Iterable.")

    def _run_single_fw(self, param: float, warmstart=None):
        if self.config.warmstart is True:
            self.optim.run(param=param, warmstart=warmstart)
            warmstart = LinearWarmstart(self.optim.flow, param)
            return self.optim.flow, warmstart
        else:
            self.optim.run(param=param)
            return self.optim.flow, None

    def _run_with_params(self, params=None, callback=None) -> None:
        # Handle callback
        run_cb = callback_to_list(callback)
        self.callback(CallBackFlag.RUN_START, run_cb)
        self.callback(CallBackFlag.ITER_PRE, run_cb)
        
        warmstart = None
        for p in params:
            self.i += 1
            self.callback(CallBackFlag.ITER_START, run_cb)
            
            # Run instance of FW for param
            flow, warmstart = self._run_single_fw(p, warmstart)
            self._add_param_solution(p, flow.copy())
            
            self.callback(CallBackFlag.ITER_END, run_cb)
            self._print_iteration_summary()
        
        self.callback(CallBackFlag.RUN_END, run_cb)
        
        self.close_run()

    def _check_adaptive_conditions(self):
        """Check if configuration matches conditions for adaptive run"""

        # Check if instance is a single commodity instance
        if (self.network.is_single_commodity is False
                or len(self.network.demand) == 0):
            raise ValueError(
                "Network must have demand function that consists of single "
                f"demand. Network demand is: {type(self.network.demand)} "
                f"consisting of {len(self.network.demand)} commodities."
            )
        
        # Check, if parameters alpha and epsilon are consistent
        if self._c.epsilon + 1>= self._c.alpha:
            msg = f"Invalid parameters alpha = {self._c.alpha} and "
            msg += f"{self._c.epsilon} for MCFI. Algorithm requires "
            msg += "epsilon < alpha - 1."
            raise ValueError(msg)

    def _run_adaptive(self, callback=None) -> None:
        # handle callback
        run_cb = callback_to_list(callback)
        self.callback(CallBackFlag.RUN_START, run_cb)
        
        param = self.config.lambda_min  # first parameter to find MCF for
        warmstart = None
        delta = None
        supp = None
        flow = None
        
        # Find an s-t-pair decomposition of the demand
        self.demand_decomposed = copy.deepcopy(self.network.demand)
        self.demand_decomposed.to_single_pairs()
        self.demand_decomposed.reset_cache()
        
        self.callback(CallBackFlag.ITER_PRE, run_cb)
        
        while True:
            self.i += 1
            self.callback(CallBackFlag.ITER_START, run_cb)
            
            # STEP 1): find next param, not needed in first iteration
            if self.i > 1:
                if self.config.adaptive_method == AdaptiveMethod.BASIC:
                    # Use only support free steps
                    delta = self._support_free_step(param,
                                                    flow,
                                                    last_delta=delta)
                    param += delta
                else:
                    # Try support step
                    tmp_delta = self._support_step(param,
                                                   flow,
                                                   supp,
                                                   last_delta=delta)
                    flow_tmp, warmstart_tmp = self._run_single_fw(param + tmp_delta,
                                                                  warmstart)
                    
                    if self.network.support_of(flow_tmp) == supp:
                        # Solution compliant with the previous support,
                        # i.e., flow_tmp is valid for next param
                        param += tmp_delta
                        warmstart = warmstart_tmp
                    else:
                        # Use support free step to recalc flow for param
                        delta = self._support_free_step(param,
                                                        flow,
                                                        last_delta=delta)
                        param += delta
                        flow_tmp = None
            
            # STEP 2): find min cost flow for param
            if (self.config.adaptive_method == AdaptiveMethod.CONSTANT_SUPPORT
                    and self.i > 1 and flow_tmp is not None):
                # Reuse fw_tmp from AdaptiveMethod.EXTENDED
                flow = flow_tmp
            else:
                # Run instance of FW for param
                flow, warmstart = self._run_single_fw(param, warmstart)
                
                # Calc support for extended method
                if self.config.adaptive_method == AdaptiveMethod.CONSTANT_SUPPORT:
                    supp = self.network.support_of(flow)
            
            # Store solution
            self._add_param_solution(param, flow.copy())
            
            self.callback(CallBackFlag.ITER_END, run_cb)
            self._print_iteration_summary()
            
            # Stop if param exceeds lambda_max
            if param > self.config.lambda_max:
                break
            
        self.callback(CallBackFlag.RUN_END, run_cb)
        self.close_run()

    def _support_free_step(
            self,
            param: float,
            flow: np.ndarray,
            last_delta=None,
            use_bisec: bool = True
            ) -> float:
        # reference for convenience
        c = self.config
        
        # decompose demand -> find source/target/rate triples
        s, t, r = self.demand_decomposed.b.get_source_sink_rate()
        unique_s = self.demand_decomposed.unique_sources
        source_map = dict(zip(unique_s, np.arange(len(unique_s))))
        R = r.sum()
        
        # compute rhs
        cost = self._net.cost(flow).sum()
        rhs = (c.alpha - 1 - c.epsilon) / (1 + c.epsilon) * cost
        rhs += c.beta / (1 + c.epsilon)
        
        def error_fct(delta: float):
            if isinstance(delta, np.ndarray):
                delta = delta[0]
            flow_estimate = np.ones_like(flow) * R * (param + delta)
            weight = self.network.cost.ddx(flow_estimate)
            D, _ = self.network.shortest_path(weight,
                                              s=unique_s,
                                              backward_edges=True,
                                              backward_positive=True)
            path_cost = (np.array([r[i] * D[source_map[s[i]], t[i]]
                                   for i in range(len(s))])
                         .sum())
            return delta * path_cost - rhs
        
        # solve the (in)equality
        # use last step size as initial guess for the solution
        if last_delta is None:
            x0 = 1e-2
        else:
            x0 = 2 * last_delta
        
        # use custom made bisection method
        if use_bisec:
            res = bisec_method(error_fct,
                               tol=.01,
                               lo=0,
                               up=x0,
                               flex_up=True)
            return res
        
        # use scipy fsolve
        res = scopt.fsolve(error_fct, x0, full_output=True)
        if res[2] != 1:
            raise Exception("Function solve did not converge")
        return res[0][0]

    def _support_step(
            self,
            param: float,
            flow: np.ndarray,
            sup: Support,
            last_delta=None,
            use_bisec: bool = True,
            ) -> float:
        # reference for convenience
        c = self.config
        net = self._net
        
        b = net.demand.ddx(param).toarray()
        
        # compute rhs
        cost = net.cost(flow).sum()
        rhs = 8 * (c.alpha - 1 - c.epsilon) / (1 + c.epsilon) * cost
        rhs += 8 * c.beta / (1 + c.epsilon)
        
        # Define error function = lhs - rhs of the inequality
        def error_fct(delta):
            if type(delta) == np.ndarray:
                delta = delta[0]
            # In case of delta == 0, the lhs is zero -> return only rhs
            if delta == 0:
                return -rhs
            # Compute norm of b wrt. inverse of Laplacian matrix
            R = net.demand.max_inflow(min_param=param,
                                      max_param=param + delta)
            # print(R, f"min = {param}, max = {param + delta}")
            flow_estimate = np.ones_like(flow) * R
            # print("fest", flow_estimate)
            ddx2 = net.cost(flow_estimate, d=2)
            # print("ddx2", ddx2)
            weight = np.zeros_like(flow)
            weight[sup.active] = 1 / ddx2[sup.active]
            norm2 = b.T.dot(net.Lstar(weight).dot(b)).flatten()[0]
            return (delta ** 2) * norm2 - rhs
        
        # solve the (in)equality
        # use last step size as initial guess for the solution
        if last_delta is None:
            x0 = 1e-2
        else:
            x0 = 2 * last_delta
        
        # use custom made bisection method
        if use_bisec:
            res = bisec_method(error_fct,
                               tol=.01,
                               lo=0,
                               up=x0,
                               flex_up=True)
            return res
        
        # use scipy fsolve
        res = scopt.fsolve(error_fct, x0, full_output=True)
        if res[2] != 1:
            raise Exception("Function solve did not converge")
        return res[0][0]

    def _print_iteration_summary(self):
        if self.config.print is True:
            out = f"Iteration {self.i:4d} | λ = {self.param_solution[-1].param:.5f}"
            print(out)

    @property
    def config(self) -> MCFIConfig:
        """Settings of MCFI.
        
        See Also
        --------
        MCFIConfig
        """
        return self._c
