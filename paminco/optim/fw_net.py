from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import FlowOptimizer
from .fw import FW, FWConfig
from .subproblem import SubproblemMethod, subproblem_solver
from paminco.callback import CallBackFlag
from paminco.net.network import Network
from paminco.net.demand import LinearDemandFunction

# TODO: delete parallel edges for network fw??


class NetworkFWConfig(FWConfig):
    """Options for Network Frank-Wolfe optimizer.
    
    Extends FWConfig with ``subproblem_method``.
    
    Parameters
    ----------
    print : bool, default=False
        Whether to print iteration summary at the end of each iteration.
    epsilon : float, default=1e-3
        Convergence threshold. FW converges if
        ``(self.funval - self.blb) / self.funval < epsilon``.
    mode : FWMode, int, or str, default=FWMode.PARTAN
    max_iter : int, default=2000
        Maximum number of iterations.
    lb : float, default=0
        Lower bound of objective function. If funval equals this bound,
        optimal solution is assumed to be found.
    subproblem_method : SubproblemMethod, int, or str, default=SubproblemMethod.AUTO
        Specifies which method should be used to solve subproblem.
    """

    options = FWConfig.options + ["subproblem_method"]

    def __init__(self, **kwargs):
        self.subproblem_method = SubproblemMethod.AUTO
        super().__init__(**kwargs)

    @property
    def subproblem_method(self) -> SubproblemMethod:
        """Method to solve the subproblem.
        
        See Also
        --------
        paminco.optim.subproblem.SubproblemMethod
        """
        return self._subproblem_method
    
    @subproblem_method.setter
    def subproblem_method(self, value) -> None:
        self._subproblem_method = SubproblemMethod.make(value)
    
    subproblem_method.__doc__ = SubproblemMethod.__doc__

    def get_fw_kwargs(self):
        d = {}
        for p in self.options:
            if p in FWConfig.options:
                d[p] = getattr(self, p)
        return d


class NetworkFW(FlowOptimizer):
    """Frank-Wolfe algorithm to find minimum cost flow.
    
    Holds an instance of :class:`paminco.optim.fw.FW` (attribute ``fw``), which 
    is supplied with proper parameters to return a minimum cost flow.
    
    Parameters
    ----------
    net : Network
        Network to find min cost flows for. A min cost flow ``f`` for
        some ``param`` is found by::
        
            minimize net.cost(f)
            s.t. net.Gamma.dot(f) = net.demand(param).
    
    name : str, optional
        Name of solver. If None (default), an automatic name will be
        generated.
    callback : (list of callable), optional
        Will be called during initialization and run() with flags indicating
        the status of the algorithm.
    use_simple_timer : bool, default=True
        Whether timestamps will be collected during initialization and run.
    copy_network : bool, default=True
        Whether to work on a copy of ``net``.
    subproblem : callable, optional
        If given, set up subproblem solver with ``subproblem(net, **kw_sub)``,
        else an auto subproblem solver is used.
    kw_sub : dict, optional
        Paramaters passed to subproblem solver.
    **kwargs : keyword arguments
        Settings for NetworkFW.
    
    See Also
    --------
    NetworkFWConfig : All Settings for NetworkFW.
    paminco.optim.fw.FW :
        Optimizer used to find minimum cost flow.
    
    Examples
    --------
    Braess Paradox: Additional edges may lead to worse travel times in
    equilibrium.
    
    >>> import paminco
    >>> braess = paminco.net.load_braess()
    >>> fw = NetworkFW(braess)
    >>> fw.run(max_iter=100)
    >>> # Delete zero-cost connection between v1 and v2
    >>> braess.delete_edges(4)
    >>> fw2 = NetworkFW(braess)
    >>> fw2.run(max_iter=100)
    >>> print(fw.cost.round(2), fw2.cost.round(2))
    35.49 35.0
    """

    def __init__(
            self,
            net: Network,
            name=None,
            callback=None,
            use_simple_timer: bool = True,
            copy_network: bool = True,
            subproblem=None,
            kw_sub=None,
            **kwargs
            ) -> None:
        # Init network
        super().__init__(net,
                         name=name,
                         callback=callback,
                         use_simple_timer=use_simple_timer,
                         copy_network=copy_network)
        self._c = NetworkFWConfig(**kwargs)
        
        # Init network subproblem
        if kw_sub is None:
            kw_sub = {}
        if subproblem is not None:
            self.subproblem_solver = subproblem(net, **kw_sub)
        else:
            self.subproblem_solver = subproblem_solver(
                method=self.config.subproblem_method,
                network=self.network,
                **kw_sub
            )
        
        self.callback(CallBackFlag.INIT_END)
        
    def run(
            self,
            param: float = 1,
            warmstart=None,
            callback=None,
            **kw
            ):
        """Find min cost flow for ``param``.
        
        Parameters
        ----------
        param : float, default=1
            Find min cost flow for demand(param).
        warmstart : LinearWarmstart, optional.
            If given, use scaled warmstart flow as initial solution.
            Else, initial solution is found by minimizing subproblem
            for weights of zero-flow.
        callback : callable, optional
            Called after each iteration:
            ``callback(CallBackFlag.ITER_END, self)``
        kw : keyword arguments
            Further options, see FWConfig.
        
        Examples
        --------
        Using a callback::
        
            import paminco
            import numpy as np
            
            network = paminco.net.load_sioux()
            network.integrate_cost()
            fw = paminco.NetworkFW(network)
            
            funvals = []
            def cb(fw, callbackflag):
                if callbackflag == CallBackFlag.ITER_END:
                    funvals.append(fw.cost)
            
            fw.run(print=True, param=1, max_iter=100, callback=cb)
            funvals = np.array(funvals)
        
        """
        self.callback(CallBackFlag.RUN_START, callback)
        
        self.subproblem_solver.param = param
        self.subproblem_solver.reset_cache()
        
        # Find initial solution
        if (warmstart is not None and
                isinstance(self.network.demand, LinearDemandFunction) and
                not np.isclose(warmstart.param, 0)):
            x0 = warmstart(param)
        else:
            f = np.zeros(self.network.m)
            s = self.network.cost.ddx(f)
            x0 = self.subproblem_solver(s)
        
        # Setup Frank-Wolfe
        def costfun(x):
            return self.network.cost(x).sum()
        
        def jac(x):
            return self.network.cost.ddx(x)
        self.fw = FW(costfun, jac, self.subproblem_solver, x0, **self._c.get_fw_kwargs())
        
        # Map callback and run FW
        def cb(optim_res):
            self.callback(CallBackFlag.ITER_END, callback)
        self.fw.run(callback=cb, **kw)
        
        self.callback(CallBackFlag.RUN_END, callback)

    @property
    def breakflag(self):
        return self.fw.breakflag
    
    @property
    def x(self):
        return self.fw.x
    
    @property
    def flow(self):
        return self.x

    @property
    def cost(self):
        return self.fw.funval

    @property
    def lb(self) -> float:
        """Lower bound."""
        return self.fw.lb

    @property
    def blb(self) -> float:
        """Best lower bound."""
        return self.fw.blb

    @property
    def config(self) -> NetworkFWConfig:
        """Settings of NetworkFW optimizer."""
        return self._c

    @property
    def flows(self) -> pd.DataFrame:
        """(Intermediate) flows of the FW routine."""
        
        df = pd.DataFrame(self.fw.xes.__dict__)
        df.columns = ["x", "x_s", "x_eta", "x_pmax", "x_bef"]
        
        # Add source and target indices
        df["s"] = self.network.edges.s
        df["t"] = self.network.edges.t
        
        # Reordering
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        return df[cols]
