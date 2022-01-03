from __future__ import annotations
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as sps

from paminco._base import ParametricSolver, Config, ParametricSolution
from paminco.callback import CallBackFlag
from paminco.linalg import InverseMethod, SingularLaplaceError
from paminco.net.network import Network
from paminco.net.demand import LinearDemandFunction, AffineDemandFunction
from paminco.net.cost import PiecewiseQuadraticCoefficients, PiecewiseQuadraticCost
from paminco.utils import callback_to_list
from paminco.utils.math import np_divide_a_by_b, find_min_col_lex
from paminco.utils.typing import IntEnum2


class PivotStepMode(IntEnum2):
    """Enum defining the pivot step mode.

    The pivot step mode determines how to select a single minimum edge
    from a set of boundary edges.

    See Also
    --------
    paminco.utils.math.find_min_col_lex

    References
    ----------
    .. [1] https://www3.math.tu-berlin.de/Vorlesungen/WS07/ADM2/lectures/2007-11-14/handout141107.pdf
    """

    LEX = 1
    """Find min edge according to lexicographic rule."""

    RANDOM = 2
    """Choose min edge randomly."""

    FIRST = 3
    """Select first edge in boundary edges."""


class EFAEdges:
    r"""
    Object that stores edge values during EFA run.
    
    Parameters
    ----------
    source_target : ndarray
        Ndarray of shape (m, 2) containing indices of sources and targets.
    
    Attributes
    ----------
    s
    t
    region : ndarray (m, )
        Region of edges, ndarray of int.
    flow : ndarray (m, )
        Induced edge flow, ndarray of float.
    gamma_pi_t : ndarray (m, )
        Gamma times potential offset::
        
            Gamma @ pi_t
    
    gamma_dpi : ndarray (m, )
        Gamma times potential direction::
        
            Gamma @ dpi_t
        
    round_gamma_dpi : ndarray (m, )
        Rounding factor for ``gamma_dpi``.
    ll : ndarray (m, )
        .. math::
        
            \underline{\lambda}_{\mathbf{t}}(e) :=
            \frac
                {
                    \sigma_{e, t_e}-\mathbf{\gamma}_{e}^{\top}
                    \mathbf{\pi}_{\mathbf{t}}
                }
                {
                    \mathbf{\gamma}_{e}^{\top}
                    \Delta \mathbf{\pi}_{\mathbf{t}}
                }
    lu : ndarray (m, )
        .. math::
        
            \overline{\lambda}_{\mathbf{t}}(e) :=
            \frac
                {
                    \sigma_{e, t_e + 1}-\mathbf{\gamma}_{e}^{\top}
                    \mathbf{\pi}_{\mathbf{t}}
                }
                {
                    \mathbf{\gamma}_{e}^{\top}
                    \Delta \mathbf{\pi}_{\mathbf{t}}
                }
    
    See Also
    --------
    NodePotentials : Definition of pi_t and dpi_t.
    paminco.net.Network.Gamma : Definition of Gamma.
    """
    
    def __init__(self, source_target):
        self.source_target = source_target
        self.region = None
        self.flow = None
        self.ll = None
        self.lu = None
        self.gamma_pi_t = None
        self.gamma_dpi = None
        self.round_gamma_dpi = None
        
    def __len__(self) -> bool:
        return len(self.source_target)
    
    def to_df(self) -> pd.DataFrame:
        cols = ["s", "t", "region", "flow", "gamma_dpi", "gamma_pi_t",
                "round_gamma_dpi", "ll", "lu"]
        data = {k: getattr(self, k) for k in cols}
        df = pd.DataFrame(data)
        return df
    
    @property
    def s(self) -> np.ndarray:
        """ndarray (m, ) of int: sources."""
        return self.source_target[:, 0]
    
    @property
    def t(self) -> np.ndarray:
        """ndarray (m, ) of int: targets."""
        return self.source_target[:, 1]


class NodePotentials:
    r"""
    Class that stores values that relate to the potential in some region.
    
    The potential :math:`\mathbf{\pi}` for some region :math:`R_{\mathbf{t}}`
    is given by:
    
    .. math::
        \mathbf{\pi} = \mathbf{\pi}_{\mathbf{t}}
        + \lambda_{\mathbf{t}}^{\text{max}} \Delta
        \mathbf{\pi}_{\mathbf{t}} = \mathbf{L}_{\mathbf{t}}^{\ast}
        (\mathbf{\Gamma}\mathbf{d}_{\mathbf{t}})
        + \lambda_{\mathbf{t}}^{\text{max}}
        (\mathbf{L}_{\mathbf{t}}^{\ast} \mathbf{b}),
        
    where :math:`\mathbf{L}_{\mathbf{t}}^{\ast}` is the generalized inverse
    of the weighted Laplacian, :math:`\Gamma` the incidence matrix and
    :math:`\mathbf{d} _{\mathbf{t}} = \frac{\mathbf{b} _{\mathbf{t}}}{\mathbf{a} _{\mathbf{t}}}`.
    Here, :math:`\mathbf{b} _{\mathbf{t}}` is the vector of offsets and
    :math:`\mathbf{a} _{\mathbf{t}}` the vector of slopes of the linear marginal
    cost functions :math:`\tilde{f}_e` for the region :math:`R_{\mathbf{t}}`.
    
    A node potential :math:`\mathbf{\pi}` induces a flow
    :math:`\mathbf{f}^{-1}(\mathbf{\pi})` by:
    
    .. math::
        \mathbf{f}^{-1}(\mathbf{\pi}) = \mathbf{C}_{\mathbf{t}}
        \Gamma^T\mathbf{\pi} - \mathbf{d}_{\mathbf{t}}. 
        
    Attributes
    ----------
    node_idx : ndarray (n, )
        Node indices, ndarray of int.
    node_lbl : ndarray (n, )
        Node labels, ndarray of str.
    pi : ndarray (n, )
        Node potential :math:`\mathbf{\pi}`.
    pi_t : ndarray (n, )
        Potential offset of the current region
        :math:`\mathbf{\pi}_{\mathbf{t}}`.
    dpi_t : ndarray (n, )
        Potential direction of the current region
        :math:`\Delta \mathbf{\pi}_{\mathbf{t}}`.
    d_tilde : ndarray (n, )
        :math:`\tilde{\mathbf{d}}_{\mathbf{t}} = \mathbf{\Gamma}\mathbf{d}_{\mathbf{t}}`,
        
    See Also
    --------
    paminco.net.cost.PiecewiseQuadraticCoefficients
    """
    
    def __init__(self, net_nodes) -> None:
        self.node_idx = net_nodes.indices
        self.node_lbl = net_nodes.labels
        self.pi = None
        self.dtilde = None
        self.pi_t = None
        self.dpi_t = None
    
    def __len__(self) -> int:
        return len(self.node_idx)
    
    def to_df(self) -> pd.DataFrame:
        cols = ["node_idx", "node_lbl", "pi", "pi_t", "dpi_t", "dtilde"]
        data = {k: getattr(self, k) for k in cols}
        df = pd.DataFrame(data)
        return df


class EFABreakFlag(IntEnum2):
    """Class that defines breakflags for the EFA algorithm."""

    NOT_SET = 0
    """Undefined breakflag."""

    LAMBDA_LARGER_MAX = 1
    """Lambda exceeds maximal value set for lambda."""

    LAMBDA_INF = 2
    """Lambda equals infinity."""

    MAX_ITER = 80
    """Algorithm has reached specified maximum number of iterations."""

    NO_BOUNDS = 90
    """No bounds have been identifed."""

    NO_BOUNDARY_EDGES = 91
    """No boundary edge to select min edge from."""

    def _execute_pivot(self) -> bool:
        # Returns whether to do a pivot step for the current breakflag
        if self == EFABreakFlag.LAMBDA_INF:
            return False
        if self == EFABreakFlag.NO_BOUNDS:
            return False
        if self == EFABreakFlag.NO_BOUNDARY_EDGES:
            return False
        return True


class EFAConfig(Config):
    """Settings for running the electrical flow algorithm (EFA).

    Parameters
    ----------
    lambda_max : float, default=1
        Maxmimum parameter (lambda) to find *minimum cost flow* for.
    inverse_method : str, int or InverseMethod, default=InverseMethod.CHOLESKY
        Method to find the inverse of the laplacian.
    pivot_mode : int, str or PivotStepMode, default=PivotStepMode.LEX
        Method to choose min edge if more than one boundary edge is
        identified.
    max_iter : int, default=99999
        Maximum number of iterations for solver.
    print : bool, default=False
        Whether to print a summary of each iteration.
    recomp_interval : int, default=1
        If == 1, recompute inverse laplacian in every iteration. 
        If < 1, never recompute the laplacian. Else
        update and recompute every `recomp_interval`. The higher the
        value, the less accurate the inverse laplacian.
    round_lambda : int, default=3
        Round lambda value for edges to ``round_lambda`` after decimal.
        Used to determine boundary edges.
    rounding_margins_base : int, default=-16
        Set gamma_dpi with low exponent (in `IEEE754 <https://en.wikipedia.org/wiki/IEEE_754>`_) to zero.
    rounding_margins_fac : int, default=-5
        Set gamma_dpi with low exponent (in `IEEE754 <https://en.wikipedia.org/wiki/IEEE_754>`_) to zero.
    """

    all_options = [
        "inverse_method",
        "lambda_max",
        "pivot_mode",
        "max_iter",
        "print",
        "recomp_interval",
        "round_lambda",
        "rounding_margins_base",
        "rounding_margins_fac",
    ]
    """All available settings for EFA."""

    kw_init = [
        "lambda_max",
    ]
    """Settings for EFA that can be passed only during initialization."""

    def __init__(self, **kwargs):
        super().__init__()
        
        self._inverse_method = InverseMethod.CHOLESKY
        self.lambda_max = 1
        self._pivot_mode = PivotStepMode.LEX
        self.max_iter = 99999
        self.print = False
        self.recomp_interval = 1
        self.round_lambda = 3
        self.rounding_margins_base = -16
        self.rounding_margins_fac = -5
        
        self.map_kwargs(run=False, **kwargs)

    def map_kwargs(self, run: bool = False, **kwargs) -> None:
        if run is True:
            for (k, v) in kwargs.items():
                if k in self.kw_init:
                    raise ValueError(
                        f"'{k}' is not a valid parameter to be set in run() "
                        "method."
                    )
                else:
                    setattr(self, k, v)
        else:
            for (k, v) in kwargs.items():
                if k.startswith("_"):
                    k = k[1:]
                    
                if hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise ValueError(f"'{k}' is not a valid parameter.")

    @property
    def inverse_method(self) -> InverseMethod:
        """Method to calculate inverse Laplacian.
        
        See Also
        --------
        paminco.linalg.InverseMethod
        """
        return self._inverse_method
    
    @inverse_method.setter
    def inverse_method(self, value) -> None:
        self._inverse_method = InverseMethod.make(value)
    
    @property
    def pivot_mode(self) -> PivotStepMode:
        """Method to perform pivot step.
        
        See Also
        --------
        PivotStepMode
        """
        return self._pivot_mode
    
    @pivot_mode.setter
    def pivot_mode(self, value) -> None:
        self._pivot_mode = PivotStepMode.make(value)


class EFA(ParametricSolver):
    r"""Electrical flow algorithm.
    
    Computes a piecewise linear optimal potential function that induces a 
    piecewise linear minimum cost flow function. The underlying network must
    have piecewiese quadratic edge costs. 
    
    Parameters
    ----------
    net : Network
        Network to find parametric min cost flows for.
    name : str, optional
        Name of solver.
    callback : callable, or list of callable, optional
        Callbacks called during initialization and run method. Can be
        used for debugging and timing.
    use_simple_timer : bool, default=True
        Whether to time EFA. If ``True``, timestamps for intializtion
        and every iteration will be saved to attribute ``timer``.
    phase1_of : EFA, default=None
        The main EFA object of a 2-phase EFA run. If set to something
        other than ``None``, it is assumed that this EFA object is
        used to compute the initial solution for the EFA run of the
        ``phase1_of`` object. If set, the ``is_phase1`` property of
        this object will be set to ``True``.
    kwargs : keyword arguments, optional
        For further options of EFA, see EFAConfig.
    
    See Also
    --------
    EFAConfig : Settings for EFA.
    
    References
    ----------
    .. [1] Klimm, Max, and P. Warode. "Parametric Computation of Minimum Cost
           Flows with Piecewise Quadratic Costs." Mathematics of Operations
           Research (2021). Available 10/25/2021 at https://www3.math.tu-berlin.de/disco/research/publications/pdf/KlimmWarode2021.pdf   
    """

    def __init__(
            self,
            network: Network,
            name=None,
            callback=None,
            use_simple_timer: bool = True,
            preprocess_network: bool = True,
            phase1_of: EFA = None,
            copy_network: bool = True,
            **kwargs
            ) -> None:

        # Store the original EFA object if this is only a phase 1 run
        self.phase1_of = phase1_of

        # init network and configs
        super().__init__(network,
                         name=name,
                         callback=callback,
                         copy_network=copy_network,
                         use_simple_timer=use_simple_timer)
        self._c = EFAConfig(**kwargs)
        
        # prepare network for run
        if preprocess_network is True:
            self._preprocess_net()
        
        self._check_consistency()
        
        # setup nodes
        self._np = NodePotentials(self._net.nodes)
        
        # setup edges(st, rounding margins and inital region)
        self._e = EFAEdges(self._net.shared.edges.indices)

        # setup properites
        self.min_edge = None
        
        # Storing breakpoint solutions
        self._param_solution = ParametricSolution()
        
        # Flag end of iteration if someone "listenes"
        self.callback(CallBackFlag.INIT_END)

    def __eq__(self, other) -> bool:
        if isinstance(other, EFA) is False:
            return False
        for att in ["_np", "_e"]:
            if getattr(self, att) != getattr(other, att):
                return False
        return True

    def _preprocess_net(self) -> None:
        # make network compatible with solver
        self._net.clean(remove_zones=True,
                        remove_parallel_edges=True,
                        remove_zero_cost_edges=True,
                        remove_isolated_nodes=True,
                        remove_unreachable_nodes=True,
                        remove_commodities=True)

    def _set_rounding_margins(self) -> None:
        margins = self._net._c.rounding_margins.astype(int)
        margins = np.minimum(margins + self._c.rounding_margins_fac,
                             self._c.rounding_margins_base,
                             out=margins)
        self._e.round_gamma_dpi = margins.astype(int)

    def _check_consistency(self) -> None:
        if (self.network.is_single_commodity is False
                or len(self.network.demand) == 0):
            raise ValueError(
                "Network must have demand function that consists of single "
                f"demand. Network demand is: {type(self.network.demand)} "
                f"consisting of {len(self.network.demand)} commodities."
            )
        
        if not isinstance(self.network.cost, PiecewiseQuadraticCost):
            raise ValueError(
                "Network cost must be piecewise quadatric, is: "
                f"{type(self.network.cost)}."
            )

    def run(self, callback=None, **kwargs) -> None:
        """Run the EFA algorithm for given network.
        
        You may pass further parameters that will be mapped to the
        configs of the solver, see EFAConfig.
        
        Parameters
        ----------
        callback : callable / list of callables, optional
            Callables are called with signature:
            
            ``cb(self, CallBackFlag)``
            
            Full callables consist of those defined in initialization
            plus those passed here.
            
                Start of run       -> ``cb(self, CallBackFlag.RUN_START)``
                
                Start of iteration -> ``cb(self, CallBackFlag.ITER_START)``
                
                End of iteration   -> ``cb(self, CallBackFlag.ITER_END)``
                
                End of run         -> ``cb(self, CallBackFlag.End_RUN)``
        
        kwargs : keyword arguments
            Further options for solver, see EFAConfig.
        
        See Also
        --------
        EFAConfig
        paminco.callback
        """
        # map kwargs of run to config object
        self.config.map_kwargs(run=True, **kwargs)
        
        run_cb = callback_to_list(callback)
        # Store run callbacks (for possible phase-1-callbacks)
        self._run_cb = run_cb
        self.callback(CallBackFlag.RUN_START, run_cb)
        
        # Set initial values for run
        self.i = 1
        self.lambda_min = 0
        self.lambda_max = None
        self.breakflag = EFABreakFlag.NOT_SET
        self._set_rounding_margins()
        self._initial_region()
        self._calculate_inv(force_recomputation=True)
        
        self.callback(CallBackFlag.ITER_PRE, run_cb)
        
        # loop through algorithm until some break condition is met
        while self.breakflag == EFABreakFlag.NOT_SET:
            self.callback(CallBackFlag.ITER_START, run_cb)
            
            # Set to None to avoid misuse
            self.lambda_max = None
            
            # Compute potentials, flows and save solution
            self._compute_potentials()
            self._compute_flows()
            self._add_param_solution(self.lambda_min,
                                     self._e.flow.copy(),
                                     potential=self._np.pi)
            self._pivot_step()
            
            # Iteration cleanup
            self.callback(CallBackFlag.ITER_END, run_cb)
            if self.i == (self._c.max_iter):
                self.breakflag = EFABreakFlag.MAX_ITER
            self.i += 1
            
        # Find flow for lambda_max if pivot step would follow
        if self.breakflag._execute_pivot():
            self.lambda_min = self.lambda_max
            self.lambda_max = np.inf
            self._compute_potentials()
            self._compute_flows()
            self._add_param_solution(self.lambda_min,
                                     self._e.flow.copy(),
                                     potential=self._np.pi)
        
        self.callback(CallBackFlag.RUN_END, run_cb)
        
        # Allow flow interpolation with slope of flow if last
        # breakpoint was infinity
        dflow = None
        dpi_ = None
        if self.breakflag == EFABreakFlag.LAMBDA_INF:
            self.i -= 1
            self._print_iteration_summary()
            dflow = (1 / (2 * self._ec.a) * self._net.times_gamma(self._np.dpi_t))
            dpi_ = self._np.dpi_t
        self.close_run(dflow=dflow, dpi=dpi_)

    def _pivot_step(self) -> None:
        self._compute_boundary()
        if self.breakflag._execute_pivot():
            self.lambda_min = self.lambda_max
            self._select_boundary()
            self._update_region()
            self._print_iteration_summary()
            self._calculate_inv()

    def _initial_region(self) -> None:
        # Determine inital region for linear or affine demands
        if isinstance(self.network.demand, LinearDemandFunction):
            # If the demand function is linear, find the initial region directly
            r0 = self._initial_region_linear()
        elif isinstance(self.network.demand, AffineDemandFunction):
            # For the affine demand case, use Phase I run to determine initial region
            r0 = self._initial_region_affine()
        else:
            raise ValueError(
                "Demand function must be linear or affine, is: "
                f"{type(self.network.demand)}."
            )
        
        # Get edge cost coefficients for initial region
        self._b0 = self.network.demand(0).toarray().ravel()
        self._region_zero = r0.copy()
        self._e.region = r0
        self._update_cost_coeffs()

    def _initial_region_linear(self):
        # Find a region that is compliant with the flow
        init_region = self.network.cost.region_of(np.zeros(self.network.m))
        
        # If the piecewise quadratic cost have no jumps (== infinite slopes a)
        # then this region is compliant with the potentials as well
        # and can be returned immediately
        if max(abs(self.network.cost.coefficients.a)) < np.inf:
            return init_region
        
        # Compute potential for the zero flow
        cost_of_zero_flow = self.network.cost.ddx(np.zeros(self.network.m))
        
        # Compute shortest path potential starting from a random source
        first_source = self.network.demand.b.source_id[0]
        pot, _ = self.network.shortest_path(weight=cost_of_zero_flow,
                                            s=first_source,
                                            backward_edges=True)
        pot_offset = self.network.times_gamma(pot) - cost_of_zero_flow

        self._np.pi = pot
        
        # Go to previous function part if potential difference does not match cost
        init_region[~np.isclose(0, pot_offset)] -= 1
        
        return init_region
    
    def _initial_region_affine(self):
        network = deepcopy(self.network)
        network._d = LinearDemandFunction(network.demand.b0, shared=network.shared)

        self.efa_phase1 = EFA(
            network,
            preprocess_network=False, 
            lambda_max=1,
            phase1_of=self)
        
        if self._c.print is True:
            print("=" * 11 + " START OF PHASE 1 " + "=" * 11)
        
        self.efa_phase1.run(
            print=self._c.print, 
            callback=self._map_phase1_callback)
        
        if self._c.print is True:
            print("=" * 12 + " END OF PHASE 1 " + "=" * 12)
        
        r0 = deepcopy(self.efa_phase1.region)
        if self.efa_phase1.breakflag._execute_pivot():
            # If phase 1 did end with a normal pivoting step,
            # revert to the previous region that contains the
            # flow for lambda = 1
            r0[self.efa_phase1.min_edge] -= self.efa_phase1.region_activate
        
        return r0

    def _map_phase1_callback(self, instance, flag):
        """Pass callbacks from Phase1 EFA to callbacks"""
        if not hasattr(self, '_phase1_cb'):
            self._phase1_cb = [cb for cb in self.callbacks]
            if hasattr(self, '_run_cb'):
                self._phase1_cb += self._run_cb
        for cb in self._phase1_cb:
            cb(instance, flag)

    def _update_region(self) -> None:
        if self.min_edge is not None:
            # update region values
            self._e.region[self.min_edge] += int(self.region_activate)
        else:
            raise RuntimeError("no min edge found.")

        self._update_cost_coeffs()

    def _update_cost_coeffs(self) -> None:
        # Update cost coefficients for edges by current region
        self._ec = self._net.cost.get_coefficients(at=self._e.region,
                                                   is_region=True)
        self._np.d_tilde = self._net.gamma_times(self._ec.d)

    def _calculate_inv(
            self,
            force_recomputation: bool = False,
            ) -> None:
        try:
            recompute = force_recomputation
            # Recompute, if recomputation interval is reached
            if self._c.recomp_interval > 0:
                if self.i % self._c.recomp_interval == 0:
                    recompute = True
            if recompute:
                # recompute inverse
                self.Lstar = self._net.Lstar(region=self.region,
                                             method=self._c.inverse_method)
            else:
                # Update inverse
                # calculate diff in c as if region has not been updated
                edge_region = self.region[self.min_edge] - self.region_activate
                dc = self._net.cost.delta_c(region=edge_region,
                                            step=self.region_activate,
                                            edge=self.min_edge)
                self.Lstar = self._net.Lstar_update(self.Lstar,
                                                    self.min_edge,
                                                    dc)
        except SingularLaplaceError as sle:
            self._fix_region(sle)
            self._calculate_inv(force_recomputation=True)

    def _fix_region(self, laplace_error) -> None:
        n_cc, cc = laplace_error.n_cc, laplace_error.cc
        if n_cc == 1:
            raise RuntimeError(
                "Trying to enter ambiguous region while "
                "the active graph is connected."
            )
        
        # Get demand direction
        demand = self._net._d.ddx(self.lambda_min).toarray().ravel()
        
        # Find nodes for which to increase potential
        if sum(demand[cc == 0]) > 0:
            nodes_pot_inc = (cc == 0)
        else:
            nodes_pot_inc = (cc != 0)
        
        # If no potential pi_t is given (because we are in the first region)
        # try to find some meaningful potential satisfying the demands
        if self._np.pi_t is None:
            # solve the (underdefined) system L = d_tile + b_0
            pi, _, _, _ = np.linalg.lstsq(
                self.network.L(region=self.region, return_as='array'),
                self._np.d_tilde + self._b0,
                rcond=None)
        else:
            # For lambda = 0 the last term is not needed
            # (also avoids problems with unset dpi_t if multiple initial
            #  regions are ambiguous)
            if self.lambda_min == 0:
                pi = self._np.pi_t.ravel()
            else:
                pi = (self._np.pi_t + self._np.dpi_t * (self.lambda_min)).ravel()
        dpi = np.zeros(self._net.n)
        dpi[nodes_pot_inc] = 1
        gamma_pi = self._net.times_gamma(pi)
        gamma_dpi = self._net.times_gamma(dpi)
        
        # Calculate lower and upper lambda
        ll = np_divide_a_by_b(self._ec.sig_l - gamma_pi,
                              gamma_dpi)
        lu = np_divide_a_by_b(self._ec.sig_u - gamma_pi,
                              gamma_dpi)
        
        # Round bounds for lex mode
        ll = ll.round(self._c.round_lambda)
        lu = lu.round(self._c.round_lambda)
        
        bounds = np.union1d(ll[gamma_dpi < 0],
                            lu[gamma_dpi > 0])
        mu = min(bounds)
        lower_bounds = np.logical_and(gamma_dpi < 0,
                                      ll == mu)
        upper_bounds = np.logical_and(gamma_dpi > 0,
                                      lu == mu)
        bound_edges = np.where(np.logical_or(lower_bounds, upper_bounds))[0]
        min_edge = np.random.choice(bound_edges)
        self.min_edge = min_edge
        
        gamma_dpi = self._net.times_gamma(dpi, edge=min_edge)
        region_activate = int(np.sign(gamma_dpi))
        self.region_activate = region_activate
        self._e.region[min_edge] += int(region_activate)
        self._update_cost_coeffs()
        
        if self._c.print is True:
            out = f"Iteration {self.i:4d}a| * AMBIGUOUS REGION * | "
            out += {1: "\u2191 ", -1: "\u2193 "}[region_activate]
            out += "e* = " + str(min_edge)
            if len(bound_edges) > 1:
                out += " ∈ E* = " + str(list(bound_edges))
            print(out)
        
        self._np.pi = pi
        # Ambiguous region has no well-defined offset potential,
        # so we just use pi
        self._np.pi_t = pi

    def _compute_potentials(self) -> None:
        self._np.pi_t = self.Lstar.dot(self._np.d_tilde + self._b0)
        b = self._net._d.ddx(self.lambda_min)
        self._np.dpi_t = self.Lstar.dot(b).ravel()
        self._np.pi = (self._np.pi_t + self._np.dpi_t * (self.lambda_min)).ravel()
        self._e.gamma_pi_t = self._net.times_gamma(self._np.pi_t)
        self._e.gamma_dpi = self._net.times_gamma(self._np.dpi_t)

    def _compute_flows(self) -> None:
        self._e.flow = (1 / (2 * self._ec.a) * self._net.times_gamma(self._np.pi)
                        - self._ec.d)

    def _compute_boundary(self) -> None:
        # Get lower and upper lambda bounds and set boundary edges
        self._set_lambda_lower_and_upper()
        self.bounds = np.union1d(self._e.ll[self._e.gamma_dpi < 0],
                                 self._e.lu[self._e.gamma_dpi > 0])
        
        # check if boundary edges are found
        if len(self.bounds) == 0:
            self.breakflag = EFABreakFlag.NO_BOUNDS
            return
        
        # Get new lambda max as min of bounds
        self.lambda_max = min(self.bounds)
        
        # set boundary edges
        self._get_all_boundary_edges()
        
        # check for various break conditions
        if len(self.boundary_edges) == 0:
            self.breakflag = EFABreakFlag.NO_BOUNDARY_EDGES
        elif self.lambda_max == np.inf:
            self.breakflag = EFABreakFlag.LAMBDA_INF
        elif self.lambda_max >= self._c.lambda_max:
            self.breakflag = EFABreakFlag.LAMBDA_LARGER_MAX

    def _set_lambda_lower_and_upper(self) -> None:
        # set gamma_dpi with low exponennt (in IEEE754) to zero
        exp_ = np.frexp(self._e.gamma_dpi)[1]
        self._e.gamma_dpi = np.where(exp_ <= self._e.round_gamma_dpi,
                                     0,
                                     self._e.gamma_dpi)
        
        # get lower and upper bounds for lambda
        self._e.ll = np_divide_a_by_b(self._ec.sig_l - self._e.gamma_pi_t,
                                      self._e.gamma_dpi)
        self._e.lu = np_divide_a_by_b(self._ec.sig_u - self._e.gamma_pi_t,
                                      self._e.gamma_dpi)
        
        # round bounds for lex mode
        self._e.ll = self._e.ll.round(self._c.round_lambda)
        self._e.lu = self._e.lu.round(self._c.round_lambda)

    def _get_all_boundary_edges(self) -> None:
        """Select all edges that fulfill boundary conditions.
        
        (a) gamma_dpi > 0 && (lambda_lower == lambda_max) or
        (b) gamma_dpi < 0 && (lambda_upper == lambda_max)
        """
        lower_bounds = np.logical_and(self._e.gamma_dpi < 0,
                                      self._e.ll == self.lambda_max)
        upper_bounds = np.logical_and(self._e.gamma_dpi > 0,
                                      self._e.lu == self.lambda_max)
        bounds = np.logical_or(lower_bounds, upper_bounds)
        self.boundary_edges = np.where(bounds)[0]

    def _select_boundary(self) -> None:
        # TODO: is das hier sinnvoll abzubrechen wenn some break conditions gefunden wurden? Wenn ja welche?
        # -> logik der validen szenarios auslagern an breakflag
        if self.breakflag not in [EFABreakFlag.NOT_SET,
                                  EFABreakFlag.LAMBDA_LARGER_MAX]:
            return
        elif (len(self.boundary_edges) == 1 or
                self._c.pivot_mode == PivotStepMode.FIRST):
            # choose only existent or first
            self.min_edge = self.boundary_edges[0]
        elif self._c.pivot_mode == PivotStepMode.RANDOM:
            # choose edge randomly
            self.min_edge = np.random.choice(self.boundary_edges)
        else:
            # lexicographic mode
            self.M = np.zeros(shape=(self._net.n, len(self.boundary_edges)))
            
            self.Lstar_cache = self.Lstar.toarray()
            
            # loop through boundary edges, compute vector and update M
            for j, edge in enumerate(self.boundary_edges):
                # self._lex_rule_compute_m_vec(edge, j)
                self._lex_rule_compute_m_vec2(edge, j)
                
            # find smallest column (lexicographically) in M matrix -> min edge
            min_m = find_min_col_lex(self.M)
            self.min_edge = self.boundary_edges[min_m]
        
        # check whether edge region was incremented or decremented
        gamma_dpi = self._net.times_gamma(self._np.dpi_t, edge=self.min_edge)
        self.region_activate = int(np.sign(gamma_dpi))

    def _lex_rule_compute_m_vec(
            self,
            edge: int,
            j: int
            ) -> None:
        # compute gamma_e * delta_pi
        gdpi = self._net.times_gamma(self._np.dpi_t, edge=edge)
        
        # compute gamma_e * L_star * L0
        # First, compute L_star * gamma_e efficiently
        w_i = self._net.edges.t[edge]
        v_i = self._net.edges.s[edge]
        
        # TODO-opt -> optimization potential
        lstar_w = self.Lstar.dot(np.eye(self._net.n, 1, -w_i)).flatten()
        lstar_v = self.Lstar.dot(np.eye(self._net.n, 1, -v_i)).flatten()
        lstar_gam = lstar_w - lstar_v
        
        # Compute L0 * (L_star * gamma_e) and add to M matrix
        glstarl = self.L0.dot(lstar_gam)
        self.M[:, j] = (-1 / gdpi) * glstarl

    def _lex_rule_compute_m_vec2(
            self,
            edge: int,
            j: int
            ) -> None:
        # compute gamma_e * delta_pi
        gdpi = self._net.times_gamma(self._np.dpi_t, edge=edge)
        
        # compute gamma_e * L_star * L0
        # First, compute L_star * gamma_e efficiently
        w_i = self._net.edges.t[edge]
        v_i = self._net.edges.s[edge]
        
        # TODO-opt -> optimization potential
        lstar_w = self.Lstar_cache[:, w_i]
        lstar_v = self.Lstar_cache[:, v_i]
        lstar_gam = lstar_w - lstar_v
        
        # Compute L0 * (L_star * gamma_e) and add to M matrix
        glstarl = self.L0.dot(lstar_gam)
        self.M[:, j] = (-1 / gdpi) * glstarl

    def _print_iteration_summary(self) -> None:
        # print summary of iteration (if enabled)
        if self._c.print is True:
            out = "Iteration {:4d} | ".format(self.i)
            lam_str = "λ ∈ [{:.3f}, {:.3f}]".format(self.param_solution[-1].param,
                                                    self.lambda_max)
            out += lam_str.ljust(20)
            if self.min_edge is not None and self.min_edge >= 0:
                out += " | "
                out += {1: "\u2191 ", -1: "\u2193 "}[self.region_activate]
                out += "e* = " + str(self.min_edge)
                if len(self.boundary_edges) > 1:
                    out += " ∈ E* = " + str(list(self.boundary_edges))
            else:
                out += " | E* = ∅"
            
            if self.breakflag != EFABreakFlag.NOT_SET:
                out += " | " + str(self.breakflag)
            
            print(out)

    @property
    def region(self) -> np.ndarray:
        """ndarray of int: piecewise cost region of edges."""
        return self.edges.region

    @property
    def edges(self) -> EFAEdges:
        """Object that keeps track of edge values for current region.
        
        See Also
        --------
        EFAEdges
        """
        return self._e

    @property
    def edge_coeffs(self) -> PiecewiseQuadraticCoefficients:
        """Edge coefficients for current region.
        
        See Also
        --------
        paminco.net.cost.sPiecewiseQuadraticCoefficients
        """
        return self._ec

    @property
    def node_potentials(self) -> NodePotentials:
        """Object that keeps track of node potentials of current region.
        
        See Also
        --------
        NodePotentials
        """
        return self._np

    @property
    def L0(self) -> sps.spmatrix:
        """spmatrix: Laplacian for initial region."""
        if hasattr(self, "_L0") is False or self._L0 is None:
            self._L0 = self._net.L(region=self.region_zero)
        return self._L0

    @property
    def region_zero(self) -> np.ndarray:
        """ndarray of int: initial region in piecewise cost funcs."""
        return self._region_zero

    @property
    def is_phase1(self) -> bool:
        """Returns ``True`` if this is a phase 1 EFA run."""
        return self.phase1_of is not None
