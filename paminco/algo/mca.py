import numpy as np
import scipy.sparse as sps

from .efa import EFA, EFAConfig, EFAEdges, NodePotentials
from paminco._base import AlphaBetaApproximativeSolver, ParametricSolution
from paminco.callback import CallBackFlag
from paminco.net.network import Network
from paminco.net.cost import InterpolationRule, PiecewiseQuadraticCoefficients, PiecewiseQuadraticCost
from paminco.utils.bisec import bisec_fast
from paminco.utils.misc import callback_to_list


class MCAInterpolationRule(InterpolationRule):
    r"""Breakpoint rule for MCA guaranteeing the :math:`(\alpha, \beta)`-approximation property.
    
    This class manages the the breakpoint computation for the interpolation
    of the cost functions in the Marginal Cost Approximation (MCA).

    The following assumptions on the cost functions :math:`F_e(x)` and the
    marginal costs :math:`f_e(x) := F'_e(x)` must be satisfied.

    * :math:`F_e` is strictly convex and smooth.
    * For :math:`x \geq 0`, :math:`f_e''(x)` is non-negative and non-decreasing
    * For :math:`x < 0`, :math:`f_e''(x)` is non-positive and non-increasing

    These properties are satisfied for functions of the form
    
    .. math::
        F_e(x) = a_e |x_e| x_e^{k-1} + b_e

    for :math:`k > 3` and therefore apply for example to BPR functions for traffic networks
    or the activation functions typically used in the Weymouth equations in gas networks.

    The step sizes :math:`\delta_i` depending on :math:`x_i` that this rule computes, satisfy
    the inequality

    .. math::
        \delta_i B \leq 2 \sqrt{2} \sqrt{(\alpha-1) |f_e (x_i)| + \frac{\beta}{m x^{\max}}}

    where :math:`B = \sqrt{ \max \{ |f''(x_i)|, |f''(x_{i+1})| \} }`.
    The breakpoint rule always includes :math:`x=0` as breakpoint.
    The breakpoints are computed by solving the above inequality numerically up to a fixed
    precision given by the parameter ``accuracy``, while rounding to the safe side.

    Parameters
    ----------
    alpha : float
        The parameter :math:`\alpha` for the relative approximation error of the MCA output
    beta : float
        The parameter :math:`\beta` for the absolute approximation error of the MCA output
    m : int
        The number of edges of the network the rule is applied to. Needed, since the absolute
        error of all edges is additive, in order to ensure an overall absolute error.
    x_max : float
        The (absolute value of the) maximum flow value on one edge, or a bound on that value.
        An estimate on this value could be
        :math:`x^{\max} := \lambda^{\max} \frac{1}{2} \sum_{v \in V} |b_v|`.
    accuracy : float, default=1e-05
        The accuracy for the numerical solution of the breakpoint inequality.

    See also
    --------
    paminco.network.cost.NetworkCostInterpolation
    """

    def __init__(
            self,
            alpha: float,
            beta: float,
            m: int,
            x_max: float,
            accuracy: float = 1e-5
            ):
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.x_max = x_max
        self.accuracy = accuracy

    def step(
            self,
            edge_cost,
            edge: int,
            x: float
            ) -> float:
        r"""Compute the next breakpoint :math:`\delta` by solving the inequality
        
        .. math::
            \delta B \leq 2 \sqrt{2} \sqrt{(\alpha-1) |f_e (x)| + \frac{\beta}{m x^{\max}}}

        Parameters
        ----------
        edge_cost: EdgeCost
            The :class:`~paminco.net.cost.EdgeCost` object representing the edge cost function
            :math:`F(x)`.
        edge: int
            The index of the edge. Only for debugging purposes, can also be None.
        x: float
            The last breakpoint.
        
        Returns
        -------
        delta: float
            The step size :math:`\delta` to the next breakpoint.
        """
        
        def f(x):
            return edge_cost(x, d=1)
        
        def f2(x):
            return edge_cost(x, d=3)
        
        # If the cost that are interpolated are polynomials
        # and the degree of the polynomial is less or equal 2
        # than no interpolation is needed => return infinite
        # step size
        if edge_cost.degree <= 2:
            return np.inf
        
        y1 = abs(f(x))
        f2atx = abs(f2(x))
        fixed_rhs_val = 8 * self.beta / (self.m * self.x_max)
        
        def err_fct(delta):
            if x < 0:
                y2 = abs(f(x + delta))
            else:
                y2 = np.inf
            
            rhs_val = 8 * (self.alpha - 1) * min(y1, y2)
            rhs_val += fixed_rhs_val
            
            mcost2val = abs(f2(delta + x)) if x >= 0 else f2atx
            return delta ** 2 * mcost2val - rhs_val
        
        try:
            delta = bisec_fast(err_fct,
                               tol=self.alpha * self.accuracy,
                               up=1,
                               exclude_lo=True,
                               val_constr='npos')
        except RuntimeError:
            return None

        # Breakpoint at 0 is required since the breakpoint
        # computation rule differs for negative and
        # positive values of x
        if x < 0 and x + delta > 0:
            delta = -x
        
        return delta


class MCAConfig(EFAConfig):
    """Settings for MCA algorithms.
    
    Parameters
    ----------
    alpha : float, default=1.01
        Relative tolerance for alpha-beta-approximation.
    beta : float, default=1
        Absolute tolerance for alpha-beta-approximation.
    interpolation_accuracy : float, default=1e-3
        # TODO-PW
    interpolation_step_size, optional
        # TODO-PW
    lambda_max : float, default=1
        Maxmimum parameter (lambda) to find *minimum cost flow* for.
    inverse_method : str, int or InverseMethod, default=InverseMethod.CHOLESKY
        Method to find the inverse Laplacian.
    pivot_mode : int, str or PivotStepMode, default=PivotStepMode.LEX
        Method to choose min edge if len(boundary_edges) > 1.
    max_iter : int, default=99999
        Maximum number of iterations for solver.
    print : bool, default=False
        Whether to print a summary of each iteration.
    recomp_interval : int, default=1
        If <= 1, recompute inverse Laplacian in every iteration. Else
        update and recompute every `recomp_interval`. The higher the
        value, the less accurate the inverse Laplacian.
    round_lambda : int, default=3
        Round lambda value for edges to `round_lambda` after decimal.
        Used to determine boundary edges.
    rounding_margins_base : int, default=-16
        Set gamma_dpi with low exponent (in `IEEE754 <https://en.wikipedia.org/wiki/IEEE_754>`_) to zero.
    rounding_margins_fac : int, default=-5
        Set gamma_dpi with low exponent (in `IEEE754 <https://en.wikipedia.org/wiki/IEEE_754>`_) to zero.
        
    """
    
    kw_init = [
        "lambda_max",
        "alpha",
        "beta",
        "interpolation_step_size",
        "interpolation_accuracy",
    ]
    
    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = 1.01
        self.beta = 1
        self.interpolation_accuracy = 1e-3
        self.interpolation_step_size = None
        
        # ? TODO-PW: alpha-beta approxim for equation check in interpolation instead of abs difference
        
        self.map_kwargs(run=False, **kwargs)
    
    def get_efa_kwargs(self):
        return {k: getattr(self, k) for k in EFAConfig.all_options}


class MCA(AlphaBetaApproximativeSolver):
    r"""Marginal Cost Approximation.
    
    This method interpolates the marginal edge cost and then runs EFA to
    compute an exact optimal potential (and thus minimum cost flow) function.
    Thus, MCA is an approximate solver, where the approximation error arises
    from the spline-interpolation, and is named as marginal cost approximation
    (MCA).
    
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
        Whether to time MCA. If ``True``, timestamps for intializtion
        and every iteration will be saved to attribute ``timer``.
    kwargs : keyword arguments, optional
        Further options for MCA, see MCFIConfig.
    
    Attributes
    ----------
    region
    edges
    edge_coeffs
    node_potentials
    
    See Also
    --------
    MCAConfig: Settings for MCA.
    MCAInterpolationRule : Piecewise linear interpolation of the edge cost.
    paminco.algo.efa.EFA:
        Electrical flow algorithm to be run on modified network with piecewise
        quadratic cost.

    References
    ----------
    .. [1] Klimm, Max, and P. Warode. "Parametric Computation of Minimum Cost
           Flows with Piecewise Quadratic Costs." Mathematics of Operations
           Research (2021). Available 10/25/2021 at https://www3.math.tu-berlin.de/disco/research/publications/pdf/KlimmWarode2021.pdf
    
    Examples
    --------
    Compute the **price of anarchy** for Sioux Falls::
    
        import numpy as np
        import copy
        import paminco
        net = paminco.net.load_sioux()
        net.set_demand(("7", "13", 100000))
        # ----------------------------------------------------------------
        # Calculate system optimum
        mca_so = MCA(net)
        mca_so.run()
        # ----------------------------------------------------------------
        # Calculate user equilibrium -> integrate costs
        net_ue = copy.deepcopy(net)
        net_ue.integrate_cost()
        mca_ue = MCA(net_ue)
        mca_ue.run()
        # ----------------------------------------------------------------
        # Calculate price of anarchy (PoA)
        x = np.linspace(0, 1, 100)
        cost_so = np.array([net.cost(mca_so.flow_at(p)).sum() for p in x])
        cost_ue = np.array([net.cost(mca_ue.flow_at(p)).sum() for p in x])
        PoA = cost_ue / cost_so
    
    Determine edge flow in user equilibrium as a function of demand factor::
    
        import paminco
        net = paminco.net.load_sioux()
        net.integrate_cost()
        net.set_demand(("2", "21", 100000))
        mca = MCA(net)
        mca.run()
        mca.plot_flow_on_edge(55)
    
    """
    def __init__(
            self,
            network: Network,
            name=None,
            callback=None,
            use_simple_timer: bool = True,
            copy_network: bool = True,
            **kwargs
            ) -> None:
        super().__init__(
            network,
            name=name,
            callback=callback,
            copy_network=copy_network,
            use_simple_timer=use_simple_timer
        )
        # MCA settings
        self._c = MCAConfig(**kwargs)
        
        # Prepare network be run with EFA
        self._cost_to_piecewise()
        
        # Pass prepped network to EFA
        self.efa = EFA(
            network=self.network,
            copy_network=False,
            preprocess_network=False,
            use_simple_timer=False,
            **self._c.get_efa_kwargs()
        )
        
        # Flag end of iteration to listeners
        self.callback(CallBackFlag.INIT_END)

    def _cost_to_piecewise(self) -> None:
        # Initialize network costs if not PiecewiseQuadraticCost
        if isinstance(self._net.cost, PiecewiseQuadraticCost) is False:
            # Get maximum value for which to egde cost splines
            x_max = self._net._d.max_inflow(max_param=self._c.lambda_max).max()
            
            # Transform edge cost to piecewise quadratic
            iprule = MCAInterpolationRule(
                alpha=self.config.alpha,
                beta=self.config.beta,
                m=self.network.m,
                x_max=x_max,
                accuracy=self.config.interpolation_accuracy
            )
            self._net._c = self._net.cost.interpolate(iprule)
    
    def run(self, callback=None, **kwargs):
        if callback:
            callback = self.callbacks + callback_to_list(callback)
        else:
            callback = self.callbacks
        self.efa.run(callback=callback, **kwargs)
        
    @property
    def i(self) -> int:
        """Get current iteration count."""
        return self.efa.i
    
    @property
    def lambda_min(self) -> float:
        """Get current lambda_min."""
        return self.efa.lambda_min
        
    @property
    def lambda_max(self) -> float:
        """Get current lambda_max."""
        return self.efa.lambda_max
        
    @property
    def breakflag(self) -> float:
        """Get current breakflag value."""
        return self.efa.breakflag
        
    @property
    def region(self) -> np.ndarray:
        return self.efa.region
    
    region.__doc__ = EFA.region.__doc__

    @property
    def edges(self) -> EFAEdges:
        return self.efa.edges
    
    edges.__doc__ = EFA.edges.__doc__

    @property
    def edge_coeffs(self) -> PiecewiseQuadraticCoefficients:
        return self.efa.edge_coeffs
    
    edge_coeffs.__doc__ = EFA.edge_coeffs.__doc__

    @property
    def node_potentials(self) -> NodePotentials:
        return self.efa.node_potentials
    
    node_potentials.__doc__ = EFA.node_potentials.__doc__

    @property
    def param_solution(self) -> ParametricSolution:
        return self.efa.param_solution

    @property
    def L0(self) -> sps.spmatrix:
        return self.efa.L0
    
    L0.__doc__ = EFA.L0.__doc__

    @property
    def region_zero(self) -> np.ndarray:
        return self.efa.region_zero
    
    region_zero.__doc__ = EFA.region_zero.__doc__
