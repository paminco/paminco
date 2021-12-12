.. _theory-paminco:

=============================
Parametric Minimum Cost Flows
=============================

.. contents::
  :local:
  :depth: 2

There exists a variety of solutions to the :ref:`minimum cost flow problem <theory-mcf>`. 
*parcmf* provides solutions to a slighly different problem.

.. _theory-mcf-func:

1. Minimum cost flow functions
==============================
The goal is to find a minimum cost flow **function** that maps a demand factor :math:`\lambda` to 
a minimum cost flow:

.. math::

  \begin{align*}
  \min \text { } & C(\mathbf{x}) \\
  \text {s.t.} \quad \mathbf{\Gamma} \mathbf{x} &= \mathbf{h}(\lambda), \\
  \mathbf{x} &\geq \mathbf{0},
  \end{align*}

.. toggle-header:: 
  :header: where **show definitions**:

  - :math:`\mathbf{\Gamma}` is the incidence matrix and
  - :math:`\mathbf{h}(\lambda): \mathbb{R} \rightarrow \mathbf{b}_{\lambda}` a demand function. 

For notational simplicity, we only discuss linear demand functions 
:math:`\mathbf{h}(\lambda) = \lambda \mathbf{b}`.


.. _theory-mcf-abapprox:

1.1 Alpha-beta-approximate minimum cost flow functions
------------------------------------------------------
.. toggle:: 

  It may not always be (computationally) feasible to compute an exact minimum cost flow function and
  the function has to be approximated.

  Consider an interval :math:`\left[0, \lambda^{\max }\right]`, for which we want to find an 
  :math:`(\alpha, \beta)`-approximate minimum cost flow function. 
  A function is an :math:`(\alpha, \beta)`-approximate minimum cost flow function if the cost of the 
  flow :math:`\mathbf{x}(\lambda)` exceed the optimal cost at most by a relative factor of 
  :math:`\alpha` and an absolute error of :math:`\beta`:

  .. math::
    \tilde{C}(\lambda) \leq \alpha C(\lambda)+\beta \quad 
    \text { for all } \lambda \in\left[0, \lambda^{\max }\right],

  where :math:`\tilde{C}(\lambda) = \tilde{C}(\mathbf{x}(\lambda)) = \sum_{e \in E} F_e(x_e(\lambda))` 
  denotes the cost of the flow :math:`\mathbf{x}(\lambda)`.

.. _theory-par-eflow:

2. Parametric electrial flows
=============================
Instead of computing a minimum cost flow :math:`\mathbf{x}` (for a fixed demand) directly, 
we can compute an :ref:`optimal potential <theory-mcf-characterization-optpot>` :math:`\mathbf{\pi}` which *induces* a flow:

.. math::
  \begin{equation*}
  \mathbf{x} = \mathbf{f}^{-1} (\mathbf{\Gamma}^T \mathbf{\pi}),
  \end{equation*}

where :math:`\mathbf{f}^{-1}` maps an optimal potential to an *electrial flow*:

.. math::
  \begin{equation*}
  \mathbf{f}^{-1} : \Pi \rightarrow \mathbb{R}^m \quad 
  \mathbf{\pi} \rightarrow \mathbf{f}^{-1}(\mathbf{\pi}) := (f_e^{-1}(\pi_w - \pi_v))_{e \in E}.
  \end{equation*}

.. _theory-par-eflow-linear:

2.1 Linear marginal cost
------------------------
Assume that the marginal cost for all edges are linear and of the following form: 
:math:`f_e(x) = 2ax + b`. 
Then we find a *linear* function :math:`\mathbf{\pi}(\lambda)` and its induced 
linear electrical flow :math:`\mathbf{x}(\lambda)`:

.. math::
  \begin{align*}
  \mathbf{\pi}(\lambda) &= \mathbf{L}^{\ast}(\lambda \mathbf{b} + \mathbf{\Gamma} \mathbf{d}) \quad \text{and}\\
  \mathbf{x}(\lambda) &= \mathbf{C} \mathbf{\Gamma}^T \mathbf{\pi}(\lambda) - \mathbf{d}
  = \mathbf{C} \mathbf{\Gamma}^T \mathbf{L}^{\ast}(\lambda \mathbf{b} + \mathbf{\Gamma} \mathbf{d}) - \mathbf{d}
  , 
  \end{align*}

.. toggle-header:: 
  :header: where **show definitions**:

  - :math:`\mathbf{C} = \text{diag}(1/2a_1, 1/2a_2, ..., 1/2a_m)`, 
  - :math:`\mathbf{d} = (b_1/2a_1, b_2/2a_2, ..., b_m/2a_m)^T`,
  - :math:`\mathbf{L} = \mathbf{\Gamma} \mathbf{C} \mathbf{\Gamma}^T` the weighted Laplacian of the graph, and 
  - :math:`\mathbf{L}^{\ast}` the generalized inverse of :math:`\mathbf{L}`.

.. admonition:: At a glance

  Linear marginal cost result in a linear minimum cost flow function.

.. toggle-header:: 
  :header: *Example*: linear resistances **show / hide**

  Consider an electrical network with linear resistances, i.e., :math:`f_e(x) = r_e \cdot x`.
  The objective function simplifies to :math:`C(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{R} \mathbf{x}` with
  :math:`\mathbf{R} = \text{diag}(r_{e_1}, r_{e_e}, ..., r_{e_m})`.
  
  Then we find the optimal potential and minimum cost flow as:

  .. math::

    \begin{align*}
      \mathbf{\pi}(\lambda) &= \mathbf{L}^{\ast} \mathbf{h}(\lambda) \\
      \mathbf{x}(\lambda) &= \mathbf{C} \mathbf{\Gamma}^T \mathbf{L}^{\ast} \mathbf{h}(\lambda),
    \end{align*}

  where :math:`\mathbf{C} = \mathbf{R}^{-1}`.

  Thus, the parametric electrial flow is just a linear transformation of the demand function :math:`\mathbf{h}(\lambda)`.


.. _theory-par-eflow-linear-piecewise:

2.2 Piecewise linear marginal costs
-----------------------------------
The assumption of linear marginal cost restricts the applicability of the algorithm.
We now discuss problems with *piecewise linear marginal costs*:

.. tabbed:: Graph
  
  .. image:: /_static/img/tikz/fig2a_graph_with_marginal_cost.png
    :align: center
    :height: 250

.. tabbed:: Regions in :math:`\Pi` and solution curve

  .. image:: /_static/img/tikz/fig3a_potential_space_and_solution_curve.png
    :align: center
    :height: 250

.. tabbed:: Induced edge flows

  .. image:: /_static/img/tikz/fig3b_induced_flow.png
    :align: center
    :height: 250

The piecewise linear marginal edge cost subdivide the potential space :math:`\Pi` into 
regions.
We denote :math:`\mathbf{t} = (t_{e_1}, t_{e_2}, ..., t_{e_m})^T` the vector of the indices 
per edge cost function.
Given an optimal potential :math:`\mathbf{\pi}`, the electrical flow is then:

.. math::

  \mathbf{x} 
  = \mathbf{C}_{\mathbf{t}} \mathbf{\Gamma}^T \mathbf{\pi} - \mathbf{d}_{\mathbf{t}}
  = \mathbf{C}_{\mathbf{t}} \mathbf{\Gamma}^T \mathbf{L}_{\mathbf{t}}^{\ast}(\mathbf{b} 
    + \mathbf{\Gamma} \mathbf{d}_{\mathbf{t}}) - \mathbf{d}_{\mathbf{t}}

In a region :math:`R_{\mathbf{t}}` the solution curve is linear. 
The full solution curve is union of the solution segments :math:`\Pi_{\mathbf{t}}` and
thus a piecewise linear function:

.. math::

  \pi\left(\mathbb{R}_{\geq 0}\right):=\{\pi(\lambda) \mid \lambda \geq 0\}=\bigcup_{t \in \bar{T}} \Pi_{t}.

Furhter, for all regions, there exist numbers :math:`\lambda_t^{\text{min}}` and :math:`\lambda_t^{\text{max}}`:

.. math::

    \Pi_{\mathbf{t}} = 
    \left\{ 
      \mathbf{\pi}_{\mathbf{t}} + \lambda \Delta \mathbf{\pi}_{\mathbf{t}} 
      \mid \lambda \in [\lambda_t^{\text{min}}, \lambda_t^{\text{max}}]
    \right\}
    .

Hence, the main challenge lies in finding the correct regions :math:`R_{\mathbf{t}}`.
The :ref:`Electrial Flow Algorithm (EFA) <efa>` [1]_ provides a solution for this problem.
  
.. admonition:: Idea behind the Electrial Flow Algorithm

  *Piecewise linear* marginal cost result in a subdivision of the potential space :math:`\Pi`.
  For every region :math:`R_{\mathbf{t}} \in \Pi`, the solution curve is linear. Thus the full
  solution curve is *piecewise linear*. 
  The difficulty thus lies in finding the correct sequence of regions.

2.3 Interpolation of cost functions
-----------------------------------
The EFA algorithm is restricted to piecewise linear marginal cost functions :math:`f_e` and thus
minimum cost flow problems with piecewise quadratic cost functions.
However, it is possible to approximate continous convex cost functions :math:`F_e` with 
piecewise quadratic convex cost functions :math:`\tilde{F}_e` [2]_.
The :ref:`Marginal Cost Approximation (MCA) <mca>` algorithm builds on that idea and "feeds"
the EFA with approximated piecewise linear marginal cost.
Hence, MCA calculates an :ref:`alpha-beta-approximate <theory-mcf-abapprox>` minimum cost flow function.
  
.. seealso:: 

  :class:`MCAInterpolationRule <paminco.algo.mca.MCAInterpolationRule>` : Criteria for proper approximation of cost functions.


.. rubric:: References

.. [1] Klimm M, Warode P (2021) Parametric Computation of Minimum Cost
       Flows with Piecewise Quadratic Costs. *Mathematics of Operations
       Research*. Available at https://www3.math.tu-berlin.de/disco/research/publications/pdf/KlimmWarode2021.pdf
.. [2] Rockafellar, RT (1984) *Network Flows and Monotropic Optimization* (John Wiley and Sons, Hoboken, NJ)