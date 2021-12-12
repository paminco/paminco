.. _theory-mcf:

=========================
Minimum Cost Flow Problem
=========================

.. contents::
  :local:
  :depth: 2

The `minimum cost flow problem <https://en.wikipedia.org/wiki/Minimum-cost_flow_problem>`_ is 
related to many real world applications such as finding Wardrop equilibria in 
:ref:`traffic networks <ug-traffic-index>`.

Consider a strongly connected graph :math:`G = (V, E)` without self loops and parallel edges.
We denote :math:`n = |V|` the number of vertices or nodes and :math:`m = |E|` the number of links
or edges.
We want to find a flow :math:`\mathbf{x}` -- a vector of shape (m, ) -- that minimizes some 
objective cost function :math:`C`:

.. math::

  \begin{align*}
  \min \text { } & C(\mathbf{x}) \\
  \text {s.t.} \quad \mathbf{\Gamma} \mathbf{x} &= \mathbf{b}, \\
  \mathbf{x} &\geq \mathbf{0}, 
  \end{align*}

.. toggle-header:: 
  :header: where **show definitions**:

  - :math:`\mathbf{\Gamma}` is the 
    `incidence matrix <https://en.wikipedia.org/wiki/Incidence_matrix>`_ of :math:`G` and
  - :math:`\mathbf{b}` a demand vector of shape (n, ) that encodes the demand values for each node,
    negative values representing sources.

Given a demand :math:`\mathbf{b}`, many flows :math:`\tilde{\mathbf{x}} \in \mathcal{X}` are 
feasible solutions.
The minimum cost flow has lower cost compared to all other feasible flows:
:math:`C(\mathbf{x}) \leq C(\tilde{\mathbf{x}})`.


.. _theory-mcf-undirected:

1. Undirected minimum cost flow problem
=======================================
The classic minimum cost flow restricts quantities to traverse edges in one direction only.
However, in many applications -- physical networks such as gas, electrical or water networks --
units may flow along the same edge in different directions. 
Solving the above optimization problem without the restriction :math:`\mathbf{x} > 0` thus yields the
*undirected minimum cost flow*:

.. math::

  \begin{align*}
  \min \text { } & C(\mathbf{x}) \\
  \text {s.t.} \quad \mathbf{\Gamma} \mathbf{x} &= \mathbf{b}.
  \end{align*}

.. _theory-mcf-multicommodity:

2. Multi-commodity minimum cost flow problem
=============================================
Consider a :math:`n \times k` demand matrix :math:`\mathbf{B}` where each column represents a 
demand vector. Here, we want to find a :math:`n \times k` flow matrix :math:`\mathbf{X}`, where 
each column :math:`\mathbf{X}_{:j}` is the minimum cost flow associated with the demand vector 
:math:`\mathbf{B}_{:j}`. A minimum cost flow matrix is the solution to the following optimization 
problem:

.. math::

  \begin{align*}
  \min \text { } & C(\mathbf{X}) \\
  \text {s.t.} \quad \mathbf{\Gamma} \mathbf{X} &= \mathbf{B}, \\
  \mathbf{X} &\geq \mathbf{0}
  \end{align*}

.. _theory-mcf-characterization:

3. Characterization of minimum cost flows
=========================================

.. _theory-mcf-characterization-optpot:

3.1 Optimal potentials
----------------------
A vector :math:`\mathbf{x} \in \mathbb{R}^m` is a minimum cost flow that solves the above optimization problem iff 
there is a vector :math:`\mathbf{\pi} \in \mathbb{R}^n` of vertex potentials so that:

.. math::

  \begin{aligned}
  f_{e}^{-}\left(x_{e}\right) \leq \pi_{w}-\pi_{v} & \leq f_{e}^{+}\left(x_{e}\right) \text { for all } e=(v, w) \in E \text { with } x_{e}>0, \\
  \pi_{w}-\pi_{v} & \leq f_{e}^{+}\left(x_{e}\right) \text { for all } e=(v, w) \in E \text { with } x_{e}=0
  ,
  \end{aligned}

.. toggle-header:: 
  :header: where **show definitions**:

  - :math:`f_e` is piecewise linear marginal cost function, 
  - :math:`f_{e}^{-}\left(x_{e}\right)` denotes the left-hand limit of :math:`f_e`, and
  - :math:`f_{e}^{-}\left(x_{e}\right)` the right-hand limit of :math:`f_e`.

The vector :math:`\mathbf{\pi}` is then termed *optimal potential (vector)*.
Optimal potentials can thus be used to *verify* if a flow :math:`\mathbf{x}` is indeed a minimum cost flow.
If we can compute an optimal potential, :math:`\mathbf{x}` is a minimum cost flow; 
if no such potential exsits, it is not.

.. admonition:: Potentials and direction of flow

  Due to our convention for demand vectors :math:`\mathbf{b}` to encode sources with negative values and sinks
  with positive values, we define the potential difference :math:`\pi_w - \pi_v`, i.e., the
  potential increases in the direction of flow. This may not be intuitive for physical applications where 
  quantities usually "flow" from higher to lower potential.

.. _theory-mcf-characterization-sppot:

3.2 Shortest path potentials
----------------------------

The above verification of minimial cost flows can be carried out by calculating *shortest paths*.

.. _theory-mcf-characterization-supp:

3.3 Support of minimum cost flows
---------------------------------
