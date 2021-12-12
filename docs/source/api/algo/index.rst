.. _algo:

:mod:`paminco.algo`

==========
Algorithms
==========

The following main algorithms are implemented in paminco.
While EFA computes an exact parametric minimum cost flow function for piecewise quadratic edge 
cost, MCA and MCFI both output a 
TODO
.. :ref:`alpha-beta-approximate <theory-mcf-abapprox>`
minimum cost flow function.

.. list-table:: Algorithms for parametric mincost flows
   :widths: 5 17 43 35
   :header-rows: 1

   * - Abbrev
     - Full name
     - Short summary
     - Source of approximation error

   * - :ref:`EFA <efa>`
     - Electrical Flow Algorithm
     - Compute an optimal potential function :math:`\lambda \rightarrow \mathbf{\pi}(\lambda)` 
       for linear piecewise marginal costs that induces a parametric minimum cost function 
       :math:`\lambda \rightarrow \mathbf{x}(\lambda)`.
     - / 
   * - :ref:`MCA <efa>`
     - Marginal Cost Approximation
     - Interpolate marginal edge cost and run EFA with piecewise quadratic cost .
     - Linear interpolation of marginal edge costs :math:`f_e`.
   * - :ref:`MCFI <efa>`
     - Minimum Cost Flow Interpolation
     - *Repeat*: 1) find next parameter :math:`\lambda` and 2) calculate minimum cost flow 
       :math:`\mathbf{x}_{\lambda}`. *Then*: interpolate between breakpoint solutions.
     - Interpolative nature and optimizer tolerance :math:`\epsilon`.

.. toctree::
  :maxdepth: 1

  efa
  mca
  mcfi
