.. _ug-traffic-background:

==========
Background
==========

.. contents::
  :local:
  :depth: 2

.. _ug-traffic-ltt:

Link Travel Time
================

For transportation networks, the link travel time :math:`l_e(x_e)` describes how much time 
it takes to traverse the edge as a function of some flow :math:`x_e`, i.e., the amount many units on
the edge. The link travel time is usually defined as:

.. math::

  l_e(x_e) = \text{fft}_e \cdot \left( 1 + B_e \cdot \left(\frac{x}{\text{cap}_e}\right) ^ {p_e} \right),

where :math:`\text{fft}_e` is the free flow travel time (how much time does it take if the road is 
empty), :math:`\text{cap}_e` the capacity and :math:`B_e` a factor that models the congestion.



.. _ug-traffic-equilibria:

Wardrop´s Principles and the Price of Anarchy
=============================================

.. _ug-traffic-ue:

Wardrop´s first principle: user equilibrium (UE)
------------------------------------------------
A user equilibrium (UE) in a traffic network exists if every route a driver might use takes the 
same amount of time:

.. math::

   \sum_{e \in P} l_e(x_e) \leq \sum_{e \in Q} l_e(x_e)

for all s-t-paths P and Q and where :math:`l_e(x_e)` is a continuous and strictly increasing 
travel time function.

It is known that a Wardrop equilibium coincides with a 
:ref:`minimum cost flow <theory-mcf>` with the objective cost function :math:`C` defined
as:

.. math::

  C = \sum_{e \in E} F_e(x_e))  = \sum_{e \in E} \int_{0}^{x_{e}} l_{e}(\xi) \mathrm{d} \xi.
  
The optimization problem thus becomes:

.. math::

  \begin{align*}
  \min \sum_{e \in E} \int_{0}^{x_{e}} l_{e}(\xi) \mathrm{d} \xi \\
  \text {s.t.} \quad \mathbf{\Gamma} \mathbf{x} &= \mathbf{b}, \\
  \mathbf{x} &\geq \mathbf{0}
  \end{align*}


.. _ug-traffic-so:

Wardrop´s second principle: system optimum (SO)
-----------------------------------------------
In contrast to the user eqiulibrium, the system optimum minimizes the total 
travel time of all traffic participants. It is obtained by finding a 
:ref:`minimum cost flow <theory-mcf>`
with an objective cost function :math:`C` defined as

.. math::

  C = \sum_{e \in E} F_e(x_e))  = \sum_{e \in E} x_{e} \cdot l_{e}(x_e).


.. _ug-traffic-poa:

The price of anarchy (PoA)
--------------------------
`The Price of Anarchy (PoA) <https://en.wikipedia.org/wiki/Price_of_anarchy>`_ measures the
influence of selfish behavior on system efficiency. For a traffic network, it is defined as the
ratio of the total system travel time (TSTT) in the user equilibrium :math:`\mathbf{x}_{\text{UE}}^{\ast}` 
to that in the system optimal :math:`\mathbf{x}_{\text{SO}}^{\ast}`:

.. math::

  \text{PoA} = \frac{\text{TSTT}(\mathbf{x}_{\text{UE}}^{\ast})}{\text{TSTT}(\mathbf{x}_{\text{SO}}^{\ast})},

where :math:`\text{TSTT}(\mathbf{x}) = \sum_{e \in E} x_{e} \cdot l_{e}(x_e)`, i.e., the objective
function of the system optimum.



