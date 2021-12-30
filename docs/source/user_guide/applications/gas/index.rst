======
Gaslib
======

GasLib [Sch17]_ is a library for instances of gas networks. Each network from GasLib includes various XML files such as a network XML file and a nomination XML file. The former contains the edge data, and the latter specifies the flow that is supplied or consumed for each network node. 
The gas network is modeled as a *directed graph* :math:`G = (V, E)` where :math:`E` consists of the *active* and *passive* edges. Active edges are compressor groups, valves, and control valves. Passive edges are comprised of pipes and resistors. 
For the gas networks, all edges are *undirected*, i.e., flow can traverse a pipe in both ways. Thus, :math:`x_e > 0` indicates to a gas flow in the direction of the edge and :math:`x_e < 0` against the specified direction. 

The gas flow for a pipe :math:`e` can be modeled with the Weymouth equations [Wey12]_:

.. math::

    \begin{equation}
        p_w^2 -p_v^2  = \beta_e x |x|,
    \end{equation}

where :math:`p_v` denotes the gas pressure for a node :math:`v` and :math:`\beta_e` is factor that models the resistance of :math:`e` [Pfe15]_ [War21]_. 
We only consider a network consisting of pipes, i.e., network elements such as compressors and valves are collapsed. 
The coefficient :math:`\beta_e` can then be calculated using physical specifications such as length and diameter. See [Pfe15]_ for further details on this conversion.

.. toctree::
    :hidden:

    Gas24 <gas24.ipynb>

.. rubric:: References

.. [Pfe15] Pfetsch ME, et al. (2015) "Validation of nominations in gas network optimization: models, methods, and solutions." *Optimization Methods and Software* 30.1: 15-53.
.. [Sch17] Schmidt M, Aßmann D, Burlacu R, Humpola J, Joormann I, Kanelakis N, ... & Sirvent M (2017). "Gaslib—A library of gas network instances." *Data*, 2(4), 40.
.. [War21] Warode P (2021). *Parametric Computation of Equilibria and Flows.* PhD thesis. TU Berlin, Berlin, Germany.
.. [Wey12] Weymouth RF (1912). "Problems in natural gas engineering." In: *Trans. Am. Soc. Mech. Eng.* 34.1349, pp. 185–231.