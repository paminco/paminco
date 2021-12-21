.. _ug-index:

==========
User Guide
==========

.. tabbed:: Traffic Networks

  .. admonition:: Motivation

    :ref:`Minimum cost flows <theory-mcf>` coincide with
    :ref:`user equilibrium <ug-traffic-ue>` and 
    :ref:`system optimum <ug-traffic-so>` in
    :ref:`traffic network <ug-traffic-index>`
    if the edge cost are transformed by:

    .. math::

      \begin{align}
      \text{User equilibrium:} \quad F_e &= \int_0^{x_e} l_e(s) ds \\ 
      \text{System optimum:} \quad F_e &= x_e \cdot l_e \\ 
      \end{align}

    .. toggle-header:: 
      :header: The link travel time is defined as: **show/hide**

      .. math::

        l_e(x_e) = \text{fft}_e \cdot \left( 1 + B_e \cdot \left(\frac{x}{\text{cap}_e}\right) ^ {p_e} \right).

  .. dropdown:: *Braess Paradox*: Adding capacity to a network might decrease network efficiency due to selfish participants.
    
    .. image:: applications/traffic/img/braess.jpg
        :width: 500
        :alt: Braess network
        :align: center

    Adding capacity to a network might decrease network efficiency due to selfish participants.
    
    .. link-button:: applications/traffic/braess_paradox.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`
  
  .. dropdown:: *Pigou's Example*: An upper bound for the Price of Anarchy

    .. image:: applications/traffic/img/pigou.png
        :width: 500
        :alt: Pigou network
        :align: center

    Simple network with a price of anarchy of :math:`\text{PoA} = \frac{4}{3}`.

    .. link-button:: applications/traffic/pigou_example.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`
  
  .. dropdown:: *SiouxFalls*: calculating and plotting a fixed user equilibrium

    .. image:: applications/traffic/img/sioux_ue.png
        :height: 500
        :alt: Pigou network
        :align: center

    We compute a user equilibrium and plot it using NetworkX.

    .. link-button:: applications/traffic/sioux_fixed_ue.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`
  
  .. dropdown:: *SiouxFalls*: parametric Price of Anarchy

    .. image:: applications/traffic/img/sioux_poa.png
        :width: 500
        :alt: Pigou network
        :align: center

    We compute a parametric price of anarchy for SiouxFalls and a single commodity.

    .. link-button:: applications/traffic/sioux_par_poa.html
        :text: Go to user guide
        :classes: btn-outline-primary

    
    :link-badge:`mca,MCA,ref,badge-success`
  
  .. dropdown:: *Barcelona*: Price of Anarchy

    .. link-button:: applications/traffic/barcelona_poa.html
        :text: Go to user guide
        :classes: btn-outline-primary

    We calculate user equilibrium and system optimum for Barcelona 
    with :class:`SymbolicCost <paminco.net.cost.SymbolicCost>` due to non-integer power factors.
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`
    :link-badge:`cost-symbolic,SymbolicCost,ref,badge-light`

.. tabbed:: Natural Gas

  .. dropdown:: GasLib: Gas24

    We compute the parametric edge flows for an affine demand function with MCA.

    .. link-button:: applications/gas/gas24.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`mca,MCA,ref,badge-success`
    :link-badge:`demandfunc-aff,"AffineDemandFunction",ref,badge-info`

  
.. tabbed:: Network Setup

  .. admonition:: Motivation

    You want to set up a custom graph with *paminco*? Learn how to specify graphs, 
    how to allow for negative flows (we treat all graphs as directed), and how to 
    specify cost and demand functions.

  .. dropdown:: *Graph Setup*: directed graphs (with negative flows)

    .. image:: ../_static/img/dir_graph.jpg
      :width: 400
      :align: center

    We set up directed graphs,
    learn about graph representations and how to set lower
    and upper boundaries for the edge flow in order to allow for negative edge flows 
    (thus creating undirected graphs).

    .. link-button:: network_setup/howto_graph_setup.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`~edges,Edges,"ref",badge-info`
    :link-badge:`~nodes,Nodes,"ref",badge-info`

  .. dropdown:: *Edge Cost*: polynomial, piecewise quadratic, or symbolic
  
      .. image:: /_static/img/tikz/fig2a_graph_with_marginal_cost.png
      :width: 400
      :align: center

    We show how to equip a network with cost functions, 
    i.e., how the cost :math:`F_e` of an edge :math:`e` vary by the flow on that edge. 

    .. link-button:: network_setup/howto_set_cost.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`cost-abstract,"NetworkCost (abstract)",ref,badge-light`
    :link-badge:`cost-poly,"PolynomialCost",ref,badge-info`
    :link-badge:`cost-piecewise,"PiecewiseQuadraticCost",ref,badge-info`
    :link-badge:`cost-symbolic,"SymbolicCost",ref,badge-info`
  
  .. dropdown:: *Demand Function*: linear and affine demand functions
    
    We illustrate how to specify the demand function for a network.
    Typically this is a linear demand funtion that scales the node rates (encoded with a demand vector :math:`\mathbf{b}`) 
    by a demand multiplier :math:`\lambda`: :math:`\mathbf{h}_\text{linear}(\lambda) = \lambda \mathbf{b}` 
    or an affine demand function that adds a base demand 
    :math:`\mathbf{h}_\text{affine}(\lambda) = \mathbf{b}_0  + \lambda \mathbf{b}`.

    .. link-button:: /notebooks/howto_set_demand.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`demandfunc,"DemandFunction (abstract)",ref,badge-light`
    :link-badge:`demandfunc-lin,"LinearDemandFunction",ref,badge-info`
    :link-badge:`demandfunc-aff,"AffineDemandFunction",ref,badge-info`
    
  .. dropdown:: *Read-in from XML*: How to read in network from XML
    
    .. link-button:: network_setup/howto_read_from_xml.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`~network,"Network",ref,badge-info`
  
  .. dropdown:: *Plotting*: How to specify node positions and plot with NetworkX

    We specify node positions and used them to plot a network
    with `NetworkX <https://networkx.org/>`_.

    .. link-button:: network_setup/howto_plot_network.html
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`~network,"Network",ref,badge-info`
    :link-badge:`~nodes,"Nodes",ref,badge-info`
  


.. toctree::
  :maxdepth: 2
  :hidden:

  applications/index
  network_setup/index
