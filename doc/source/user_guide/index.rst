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
    
    .. image:: /_static/img/Braess_paradox_road_example.svg
        :width: 700
        :alt: Parametric User Equilibria for Sioux Falls
        :align: center

    Adding capacity to a network might decrease network efficiency due to selfish participants.
    
    .. link-button:: applications/traffic/braess_paradox.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`
  
  .. dropdown:: *Pigou's Example*: An upper bound for the Price of Anarchy

    .. link-button:: applications/traffic/pigou_example.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`
  
  .. dropdown:: *SiouxFalls*: calculating and plotting a fixed user equilibrium

    .. link-button:: applications/traffic/sioux_fixed_ue.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`
  
  .. dropdown:: *SiouxFalls*: parametric Price of Anarchy

    .. link-button:: applications/traffic/sioux_par_poa.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`mca,MCA,ref,badge-success`
  
  .. dropdown:: *Barcelona*: Price of Anarchy

    .. link-button:: applications/traffic/barcelona_poa.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`

.. tabbed:: Natural Gas

  .. dropdown:: *SiouxFalls*: calculating and plotting a fixed user equilibrium

    .. link-button:: applications/traffic/sioux_fixed_ue.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`fw-net,NetworkFW,ref,badge-success`

  
.. tabbed:: Network Setup

  .. admonition:: Motivation

    You want to setup a custom graph with *paminco*? Learn how to specify graphs, 
    how to allow for negative flows (we treat all graphs as directed), and how to 
    specify cost and demand functions.

  .. dropdown:: *Graph Setup*: directed graphs (with negative flows)

    We setup directed graphs,
    learn about graph representations and how to set lower
    and upper boundaries for the edge flows in order to allow for negative edge flows 
    (thus creating undirected graphs).

    .. image:: ../_static/img/dir_graph.jpg
      :width: 300
      :align: center

    .. link-button:: network_setup/howto_graph_setup.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`edges,Edges,"ref",badge-info`
    :link-badge:`nodes,Nodes,"ref",badge-info`

  .. dropdown:: *Edge Cost*: polynomial, piecewise quadratic, or symbolic
    
    We show how to equip a network with cost functions, 
    i.e., how the cost :math:`F_e` of an edge :math:`e` vary by the flow on that edge. 

    .. image:: /_static/img/tikz/fig2a_graph_with_marginal_cost.png
      :width: 300
      :align: center

    .. link-button:: network_setup/howto_set_cost.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`api-network-cost-NetworkCost,"NetworkCost (abstract)",ref,badge-light`
    :link-badge:`api-network-cost-PolynomialCost,"PolynomialCost",ref,badge-info`
  
  .. dropdown:: *Demand Function*: linear and affine demand functions
    
    We illustrate how to specify the demand function for a network.
    Typically this is a linear demand funtion that scales the node rates (encoded with a demand vector :math:`\mathbf{b}`) 
    by a demand multiplier :math:`\lambda`: :math:`\mathbf{h}_\text{linear}(\lambda) = \lambda \mathbf{b}` 
    or an affine demand function that adds a base demand 
    :math:`\mathbf{h}_\text{affine}(\lambda) = \mathbf{b}_0  + \lambda \mathbf{b}`.

    .. link-button:: /notebooks/howto_set_demand.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`demandfunc,"DemandFunction (abstract)",ref,badge-light`
    :link-badge:`demandfunc-lin,"LinearDemandFunction",ref,badge-info`
    :link-badge:`demandfunc-aff,"AffineDemandFunction",ref,badge-info`
    
  .. dropdown:: *Read-in from XML*: How to read in network from XML
    
    .. link-button:: network_setup/howto_read_from_xml.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`network,"Network","ref",badge-info`
  
  .. dropdown:: *Plotting*: How to specify node positions and plot with networkx

    We specify node positions and used them to plot a network
    with `NetworkX <https://networkx.org/>`_.

    .. image:: /_static/img/sioux_ue.gif
      :width: 400
      :align: center
    
    .. link-button:: network_setup/howto_plot_network.ipynb
        :text: Go to user guide
        :classes: btn-outline-primary
    
    :link-badge:`network,"Network","ref",badge-info`
    :link-badge:`nodes,"Nodes","ref",badge-info`
  


.. toctree::
  :maxdepth: 2
  :hidden:

  applications/index
  network_setup/index
