.. _~network:

=======
Network
=======

Class Description
=================
.. currentmodule:: paminco.net
.. autoclass:: Network

Methods
=======

Constructors
------------
.. autosummary::
   :template: base_short.rst
   :toctree: generated/

   Network.from_xml
   Network.from_npz


Specifying demand and cost
--------------------------
.. autosummary::
   :template: base_short.rst
   :toctree: generated/

   Network.set_demand
   Network.set_cost
   Network.integrate_cost
   .. TODO: remove from network?
   Network.differentiate_cost


Math  
----
.. autosummary::
   :template: base_short.rst
   :toctree: generated/

   Network.adjacency_matrix
   Network.csgraph
   Network.incidence_matrix
   Network.Gamma
   Network.gamma_times
   Network.times_gamma
   Network.laplacian
   Network.L
   Network.Lstar
   Network.Lstar_update
   Network.potential


Path and Connectedness
----------------------
.. autosummary::
   :template: base_short.rst
   :toctree: generated/

   Network.shortest_path
   Network.flow_on_shortest
   Network.connected_components
   Network.is_connected
   Network.support_of


Cleaning
--------
.. autosummary::
   :template: base_short.rst
   :toctree: generated/

   Network.delete_edges
   Network.delete_nodes
   Network.clean


Attributes
==========
.. autosummary::
   :template: base_short.rst
   :toctree: generated/

   Network.cost
   Network.demand
   Network.shared
   Network.edges
   Network.flow_direction
   Network.is_directed
   Network.is_single_commodity
   Network.is_multi_commodity
   Network.nodes
   Network.n
   Network.m
   Network.k
   Network.size
   Network.dtype_int
   Network.dtype_float
