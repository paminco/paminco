.. _~shared:

======
Shared
======

Class Description
=================
.. currentmodule:: paminco.net
.. autoclass:: Shared

Methods
=======

.. autosummary::
   :toctree: generated/

   Shared.__init__
   Shared.update
   Shared.reset_cache
   Shared.delete_edges
   Shared.delete_nodes
   Shared.delete_nodes_in_edges
   Shared.incidence_matrix
   Shared.Gamma
   Shared.adjacency_matrix
   Shared.csgraph
   Shared.get_edge_id
   Shared.get_node_id
   Shared.get_node_label
   Shared.from_xml
   Shared.make_save_dict
   Shared.save_to_numpy
   Shared.from_npz


Attributes
==========
+-----------------+------------------------------------------------------------+
| edges           | Get :class:`Edges <paminco.net.shared.Edges>` object.      |
+-----------------+------------------------------------------------------------+
| nodes           | Get :class:`Nodes <paminco.net.shared.Nodes>` object.      |
+-----------------+------------------------------------------------------------+

.. autosummary::

   :toctree: generated/

   Shared.flow_direction
   Shared.n
   Shared.m
   Shared.nodes2edge
   Shared.node2id
   Shared.dtype_int
   Shared.dtype_float
