.. _demandfunc:

==========================
Demand Function (Abstract)
==========================

Class Description
=================
.. currentmodule:: paminco.net.demand
.. autoclass:: DemandFunction

Attributes
==========
.. autosummary::
   :toctree: generated/
   :template: base_short.rst

   DemandFunction._dv0
   DemandFunction.all_single_pairs
   DemandFunction.n
   DemandFunction.shared
   DemandFunction.dtype_int
   DemandFunction.dtype_float
   DemandFunction.demand_vectors
   DemandFunction.is_single_commodity
   DemandFunction.is_multi_commodity
   DemandFunction.value_at_1
   DemandFunction.unique_sources

Methods
=======
.. autosummary::
   :toctree: generated/
   :template: base_short.rst

   DemandFunction.reset_cache
   DemandFunction.inflow
   DemandFunction.ddx_inflow
   DemandFunction.max_inflow
   DemandFunction.delete_nodes
   DemandFunction.map_node_label_to_id
   DemandFunction.map_node_id_to_label
   DemandFunction.to_single_pairs
   DemandFunction.add_to_etree
   DemandFunction.from_xml
   DemandFunction.make_save_dict
   DemandFunction.save_to_numpy
   DemandFunction.from_npz