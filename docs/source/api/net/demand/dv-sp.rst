.. _demandvec-sp:

==============
DemandVectorSP
==============

Class Description
=================
.. currentmodule:: paminco.net.demand
.. autoclass:: DemandVectorSP

Methods
=======
Value
-----
.. autosummary::
   :toctree: generated/
   :template: base_short.rst

   DemandVectorSP.__call__
   DemandVectorSP.sparse

Load / Save
-----------
.. autosummary::
   :toctree: generated/
   :template: base_short.rst

   DemandVectorSP.make_save_dict
   DemandVectorSP.save_to_numpy
   DemandVectorSP.from_npz
   DemandVectorSP.add_to_etree
   DemandVectorSP.from_xml

Misc
----
.. autosummary::
   :toctree: generated/
   :template: base_short.rst

   DemandVectorSP.delete_nodes
   DemandVectorSP.to_single_pairs
   DemandVectorSP.reset_cache
   DemandVectorSP.map_node_label_to_id
   DemandVectorSP.map_node_id_to_label
   DemandVectorSP.scaled_copy

Attributes
==========
.. autosummary::
   :toctree: generated/
   :template: base_short.rst

   DemandVectorSP.k
   DemandVectorSP.n
   DemandVectorSP.shared
   DemandVectorSP.dtype_int
   DemandVectorSP.dtype_float
   DemandVectorSP.all_single
   DemandVectorSP.rate
   DemandVectorSP.source_lbl
   DemandVectorSP.sink_lbl
   DemandVectorSP.source_id
   DemandVectorSP.sink_id
   DemandVectorSP.total_rate