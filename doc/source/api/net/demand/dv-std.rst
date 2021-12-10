.. _demandvec-std:

============
DemandVector
============

Class Description
=================
.. currentmodule:: paminco.net.demand
.. autoclass:: DemandVector

Methods
=======
Value
-----
.. autosummary::
   :toctree: generated/

   DemandVector.__call__
   DemandVector.sparse

Load / Save
-----------
.. autosummary::
   :toctree: generated/

   DemandVector.make_save_dict
   DemandVector.save_to_numpy
   DemandVector.from_npz
   DemandVector.add_to_etree
   DemandVector.from_xml

Misc
----
.. autosummary::
   :toctree: generated/

   DemandVector.delete_nodes
   DemandVector.to_single_pairs
   DemandVector.reset_cache
   DemandVector.map_node_label_to_id
   DemandVector.map_node_id_to_label
   DemandVector.scaled_copy

Attributes
==========
.. autosummary::
   :toctree: generated/

   DemandVector.k
   DemandVector.n
   DemandVector.shared
   DemandVector.dtype_int
   DemandVector.dtype_float
   DemandVector.all_single
   DemandVector.rate
   DemandVector.source_lbl
   DemandVector.sink_lbl
   DemandVector.source_id
   DemandVector.sink_id
   DemandVector.total_rate
