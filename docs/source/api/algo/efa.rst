.. _efa:

==============================
EFA: Electrical Flow Algorithm
==============================
.. currentmodule:: paminco.algo.efa

Class Description
=================
.. autoclass:: EFA

Settings and Subclasses
=======================
.. autosummary::
   :toctree: generated/ 
   
   EFAConfig
   NodePotentials
   EFAEdges
   EFABreakFlag
   PivotStepMode

Attributes
==========
.. autosummary::
   :toctree: generated/

   EFA.network
   EFA.name
   EFA.config
   EFA.region
   EFA.edges
   EFA.edge_coeffs
   EFA.node_potentials
   EFA.L0
   EFA.region_zero

Methods
=======
.. autosummary::
   :toctree: generated/

   EFA.__init__
   EFA.run
