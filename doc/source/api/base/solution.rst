.. _parsolution:

==================
ParametricSolution
==================

Class Description
=================
.. currentmodule:: paminco._base
.. autoclass:: ParametricSolution

Attributes
==========
.. autosummary::
   :toctree: generated/

   ParametricSolution.has_potentials
   ParametricSolution.has_costs
   ParametricSolution.arr_param
   ParametricSolution.arr_flow
   ParametricSolution.arr_cost
   ParametricSolution.arr_potential
   ParametricSolution.dflow
   ParametricSolution.dpi

Methods
=======

Value, Plot and Param
---------------------
.. autosummary::
   :toctree: generated/

   ParametricSolution.flow_at
   ParametricSolution.potential_at
   ParametricSolution.all_params
   ParametricSolution.plot_flow_on_edge

Save and Load
-------------
.. autosummary::
   :toctree: generated/

   ParametricSolution.from_arrays
   ParametricSolution.to_df
   ParametricSolution.from_df
   ParametricSolution.to_csv
   ParametricSolution.from_csv
   ParametricSolution.make_save_dict
   ParametricSolution.save_to_numpy
   ParametricSolution.from_npz
