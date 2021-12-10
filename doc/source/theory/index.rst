.. _theory-index:

======================
Theoretical Background
======================

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    Minimum Cost Flows
    ^^^^^^^^^^^^^^^^^^

    .. math::

      \begin{align*}
      \min \text { } & C(\mathbf{x}) \\
      \text {s.t.} \quad \mathbf{\Gamma} \mathbf{x} &= \mathbf{b}, \\
      \mathbf{x} &\geq \mathbf{0}
      \end{align*}

    +++

    .. link-button:: min_cost_flow
        :type: ref
        :text: Learn more
        :classes: btn-block btn-secondary stretched-link

    ---
    Parametric Minimum Cost Flows
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Find minimum cost flows for parametric demands :math:`\mathbf{h}(\lambda)`
    with :math:`\lambda \in \left[0, \lambda^{\max }\right]`

    +++

    .. link-button:: par_mcf
        :type: ref
        :text: Learn more
        :classes: btn-block btn-secondary stretched-link


.. toctree::
  :hidden:
  
  min_cost_flow
  par_mcf
