from __future__ import annotations

import warnings
import xml.etree.ElementTree as et
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sps

from .cost import NetworkCost, PolynomialCost, SymbolicCost, PiecewiseQuadraticCost
from .demand import (
    DemandFunction,
    LinearDemandFunction,
    AffineDemandFunction,
)
from .path import csr_dijkstra, csr_dijkstra_mp, get_path_edges
from .shared import Shared, Edges, Nodes, FlowDirection
from ._convert_traffic import read_tntp
from ._convert_gas import gaslib_to_network_data
from paminco.utils.io import prettify_xml
from paminco.linalg import (
    SingularLaplaceError,
    star_inv,
    star_update_by_edge,
    InverseMethod,
)
from paminco.utils.misc import Cache
from paminco.utils.typing import is_iterable, sparse_format
from paminco import _doc
import paminco.net.demand as netdemand


class Support:
    """A class representing the support (of a flow).
    
    Parameters
    ----------
    ecost : ndarray
        The edge cost of the flow.
    p_diff : ndarray
        The potential differences of the flow.
    
    Attributes
    ----------
    rtol : float, default=1e-5
        The relative tolerance when comparing ecost - p_diff to zero.
    atol : float, default=1e-8
        The absolute tolerance when comparing ecost - p_diff to zero.
    active
    inactive
    
    See Also
    --------
    numpy.isclose
    """

    def __init__(
            self,
            ecost: np.ndarray,
            p_diff: np.ndarray,
            rtol: float = 1e-5,
            atol: float = 1e-8,
            ) -> None:
        self.slack = ecost - p_diff
        self.rtol = rtol
        self.atol = atol

    def __contains__(self, edge) -> bool:
        """Return True if edge is an active edge of this edge."""
        return edge in self.active

    def __eq__(self, other) -> bool:
        """Compare this support to other support or array of edges."""
        if isinstance(other, Support):
            for e in other.active:
                if e not in self:
                    return False
            return True
        if isinstance(other, (list, np.ndarray, tuple)):
            for e in other:
                if e not in self:
                    return False
            return True
        raise TypeError(
            f"Support cannot be compared to object of type {type(other)}."
        )
    
    def __len__(self) -> int:
        """Return the number of active edges."""
        return len(self.active)

    def __str__(self) -> str:
        """Return a string representation of this support."""
        out = "<Support> Active Edges: "
        out += str(list(self.active))
        return out

    def _active(self) -> np.ndarray:
        return np.isclose(self.slack, 0, rtol=self.rtol, atol=self.atol)

    @property
    def active(self) -> np.ndarray:
        """ndarray of int: indices of the active edges."""
        return np.arange(len(self.slack))[self._active()]

    @property
    def inactive(self) -> np.ndarray:
        """ndarray of int: indices of the inactive edges."""
        return np.arange(len(self.slack))[np.logical_not(self._active())]


class Network:
    """A Network.
    
    Parameters
    ----------
    edge_data : edge_data
        Data that specifies edges. See also
        :class:`paminco.net.shared.Edges`.
    node_data : node_data, optional
        Data that specifies nodes. If None, nodes will be inferred from
        ``edge_data``. See also  :class:`paminco.net.shared.Nodes`
    cost_data : cost_data, optional
        Edge costs. If None, linear cost functions ``F_e(x_e) = x_e``
        are assumed for all edges.
    demand_data : demand_data, optional
        Commodity data to initialize a DemandFunction. See also
        :meth:`Network.set_demand() <Network.set_demand>`.
    dtype_float : dtype, default=float
        Datatype for float data arrays.
    dtype_int : dtype, default=int
        Datatype for integer data arrays.
    update_shared : bool, default=True
        Whether to update shared data after initialization.
    directed_flow : bool, default=True
        Controls default values for ``None`` in edge bounds. If ``True``,
        lower bounds are set to 0 and ``False`` to -inf. Missing upper
        bounds are set to inf.
    demand_mode : str, default="linear"
        How to interpret ``demand_data``, see also
        :meth:`Network.set_demand() <Network.set_demand>`.
    kw_edge : keyword arguments, optional
        Further keyword arguments passed to Edge constructor.
    kw_demand : keyword arguments, optional
        Further keyword arguments passed to
        :meth:`Network.set_demand() <Network.set_demand>`.
    kw_cost : keyword arguments, optional
        Further keyword arguments passed to
        :meth:`Network.set_cost() <Network.set_cost>`.
    
    See Also
    --------
    paminco.net.shared.Shared : Nodes, edge initialization.
    Network.set_demand : Demand initialization.
    Network.set_cost : Cost initialization.
    
    Examples
    --------
    """

    # TODO: def _check_consistency...
    def __init__(
            self,
            edge_data,
            node_data=None,
            cost_data=None,
            demand_data=None,
            dtype_float=float,
            dtype_int=int,
            update_shared: bool = True,
            directed=None,
            demand_mode=None,
            clean=None,
            kw_edge=None,
            kw_demand=None,
            kw_cost=None,
            ) -> None:
        if kw_edge is None:
            kw_edge = {}
        if directed is not None and "directed" not in kw_edge:
            kw_edge["directed"] = directed
            
        if kw_demand is None:
            kw_demand = {}
        if demand_mode is not None and "mode" not in kw_demand:
            kw_demand["mode"] = demand_mode
        
        if kw_cost is None:
            kw_cost = {}
        
        # init shared object (edges, nodes)
        self._s = Shared(edge_data,
                         node_data,
                         dtype_float=dtype_float,
                         dtype_int=dtype_int,
                         **kw_edge)
        
        # Init demand if given
        if demand_data is not None:
            self.set_demand(demand_data, **kw_demand)
            
        # Init costs: if not given, linear cost functions with slope of
        # one are assumed
        if cost_data is None:
            cost_data = np.c_[np.zeros(self.m), np.full(self.m, False)]
        self.set_cost(cost_data, **kw_cost)
        
        # Init cache
        self.cache = Cache()
        
        # Clean network
        if clean is True:
            clean = [
                "zones",
                "parallel_edges",
                "zero_cost_edges",
                "isolated_nodes",
                "unreachable_nodes",
                "commodities",
            ]
        elif isinstance(clean, str):
            clean = [clean]
        if is_iterable(clean):
            clean_kw = {("remove_" + k): True for k in clean}
            self.clean(**clean_kw)
        
        # Update shared network object for id <-> lbl mappings
        if update_shared is True:
            self.update_shared()
        
    def __eq__(self, other) -> bool:
        if isinstance(other, Network) is False:
            return False
        for att in ["_s", "_c", "_d"]:
            if getattr(self, att) != getattr(other, att):
                return False
        return True

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "Network with {:d} nodes and {:d} edges.".format(self.n, self.m)

    def reset_cache(self, hard: bool = False) -> None:
        self.cache.reset(hard=hard)

    def h(self, param: float = 1) -> sps.csc_matrix:
        """Demand function."""
        return self.demand(param)

    def set_demand(
            self,
            data=None,
            is_label: bool = True,
            map_label: bool = True,
            mode: str = "linear",
            **kwargs
            ) -> None:
        """Set demand of network.
        
        Parameters
        ----------
        data : demand-like, optional
            If ``data`` is a
            :class:`~paminco.net.demand.DemandFunction`,
            demand is set to ``data``. Else, demand will be set based
            on ``mode``.
        is_label : bool, default=True
            Whether commodities in ``data`` have labels.
        map_label : bool, default=True
            Whether to map node labels in ``data`` to node ids.
        mode : str, default='linear'
            1.'linear':
              ``data`` is used to build
              :class:`~paminco.net.demand.LinearDemandFunction`.
            2.'affine':
              ``data`` is used to build
              :class:`~paminco.net.demand.AffineDemandFunction`.
            3.'to_linear':
               Use current demand direction to build
               :class:`~paminco.net.demand.LinearDemandFunction`,
               ``data`` must be None.
            4.'to_affine':
               Build
               :class:`~paminco.net.demand.AffineDemandFunction`
               where base demand is set by ``data``. Demand direction
               is used from current demand function.
        kwargs : keyword arguments, optional
            Arguments passed to LinearDemandFunction or
            :class:`~paminco.net.demand.AffineDemandFunction`.
        
        Raises
        ------
        ValueError
            If ``mode == 'affine'`` and ``data`` is not a tuple of
            len = 2.
        ValueError
            If ``mode == 'to_linear'`` and ``data`` is not None.
        TypeError
            If ``mode == 'to_affine'`` and demand of network is not a
            :class:`~paminco.net.demand.LinearDemandFunction`.
        ValueError
            If ``mode`` is not in ['linear', 'affine', 'to_linear',
            'to_affine].
        
        Examples
        --------
        Setting a linear demand function that consists of two
        commodities. A commodity may be specified by a tuple (source,
        target, rate) or a dict {source: -rate, target1: rate/2,
        target2: rate/2}.
        
        >>> net = paminco.net.load_sioux()
        >>> net.set_demand([(1, 2, 400), {7: -100, 4: 100}], is_label=False)
        >>> net.demand(1.5).toarray()[:8]
        array([[   0.,    0.],
               [-600.,    0.],
               [ 600.,    0.],
               [   0.,    0.],
               [   0.,  150.],
               [   0.,    0.],
               [   0.,    0.],
               [   0., -150.]])
               
        Setting an affine demand function.
        
        >>> net = paminco.net.load_sioux()
        >>> net.set_demand(((1, 2, 400), (7, 4, 100)), is_label=False, mode="affine")
        >>> net.demand(3).toarray()[:8]
        array([[   0.],
               [-400.],
               [ 400.],
               [   0.],
               [ 300.],
               [   0.],
               [   0.],
               [-300.]])
        """
        if isinstance(data, DemandFunction):
            # Data is demand func
            self._d = data
        elif isinstance(data, str):
            # Data will be read from xml
            self._d = DemandFunction.from_xml(data,
                                              shared=self.shared,
                                              is_label=is_label,
                                              **kwargs)
        elif mode == "linear":
            self._d = LinearDemandFunction(data,
                                           shared=self.shared,
                                           is_label=is_label,
                                           **kwargs)
        elif mode == "affine":
            if isinstance(data, tuple) is False or len(data) != 2:
                raise ValueError(f"mode=='{mode}' requires tuple of data.")
            self._d = AffineDemandFunction(*data,
                                           shared=self.shared,
                                           is_label=is_label,
                                           **kwargs)
        elif mode == "to_linear":
            if data is not None:
                raise ValueError(f"mode=='{mode}' accepts no data.")
            self._d = LinearDemandFunction(self.demand.b,
                                           shared=self.shared,
                                           is_label=is_label,
                                           **kwargs)
        elif mode == "to_affine":
            if isinstance(self.demand, LinearDemandFunction) is False:
                raise TypeError(
                    f"mode=='{mode}' requires demand to be of type "
                    "LinearDemandFunction."
                )
            self._d = AffineDemandFunction(b0=data,
                                           b=self.demand.b,
                                           shared=self.shared,
                                           is_label=is_label,
                                           **kwargs)
        else:
            raise ValueError(f"mode=='{mode}' not valid.")
        
        if map_label is True:
            if is_label is False:
                self._d.map_node_id_to_label()
            else:
                self._d.map_node_label_to_id()

    def C(self, flow: np.ndarray) -> float:
        return self.cost(flow).sum()

    def set_cost(self, edge_cost, cost_type=None, **kw) -> None:
        """Set edge costs.
        
        Parameters
        ----------
        edge_cost : edge_cost
            Cost of edges.
        cost_type : str, optional
            Type of cost data.
            TODO-PW
        kw : keyword arguments
            Further keyword arguments passed to cost constructor.
        
        See Also
        --------
        paminco.net.cost.NetworkCost.from_data
        """
        if (isinstance(edge_cost, str)
                or isinstance(edge_cost, et.ElementTree)
                or isinstance(edge_cost, et.Element)):
            # Read from XML
            self._c = NetworkCost.from_xml(edge_cost, shared=self.shared)
        elif isinstance(edge_cost, NetworkCost):
            # Is already network cost
            self._c = edge_cost
            self._c._s = self.shared
        else:
            if (cost_type is None
                    and isinstance(edge_cost, tuple)
                    and isinstance(edge_cost[0], dict)):
                cost_type = "symbolic"
            
            if cost_type == "symbolic":
                self._c = SymbolicCost(edge_cost[0], shared=self.shared, **edge_cost[1], **kw)
            elif cost_type == "piecewisequadratic":
                self._c = PiecewiseQuadraticCost(edge_cost, shared=self.shared, **kw)
            else:
                # If data is given, assume polynomial cost
                self._c = PolynomialCost(edge_cost, shared=self.shared, **kw)

    def integrate_cost(self, *args, **kw) -> None:
        """Integrate edge costs.
        
        See Also
        --------
        paminco.net.cost.PolynomialCost.integrate
        paminco.net.cost.PiecewiseQuadraticCost.integrate
        """
        self.cost.integrate(*args, inplace=True, **kw)

    def derivate_cost(self, *args, **kw) -> None:
        """Derivate edge costs.
        
        See Also
        --------
        paminco.net.cost.PolynomialCost.derivate
        paminco.net.cost.PiecewiseQuadraticCost.derivate
        """
        if isinstance(self.cost, PolynomialCost):
            self.cost.derivate(d=1, inplace=True)

    def adjacency_matrix(self, *args, **kw) -> sps.csr_matrix:
        """Alias for :func:`~Network.csgraph`."""
        return self.shared.csgraph(*args, **kw)
    
    def csgraph(self, *args, **kw) -> sps.csr_matrix:
        return self.shared.csgraph(*args, **kw)
    
    csgraph.__doc__ = Shared.csgraph.__doc__

    def incidence_matrix(self, *args, **kwargs) -> sps.spmatrix:
        """Alias for :func:`~Network.Gamma`."""
        return self.Gamma(*args, **kwargs)
    
    def Gamma(self, *args, **kwargs) -> sps.spmatrix:
        return self.shared.Gamma(*args, **kwargs)
    
    Gamma.__doc__ = Shared.Gamma.__doc__

    def gamma_times(
            self,
            x,
            node=None,
            return_as: str = 'csr'
            ) -> sps.spmatrix:
        """Calculate Gamma @ x.
        
        Parameters
        ----------
        x : ndarray
            Argument.
        node : int, optional
            If given, result of dot product from row slice
            Gamma[node, :] and x is returned::
            
                Gamma[node, :] @ x
                
        return_as : str, default="csr"
            Return type of dot product if result is spmatrix.
        
        Returns
        -------
        ndarray or spmatrix
            Dot product of Gamma and x.
            
        See Also
        --------
        numpy.dot
        """
        G = self.Gamma(return_as=return_as)
        if node is None:
            out = G @ x
        else:
            out = G[node, :].dot(x)
            
        if sps.isspmatrix(out):
            return sparse_format(out, return_as)
        
        return out

    def times_gamma(
            self,
            x,
            edge=None,
            return_as: str = 'csr'
            ):
        """Calculate x @ Gamma.
        
        Parameters
        ----------
        x : ndarray or spmatrix
            Argument.
        edge : int, optional
            If given, result of dot product of x and column slice is
            returned::
            
                x @ Gamma[:, edge]
            
        return_as : str, default="csr"
            Return type of dot product if result is spmatrix.
        
        Returns
        -------
        ndarray or spmatrix
            Dot product of x and Gamma.
        
        See Also
        --------
        numpy.dot
        """
        if edge is None:
            # Standard computation via dot product
            out = x @ self.Gamma(return_as='csr')
        else:
            # Use sparsity of Gamma
            s, t = self.edges.indices[edge].T
            if len(x.shape) == 1:
                out = x[t] - x[s]
            else:
                out = x[:, t] - x[:, s]
        
        if sps.isspmatrix(out):
            return sparse_format(out, return_as)
        
        return out

    def laplacian(
            self,
            weight=None,
            reduced: bool = False,
            return_as: str = 'csr'
            ) -> sps.spmatrix:
        """Get the (weighted) Laplacian matrix L.
        
        Parameters
        ----------
        weight : float or np.ndarray, optional
            If None, unit weight is assumed for all edges. If float,
            all edges have the same weight.
        reduced : bool, default=False
            Whether to return the reduced form, i.e., without first row
            and first column.
        return_as : str, default='csr'
            Return format for sparse matrix.
        
        Returns
        -------
        L : spmatrix
            Laplacian matrix of shape (n, n) if ``reduced`` is False,
            else of shape (n-1, n-1). Type of return matrix is
            determined by parameter ``return_as``.
        
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Laplacian_matrix
        """
        # Cast weight to array of len(m) if not np.ndarray
        if weight is None:
            w = np.ones(self.m)
        elif isinstance(weight, float):
            w = np.full(self.m, weight, dtype=self.dtype_float)
        elif isinstance(weight, np.ndarray):
            w = weight
        else:
            raise TypeError(f"Invalid argument for weight: {weight}.")
        
        w = np.hstack([-w, -w, w, w])
        if self.cache.is_valid("lap") is False:
            # Rebuild if not valid in cache
            s, t = self.edges.indices.T
            i = np.hstack([s, t, s, t])
            j = np.hstack([t, s, s, t])
            coo = sps.coo_matrix((w, (i, j)), shape=(self.n, self.n))
            self.cache["lap"] = coo
        else:
            # Pull from cache
            coo = self.cache["lap"]
            coo.data = w
            
        if reduced is True:
            return sparse_format(coo, return_as)[1:, 1:]
        return sparse_format(coo, return_as)
    
    def L(
            self,
            weight=None,
            flow=None,
            reduced: bool = False,
            return_as: str = 'csr',
            **kwargs
            ) -> sps.spmatrix:
        r"""Get weighted Laplacian.
        
        Parameters
        ----------
        weight : np.ndarray, optional
            If None, weigths are determined based on flow.
            The Laplace weight of an edge e for a flow x_e is defined
            as ``1 / c''(x_e)``, where c'' is the second derivative of
            the edge costs.
        flow : np.ndarray, optional
            Use flow to determine weights if ``weights is None``.
        reduced : bool, default=False
            Whether to return the reduced form, i.e., without first row
            and first column.
        return_as : str, default='csr'
            Return format for sparse matrix.
        
        Returns
        -------
        spmatrix
            Laplacian matrix of shape (n, n) if ``reduced`` is False,
            else of shape (n-1, n-1). Type of return matrix is
            determined by parameter ``return_as``.
        
        See Also
        --------
        Network.laplacian
        paminco.net.cost.NetworkCost.laplace_weights
        paminco.net.cost.PiecewiseQuadraticCost.laplace_weights
        """
        if weight is None:
            weight = self.cost.laplace_weights(flow, **kwargs)
        return self.laplacian(weight, reduced, return_as)

    def Lstar(
            self,
            weight=None,
            flow=None,
            reduced: bool = False,
            method: InverseMethod = InverseMethod.CHOLESKY,
            safe: bool = True,
            **kwargs
            ):
        """Pseudo-inverse of weighted Laplacian.
        
        Parameters
        ----------
        weight : np.ndarray, optional
            If None, weigths of Laplacian are determined based on flow.
        flow : np.ndarray, optional
            Use flow to determine weights if ``weights is None``.
        reduced : bool, default=False
            Whether to return the reduced form, i.e., without first row
            and first column.
        method: int, str, or InverseMethod, default=InverseMethod.CHOLESKY
            Method used to calculate inverse.
        safe : bool, default=True
            If True (default), matrix will only be inverted if network is
            connected.
        
        Returns
        -------
        ndarray or CholeskyInverse
            The pseudo inverse of the weighted Laplacian.
        
        Raises
        ------
        SingularLaplaceError
            Network is not connected.
        numpy.linalg.LinAlgError
            Inverting matrix failed.
        
        See Also
        --------
        Network.connected_components : Connectedness of the network.
        Network.L : Laplacian matrix.
        paminco.linalg.star_inv : generalized inverse of a matrix.
        paminco.linalg.CholeskyInverse : Inverting a matrix using choleksy decomposition.
        """
        method = InverseMethod.make(method)
        
        # Safemode checks connectedness of graph beforehand in order to
        # detect singular laplace matrix
        if safe is True:
            if weight is None:
                weight = self.cost.laplace_weights(flow, **kwargs)
            n_cc, cc = self.connected_components(np.where(weight != 0)[0])
            if n_cc > 1:
                raise SingularLaplaceError(network=self,
                                           weight=weight,
                                           n_cc=n_cc,
                                           cc=cc)
        
        lap = self.L(weight, flow, reduced=reduced, return_as='array', **kwargs)
        raise_laplace_error = False
        
        try:
            lstar = star_inv(lap, reduced=reduced, method=method)
        except np.linalg.LinAlgError as lae:
            # Laplace inverse can not be obtained
            if weight is None:
                # Recover the weights used, if not specified
                weight = self.cost.laplace_weights(flow, **kwargs)
            if not self.is_connected(edges=np.where(weight != 0)[0]):
                # If the error is due to multiple connected components
                # of active edges, raise SingularLaplaceError
                raise_laplace_error = True
            else:
                # Unknown source of LinAlgError -> Raise the normal error
                raise lae
            
        if raise_laplace_error:
            raise SingularLaplaceError(network=self, weight=weight)
        
        return lstar

    def Lstar_update(self, *args, **kwargs) -> None:
        """Update pseudo-inverse of weighted Laplacian.
        
        See Also
        --------
        Network.Lstar
        paminco.linalg.star_update_by_edge
        """
        try:
            updated_matrix = star_update_by_edge(self, *args, **kwargs)
        except np.linalg.LinAlgError:
            raise SingularLaplaceError(network=self)
        return updated_matrix

    def shortest_path(
            self,
            weight=None,
            s=None,
            backward_edges: bool = False,
            backward_positive: bool = False,
            multiprocessing: bool = False,
            return_source_indices: bool = False,
            ):
        """Compute shortest path wrt. weight.
        
        Parameters
        ----------
        weight : array_like, optional
            Edge weights, default: unit weight (1) for all edges.
        s : int or array_like, optional
            If specified, only compute the paths originating at the
            given indices. Default: all nodes in edges.
        backward_edges : bool, default=False
            Whether to use separate entries in adjacency matrix H for
            undirected edges, i.e., H[s, t] = w and H[t, s] = w.
        backward_positive : bool, default=False
            Whether to negate weight for undirected edges, i.e.,
            H[s, t] = w and H[t, s] = -w.
        multiprocessing : bool, default=False
            Whether to calculate D and Pr using multiple processes.
        return_source_indices : bool, default=False
            Whether to return dict that maps node_id in ``s`` to indices
            in return matrices D and Pr.
        
        Returns
        -------
        D : ndarray
            Distance matrix.
        Pr: ndarray
            Predecessor matrix.
        dict, optional
            Mapping of node_id in ``s`` to indices in D and Pr if
            ``return_source_indices`` is True.
        
        See Also
        --------
        paminco.net.shared.Shared.csgraph :
            Setup of adjacency matrix used to find shortest path.
        scipy.sparse.csgraph.dijkstra : Shortest path computation.
        """
        # set weight to ones if not given
        if weight is None:
            weight = np.ones(self.shared.m)
        
        # get adjacency matrix form edge class
        adj = self.shared.csgraph(weight,
                                  respect_bounds=backward_edges,
                                  backward_positive=backward_positive)
        
        # set sources to all nodes in network if not given
        if s is None:
            s = np.arange(adj.shape[0])
            
        if adj.data.min() >= 0:
            if sys.platform.startswith('win') is False and multiprocessing is True:
                D, Pr = csr_dijkstra_mp(adj, indices=s)
            else:
                D, Pr = csr_dijkstra(adj, indices=s)
        else:
            try:
                D, Pr = sps.csgraph.shortest_path(adj,
                                                  indices=s,
                                                  return_predecessors=True)
            except sps.csgraph.NegativeCycleError as ex:
                self._last_adj_matrix = adj
                raise ex
        
        if return_source_indices is True:
            return D, Pr, dict(zip(s, np.arange(len(s))))
        return D, Pr

    def flow_on_shortest(
            self,
            demand_triples,
            weight=None,
            unique_sources=None,
            multiprocessing: bool = False,
            commodity_wise: bool = False,
            ):
        """Find flow on shortest path.
        
        Parameters
        ----------
        weight : ndarray, optional
            Weight to compute predecessor matrix.
        demand_triples : list of tuple
            Find flow on shortest path for all commoditiies specified by
            ``(source, target, demand)``.
        unique_sources : ndarray, optional
            If specified, only compute the paths from the nodes at the
            given indices.
        multiprocessing : bool, default=False
            Whether to use multiprocessing to compute the shortest path
            matrices D and Pr.
        commodity_wise : bool, default=False
            Whether to return to return flow on shortest path for each
            commodity individually, i.e., a matrix F of shape (m, c) is
            returned where F[:, i] is the flow for commodity at index
            i. Warning: is considerably slower for a large c.
        
        Returns
        -------
        ndarray or lil_matrix
            Aggregated or commodity wise flow.
        
        """
        _, Pr, d = self.shortest_path(weight,
                                      s=unique_sources,
                                      multiprocessing=multiprocessing,
                                      return_source_indices=True)
        if Pr.shape[0] == 1:
            Pr = Pr.ravel()
        
        # For all commodities: update flow on shortest path between pairs
        # (s -> t with rate r)
        if commodity_wise is True:
            # Individual flow for each commodity
            flow = sps.lil_matrix((self.shared.m, len(demand_triples)))
            for (s, t, r), idx in enumerate(demand_triples):
                path_edges = get_path_edges(Pr=Pr,
                                            s=d[s],
                                            t=t,
                                            lookup_id=self.shared.nodes2edge,
                                            reversed=True)
                flow[path_edges, idx] = r
        else:
            # Aggregated edge flow across commodities
            flow = np.zeros(self.shared.m)
            for (s, t, r) in demand_triples:
                path_edges = get_path_edges(Pr=Pr,
                                            s=d[s],
                                            t=t,
                                            lookup_id=self.shared.nodes2edge,
                                            reversed=True)
                flow[path_edges] += r
        
        return flow

    def connected_components(
            self,
            edges=None,
            return_labels: bool = True
            ):
        """Extract connected components from network.
        
        Parameters
        ----------
        edges : int or array_like, optional
            If not None, get connected components for sub-network build
            from ``edges``.
        return_labels : bool, default=True
            If True (default), return the labels for each of the
            connected components.
        
        Returns
        -------
        n_components: int
            The number of connected components.
        labels: ndarray, optional
            The length-N array of labels of the connected components.
        
        See Also
        --------
        scipy.sparse.csgraph.connected_components
        
        Examples
        --------
            >>> import paminco
            >>> net = paminco.net.load_sioux()
            >>> net.delete_nodes(["6", "9", "11", "13"], is_label=True)
            >>> net.connected_components(return_labels=False)
            2
        """
        if edges is None:
            if self.cache.is_valid("csgraph_unitweight") is False:
                # Rebuild adj
                s, t = self.shared.edges.indices.T
                w = np.ones(self.m)
                adj = sps.coo_matrix((w, (s, t)), (self.n, self.n))
                self.cache["csgraph_unitweight"] = adj
            else:
                # Pull unit weighted adj from cache
                adj = self.cache["csgraph_unitweight"]
        else:
            s, t = self.shared.edges.indices[edges, :].T
            adj = sps.coo_matrix((np.ones(len(s)), (s, t)), (self.n, self.n))
        
        return sps.csgraph.connected_components(adj,
                                                directed=False,
                                                return_labels=return_labels)

    def is_connected(self, edges=None) -> bool:
        """Check whether edges are connected.
        
        Parameters
        ----------
        edges : int or array_like, optional
            Indices of network edges to check connectedness for.
        
        Returns
        -------
        bool
            Whether edges are connected.
        
        See Also
        --------
        Network.connected_components
        
        Examples
        --------
            >>> import paminco
            >>> net = paminco.net.load_sioux()
            >>> net.delete_nodes(["6", "9", "11", "13"], is_label=True)
            >>> net.is_connected()
            False
        """
        num_cc = self.connected_components(edges, return_labels=False)
        return (num_cc == 1)

    def support_of(
            self,
            flow: np.ndarray
            ) -> Support:
        """Get the support of the flow.
        
        Parameters
        ----------
        flow : numpy.ndarray
            Flow to find support for
            
        Returns
        -------
        Support
        """
        cost = self.cost.ddx(flow)
        cost[self.edges.lb < 0] = 0
        
        # Compute potential
        D, _ = self.shortest_path(weight=cost,
                                  s=0,
                                  backward_edges=True)
        
        gamma_pi = self.times_gamma(D)
        return Support(cost, gamma_pi)

    def potential(self, flow: np.array, fixed_vertex: int = 0) -> np.ndarray:
        """Calculate potential of a flow.
        
        Parameters
        ----------
        flow : ndarray
            Flow on network edges.
        fixed_vertex : int, default=0
            Specifies which node/vertex used as reference.
        
        Returns
        -------
        pot : ndarray
            Potential of flow.
        
        See Also
        --------
        Network.shortest_path
        """
        cost = self.cost.ddx(flow)
        pot, _ = self.shortest_path(weight=cost,
                                    s=fixed_vertex,
                                    backward_edges=True)
        return pot

    def _fullfills_flow_cond(
            self,
            flow: np.ndarray,
            param: float = 1,
            return_actual_target: bool = False,
            **kwargs,
            ):
        # TODO-PW umbennen, etc
        # Check whether total in/outflow in nodes corresponds to demand vector:
        # Gamma * flow = b
        actual = self.gamma_times(flow)
        target = np.ravel(self.demand(param).sum(axis=1))
        valid = np.isclose(actual, target, **kwargs).all()
        if return_actual_target is True:
            return valid, (actual, target)
        return valid

    def clean(
            self,
            remove_zones: bool = False,
            remove_parallel_edges: bool = True,
            remove_zero_cost_edges: bool = True,
            remove_isolated_nodes: bool = False,
            remove_unreachable_nodes: bool = False,
            remove_commodities: bool = False,
            start_node_unreachables=None,
            ) -> str:
        """Clean network.
        
        Parameters
        ----------
        remove_zones : bool, default=False
            Whether to remove all nodes in network.nodes that are
            flagged as a zone node.
        remove_parallel_edges : bool, default=True
            Whether to remove parallel edges. E.g., if two edges exist
            that have the same source and the same target, only the
            first edge is kept.
        remove_zero_cost_edges : bool, default=True
            Whether to remove edges that have zero costs.
        remove_isolated_nodes : bool, default=False
            Whether to remove nodes that are isolated, i.e., nodes in
            the network that do not have any edges associated with
            them.
        remove_unreachable_nodes : bool, default=False
            Whether to remove nodes that are unreachable from
            ``start_node_unreachables``. If this is not given,
            unreachable nodes from the first node in network.nodes are
            removed.
        remove_commodities : bool, default=False
            Whether to remove commodities made up by nodes which are
            deleted in the cleaning process.
        
        Returns
        -------
        str
            Summary of cleaning process.
        """
        # Get network size before cleaning (nodes, edges, commodities)
        n, m, c = self.n, self.m, len(self.demand)
        
        out = ""
        if remove_zones is True:
            out += self._remove_zones(remove_commodities=remove_commodities,
                                      update_shared=True)
        if remove_parallel_edges is True:
            out += self._remove_parallel_edges(update_shared=True)
        if remove_zero_cost_edges is True:
            out += self._remove_zero_cost_edges(update_shared=True)
        if remove_isolated_nodes is True:
            # nodes that are in nodes but not in edges
            out += self._remove_isolated_nodes(remove_commodities=remove_commodities,
                                               update_shared=True)
        if remove_unreachable_nodes is True:
            # update indices if a previous method has manipulated network
            out += self._remove_unreachable_nodes(start_node=start_node_unreachables,
                                                  remove_commodities=remove_commodities,
                                                  update_shared=True)
        
        if (self.m == 0 and not remove_zones and remove_zero_cost_edges and
                remove_unreachable_nodes):
            raise RuntimeError(
                "The combination of "
                "\n\tremove_zones==False"
                "\n\tremove_zero_cost_edges==True"
                "\n\tremove_unreachable_nodes==True"
                "\ndid result in 0 edges left in network"
            )
            
        if len(self.demand) == 0:
            warnings.warn("Network does not have commodities!")
        
        # self.update_shared(map_demand_labels=remove_commodities)
        
        out += ("\nNodes: {:d} -> {:d}"
                "\nEdges: {:d} -> {:d}"
                "\nCommodities: {:d} -> {:d}"
                .format(n, self.n, m, self.m, c, len(self._d)))
        return out

    def update_shared(
            self,
            map_demand_labels: bool = True
            ) -> None:
        """Update network elements by shared object.
        
        Keeps consistency across edges, nodes, demand and cost.
        
        Parameters
        ----------
        map_demand_labels : bool, default=True
            Whether to remap labels to ids in network.demand
        """
        self.shared.update()
        if (map_demand_labels is True and hasattr(self, "_d") and
                len(self.demand) > 0):
            self.demand.map_node_label_to_id()

    def delete_edges(
            self,
            edges,
            update_shared: bool = True
            ) -> int:
        """Delete some edges from network.
        
        Parameters
        ----------
        edges : int or array of ints
            Edges to delete from network.edges and network.cost
            objects.
        update_shared : bool, default=True
            Whether to updated shared network elements. Keeps
            consistency of network.edges to other network elements.
        
        Returns
        -------
        int
            Amount of deleted edges.
        
        See Also
        --------
        paminco.net.shared.Shared.delete_edges :
            Delete edges from Edges.
        paminco.net.cost.PolynomialCost.delete_edges :
            Delete edges from PolynomialCost.
        paminco.net.cost.PiecewiseQuadraticCost.delete_edges :
            Delete edges from PiecewiseQuadraticCost.
        """
        self.reset_cache()
        
        # Remove edges from edgelist
        del_edges = self.shared.delete_edges(edges, return_indices=True)
        
        # Remove edges from cost
        self.cost.delete_edges(del_edges)
        
        # Update shared: mapping (nodeid, nodeid) -> edgeid
        if len(del_edges) > 0 and update_shared is True:
            self.shared._set_edge_id_mapping()
        
        return len(del_edges)

    def delete_nodes(
            self,
            nodes,
            update_shared: bool = True,
            is_label: bool = False,
            remove_commodities: bool = True,
            ) -> tuple:
        """Delete some nodes from network.
        
        Parameters
        ----------
        nodes : int, array of ints, str, or array of str
            Indices or labels of nodes to delete.
        update_shared : bool, default=True
            Whether to updated shared network elements. Keeps
            consistency to other network elements.
        is_label : bool, default=True
            Whether to delete by label or internal node index.
        remove_commodities : bool, default=True
            Whether to remove commodities made up by ``nodes``.
        
        Returns
        -------
        int
            Amount of deleted nodes.
        int
            Amount of deleted commodities.
        
        See Also
        --------
        paminco.net.shared.Shared.delete_nodes :
            Delete nodes from network.nodes.
        """
        self.reset_cache()
        
        # Remove nodes from nodelist
        del_nodes = self.shared.delete_nodes(nodes,
                                             is_label=is_label,
                                             return_indices=True,
                                             update=False)
        
        # Remove nodes from demand
        n_del_coms = 0
        if remove_commodities is True:
            n_del_coms = self.demand.delete_nodes(del_nodes)
        
        # Remove nodes from edges and costs
        del_edges = self.shared.delete_nodes_in_edges(del_nodes,
                                                      return_indices=True)
        self.cost.delete_edges(del_edges)
        
        if update_shared is True:
            self.update_shared(map_demand_labels=remove_commodities)
        
        return len(del_nodes), len(del_edges), n_del_coms

    def get_node_pos(self) -> dict:
        return self.nodes.get_pos()

    def get_flow_df(self, x: np.ndarry, labels: bool = True) -> pd.DataFrame:
        return self.edges.get_flow_df(x, labels=labels)

    def _remove_isolated_nodes(
            self,
            update_shared: bool = True,
            remove_commodities: bool = True,
            ) -> str:
        nodes_in_edges = np.unique(self.edges.labels.ravel()).tolist()
        all_nodes = self.nodes.labels.tolist()
        isolated_nodes = list(set(all_nodes) - set(nodes_in_edges))

        n_n, n_m, n_c = 0, 0, 0
        if len(isolated_nodes) > 0:
            n_n, n_m, n_c = self.delete_nodes(isolated_nodes,
                                              is_label=True,
                                              update_shared=update_shared,
                                              remove_commodities=remove_commodities)

        out = "Removing isolated nodes:\n"
        out += "\t{:d} nodes\n".format(n_n)
        out += "\t{:d} edges\n".format(n_m)
        out += "\t{:d} commodities\n".format(n_c)
        return out

    def _remove_parallel_edges(
            self,
            update_shared: bool = True,
            ) -> str:
        duplicate_idx = self.edges.get_duplicate_edges()
        
        # remove duplicates from data structs
        n_m = self.delete_edges(duplicate_idx, update_shared=update_shared)
        
        out = "Removing parallel edges:\n"
        out += "\t{:d} edges\n".format(n_m)
        return out

    def _remove_unreachable_nodes(
            self,
            start_node=None,
            update_shared: bool = True,
            remove_commodities: bool = True,
            ) -> str:
        # Build adjaceny matrix with uniform weights = 1
        adj_matrix = self.shared.csgraph(np.ones(self.m),
                                         respect_bounds=True,
                                         backward_positive=True)
        
        # If no start_node is specified, use the first node from the
        # largest strongly connected component
        if start_node is None:
            coco = sps.csgraph.connected_components
            num_cc, cc = coco(adj_matrix,
                              directed=True,
                              connection='weak',
                              return_labels=True)

            id_u, cnt_u = np.unique(cc, return_counts=True)
            comp = id_u[np.argmax(cnt_u)]
            start_node = np.argmin(id_u == comp)
        
        elif isinstance(start_node, str):
            start_node = self.shared.get_node_id(start_node)
        
        node_name = self.shared.get_node_label(start_node)
        
        # Get all distances between nodes for weight = 1 and identity
        # unreachables where distance matrix is infinite
        D, _ = csr_dijkstra(adj_matrix, indices=start_node)
        idx_unreachables = np.where(~np.isfinite(D))[0]
        
        n_n, n_m, n_c = 0, 0, 0
        if len(idx_unreachables) > 0:
            # Delete nodes in nodes and reset mapping
            n_n, n_m, n_c = self.delete_nodes(idx_unreachables,
                                              is_label=False,
                                              update_shared=update_shared,
                                              remove_commodities=remove_commodities)
        
        out = "Removing unreachable nodes from '{:s}':\n".format(node_name)
        out += "\t{:d} nodes\n".format(n_n)
        out += "\t{:d} edges\n".format(n_m)
        out += "\t{:d} commodities\n".format(n_c)
        return out

    def _remove_zero_cost_edges(
            self,
            update_shared: bool = True,
            ) -> str:
        if isinstance(self._c, PolynomialCost) is False:
            return "No zero cost edges removed. Cost are not polynomial.\n"
        zc_edges = np.where(abs(self._c.coefficients).sum(axis=1) == 0)[0]
        
        self.delete_edges(zc_edges, update_shared=update_shared)
        
        out = "Removing zero cost edges:\n"
        out += "\t{:d} edges\n".format(len(zc_edges))
        return out

    def _remove_zones(
            self,
            update_shared: bool = True,
            remove_commodities: bool = True,
            ) -> str:
        out = "Cleaning zones:\n"
        if self.nodes.has_zones is False:
            return out + "\tNo zone nodes in network."
        # Get zone nodes and delete
        zone_nodes = self.nodes.labels[self.nodes.zone].tolist()
        n_n, n_m, n_c = self.delete_nodes(zone_nodes,
                                          is_label=True,
                                          update_shared=update_shared,
                                          remove_commodities=remove_commodities)

        out += "\t{:d} nodes\n".format(n_n)
        out += "\t{:d} edges\n".format(n_m)
        out += "\t{:d} commodities\n".format(n_c)
        return out

    def _flow_to_nx(self, x, return_pos: bool = True):
        if isinstance(x, (int, float)):
            x = np.full(self.m, x)
        
        df = self.edges.to_df()[["source_lbl", "target_lbl"]]
        df["flow"] = x
        df.columns = ["source", "target", "weight"]
        
        if return_pos is True:
            self.nodes.get_pos()
            if self.nodes.xy is None:
                raise ValueError("No node coordinates set.")
            xy = self.nodes.xy
            lbl = self.nodes.labels
            pos = dict(zip(lbl, xy))
            return df, pos
        
        return df

    @staticmethod
    def convert_tntp_to_xml(
            outfile,
            netfile,
            tripsfile=None,
            nodefile=None,
            cost_type: str = "auto",
            prettify: bool = True,
            kw_init=None,
            kw_write=None,
            ):
        if kw_init is None:
            kw_init = {}
        if kw_write is None:
            kw_write = {}
        
        # Build network from tntp files
        net = Network.from_tntp(
            netfile, tripsfile, nodefile, cost_type, **kw_init
        )
        
        # Dump to xml
        return net.to_xml(outfile, prettify=prettify, **kw_write)
    
    @classmethod
    def from_tntp(
            cls,
            netfile,
            tripsfile=None,
            nodefile=None,
            cost_type: str = "auto",
            **kw
            ) -> Network:
        edge_data, node_data, cost_data, demand_data = read_tntp(
            netfile, tripsfile, nodefile, cost_type
        )
        # if cost_type != "auto":
        #     if "kw_cost" not in kw:
        #         kw["kw_cost"] = {}
        #     kw["kw_cost"]["cost_type"] = cost_type
        return cls(edge_data, node_data, cost_data, demand_data, **kw)

    @classmethod
    def from_gaslib(
            cls,
            network_file,
            scenario_file,
            contract_aux_elements=True,
            debug=False
            ) -> Network:
        """Read a gaslib instance from files and a network from the data.
        
        Parameters
        ----------
        network_file : str
            The filename of the gaslib XML network file
        scenario_file : str
            The filename of the gaslib XML scenario file
            to be used as demands
        contract_aux_elements : boolean, default=True
            If set to True, auxiliary components (i.e. componentss that are not
            pipes, such as vales and compressors) are contracted. If set to False
            these components are modeled as pipes with very small resistances
        debug : boolean, default=False
            If set to True, some debug information are printed

        Returns
        -------
        network : Network
            The network read from the files
        """
        data = gaslib_to_network_data(
            network_file, scenario_file, contract_aux_elements, debug
        )
        return cls(*data)

    @staticmethod
    def convert_gaslib_to_xml(
            outfile,
            network_file,
            scenario_file,
            contract_aux_elements=True,
            debug=False,
            prettify=True,
            kw_write=None
            ):
        """Read a gaslib instance from files and a network from the data.
        
        Parameters
        ----------
        outfile : str
            The filename of the output xml file
        network_file : str
            The filename of the gaslib XML network file
        scenario_file : str
            The filename of the gaslib XML scenario file
            to be used as demands
        contract_aux_elements : boolean, default=True
            If set to True, auxiliary components (i.e. componentss that are not
            pipes, such as vales and compressors) are contracted. If set to False
            these components are modeled as pipes with very small resistances
        debug : boolean, default=False
            If set to True, some debug information are printed
        prettify : boolean, default=True
            If set to True, the xml code in the outputfile is written in a
            more readeable form
        kw_write : dict, default=None
            Additional keywords for Elementree.write

        See Also
        --------
        xml.etree.ElementTree.ElementTree.write
        """
        data = gaslib_to_network_data(
            network_file, scenario_file, contract_aux_elements, debug
        )
        network = Network(data)
        network.to_xml(outfile, prettify=prettify, **kw_write)

    def to_xml(
            self,
            file,
            prettify: bool = False,
            **write_kw
            ):
        """Save Network as XML.
        
        Parameters
        ----------
        file : str or file object
            File name, or a file object opened for writing.
        prettify : bool, default=False
            Whether to prettify resulting XML file.
        
        See Also
        --------
        xml.etree.ElementTree.ElementTree.write
        """
        # Init tree
        root = et.Element('network')
        tree = et.ElementTree(root)
        
        # Add metadata subelement
        metadata = et.SubElement(root, "metadata")
        
        # Fill tree with node, edges, cost, and demand
        root = self.edges.add_to_etree(root,
                                       overwrite=True,
                                       cost_writer=self.cost.add_to_etree)
        if isinstance(self.cost, SymbolicCost):
            self.cost.costfuncs_to_metadata(metadata)
        root = self.nodes.add_to_etree(root, overwrite=True)
        root = self.demand.add_to_etree(root, overwrite=True)
        
        tree.write(file, **write_kw)
        if prettify:
            prettify_xml(file)

    @classmethod
    def from_xml(
            cls,
            file: str,
            return_dict: bool = False,
            **kwargs
            ) -> Network:
        """Read network from ``XML``.
        
        Parameters
        ----------
        data : str, file, ElementTree or Element
            Initialize object from xml file by passing
                * filename as str,
                * file object that contains XML data,
                * the XML ElementTree,
                * or the root Element of the ElementTree.
        return_dict : bool, default=False,
            If ``True``, the data is returned as a dict with entries
            ``edge_data``, ``node_data``, ``demand_data`` and ``cost_data``.
            If ``False`` (default), a ``Network`` object created from
            this data is returned
        
        Returns
        -------
        net : Network or dict
            The network read from the data. If ``return_dict`` is True,
            a dict with the raw data is returned.
        
        See Also
        --------
        xml.etree.ElementTree
        :func:`~paminco.net.shared.Edges.from_xml`
            ``from_xml`` method of Edges class
        :func:`~paminco.net.shared.Nodes.from_xml`
            ``from_xml`` method of Nodes class
        :func:`~paminco.net.cost.NetworkCost.from_xml`
            ``from_xml`` method of NetworkCost class
        """
        edge_data = Edges.from_xml(file, return_data=True)
        node_data = Nodes.from_xml(file, return_data=True)
        demand_data = file
        # Reading and initialization of Cost Function is
        # deferred to set_cost method
        cost_data = file

        if return_dict:
            return {
                'edge_data': edge_data,
                'node_data': node_data,
                'cost_data': cost_data,
                'demand_data': demand_data,
            }

        return cls(edge_data, node_data, cost_data, demand_data, **kwargs)

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None,
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        save_dict = self._s.make_save_dict(prefix=prefix,
                                           save_dict=save_dict)
        save_dict = self._d.make_save_dict(prefix=prefix + "dem_",
                                           save_dict=save_dict)
        save_dict = self._c.make_save_dict(prefix=prefix + "cost_",
                                           save_dict=save_dict)
        return save_dict
    
    make_save_dict.__doc__ = _doc.make_save_dict.__doc__

    def save_to_numpy(
            self,
            file: str,
            **kwargs
            ) -> None:
        save_dict = self.make_save_dict()
        save_dict.update(kwargs)
        np.savez(file, **save_dict)
    
    save_to_numpy.__doc__ = _doc.save_to_numpy.__doc__

    @classmethod
    def from_npz(
            cls,
            data,
            prefix: str = "",
            **kwargs
            ) -> Network:
        net = cls.__new__(cls)
        if isinstance(data, str):
            data = np.load(data)
        
        # load shared object (edges, nodes)
        net._s = Shared.from_npz(data, prefix=prefix)
        
        # load demand
        class_ = getattr(netdemand, str(data["demand_type"]))
        demand = class_.from_npz(data, shared=net.shared, prefix=prefix + "dem_")
        net.set_demand(demand)
        
        # load cost
        net._c = NetworkCost.from_npz(data, shared=net.shared, prefix=prefix + "cost_")
        
        return net
    
    from_npz.__func__.__doc__ = _doc.from_npz.__doc__

    @property
    def cost(self):
        """Cost associated with network.
        
        Can be either :class:`~paminco.net.cost.PolynomialCost`. or
        :class:`~paminco.net.cost.PiecewiseQuadraticCost`.
        
        See Also
        --------
        paminco.net.cost
        """
        return self._c

    @cost.setter
    def cost(self, *args, **kw) -> None:
        self.set_cost(*args, **kw)

    @property
    def demand(self) -> DemandFunction:
        """Demand associated with network.
        
        Can be either
        :class:`~paminco.net.demand.LinearDemandFunction`. or
        :class:`~paminco.net.demand.AffineDemandFunction`.
        
        See Also
        --------
        paminco.net.demand
        """
        if hasattr(self, "_d") is False:
            raise AttributeError(
                "Network has no demand, must be set with net.set_demand()."
            )
        return self._d

    @demand.setter
    def demand(self, *args, **kw) -> None:
        self.set_demand(*args, **kw)

    @property
    def shared(self) -> Shared:
        """Shared object for network objects.
        
        See Also
        --------
        paminco.net.shared.Shared
        """
        return self._s

    @property
    def edges(self) -> Edges:
        """Network edges.
        
        See Also
        --------
        paminco.net.shared.Edges
        """
        return self.shared.edges

    @property
    def flow_direction(self) -> FlowDirection:
        return self.shared.graph_type
    
    flow_direction.__doc__ = Shared.flow_direction.__doc__

    @property
    def is_directed(self) -> bool:
        """Whether the network's edges are directed."""
        return (self.edges.lb.min() >= 0)

    @property
    def is_single_commodity(self) -> bool:
        """Whether network demand consists of single commodity only."""
        return (self.demand().shape[1] == 1)

    @property
    def is_multi_commodity(self) -> bool:
        """Whether network demand consists of more than one commodity."""
        return (self.demand().shape[1] > 1)

    @property
    def nodes(self) -> Nodes:
        """Network nodes.
        
        See Also
        --------
        paminco.net.shared.Nodes
        """
        return self.shared.nodes

    @property
    def n(self) -> int:
        return self.shared.n
    
    n.__doc__ = Shared.n.__doc__

    @property
    def m(self) -> int:
        return self.shared.m
    
    m.__doc__ = Shared.m.__doc__

    @property
    def k(self) -> int:
        """Get the number of commodities in network."""
        return len(self.demand)
    
    @property
    def size(self) -> tuple:
        """Get number of nodes, edges and commodities."""
        return (self.n, self.m, self.k)

    @property
    def dtype_int(self):
        return self.shared.dtype_int
    
    dtype_int.__doc__ = Shared.dtype_int.__doc__

    @property
    def dtype_float(self):
        return self.shared.dtype_float
    
    dtype_float.__doc__ = Shared.dtype_float.__doc__
