"""Module that handles shared information for all network objects."""
import xml.etree.ElementTree as et
import numbers

import numpy as np
import pandas as pd
import scipy.sparse as sps

from paminco.utils.readin import parse_number, xml_find_root
from paminco.utils.misc import Cache
from paminco.utils.typing import sparse_format, is_int, is_iterable, IntEnum2
import paminco._doc as _doc


ID_UNMAPPED = -9999
LBL_UNMAPPED = "Invalid"


class FlowDirection(IntEnum2):
    """Enum defining the type flow for the graph."""

    DIRECTED = 0
    """All edges can only take flow >= 0."""

    UNDIRECTED = 1
    """All edges can take any flow."""

    MIXED = 2
    """Some edges may only take flow >= 0."""


class Edges:
    """
    Class that contains the edges/links of a network.
    
    An edges object can be instantiated in several ways:
        Edges(e)
            where ``e`` is an Edges object. Data in ``e`` will be
            copied if specfied by parameter ``copy``.
        Edges(st)
            where ``st`` is array_like. Parameter st is converted to
            ndarray and is expected to me of shape (m, 2). Can be
            node indices or node labels specifying an edge. If labels
            are given, indices are mapped by ``map_labels_to_indices``,
            given indices are mapped by ``map_indices_to_labels``. Edge
            bounds are determined by the parameter ``directed``.
        Edges((st, bounds))
            where ``st`` is array_like and ``bounds`` is tuple (lower,
            upper) specifying bounds used for all edges or array_like
            of shape (m, 2) marking individual bounds for all edges.
        Edges((labels, indices, bounds))
            where ``labels``, ``indices`` are array_like of shape
            (m, 2) and ``bounds`` is tuple (lower, upper) specifying
            bounds used for all edges or array_like of shape (m, 2)
            containing individual bounds for all edges.
        
    Parameters
    ----------
    data : ndarray, or tuple of ndarray
        Edge data.
    directed_flow : bool, default=True
        Controls default values for ``None`` in bounds. If ``True``, lower
        bounds are set to 0 and ``False`` to -inf. Missing upper bounds are
        set to inf.
    map_labels_to_indices : None, bool, dict, or callable, default=True
        Determines mapping of labels to indices if no indices are
        given. If ``None`` or ``False``, indices of edges will be set
        to -9999, denoting invalid edge indices. If ``dict``,
        labels will be mapped by this dict. If ``True``, node indices
        are set to 0, 1, ..., n-1. If ``callable``, use callable with
        signature ``indices = callable(labels)``.
    map_indices_to_labels : None, bool, dict, or callable, default=True
        Determines mapping of indices to indices if no labels are
        given. If ``None`` or ``False``, indices of edges will be set
        to 'invalid', denoting invalid edge labels. If ``dict``,
        indices will be mapped by this dict. If ``True``, node labels
        are set to node indices as str. If ``callable``, use callable
        with signature ``labels = callable(indices)``.
    dtype_float : dtype, default=numpy.float_
        Datatype for edge bounds.
    dtype_int : dtype, default=int
        Datatype for edge bounds.
    copy : bool, default=False
        Whether to create a copy of the inputs in data.
    
    Attributes
    ----------
    flow_directions : ndarray
        Ndarray of shape (m, ). A ``-1`` denotes an edge with lb < 0 and
        ub <= 0. A ``0`` denotes an edge with lb < 0 and ub > 0.
        A ``1`` denotes an edge with lb >=0 and ub > 0.
    """
    
    def __init__(
            self,
            data,
            directed_flow: bool = True,
            map_labels_to_indices=True,  # optional
            map_indices_to_labels=True,  # optional
            dtype_float=None,
            dtype_int=None,
            copy: bool = False,
            ) -> None:
        # Collect kwargs
        kw = {
            "directed_flow": directed_flow,
            "map_labels_to_indices": map_labels_to_indices,
            "map_indices_to_labels": map_indices_to_labels,
            "dtype_float": dtype_float,
            "dtype_int": dtype_int,
            "copy": copy,
        }
        if isinstance(data, Edges):
            d = (data.labels, data.indices, data.bounds)
            return self.__init__(d,
                                 dtype_float=data.dtype_float,
                                 dtype_int=data.dtype_int)
        elif isinstance(data, tuple):
            if len(data) == 3:
                pass
            elif len(data) == 2:
                # (labels or indices, bounds)
                st, bounds = data
                st = np.array(st)
                if st.dtype.kind in {'U', 'S'}:
                    # (labels, bounds)
                    if isinstance(map_labels_to_indices, dict):
                        st_ids = np.vectorize(map_labels_to_indices.__getitem__)(st)
                    elif map_labels_to_indices is True:
                        # Automap labels
                        # Get unique labels and sort them if quasi-ints
                        unique_lbl = np.unique(st)
                        try:
                            unique_lbl = sorted(unique_lbl, key=int)
                        except ValueError:
                            pass
                        d = dict(zip(unique_lbl, np.arange(len(unique_lbl))))
                        st_ids = np.vectorize(d.__getitem__)(st)
                    elif map_labels_to_indices is None or map_labels_to_indices is False:
                        # Set to invalid indices
                        st_ids = np.full(st.shape, ID_UNMAPPED, dtype=int)
                    else:
                        # Map labels by callable
                        st_ids = map_labels_to_indices(st)
                    return self.__init__((st, st_ids, bounds), **kw)
                elif issubclass(st.dtype.type, numbers.Integral):
                    # (indices, bounds)
                    unique_st = np.unique(st)
                    if np.array_equal(np.sort(unique_st), np.arange(len(unique_st))) is False:
                        raise ValueError(f"Indices must be all integers from 0 to  {len(unique_st) - 1}.")
                    
                    if isinstance(map_indices_to_labels, dict):
                        st_lbl = np.vectorize(map_indices_to_labels.__getitem__)(st)
                    elif map_indices_to_labels is True:
                        st_lbl = st.astype(str)
                    elif map_indices_to_labels is None or map_indices_to_labels is False:
                        # Set to invalid indices
                        st_lbl = np.full(st.shape, LBL_UNMAPPED)
                    else:
                        st_lbl = map_indices_to_labels(st)
                    return self.__init__((st_lbl, st, bounds), **kw)
                else:
                    raise ValueError(f"Invalid edge data: {data}.")
            else:
                raise ValueError(f"Invalid edge data: {data}.")
        else:
            # Only labels or indices given -> build lower and upper bounds by directed
            if directed_flow is True:
                return self.__init__((data, (0, np.inf)), **kw)
            return self.__init__((data, (-np.inf, np.inf)), **kw)
        
        # Handle datatypes
        if dtype_float is None:
            dtype_float = np.float64
        if dtype_int is None:
            dtype_int = int
        self._dtype_float = dtype_float
        self._dtype_int = dtype_int
        
        # Unpack data
        labels, indices, bounds = data
        self.labels = np.array(labels, dtype=str, copy=copy)
        self.indices = np.array(indices, dtype=dtype_int, copy=copy)
        
        # Broadcast bounds if lower, upper for all edges given
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
            if len(bounds.shape) == 1:
                bounds = bounds.reshape(1, -1)
                bounds = np.repeat(bounds, len(labels), axis=0)
                
        # Handle 'None' bounds
        bkw = {"posinf": np.inf, "neginf": -np.inf}
        self.bounds = np.array(bounds, dtype=dtype_float, copy=copy)
        self.bounds[:, 1] = np.nan_to_num(self.bounds[:, 1],
                                          copy=False,
                                          nan=np.inf,
                                          **bkw)
        if directed_flow is True:
            self.bounds[:, 0] = np.nan_to_num(self.bounds[:, 0],
                                              copy=False,
                                              nan=0.,
                                              **bkw)
        else:
            self.bounds[:, 0] = np.nan_to_num(self.bounds[:, 0],
                                              copy=False,
                                              nan=-np.inf,
                                              **bkw)
        
        # Check consistency of labels, indices and bounds
        if self.labels.shape[1] != 2:
            raise ValueError(
                f"Invalid edge data, labels are of shape {self.labels.shape}."
            )
        if (self.labels.shape == self.indices.shape == self.bounds.shape) is False:
            raise ValueError(
                "Inconsistent shapes. "
                f"Labels: {self.labels.shape}, "
                f"indices: {self.indices.shape}, "
                f"bounds: {self.bounds.shape}."
            )
        
        # Set edge directions and get type of graph
        self.flow_directions = np.zeros(len(self))
        self.flow_directions[self.lb < 0] -= 1
        self.flow_directions[self.ub > 0] += 1
        
        if len(self.flow_undirected) == len(self):
            self.flow_dir = FlowDirection.UNDIRECTED
        elif len(self.flow_undirected) == 0:
            self.flow_dir = FlowDirection.DIRECTED
        else:
            self.flow_dir = FlowDirection.MIXED
            
        self.cache = Cache()

    def __eq__(self, other) -> bool:
        for att in ["labels", "indices", "bounds"]:
            if np.array_equal(getattr(self, att), getattr(other, att)) is False:
                return False
        return True

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        if is_iterable(idx):
            return [self[i] for i in idx]
        return {att: getattr(self, att)[idx]
                for att in ["source_lbl", "target_lbl", "s", "t", "lb", "ub"]}

    def to_df(self, **kwargs) -> pd.DataFrame:
        """Get object as DataFrame.
        
        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Passed to DataFrame constructor.
        
        Returns
        -------
        df : pandas.DataFrame
            Edges with source/target labels, source/target ids, lower
            and upper bounds.
        """
        data = np.hstack([self.labels, self.indices, self.bounds])
        df = pd.DataFrame(data, **kwargs)
        df.columns = ["source_lbl", "target_lbl", "s", "t", "lb", "ub"]
        df[["source_lbl", "target_lbl"]] = df[["source_lbl", "target_lbl"]].astype(str)
        df[["s", "t"]] = df[["s", "t"]].astype(self._dtype_int)
        df[["lb", "ub"]] = df[["lb", "ub"]].astype(self._dtype_float)
        return df

    def get_flow_df(
            self,
            x,
            labels: bool = True,
            colname_flow: str = "flow"
            ) -> pd.DataFrame:
        if isinstance(x, (int, float)):
            x = np.full(len(self), x)
        
        if labels is True:
            s, t = self.source_lbl, self.target_lbl
            dtype = str
        else:
            s, t = self.s, self.t
            dtype = int
        df = pd.DataFrame({"source": s,
                           "target": t,
                           colname_flow: x})
        df[["source", "target"]] = df[["source", "target"]].astype(dtype)
            
        return df

    def get_directed(
            self,
            w=None,
            backward_positive: bool = True
            ) -> tuple:
        if self.cache.is_valid("directed_elements") is False:
            forward = self.ub > 0
            backward = self.lb < 0
            
            s_fw, t_fw = self.indices[forward, :].T
            t_bw, s_bw = self.indices[backward, :].T
            s = np.hstack((s_fw, s_bw))
            t = np.hstack((t_fw, t_bw))
            
            self.cache["directed_elements"] = (forward, backward, s, t)
        else:
            (forward, backward, s, t) = self.cache["directed_elements"]
        
        if w is not None:
            w_fw = w[forward]
            w_bw = w[backward]
            
            if backward_positive is False:
                w_bw = - w_bw
                
            # Stack weight similar to source, target
            w = np.hstack((w_fw, w_bw))
            
            return s, t, w
        
        return s, t

    def get_duplicate_edges(self) -> np.ndarray:
        # Dubplicates -> s/t both are the same
        st = pd.Series([str(a) + "-" + str(b) for (a, b) in self.indices])
        return np.where(st.duplicated())[0]

    def map_labels(self, d: dict) -> None:
        """Map edge labels by d."""
        self.indices = np.vectorize(d.__getitem__)(self.labels).astype(self.dtype_int)

    def _delete_edges(
            self,
            del_idx,
            return_indices: bool = False
            ):
        del_idx = np.array(del_idx)
        
        # Delete edges in all numpy arrays
        self.labels = np.delete(self.labels, del_idx, axis=0)
        self.indices = np.delete(self.indices, del_idx, axis=0)
        self.bounds = np.delete(self.bounds, del_idx, axis=0)
        self.flow_directions = np.delete(self.flow_directions, del_idx, axis=0)
        
        # Recompute bounded edges for cs graph
        self.cache.set_invalid("directed_elements")
        
        if return_indices is True:
            if del_idx.dtype == "bool":
                return np.where(del_idx)[0].reshape(-1,)
            return del_idx.reshape(-1,)

    def _delete_nodes(
            self,
            nodes,
            return_indices: bool = False
            ):
        nodes = np.array(nodes).reshape(1, -1)
        # get indices of edges to delete
        idx_s = ((self.s.reshape(-1, 1) - nodes) == 0).any(axis=1)
        idx_t = ((self.t.reshape(-1, 1) - nodes) == 0).any(axis=1)
        del_idx = idx_s | idx_t
        
        return self._delete_edges(del_idx, return_indices=return_indices)

    def add_to_etree(
            self,
            root: et.Element,
            overwrite: bool = True,
            cost_writer=None,
            ) -> None:
        """Add edge data to XML Element.
        
        Parameters
        ----------
        root : Element
            Element to which 'edges' will be appended to.
        overwrite : bool, default=True
            If True, existing 'edges' Element in `root` will be
            deleted. If False, edge data will be appended to the
            existing data.
        cost_writer: callable, optional
            Function that adds cost information for every edge.
            Will be called with ``cost_writer(edge_node, index,
            overwrite)`` where ``edge_node`` is xml Element,
            ``index`` is int and ``overwrite`` = overwrite.
        """
        edges = root.find("edges")
        if overwrite is True and edges is not None:
            root.remove(edges)
        
        edges = root.find("edges")
        if edges is None:
            root.append(et.Element("edges"))
        
        for (i, edge) in enumerate(self.to_df().T.to_dict().values()):
            edge_node = et.SubElement(root.find("edges"), 'edge')
            edge_node.attrib['from'] = edge['source_lbl']
            edge_node.attrib['to'] = edge['target_lbl']
            edge_node.attrib['lb'] = str(edge['lb'])
            edge_node.attrib['ub'] = str(edge['ub'])
            if cost_writer is not None:
                cost_writer(edge_node, i, overwrite=overwrite)
        
        return root

    def _read_edge(edge_node):
        source_target = (edge_node.attrib["from"], edge_node.attrib["to"])
        lb = parse_number(edge_node.attrib.get("lb", None))
        ub = parse_number(edge_node.attrib.get("ub", None))
        return source_target, (lb, ub)

    @classmethod
    def from_xml(
            cls,
            data,
            return_data: bool = False,
            **kwargs
            ):
        data = xml_find_root(data)
        edges = data.find("edges")
        if edges is None:
            return None
        
        source_target = []
        bounds = []
        
        for e in edges:
            st, b = Edges._read_edge(e)
            source_target.append(st)
            bounds.append(b)
            
        if return_data is True:
            return source_target, bounds
        
        return cls((source_target, bounds), **kwargs)
    
    from_xml.__func__.__doc__ = _doc.from_xml.__doc__

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        
        for k in ["labels", "indices", "bounds"]:
            save_dict[prefix + k] = getattr(self, k)
        return save_dict
    
    make_save_dict.__doc__ = _doc.make_save_dict.__doc__

    def save_to_numpy(
            self,
            file: str,
            **kwargs
            ) -> None:
        save_dict = self.make_save_dict()
        save_dict.update(**kwargs)
        np.savez(file, **save_dict)
    
    save_to_numpy.__doc__ = _doc.save_to_numpy.__doc__

    @classmethod
    def from_npz(
            cls,
            data,
            prefix: str = "",
            **kwargs
            ):
        if isinstance(data, str):
            data = np.load(data)
        
        edge_data = (data[prefix + "labels"],
                     data[prefix + "indices"],
                     data[prefix + "bounds"])
        
        return cls(edge_data, **kwargs)
    
    from_npz.__func__.__doc__ = _doc.from_npz.__doc__

    @property
    def flow_forward(self) -> np.ndarray:
        return np.where(self.flow_directions == 1)[0]

    @property
    def flow_backward(self) -> np.ndarray:
        return np.where(self.flow_directions == -1)[0]

    @property
    def flow_undirected(self) -> np.ndarray:
        return np.where(self.flow_directions == 0)[0]

    @property
    def lb(self) -> np.ndarray:
        """ndarray (m, ) of floats: lower bound."""
        return self.bounds[:, 0]

    @property
    def ub(self) -> np.ndarray:
        """ndarray (m, ) of floats: upper bound."""
        return self.bounds[:, 1]

    @property
    def s(self) -> np.ndarray:
        """ndarray (m, ) of int: source ids."""
        return self.indices[:, 0]

    @property
    def t(self) -> np.ndarray:
        """ndarray (m, ) of int: target ids."""
        return self.indices[:, 1]

    @property
    def source_lbl(self) -> np.ndarray:
        """ndarray (m, ) of str: sources labels."""
        return self.labels[:, 0]

    @property
    def target_lbl(self) -> np.ndarray:
        """ndarray (m, ) of str: target labels."""
        return self.labels[:, 1]

    @property
    def dtype_int(self):
        """dtype of int data."""
        return self.indices.dtype

    @property
    def dtype_float(self):
        """dtype of float data."""
        return self.bounds.dtype


class Nodes:
    """Class that contains the nodes/vertices of a network.
    
    A Nodes object can be instantiated in several ways:
        Nodes(n)
            where ``n`` is a Nodes object.
        Nodes(nodes)
            where ``nodes`` is array_like and contains either node
            labels or node indices. If no indices are given, they
            are set automatically.
        Nodes(nodes, zone)
            where ``nodes`` and ``zone`` are array_like of shape (n, ).
            ``zone`` must be boolean array denoting if a node is a
            zone, mostly used for traffic networks.
        Nodes(nodes, xy, zone)
            where ``nodes`` and ``zone`` are array_like of shape (n, )
            and ``xy`` is array_like of shape (n, 2) and contains the
            coordinates of the nodes.
        Nodes(node_labels, node_indices, xy, zone)
            where ``node_labels`` and ``node_indices`` and ``zone``
            are are array_like of shape (n, ) and ``xy`` is array_like
            of shape (n, 2).
    
    Parameters
    ----------
    data : node_data
        Input data.
    dtype_float : dtype, default=numpy.float_
        Datatype for X and Y coordinates.
    dtype_int : dtype, default=int
        Datatype for node indices.
    map_labels: None, bool, dict, or callable, default=True
        Determines mapping of labels to indices if no indices are
        given or vice versa. If ``None`` or ``False``, indices / labels
        are set to -9999 / 'invalid'. If ``dict``, mapping by this
        dict. If ``True``, indices are set to 0, 1, ..., n-1, labels to
        indices as str. If ``callable``, use callable with
        signature ``indices = callable(labels)`` or
        ``labels = callable(indices)``.
    copy : bool, default=False
        Whether to create a copy of the inputs in data.
    
    Attributes
    ----------
    index
    node
    zone
    has_zones
    x
    y
    """
    
    def __init__(
            self,
            data,
            dtype_float=None,
            dtype_int=None,
            map_labels=True,  # optional
            copy: bool = False,
            ) -> None:
        # Collect kwargs
        kw = {
            "map_labels": map_labels,
            "dtype_float": dtype_float,
            "dtype_int": dtype_int,
            "copy": copy,
        }
        
        if isinstance(data, Nodes):
            d = (data.labels, data.indices, data.xy, data.zone)
            return self.__init__(d,
                                 dtype_float=data.dtype_float,
                                 dtype_int=data.dtype_int)
        elif isinstance(data, tuple):
            if len(data) == 4:
                pass
            elif len(data) == 3:
                # (labels or indices, xy, zone)
                node, xy, zone = data
                node = np.array(node)
                if node.dtype.kind in {'U', 'S'}:
                    # Case Labels
                    if map_labels is True:
                        # Automap labels
                        indices = np.arange(len(node))
                    elif map_labels is None or map_labels is False:
                        indices = np.full(len(node), ID_UNMAPPED, dtype=int)
                    else:
                        indices = map_labels(node)
                    return self.__init__((node, indices, xy, zone), **kw)
                elif issubclass(node.dtype.type, numbers.Integral):
                    if np.array_equal(np.sort(node), np.arange(len(node))) is False:
                        raise ValueError(f"Indices must of integers from 0 to {len(node) - 1} in any order.")
                    if map_labels is True:
                        labels = node.astype(str)
                    elif map_labels is None or map_labels is False:
                        # Set to invalid indices
                        labels = np.full(len(node), "None")
                    else:
                        labels = map_labels(node)
                    kw["dtype_int"] = node.dtype
                    return self.__init__((labels, node, xy, zone), **kw)
                else:
                    raise ValueError(f"Nodes must be ints or strings, are: {node.dtype}.")
            elif len(data) == 2:
                # (labels or indices, xy)
                zone = np.full(len(data[0]), False, dtype=bool)
                return self.__init__((*data, zone), **kw)
            else:
                raise ValueError("TODO")
        else:
            # Only labels or Id's specified
            data = np.array(data, copy=False)
            return self.__init__((data, None), **kw)
        
        # Handle datatypes
        if dtype_float is None:
            dtype_float = np.float64
        if dtype_int is None:
            dtype_int = int
        self.dtype_int = dtype_int
        self.dtype_float = dtype_float
        
        labels, indices, xy, zone = data
        indices = np.argsort(indices)
        self.labels = np.array(labels, dtype=str, copy=copy)[indices]
        if isinstance(zone, bool):
            zone = [zone] * len(self.labels)
        self.zone = np.array(zone, dtype=bool, copy=copy)[indices]

        if xy is not None:
            xy = np.array(xy)
            if xy.shape != (len(self.labels), 2):
                raise ValueError(f"Coordinates have wrong shape: {xy.shape}, should be {len(labels), 2}.")
            self.xy = np.array(xy, dtype=dtype_float, copy=copy)[indices]
        else:
            self.xy = None
            
        if (len(self.labels) == len(indices) == len(self.zone)) is False:
            raise ValueError(
                "Invalid shape of node data, "
                f"labels: {self.labels.shape}, "
                f"indices: {indices.shape}, "
                f"zone: {self.zone.shape}."
            )
        
        self.set_mappings()

    def __len__(self) -> int:
        return len(self.labels)

    def __eq__(self, other) -> bool:
        for att in ["labels", "indices", "zone"]:
            if np.array_equal(getattr(self, att), getattr(other, att)) is False:
                return False
        return True

    def set_mappings(self) -> None:
        """(Re)-set labels <-> indices mappings."""
        self.lbl2id = dict(zip(self.labels, self.indices))
        self.id2lbl = dict(zip(self.indices, self.labels))

    def get_pos(self) -> dict:
        if self.xy is None:
            raise ValueError("No node coordinates set.")
        return dict(zip(self.labels, self.xy))

    def delete_nodes(
            self,
            nodes,
            return_indices: bool = False
            ):
        """Delete nodes from Nodes object.
        
        Parameters
        ----------
        nodes : int, ndarray of int
            Indices of nodes to delete.
        return_indices : bool, default=False
            If True, node indices of deleted nodes are returned.
        
        Returns
        -------
        ndarray, optional
            If return_indices is True, node indices of deleted
            nodes are returned.
        
        See Also
        --------
        numpy.delete
        """
        self.labels = np.delete(self.labels, nodes)
        self.zone = np.delete(self.zone, nodes)
        if self.xy is not None:
            self.xy = np.delete(self.xy, nodes, axis=0)
        
        if return_indices is True:
            return np.array(nodes)

    def to_df(
            self,
            **kwargs
            ) -> pd.DataFrame:
        """Get object as pandas DataFrame.
        
        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Keyword arguments passed to pd.DataFrame constructor.
        
        Returns
        -------
        pandas.DataFrame
            Nodes data with node label, coordinates (x, y) and zone
            (bool).
        """
        df = pd.DataFrame({"label": self.labels, "zone": self.zone}, **kwargs)
        df["label"] = df["label"].astype(str)
        df["zone"] = df["zone"].astype(bool)
        if self.xy is not None:
            df[["x", "y"]] = self.xy.astype(self.dtype_float)
        return df

    def add_to_etree(
            self,
            root: et.Element,
            overwrite: bool = True
            ):
        """Add node data to xml.etree.ElementTree.Element.

        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Element to which 'nodes' will be appended to.
        overwrite : bool, default=True
            If True, existing 'nodes' Element in root will be
            deleted. If False, node data will be appended to the
            existing data.
        """
        nodes = root.find("nodes")
        if overwrite is True and nodes is not None:
            root.remove(nodes)
        
        nodes = root.find("nodes")
        if nodes is None:
            root.append(et.Element("nodes"))
        
        for node in self.to_df().T.to_dict().values():
            n_node = et.SubElement(root.find("nodes"), 'node')
            n_node.attrib['node'] = node['label']
            if 'x' in node:
                n_node.attrib['x'] = str(node['x'])
                n_node.attrib['y'] = str(node['y'])
            if node['zone'] is True:
                n_node.attrib['zone'] = "true"
        
        return root

    @classmethod
    def from_edges(
            cls,
            edges: Edges,
            **kw):
        d = dict(zip(edges.labels.ravel(), edges.indices.ravel()))
        labels = np.array(list(d.keys()))
        indices = np.array(list(d.values()))
        if np.array_equal(np.sort(indices), np.arange(len(indices))) is False:
            raise ValueError("Invalid edge indices.")
        return cls(labels[indices.argsort()], **kw)

    @classmethod
    def from_xml(
            cls,
            data,
            return_data: bool = False,
            **kwargs
            ):
        data = xml_find_root(data)
        nodes = data.find("nodes")
        
        if nodes is None:
            return None
        
        lbl = []
        xy = []
        zone = []
        for n in nodes:
            lbl.append(n.get('node'))
            x = parse_number(n.get('x', 0.0))
            y = parse_number(n.get('y', 0.0))
            xy.append([x, y])
            zone.append(n.get('zone', 'false').strip().lower() == 'true')
            
        if return_data is True:
            return (lbl, xy, zone)
        
        return cls((lbl, xy, zone), **kwargs)
    
    from_xml.__func__.__doc__ = _doc.from_xml.__doc__

    def make_save_dict(self, prefix: str = "", save_dict=None) -> dict:
        if save_dict is None:
            save_dict = {}
        for k in ["labels", "zone", "xy"]:
            save_dict[prefix + k] = getattr(self, k)
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
            **kwargs,
            ):
        if isinstance(data, str):
            data = np.load(data)
        # make empty edge object and fill with data
        node_data = (
            data[prefix + "labels"],
            data[prefix + "xy"],
            data[prefix + "zone"],
        )
        return cls(node_data, **kwargs)
    
    from_npz.__func__.__doc__ = _doc.from_npz.__doc__

    def _get_node(self, idx):
        return {att: getattr(self, att)[idx]
                for att in ["index", "node", "x", "z", "zone"]}

    @property
    def indices(self) -> np.ndarray:
        """ndarray (m, ) of int: node indices."""
        return np.arange(len(self.labels), dtype=self.dtype_int)

    @property
    def has_zones(self) -> bool:
        """bool: Whether Nodes object has any zone."""
        return self.zone.any()

    @property
    def x(self) -> np.ndarray:
        """ndarray (m, ) of floats: X coordinates."""
        if self.xy is None:
            raise AttributeError("Nodes has no coordinates.")
        return self.xy[:, 0]

    @property
    def y(self) -> np.ndarray:
        """ndarray (m, ) of floats: Y coordinates."""
        if self.xy is None:
            raise AttributeError("Nodes has no coordinates.")
        return self.xy[:, 1]


class Shared:
    """Class that acts as a shared object for a Network.
    
    Consists mainly of nodes and edges object, handles label and index
    mappings.
    
    Parameters
    ----------
    edge_data : tuple of ndarray
        Data to construct Edges object.
    node_data : array_like or tuple of ndarray, optional
        Data to construct Nodes object.
    dtype_float : dtype, default=numpy.float_
        Datatype for all float ndarray.
    dtype_int : dtype, default=int
        Datatype for all int ndarray.
    
    See Also
    --------
    Edges
    Nodes
    """

    def __init__(
            self,
            edge_data,
            node_data=None,
            dtype_float=np.float_,
            dtype_int=int,
            **kwargs
            ) -> None:
        if node_data is None:
            self.edges = Edges(edge_data,
                               dtype_float=dtype_float,
                               dtype_int=dtype_int,
                               **kwargs)
            self.nodes = Nodes.from_edges(self.edges,
                                          dtype_float=dtype_float,
                                          dtype_int=dtype_int)
        else:
            self.nodes = Nodes(node_data,
                               dtype_float=dtype_float,
                               dtype_int=dtype_int)
            self.edges = Edges(edge_data,
                               map_labels_to_indices=self.nodes.lbl2id,
                               map_indices_to_labels=self.nodes.id2lbl,
                               dtype_float=dtype_float,
                               dtype_int=dtype_int,
                               **kwargs)

        self._update_edges()
        self.cache = Cache()

    def __eq__(self, other) -> bool:
        for att in ["edges", "nodes"]:
            if getattr(self, att) != getattr(other, att):
                return False
        return True

    def update(self):
        """Update internal node and edge mappings."""
        self._update_nodes()
        self._update_edges()

    def reset_cache(self, hard: bool = False) -> None:
        self.cache.reset()

    def _update_nodes(self):
        self.nodes.set_mappings()

    def _update_edges(self):
        self._set_edge_indices()

    def _set_edge_indices(self) -> None:
        self.edges.map_labels(self.node2id)
        self._set_edge_id_mapping()

    def _set_edge_id_mapping(self) -> None:
        # create mapping (nodeid, nodeid) -> edgeid
        k_id = [tuple(row) for row in self.edges.indices]
        self._nodes2edge = dict(zip(k_id, range(len(k_id))))

    def _get_dtypes(
            self,
            dtype_int=None,
            dtype_float=None
            ) -> tuple:
        if dtype_int is None:
            dtype_int = self.dtype_int
        if dtype_float is None:
            dtype_float = self.dtype_float
        return (dtype_int, dtype_float)

    def delete_edges(
            self,
            edges,
            update: bool = True,
            **kwargs
            ):
        """Delete edge(s) from Edges object.
        
        Parameters
        ----------
        edges : int or ndarray of ints
            Indices to delete.
        update : bool, default=True
            Whether to reset mapping of node labels to node ids.
        return_indices, default=False
            If True, edge indices of deleted edges are returned.
        
        Returns
        -------
        ndarray, optional
            If return_indices is True, edge indices of deleted nodes are
            returned.
        
        See Also
        --------
        numpy.delete
        """
        self.cache.set_invalid("gamma", "gamma_T")
        ret = self.edges._delete_edges(edges, **kwargs)
        if update is True:
            self._update_edges()
        return ret

    def delete_nodes(
            self,
            nodes,
            update: bool = True,
            is_label: bool = True,
            **kwargs
            ):
        """Delete node(s) from Nodes object.
        
        Parameters
        ----------
        nodes : int, array of ints, str, or array of str
            Indices or labels of nodes to delete.
        update : bool, default=True
            Whether to reset mapping of node labels to node ids.
        is_label : bool, default=True
            Whether to delete by label or internal node index.
        return_indices : bool, default=False
            If True, node indices of deleted nodes are returned.
        
        Returns
        -------
        ndarray, optional
            If return_indices is True, node indices of deleted nodes are
            returned.
        
        See Also
        --------
        numpy.delete
        """
        if is_label is True:
            nodes = self.get_node_id(nodes, vectorize=True)
        ret = self.nodes.delete_nodes(nodes, **kwargs)
        if update is True:
            self._update_nodes()
        return ret

    def delete_nodes_in_edges(
            self,
            nodes,
            update: bool = True,
            is_label: bool = False,
            **kwargs
            ):
        """Delete node(s) from Edges object.
        
        An edge is deleted if either source or target id equals that of
        a node in nodes.
        
        Parameters
        ----------
        nodes : int, array of ints, str, or array of str
            Nodes to delete.
        return_indices : bool, default=False
            If True, edge indices of deleted edges are returned.
        
        Returns
        -------
        ndarray, optional
            If return_indices is True, edge indices of deleted edges
            are returned.
        """
        self.cache.set_invalid("gamma", "gamma_T")
        if is_label is True:
            nodes = self.get_node_id(nodes, vectorize=True)
        ret = self.edges._delete_nodes(nodes, **kwargs)
        if update is True:
            self._update_edges()
        return ret

    def incidence_matrix(self, *args, **kwargs) -> sps.spmatrix:
        """Alias for :func:`Shared.Gamma`."""
        return self.Gamma(*args, **kwargs)

    def Gamma(
            self,
            return_as: str = 'csr',
            transpose: bool = False,
            ) -> sps.spmatrix:
        """Return the incidence matrix Gamma of the network.
        
        Gamma is of shape (m, n) and is defined as::
        
            Gamma[v, e] = 1 if edge e enters vertex v,
            Gamma[v, e] = -1 if edge e leaves vertex v,
            Gamma[v, e] = 0 otherwise.
        
        Parameters
        ----------
        return_as : str, default='csr'
            Sparse matrix type to be returned.
        transpose : bool, default=True
            Whether to transpose Gamma matrix.
        
        Returns
        -------
        Gamma : spmatrix
            Incidence matrix of the network.
        
        See Also
        --------
        scipy.sparse
        
        References
        ----------
        https://en.wikipedia.org/wiki/Incidence_matrix
        
        Examples
        --------
        >>> net = paminco.net.load_sioux()
        >>> net.Gamma().toarray()[:5, :5]
        array([[-1, -1,  1,  0,  1],
               [ 1,  0, -1, -1,  0],
               [ 0,  1,  0,  0, -1],
               [ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0]])
        """
        # Rebuild gamma if neccessary
        if self.cache.is_valid("gamma") is False:
            i = self.edges.indices.T.ravel()
            j = np.hstack([np.array(range(self.m))] * 2)
            vals = np.hstack(([-1] * self.m, [1] * self.m))
            coo = sps.coo_matrix((vals, (i, j)), shape=(self.n, self.m))
            
            # Cache gamma and transpose
            gamma = sparse_format(coo, return_as)
            self.cache["gamma"] = gamma
            self.cache["gamma_T"] = gamma.T.tocsr()
            
        if transpose:
            return sparse_format(self.cache["gamma_T"], return_as)
        return sparse_format(self.cache["gamma"], return_as)

    def adjacency_matrix(self, *args, **kw) -> sps.csr_matrix:
        """Alias for :func:`~Shared.csgraph`."""
        return self.csgraph(*args, **kw)

    def csgraph(
            self,
            weight=None,
            respect_bounds: bool = True,
            backward_positive: bool = False,
            dtype=None,
            ) -> sps.csr_matrix:
        """Get the compressed sparse graph, shape (n, n).
        
        A network/graph with n nodes can be represented by an node to
        node adjacency matrix H. If there is a connection from node i
        to node j, then H[i, j] = w, where w is the weight of the
        connection.
        
        Parameters
        ----------
        weight : ndarray
            Weight on network edges, shape (m, ).
        respect_bounds : bool, default=True
            If True, an undirected edge from s to t with
            ``lb<0 and ub>0`` will lead to separate entries in H. I.e.,
            H[s, t] = w and H[t, s] = w.
        backward_positive : bool, default=False
            Whether to negate weight for undirected edges if
            ``respect_bounds`` is True. I.e.,  H[s, t] = w and
            H[t, s] = -w.
        dtype : dtype, optional
            Datatype of csgraph.
        
        Returns
        -------
        csr_matrix
            Compressed sparse network graph.
        
        Examples
        --------
        >>> net = paminco.net.load_sioux()
        >>> H = net.shared.csgraph(np.arange(net.m) + 1)
        >>> H[:5, :5].toarray()
        array([[ 0.,  1.,  2.,  0.,  0.],
               [ 3.,  0.,  0.,  0.,  0.],
               [ 5.,  0.,  0.,  6.,  0.],
               [ 0.,  0.,  8.,  0.,  9.],
               [ 0.,  0.,  0., 11.,  0.]])
        """
        if dtype is None:
            dtype = self.dtype_float
        
        if weight is None:
            weight = np.ones(self.m)
        
        if respect_bounds is True:
            s, t, w = self.edges.get_directed(weight, backward_positive=backward_positive)
        else:
            if self.cache.is_valid("csgraph") is False:
                s, t = self.edges.indices.T
                w = weight
        
        return sps.csr_matrix((w, (s, t)),
                              shape=(self.n, self.n),
                              dtype=dtype)

    def get_edge_id(
            self,
            nodes
            ):
        """Map node indices to edge indices.
        
        Parameters
        ----------
        nodes : (sequence of) tuple (int, int)
            Node indices (source, target) to be mapped to edge indices.
        
        Returns
        -------
        int, or list of int
            Edge indices for nodes.
        """
        if isinstance(nodes, tuple):
            return self.nodes2edge[nodes]
        return [self.get_edge_id(n) for n in nodes]

    def get_node_id(
            self,
            nodes,
            vectorize: bool = False
            ):
        """Map labels in nodes to node indices.
        
        Parameters
        ----------
        nodes : str or array_like
            Node labels to map to node indices.
        vectorize : bool, default=False
            Whether to vectorize over ``nodes`` (must be ndarray).
            Better performance for larger arrays.
        
        Returns
        -------
        int, list or ndarray
            Node indices.
            
            ``int`` : If nodes is str.
            
            ``ndarray`` : If nodes is ndarray and vectorize is True.
            
            ``list of int`` : else.
        """
        # Single entry
        if isinstance(nodes, str):
            return self.node2id[nodes]
        
        # Vectorize for arrays, better performance for larger arrays
        if (isinstance(nodes, np.ndarray) and 
                (vectorize is True or nodes.ndim > 1)):
            return np.vectorize(self.node2id.__getitem__)(nodes)
        
        if is_iterable(nodes):
            return [self.node2id[n] for n in nodes]
        
        return ValueError("'nodes' must be either str, iterable or array.")

    def get_node_label(
            self,
            nodes,
            vectorize: bool = False
            ):
        """Map indices in nodes to node labels.
        
        Parameters
        ----------
        nodes : int or array_like
            Node indices to map to node labels.
        vectorize : bool, default=False
            Whether to vectorize over ``nodes`` (must be ndarray).
            Better performance for larger arrays.
        
        Returns
        -------
        str, list or ndarray
            Node label(s).
            
            ``str`` : If nodes is int.
            
            ``ndarray`` : If nodes is ndarray and vectorize is True.
            
            ``list of str`` : else.
        
        Raises
        ------
        ValueError:
            Nodes is neither int, ndarray or iterable.
        """
        # Reverse dict
        d = self.nodes.id2lbl
        
        # Case single entry
        if is_int(nodes):
            return d[nodes]
        
        # Vectorize for numpy arrays, better performance for larger arrays
        if (isinstance(nodes, np.ndarray) and 
                (vectorize is True or nodes.ndim > 1)):
            return np.vectorize(d.__getitem__)(nodes)
        
        if is_iterable(nodes):
            return [d[n] for n in nodes]
        
        return ValueError("'nodes' must be either int, iterable or array.")

    @classmethod
    def from_xml(cls, data, **kwargs):
        edges_data = Edges.from_xml(data, return_data=True)
        nodes_data = Nodes.from_xml(data, return_data=True)
        return cls(edges_data, nodes_data, **kwargs)
    
    from_xml.__func__.__doc__ = _doc.from_xml.__doc__

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        sd = self.edges.make_save_dict(prefix=prefix + "edge_",
                                       save_dict=save_dict)
        sd = self.nodes.make_save_dict(prefix=prefix + "node_",
                                       save_dict=sd)
        return sd
    
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
            kw_edges=None,
            kw_nodes=None,
            ):
        """Load Shared from .npz file.
        
        Parameters
        ----------
        data : str or NpzFile
            Filename as str or :class:`~numpy.lib.npyio.NpzFile`.
        prefix : str, default=""
            Object data is stored with ``key = (prefix + internal_name)``.
        kw_edges : keyword arguments, optional
            Further arguments passed to edge constructor.
        kw_nodes : keyword arguments, optional
            Further arguments passed to nodes constructor.
        
        Returns
        -------
        s : Shared
            Object to be shared among network objects.
        """
        if kw_edges is None:
            kw_edges = {}
        if kw_nodes is None:
            kw_nodes = {}
        
        if isinstance(data, str):
            data = np.load(data)
        
        nodes = Nodes.from_npz(data,
                               prefix + "node_",
                               **kw_nodes)
        
        edges = Edges.from_npz(data,
                               prefix + "edge_",
                               **kw_edges)
        return cls(edges, nodes)

    @property
    def flow_direction(self) -> FlowDirection:
        """The direction of flow on the edges.
        
        See Also
        --------
        paminco.net.shared.FlowDirection
        """
        return self.edges.flow_dir

    @property
    def n(self) -> int:
        """Get number of nodes in network."""
        return len(self.nodes)

    @property
    def m(self) -> int:
        """Get number of edges in network."""
        return len(self.edges)

    @property
    def nodes2edge(self) -> dict:
        """Get dict that maps (node_id, node_id) -> edge_id."""
        return self._nodes2edge

    @property
    def node2id(self) -> dict:
        """Get dict that maps node label (str) -> node id (int)."""
        return self.nodes.lbl2id

    @property
    def dtype_int(self):
        """Get int data type, used for node ids in network."""
        return self.edges.dtype_int

    @property
    def dtype_float(self):
        """Get float dtype for network."""
        return self.edges.dtype_float
