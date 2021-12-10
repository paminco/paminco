from __future__ import annotations

import abc
from copy import deepcopy
import xml.etree.ElementTree as et

import numpy as np
import scipy.sparse as sps

from .shared import Shared, ID_UNMAPPED, LBL_UNMAPPED
from paminco.utils.readin import xml_find_root, xml_add_element
from paminco.utils.typing import is_int
from paminco.utils.misc import Cache
from paminco import _doc


TAG_COMMODITIES = "commodities"
DTYPE_INT = np.int32
DTYPE_FLOAT = np.float64


class Commodity(abc.ABC):
    """Abstract class of a commodity."""

    def __repr__(self) -> str:
        return self.__class__.__name__ + " @ " + str(hex(id(self))) + "\n" + str(self)

    @staticmethod
    def from_data(data, **kwargs):
        """Construct commodity from ``data``.
        
        Parameters
        ----------
        data : tuple or dict
        
            ``(source, target, rate)``: a CommoditySingleSourceSink is constructed.
            
            ``dict``: a CommodityMultiSourceSink is constructed.
        
        Returns
        -------
        CommoditySingleSourceSink or CommodityMultiSourceSink
        
        See Also
        --------
        CommoditySingleSourceSink
        CommodityMultiSourceSink
        """
        if isinstance(data, Commodity):
            return data
        if isinstance(data, tuple) and len(data) == 3:
            if data[0] == data[1] or data[2] == 0:
                return None
            return CommoditySingleSourceSink.from_tuple(data, **kwargs)
        elif isinstance(data, dict):
            return CommodityMultiSourceSink(data, **kwargs)
        raise TypeError("Data for commodity must be tuple of len 3 or dict.")

    @abc.abstractmethod
    def get_scaled_copy(self): ...
    @abc.abstractmethod
    def map_node_labels(self): ...
    @abc.abstractmethod
    def map_node_ids(self): ...
    @property
    @abc.abstractmethod
    def source_lbl(self): ...
    @property
    @abc.abstractmethod
    def sink_lbl(self): ...
    @property
    @abc.abstractmethod
    def source_id(self): ...
    @property
    @abc.abstractmethod
    def sink_id(self): ...
    @property
    @abc.abstractmethod
    def rate(self): ...


class CommoditySingleSourceSink(Commodity):
    """Commodity with exactly one source and one sink.
    
    Parameters
    ----------
    source : str or int
        Label or index of source.
    sink : str or int
        Label or index of sink.
    rate : float
        Rate of flow from source to sink.
    is_label : bool, default=True
        If True, source and sink are expected to be str, else as int.
    
    Attributes
    ----------
    has_valid_indices
    has_valid_labels
    source_lbl
    source_id
    sink_lbl
    sink_id
    rate
    total_rate
    """

    def __init__(
            self,
            source,
            sink,
            rate: float,
            is_label: bool = True,
            ) -> None:
        if is_label is True:
            if (isinstance(source, str) is False or
                    isinstance(sink, str) is False):
                raise TypeError(
                    "source and sink must be str if is_label==True."
                )
            self._source_lbl = source
            self._sink_lbl = sink
            self._source_id = ID_UNMAPPED
            self._sink_id = ID_UNMAPPED
        else:
            if is_int(source) is False or is_int(sink) is False:
                raise TypeError(
                    "source and sink must be int if is_label==False."
                )
            self._source_lbl = LBL_UNMAPPED
            self._sink_lbl = LBL_UNMAPPED
            self._source_id = source
            self._sink_id = sink
        self._rate = rate

    def __len__(self) -> int:
        return 2

    def __eq__(self, other) -> bool:
        if isinstance(other, CommoditySingleSourceSink) is False:
            raise TypeError(
                "Comparison object must be of type 'CommoditySingleSourceSink'."
            )
        for att in ["_source_lbl", "_sink_lbl", "_source_lbl", "_sink_lbl", "_rate"]:
            if getattr(self, att) != getattr(other, att):
                return False
        return True

    def __str__(self) -> str:
        if self.has_valid_indices and self.has_valid_labels:
            return (
                f"'{self.source_lbl}' "
                f"({self.source_id}) \u2192 '{self.sink_lbl}' "
                f"({self.sink_id}) | {self.rate}."
            )
        elif self.has_valid_labels:
            return (
                f"'{self.source_lbl}' "
                f"\u2192 '{self.sink_lbl}' | {self.rate}."
            )
        elif self.has_valid_indices:
            return (
                f"{self.source_id} "
                f"\u2192 {self.sink_id} | {self.rate}."
            )
        else:
            raise ValueError("Neither ids not labels have been mapped.")

    def get_scaled_copy(self, scale_factor: float = 1) -> CommoditySingleSourceSink:
        """Get a copy of commodity with rate scaled by ``scale_factor``.
        
        Parameters
        ----------
        scale_factor : float, default=1
            Scale rate from source to sink by scale_factor.
        
        Returns
        -------
        CommoditySingleSourceSink
            Commodity with rate scaled by ``scale_factor``.
        """
        cpy = self.__new__(CommoditySingleSourceSink)
        for att in ["_source_lbl", "_source_id", "_sink_lbl", "_sink_id",
                    "_rate"]:
            if hasattr(self, att):
                setattr(cpy, att, getattr(self, att))
        cpy._rate *= scale_factor
        return cpy

    def get_sparse_elements(self, col_number: int = 0):
        """Extract commodity data for sparse demand matrix.
        
        Parameters
        ----------
        col_number : int, default=0
            Column indices of the matrix entries.
        
        Returns
        -------
        list
            ``[source_id, sink_id]``: Row indices of the matrix entries .
        list
            ``[col_number, col_number]``: Column indices.
        list
            ``[-rate, rate]``: Data.
        """
        rows = [self.source_id, self.sink_id]
        cols = [col_number, col_number]
        data = [-self.rate, self.rate]
        return (rows, cols, data)

    def map_node_labels(self, d: dict) -> None:
        """Map labels -> ids by d."""
        self._source_id = d[self.source_lbl]
        self._sink_id = d[self.sink_lbl]

    def map_node_ids(self, d: dict) -> None:
        """Map ids -> labels by d."""
        self._source_lbl = d[self.source_id]
        self._sink_lbl = d[self.sink_id]

    def node_in_comm(self, node, is_label: bool = True) -> bool:
        """Check whether commodity is made up by node in ``node``.
        
        Parameters
        ----------
        node : int, str or array_like
            Nodes to check for.
        is_label : bool, default=True
            Whether nodes in ``node`` are labels.
        
        Returns
        -------
        bool
            Whether commodity is made up with ``node``.
        """
        # Determine search array
        if is_label is True:
            search_arr = [self.source_lbl, self.sink_lbl]
        else:
            search_arr = [self.source_id, self.sink_id]

        node = np.array(node, copy=False)
        if len(np.intersect1d(node, search_arr)) > 0:
            return True
        return False

    def to_tuple(self, as_label: bool = True) -> tuple:
        """Get Commodity data as tuple.
        
        Parameters
        ----------
        as_label : bool, default=True
            If True, return (source_label, sink_label, rate), else
            (source_id, sink_id, rate).
        
        Returns
        -------
        tuple
            War commodity data.
        """
        if as_label is True:
            return (self.source_lbl, self.sink_lbl, self.rate)
        return (self.source_id, self.sink_id, self.rate)

    @classmethod
    def from_tuple(cls, t, **kwargs) -> CommoditySingleSourceSink:
        """Construct from tuple (source, sink, rate).
        
        Parameters
        ----------
        t : tuple
            ``(source, sink, rate)``: Raw commodity data.
        kwargs : keyword arguments, optional
            Further keyword arguments passed to CommoditySingleSourceSink
            constructor.
        
        Returns
        -------
        CommoditySingleSourceSink
            Commodity object created from ``t``.
        """
        assert isinstance(t, tuple) is True and len(t) == 3
        return cls(*t, **kwargs)

    @classmethod
    def from_labels_and_id(cls, data) -> CommoditySingleSourceSink:
        """Construct from fully specified data (labels and ids).
        
        Parameters
        ----------
        data : tuple
            ``(node_labels, ndoes_ids, rate)`` where ``node_labels``
            is a tuple of str (source_label, sink_label), ``node_ids``
            a tuple of int (source_id, sink_id) and ``rate`` a float.
        
        Returns
        -------
        CommoditySingleSourceSink
            Commodity object created from ``data``.
        """
        c = cls.__new__(cls)
        c._source_lbl, c._sink_lbl = data[0]
        c._source_id, c._sink_id = data[1]
        c._rate = data[2]
        return c

    @property
    def has_valid_indices(self) -> bool:
        """Whether node labels have been mapped to node indices."""
        if self.source_id == ID_UNMAPPED or self.sink_id == ID_UNMAPPED:
            return False
        return True

    @property
    def has_valid_labels(self) -> bool:
        """Whether node indices have been mapped to node labels."""
        if self.source_lbl == LBL_UNMAPPED or self.sink_lbl == LBL_UNMAPPED:
            return False
        return True

    @property
    def source_lbl(self) -> str:
        """str: label of source node."""
        return self._source_lbl

    @property
    def source_id(self) -> int:
        """int: index of source node."""
        return self._source_id

    @property
    def sink_lbl(self) -> str:
        """str: label of sink node."""
        return self._sink_lbl

    @property
    def sink_id(self) -> int:
        """int: index of sink node."""
        return self._sink_id

    @property
    def rate(self) -> float:
        """float: rate of flow from source to sink."""
        return self._rate

    @property
    def total_rate(self) -> float:
        """float: rate of flow from source to sink."""
        return self._rate


class CommodityMultiSourceSink(Commodity):
    """Commodity with multiple sources and/or multiple sinks.
    
    A CommodityMultiSourceSink can be comprised of any number of sources (N_in) and
    any number of sinks (N_out). The total number of nodes that make up a
    CommodityMultiSourceSink is thus N_tot = N_in + N_out.
    
    Parameters
    ----------
    data : dict
        Keys are node labels or ids, values are rate (in/outflow).
    is_label : bool, default=True,
        Whether keys in ``data`` are labels.
    check_validity : bool, default=False
        Check whether sum of values in ``data`` equals 0.
    dtype_float : type, default=numpy.float_
        Datatype to save rate values.
    
    """

    def __init__(
            self,
            data: dict,
            is_label=True,
            check_validity=False,
            dtype_float=np.float_
            ) -> None:
        # get nodes and rates from dict
        node, self._rate = np.array(list(data.items())).T
        self._rate = self._rate.astype(dtype_float)

        if (check_validity is True and
                np.isclose(self._rate.sum(), 0) is False):
            raise ValueError(
                f"Total in/outflow: {self._rate.sum()} ""does not equal 0."
            )
        
        if is_label is True:
            # retrieve nodes as labels
            self._node_labels = node.astype(str)
            self._node_ids = np.ones_like(self._node_labels).astype(int) * ID_UNMAPPED
        else:
            # retrieve nodes as indices
            self._node_ids = node.astype(int)
            self._node_labels = np.array([LBL_UNMAPPED] * len(self._node_ids))

    def __eq__(self, other) -> bool:
        if isinstance(other, CommodityMultiSourceSink) is False:
            raise TypeError(
                "Comparison object must be of type 'CommodityMultiSourceSink'."
            )
        for att in ['_node_labels', '_node_ids', 'rate']:
            if np.array_equal(getattr(self, att), getattr(other, att)) is False:
                return False
        return True

    def __len__(self) -> int:
        return len(self._rate)

    def __str__(self) -> str:
        out = ""
        labels, indices, rates = self._node_labels, self._node_ids, self.rate
        
        def nl(i):
            if i != 0:
                return "\n"
            return ""
        
        if self.has_valid_indices and self.has_valid_labels:
            for i in range(len(self)):
                out += f"{nl(i)}'{labels[i]}' ({indices[i]}) : {rates[i]}"
        elif self.has_valid_labels:
            for i in range(len(self)):
                out += f"{nl(i)}'{labels[i]}' : {rates[i]}"
        elif self.has_valid_indices:
            for i in range(len(self)):
                out += f"{nl(i)}{indices[i]} : {rates[i]}"
        else:
            raise ValueError("Neither ids not labels have been mapped.")
        return out

    def decompose(self, max_iter=None, as_label: bool = True):
        """Decompose CommodityMultiSourceSink into several single commodities.

        Parameters
        ----------
        max_iter : int, optional
            Number of iterations for decomposition
        as_label : bool, default=True
            If True return tuples with node labels, else with node ids.
        
        Returns
        -------
        list
            Single commodities as tuples (source, sink, rate).
        """
        if max_iter is None:
            max_iter = len(self)**2
        
        # Init search and results
        search_array = np.copy(self.rate)
        st_pairs = []
        total_matched = 0
        
        for i in range(max_iter):
            # get most negative source
            idx_s = np.argmin(search_array)
            s = search_array[idx_s]
            
            # get unmatched sink
            idx_t = np.argmax(search_array)
            t = search_array[idx_t]
            
            # matched value
            mv = min(abs(s), t)
            
            # consolidate sink/source
            search_array[idx_s] += mv
            search_array[idx_t] -= mv
            total_matched += mv
            
            # store found pair
            if as_label is True:
                pair = (
                    self._node_labels[idx_s],
                    self._node_labels[idx_t],
                    mv
                )
            else:
                pair = (
                    self._node_ids[idx_s],
                    self._node_ids[idx_t],
                    mv
                )
            st_pairs.append(pair)
            
            # check if consolidation successful
            if (np.isclose(total_matched, self.total_rate)
                    and np.all(np.isclose(search_array, 0))):
                return st_pairs
        
        raise RuntimeError("Could not find s-t-pair decomposition.")

    def node_in_comm(self, node, is_label: bool = True) -> bool:
        # determine search array
        if is_label is True:
            search_arr = self._node_labels
        else:
            search_arr = self._node_ids

        node = np.array(node, copy=False)
        if len(np.intersect1d(node, search_arr)) > 0:
            return True
        return False

    node_in_comm.__doc__ = CommoditySingleSourceSink.node_in_comm.__doc__

    def to_dict(self, labels: bool = True):
        """Get commodity as dict.
        
        Parameters
        ----------
        labels : bool, default=True
            If True, returned dict has node labels as keys, else node indices.
        
        Returns
        -------
        dict
            Commodity as dict.
        """
        if labels is True:
            if self.has_valid_labels:
                return dict(zip(self.node_labels, self.rate))
            if self.has_valid_indices:
                return dict(zip(self.node_ids, self.rate))
            raise ValueError(
                "Commodity has neither valid indices nor valid labels"
            )
        if self.has_valid_indices:
            return dict(zip(self.node_ids, self.rate))
        raise ValueError(
            "Commodity has neither valid indices nor valid labels"
        )

    def get_scaled_copy(self, scale_factor: float = 1):
        """Get copy where every rate is scaled by `scale_factor`.

        Returns
        -------
        CommodityMultiSourceSink
            Commodity with rate scaled by ``scale_factor``.
        """
        cpy = self.__new__(CommodityMultiSourceSink)
        cpy._node_labels = np.copy(self._node_labels)
        cpy._node_ids = np.copy(self._node_ids)
        cpy._rate = np.copy(self._rate) * scale_factor
        return cpy

    def get_sparse_elements(self, col_number: int = 0):
        """Extract commodity data for sparse demand matrix.

        Parameters
        ----------
        col_number : int, default=0
            Column indices of the matrix entries.
        
        Returns
        -------
        list
            Row indices of the matrix entries of len N_tot.
        list
            Column indices of the matrix entries of len N_tot. Equals
            [col_number, ..., col_number].
        list
            Matrix entries of len N_tot.
        """
        rows = self.node_ids.tolist()
        cols = np.repeat(col_number, len(rows)).tolist()
        data = self.rate.tolist()
        return (rows, cols, data)

    def map_node_labels(self, d) -> None:
        self._node_ids = np.vectorize(d.__getitem__)(self._node_labels).astype(int)

    map_node_labels.__doc__ = CommoditySingleSourceSink.map_node_labels.__doc__

    def map_node_ids(self, d) -> None:
        self._node_labels = np.vectorize(d.__getitem__)(self._node_ids).astype(str)

    map_node_ids.__doc__ = CommoditySingleSourceSink.map_node_ids.__doc__

    @classmethod
    def from_label_and_id(cls, data) -> CommodityMultiSourceSink:
        """Construct from fully specified data (labels and ids).
        
        Parameters
        ----------
        data : tuple of ndarray
            ``(labels, ids, rate)`` where every element is ndarray of
            shape (N_tot, ).
        
        Returns
        -------
        CommodityMultiSourceSink
            Commodity constructed from data.
        """
        # make empty object and fill with data
        cm = cls.__new__(cls)
        cm._node_labels = data[0]
        cm._node_ids = data[1]
        cm._rate = data[2]
        return cm

    @property
    def N_tot(self) -> int:
        """Total number of nodes in commodity: sinks + sources."""
        return len(self)

    @property
    def N_in(self) -> int:
        """Number of sources."""
        return self.N_tot - self.N_out

    @property
    def N_out(self) -> int:
        """Number of sinks."""
        return sum(self.is_sink)

    @property
    def is_sink(self) -> np.ndarray:
        """ndarray (N_tot, ) of bool: whether node is sink."""
        return (np.sign(self.rate) + 1).astype(bool)

    @property
    def has_valid_indices(self) -> bool:
        """bool: whether nodes have proper indices."""
        return not (ID_UNMAPPED in self.node_ids)
    
    @property
    def has_valid_labels(self) -> bool:
        """bool: whether nodes have proper labels."""
        return not (LBL_UNMAPPED in self.node_labels)

    @property
    def is_single(self) -> bool:
        """bool: whether commodity has single source and single sink."""
        if len(self) == 2 and len(self.sink_id) == 1:
            return True
        return False

    @property
    def node_ids(self) -> np.ndarray:
        """ndarray (N_tot, ) of int: node indices."""
        return self._node_ids

    @property
    def node_labels(self) -> np.ndarray:
        """ndarray (N_tot, ) of str: node labels."""
        return self._node_labels

    @property
    def source_lbl(self) -> np.ndarray:
        """ndarray (N_in, ) of str: source labels."""
        return self._node_labels[~self.is_sink]

    @property
    def source_id(self) -> np.ndarray:
        """ndarray (N_in, ) of int: source indices."""
        return self._node_ids[~self.is_sink]

    @property
    def sink_lbl(self) -> np.ndarray:
        """ndarray (N_out, ) of str: sink labels."""
        return self._node_labels[self.is_sink]

    @property
    def sink_id(self) -> np.ndarray:
        """ndarray (N_out, ) of int: sink indices."""
        return self._node_ids[self.is_sink]

    @property
    def rate(self) -> np.ndarray:
        """ndarray (N_tot, ) of float: in/outflow per node."""
        return self._rate

    @property
    def total_rate(self) -> float:
        """float: total flow on commodity."""
        return self._rate[self.is_sink].sum()


class _DV(abc.ABC):
    """Base class of a demand vector."""
    
    def __init__(
            self,
            shared=None,
            dtype_int=None,
            dtype_float=None,
            ) -> None:
        self._shared = shared
        self._dtype_int = dtype_int
        self._dtype_float = dtype_float

    def __call__(self, *args, **kw) -> sps.csc_matrix:
        """Get demand as sparse matrix of shape (n, k).
        
        Every column represents the in/outflow for the network nodes
        for one commodity.
        """
        return self.sparse(*args, **kw)

    def get_source_sink_rate(self, scale_by: float = 1):
        """Return a source-sink-rate decomposition of this demand vector.
        
        Parameters
        ----------
        scale_by : float, default=1
            The demand vector is scaled by this factor before determining
            the s-t-r-decompositions

        Returns
        -------
        decomposition : tuple
            A tuple ``(s, t, r)`` containing sources ``s`` sinks ``t`` and
            associated demand rates ``r`` that induce this demand vector
        """
        if self.all_single is False:
            raise ValueError("TODO.")

        # Retrieve rates
        b = self()
        rates = scale_by * b.data[b.data > 0]
        
        # Collect (cached) sources
        if self.cache.is_valid("source_indices") is False:
            sources = self.cache["source_indices"] = b.indices[b.data < 0]
        else:
            sources = self.cache["source_indices"]
        
        # Collect (cached) sinks
        if self.cache.is_valid("sink_indices") is False:
            sinks = self.cache["sink_indices"] = b.indices[b.data > 0]
        else:
            sinks = self.cache["sink_indices"]
        
        return sources, sinks, rates

    def add_to_etree(
            self,
            root: et.Element,
            name=None,
            ) -> None:
        """Add demand data to xml.etree.ElementTree.Element.
        
        Data will be added based on self.value(param=1). Note that only
        LinearDemandFunctions can be properly saved to xml.
        
        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Element to which 'commodities' will be appended to.
        name : str, optional
            Attribute name of 'commodities' element. If None, no attribute
            will be set.
        """
        commodities = et.SubElement(root, 'commodities')
        if name is not None:
            commodities.attrib["name"] = name
        
        spmat = self.sparse()
        for i in range(spmat.shape[1]):
            col = spmat[:, i]
            lbls = self.shared.get_node_label(col.indices)
            data = col.data
            com_node = et.SubElement(commodities, 'commodity')
            
            if len(data) == 2:
                # Commodity with single source and single sink
                com_node.attrib['from'] = lbls[np.where(data < 0)[0][0]]
                com_node.attrib['to'] = lbls[np.where(data > 0)[0][0]]
                com_node.attrib['rate'] = str(abs(data[0]))
            else:
                # Commodity with multiple sources and/or multipls sinks
                com_dict = dict(zip(lbls, data))
                for (elem_lbl, elem_val) in com_dict.items():
                    if elem_val != 0:
                        b_node = et.SubElement(com_node, 'b')
                        b_node.attrib['node'] = elem_lbl
                        b_node.attrib['value'] = str(elem_val)

    @classmethod
    def from_npz(
            cls,
            data,
            shared=None,
            prefix: str = "",
            **kwargs
            ) -> DemandVector | DemandVectorSP:
        if isinstance(data, str):
            data = np.load(data)
        if prefix + "node_labels" in data:
            return DemandVectorSP.from_npz(data, shared=shared, prefix=prefix, **kwargs)
        return DemandVector.from_npz(data, shared=shared, prefix=prefix, **kwargs)
    
    from_npz.__func__.__doc__ = _doc.from_npz_shared.__doc__

    @property
    def k(self) -> int:
        """Number of commodities in demand vector."""
        return len(self)
    
    @property
    def n(self) -> int:
        if self.shared is not None:
            return self.shared.n
        elif hasattr(self, "_n"):
            return self._n
        raise AttributeError(
            "Number of nodes has not been set: Neither shared object nor "
            "number of nodes specified."
        )
    
    @n.setter
    def n(self, val: int) -> None:
        self._n = val
    
    @property
    def shared(self):
        """Shared object for network objects.
        
        See Also
        --------
        paminco.net.shared.Shared
        """
        return self._shared
    
    @shared.setter
    def shared(self, val: int) -> None:
        self._shared = val
    
    @property
    def dtype_int(self):
        if self._dtype_int is not None:
            return self._dtype_int
        elif self.shared is not None:
            return self.shared.dtype_int
        return DTYPE_INT
    
    @dtype_int.setter
    def dtype_int(self, val) -> None:
        self._dtype_int = val
    
    @property
    def dtype_float(self):
        if self._dtype_float is not None:
            return self._dtype_float
        elif self.shared is not None:
            return self.shared.dtype_float
        return DTYPE_FLOAT
    
    @dtype_float.setter
    def dtype_float(self, val) -> None:
        self._dtype_float = val

    @abc.abstractmethod
    def delete_nodes(self): ...
    @abc.abstractmethod
    def to_single_pairs(self): ...
    @abc.abstractmethod
    def map_node_label_to_id(self): ...
    @abc.abstractmethod
    def map_node_id_to_label(self): ...
    @abc.abstractmethod
    def scaled_copy(self): ...
    @abc.abstractmethod
    def sparse(self): ...
    @abc.abstractmethod
    def make_save_dict(self): ...
    @abc.abstractmethod
    def save_to_numpy(self): ...
    @property
    @abc.abstractmethod
    def all_single(self): ...
    @property
    @abc.abstractmethod
    def rate(self): ...
    @property
    @abc.abstractmethod
    def source_lbl(self): ...
    @property
    @abc.abstractmethod
    def sink_lbl(self): ...
    @property
    @abc.abstractmethod
    def source_id(self): ...
    @property
    @abc.abstractmethod
    def sink_id(self): ...


class DemandVector(_DV):
    """Class that builds on a number of commodities (k).
    
    A DemandVector object can be instantiated in several ways:
        DemandVector(dv)
            where ``dv`` is a DemandVector.
        DemandVector(M)
            where ``M`` is matrix (ndarray or spmatrix) with node
            indices as rows and commodities as columns.
        DemandVector(it)
            where ``it`` an iterable with elements that are tuples or
            dicts.
    
    Parameters
    ----------
    shared : Shared
        Shared object for all network objects.
    data : DemandVector, ndarray, spmatrix, or iterable
        (Processed) commodity data.
    copy : bool, default=True
        Whether to copy data in ``data``.
    is_label : bool, default=True
        Whether commodity nodes are given as labels or ids in data if
        data is iterable.
    
    Attributes
    ----------
    k
    shared
    all_single
    commodities
    source_lbl
    sink_lbl
    source_id
    sink_id
    rate
    total_rate
    
    See Also
    --------
    Commodity
    CommoditySingleSourceSink
    CommodityMultiSourceSink
    """

    def __init__(
            self,
            data,
            shared=None,
            dtype_int=None,
            dtype_float=None,
            copy: bool = True,
            is_label: bool = True
            ) -> None:
        super().__init__(
            shared=shared,
            dtype_int=dtype_int,
            dtype_float=dtype_float,
        )
        if isinstance(data, DemandVector):
            if copy is True:
                self._comm = deepcopy(data._comm)
            else:
                self._comm = data._comm
        elif isinstance(data, (np.ndarray, sps.spmatrix)):
            self._comm = []
            for j in range(data.shape[1]):
                idx = data[:, j] != 0
                ids = idx.nonzero()[0]
                vals = np.array(data[:, j][idx]).flatten()
                d = dict(zip(ids, vals))
                self._comm.append(CommodityMultiSourceSink(d, is_label=False))
        else:
            self._comm = [Commodity.from_data(c, is_label=is_label) for c in data]
            self._comm = [c for c in self._comm if c is not None]
        
        self.cache = Cache(counter=True)

    def __eq__(self, other) -> bool:
        assert len(self) == len(other), \
            "DemandVectors must have same length."
        for (c1, c2) in list(zip(self._comm, other._comm)):
            if c1 != c2:
                return False
        return True

    def __len__(self) -> int:
        return len(self._comm)

    def reset_cache(self, hard: bool = False) -> None:
        self.cache.reset(hard=hard)
    
    reset_cache.__doc__ = _doc.reset_cache.__doc__

    def delete_nodes(self, nodes) -> int:
        """Delete all commodities that are made up by ``nodes``.
        
        Parameters
        ----------
        nodes : int or array_like
            Indices of nodes to remove.
        
        Returns
        -------
        int
            Number of removed commodities.
        """
        len_before = len(self)
        self._comm = [c for c in self._comm if not c.node_in_comm(nodes, is_label=False)]
        self.reset_cache()
        
        return len_before - len(self)

    def map_node_label_to_id(self) -> None:
        """Map labels -> ids for all commodities."""
        for c in self._comm:
            try:
                c.map_node_labels(self.shared.node2id)
            except KeyError as e:
                raise KeyError(f"Node with label {e} not in network nodes.")
        self.cache.set_invalid("source_id", "sink_id")

    def map_node_id_to_label(self) -> None:
        """Map ids -> labels for all commodities."""
        d = {v: k for k, v in self.shared.node2id.items()}
        for c in self._comm:
            try:
                c.map_node_ids(d)
            except KeyError as e:
                raise KeyError(f"Node with id {e} not in network nodes.")
        self.cache.set_invalid("source_lbl", "sink_lbl")

    def scaled_copy(self, f: float = 1):
        """Get a copy where rate of every commodity is scaled by ``f``.
        
        Parameters
        ----------
        f : float, default=1
            Scale factor.
        
        Returns
        -------
        DemandVector
            Copy of demand vector with scaled commodites.
        """
        cpy = self.__new__(DemandVector)
        cpy._s = self._s
        cpy._comm = [c.get_scaled_copy(f) for c in self._comm]
        cpy.cache = Cache()
        return cpy

    def to_single_pairs(self, as_label: bool = True):
        """Decompose all CommodityMultiSourceSink.
        
        Parameters
        ----------
        as_label : bool, default=True
            If True decomposed commodites are set up by labels, else
            by id.
        
        Returns
        -------
        DemandVectorSP
            Demand vector with single commodities only.
        
        See Also
        --------
        CommodityMultiSourceSink.decompose
        """
        data = []
        for c in self._comm:
            if isinstance(c, CommodityMultiSourceSink):
                data.extend(c.decompose(as_label=as_label))
            else:
                data.append(c.to_tuple(as_label=as_label))
        dv = DemandVectorSP(data, shared=self.shared, is_label=as_label)
        
        if as_label is True:
            dv.map_node_label_to_id()
        
        return dv

    def sparse(self, dtype=None) -> sps.csc_matrix:
        """Get demand as sparse matrix.

        Total in/outflow per node (row) per commodity (column), e.g.:
            Data::
            
                (0, 2, 100)            [single commodity]
                {1: 33, 2: 33, 3: -66} [multi commodity]
            
            Resulting matrix::
            
                [[-100,    0],
                 [   0,   33],
                 [ 100,   33],
                 [   0,  -66]]
        
        Parameters
        ----------
        dtype: dtype, optional
            Datatype of returned sparse matrix.
        
        Returns
        -------
        scipy.sparse.csc_matrix
            Demand as sparse matrix.
        """
        if dtype is None:
            dtype = self.dtype_float
        
        if self.cache.is_valid("sparse") is False:
            rows, cols, data = [], [], []
            
            for (i, comm) in enumerate(self._comm):
                r, c, d = comm.get_sparse_elements(col_number=i)
                rows += r
                cols += c
                data += d
            
            sparse = sps.coo_matrix((data, (rows, cols)),
                                    shape=(self.n, len(self)),
                                    dtype=dtype)
            self.cache["sparse"] = (sps.csc_matrix(sparse), sparse)
        
        # recast datatype if necessary for return
        return sps.csc_matrix(self.cache["sparse"][0], dtype=dtype)

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        if "demand_type" not in save_dict:
            save_dict["demand_type"] = self.__class__.__name__
        
        idx_single = []
        idx_multi = []
        labels = []
        ids = []
        rates = []
        
        # collect all single commodites and remember indices
        for (i, c) in enumerate(self._comm):
            if isinstance(c, CommoditySingleSourceSink):
                idx_single.append(i)
                labels.append([getattr(c, "_source_lbl"), getattr(c, "_sink_lbl")])
                ids.append([getattr(c, "_source_id"), getattr(c, "_sink_id")])
                rates.append([getattr(c, "_rate")])
            elif isinstance(c, CommodityMultiSourceSink):
                idx_multi.append(i)
                save_dict[prefix + "cm_" + str(i) + "id"] = getattr(c, "_node_ids")
                save_dict[prefix + "cm_" + str(i) + "label"] = getattr(c, "_node_labels")
                save_dict[prefix + "cm_" + str(i) + "rate"] = getattr(c, "_rate")
        
        save_dict[prefix + "cs_labels"] = np.array(labels)
        save_dict[prefix + "cs_ids"] = np.array(ids)
        save_dict[prefix + "cs_rates"] = np.array(rates)
        save_dict[prefix + "idx_single"] = np.array(idx_single)
        save_dict[prefix + "idx_multi"] = np.array(idx_multi)
        
        return save_dict

    make_save_dict.__doc__ = _doc.make_save_dict.__doc__

    def save_to_numpy(self, file: str, **kwargs) -> None:
        save_dict = self.make_save_dict()
        save_dict.update(kwargs)
        np.savez(file, **save_dict)
    
    save_to_numpy.__doc__ = _doc.save_to_numpy.__doc__

    @classmethod
    def from_xml(cls): pass
    # TODO-PJ linmk to func

    @classmethod
    def from_npz(
            cls,
            data,
            shared=None,
            prefix: str = "",
            **kwargs
            ) -> DemandVector:
        # load data
        if isinstance(data, str):
            data = np.load(data)
        
        cs_labels = data[prefix + "cs_labels"]
        cs_ids = data[prefix + "cs_ids"]
        cs_rates = data[prefix + "cs_rates"]
        idx_single = data[prefix + "idx_single"].tolist()
        idx_multi = data[prefix + "idx_multi"].tolist()
        if isinstance(idx_single, int):
            idx_single = [idx_single]
        if isinstance(idx_multi, int):
            idx_multi = [idx_multi]
        
        comm = []
        # load single commodites
        if len(cs_labels) > 0:
            for d in list(zip(cs_labels, cs_ids, cs_rates.ravel())):
                comm.append(CommoditySingleSourceSink.from_labels_and_id(d))
        
        # load multi commodities
        if len(idx_multi) > 0:
            for idx_m in idx_multi:
                b = prefix + "cm_" + str(idx_m)
                d = [data[b + "label"], data[b + "id"], data[b + "rate"]]
                comm.append(CommodityMultiSourceSink.from_label_and_id(d))
        
        # reindex commodities
        idx_full = idx_single + idx_multi
        comm = [c for idx, c in sorted(zip(idx_full, comm))]
        
        return cls(comm, shared=shared, **kwargs)
    
    from_npz.__func__.__doc__ = _doc.from_npz_shared.__doc__

    @property
    def all_single(self) -> bool:
        """Whether is comprised only of commodities with single sink / source."""
        if self.cache.is_valid("sparse") is False:
            m = self.sparse()
            return (m.getnnz(axis=0) == 2).all()
        return (self.cache["sparse"][1].getnnz(axis=0) == 2).all()

    @property
    def commodities(self) -> list:
        """List [len k] of commdodities: CommoditySingleSourceSink or CommodityMultiSourceSink."""
        return self._comm

    @property
    def source_lbl(self) -> list:
        """List [len k] of str or ndarray: source labels per commodity."""
        if self.cache.is_valid("source_lbl") is False:
            v = self.cache["source_lbl"] = [c.source_lbl for c in self._comm]
            return v
        return self.cache["source_lbl"]

    @property
    def sink_lbl(self) -> list:
        """List [len k] of str or ndarray: sink labels per commodity."""
        if self.cache.is_valid("sink_lbl") is False:
            v = self.cache["sink_lbl"] = [c.sink_lbl for c in self._comm]
            return v
        return self.cache["sink_lbl"]

    @property
    def source_id(self) -> list:
        """List [len k] of int or ndarray: source indices per commodity."""
        if self.cache.is_valid("source_id") is False:
            v = self.cache["source_id"] = [c.source_id for c in self._comm]
            return v
        return self.cache["source_id"]

    @property
    def sink_id(self) -> list:
        """List [len k] of int or ndarray: sink indices per commodity."""
        if self.cache.is_valid("sink_id") is False:
            v = self.cache["sink_id"] = [c.sink_id for c in self._comm]
            return v
        return self.cache["sink_id"]

    @property
    def rate(self) -> list:
        """List [len k] of float or ndarray: node in/outflow per commodity."""
        if self.cache.is_valid("rate") is False:
            v = self.cache["rate"] = [c.rate for c in self._comm]
            return v
        return self.cache["rate"]

    @property
    def total_rate(self) -> np.ndarray:
        """np.ndarray (k, ): total flow per commodity."""
        if self.cache.is_valid("total_rate") is False:
            v = self.cache["total_rate"] = np.array([c.total_rate for c in self._comm])
            return v
        return self.cache["total_rate"]


class DemandVectorSP(_DV):
    """Class that builds on a number of commodities `k`.
    
    A DemandVectorSP object can be instantiated in several ways:
        DemandVectorSP(dv)
            where ``dv`` is a DemandVectorSP.
        DemandVectorSP(arr)
            where ``arr`` is array_like. The resulting array from
            ``numpy.array(arr)`` is expected to look like::

                [source_1,   sink_1,   rate_1]
                            ...
                [source_k,   sink_k,   rate_k]
    
    Parameters
    ----------
    data : DemandVector or array_like
        Commodity data in processed or unprocessed form.
    shared : Shared, optional
        Shared object for all network objects.
    copy : bool, default=True
        Whether to copy data in ``data``.
    is_label : bool, default=True
        Whether commodity nodes are given as labels or ids in data.
    
    Attributes
    ----------
    k
    shared
    all_single
    rate
    total_rate
    source_lbl
    sink_lbl
    source_id
    sink_id
    """

    def __init__(
            self,
            data,
            shared=None,
            dtype_int=None,
            dtype_float=None,
            copy: bool = True,
            is_label: bool = True
            ) -> None:
        super().__init__(
            shared=shared,
            dtype_int=dtype_int,
            dtype_float=dtype_float,
        )
        if isinstance(data, DemandVectorSP):
            self._node_labels = np.array(data._node_labels, copy=copy)
            self._node_ids = np.array(data._node_ids, copy=copy)
            self._rates = np.array(data._rates, copy=copy)
        elif isinstance(data, (list, np.ndarray)):
            data = np.array(data, copy=False)
            if is_label is True:
                self._node_labels = data[:, :2].astype(str)
                self._node_ids = np.ones_like(self._node_labels, dtype=int) * ID_UNMAPPED
            else:
                self._node_ids = data[:, :2].astype(int)
                self._node_labels = np.full(self._node_ids.shape, LBL_UNMAPPED)
            self._rates = data[:, 2].astype(float)
        else:
            raise TypeError("Invalid input data.")
        
        # remove zero rate commodities
        zc = (self._rates == 0)
        self._rates = np.delete(self._rates, zc)
        self._node_ids = np.delete(self._node_ids, zc, axis=0)
        self._node_labels = np.delete(self._node_labels, zc, axis=0)
        
        self.cache = Cache()

    def __len__(self) -> int:
        return len(self._rates)

    def __eq__(self, other) -> bool:
        for att in ["_rates", "_node_ids", "_node_labels"]:
            if np.array_equal(getattr(self, att), getattr(other, att)) is False:
                return False
        return True

    def reset_cache(self, hard: bool = False) -> None:
        self.cache.reset(hard=hard)
    
    reset_cache.__doc__ = _doc.reset_cache.__doc__

    def map_node_label_to_id(self) -> None:
        try:
            self._node_ids = (self.shared.get_node_id(self._node_labels,
                                                      vectorize=True)
                              .astype(int))
        except KeyError as e:
            raise KeyError("Node id " + str(e) + " not in network nodes.")
    
    map_node_label_to_id.__doc__ = DemandVector.map_node_label_to_id.__doc__

    def map_node_id_to_label(self) -> None:
        try:
            self._node_labels = (self.shared.get_node_label(self._node_ids,
                                                            vectorize=True)
                                 .astype(str))
        except KeyError as e:
            raise KeyError("Node id " + str(e) + " not in network nodes.")

    map_node_id_to_label.__doc__ = DemandVector.delete_nodes.__doc__

    def scaled_copy(self, f: float = 1):
        """Get a copy where every commodity is scaled by `f`.
        
        Parameters
        ----------
        f : float, default=1
            Scale factor.
        
        Returns
        -------
        DemandVectorSP
            Copy of DemandVectorSP with scaled commodites.
        """
        cpy = DemandVectorSP(self, copy=True)
        cpy._rates *= f
        return cpy

    def to_single_pairs(self, as_label: bool = True) -> DemandVectorSP:
        """Returns self.
        
        Used for consistency with DemandVector.to_single_pairs
        """
        return self

    def delete_nodes(self, nodes) -> int:
        len_before = len(self)
        nodes = np.array(nodes).reshape(1, -1)
        
        # identify commodities where source or sink equals nodes
        idx_source = ((self.source_id.reshape(-1, 1) - nodes) == 0).any(axis=1)
        idx_sink = ((self.sink_id.reshape(-1, 1) - nodes) == 0).any(axis=1)
        all_idx = idx_source | idx_sink
        
        # delete indices in data arrays
        self._rates = np.delete(self._rates, all_idx, axis=0)
        self._node_labels = np.delete(self._node_labels, all_idx, axis=0)
        self._node_ids = np.delete(self._node_ids, all_idx, axis=0)
        
        self.reset_cache()
        
        return len_before - len(self)
    
    delete_nodes.__doc__ = DemandVector.delete_nodes.__doc__

    def sparse(self, dtype=None) -> sps.csc_matrix:
        """Get demand as sparse matrix.
        
        Total in/outflow per node (row) per commodity (column), e.g.,:
            Data::
            
                (0, 2, 100)
                (3, 1, 66)
            
            Resulting matrix::
            
                [[-100,    0],
                 [   0,   66],
                 [ 100,    0],
                 [   0,  -66]]
        
        Parameters
        ----------
        dtype: dtype, optional
            Datatype of returned sparse matrix.
        
        Returns
        -------
        scipy.sparse.csc_matrix
            Demand as sparse matrix.
        """""
        if dtype is None:
            dtype = self.dtype_float
        
        if self.cache.is_valid("sparse") is False:
            rows = self._node_ids.ravel()
            cols = np.repeat(range(len(self)), 2)
            data = (np.array([-1, 1]).reshape(-1, 1) * self.rate.T).T.ravel()
            sparse = sps.coo_matrix((data, (rows, cols)),
                                    shape=(self.n, len(self)),
                                    dtype=dtype)
            self.cache["sparse"] = (sps.csc_matrix(sparse), sparse)
            return sps.csc_matrix(sparse, dtype=dtype)
        # recast datatype if necessary for return
        return sps.csc_matrix(self.cache["sparse"][0], dtype=dtype)

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        if "demand_type" not in save_dict:
            save_dict["demand_type"] = self.__class__.__name__
        
        save_att = {
            "node_labels": "_node_labels",
            "node_ids": "_node_ids",
            "rates": "_rates",
        }
        for (k, v) in save_att.items():
            save_dict[prefix + k] = getattr(self, v)
        
        return save_dict
    
    make_save_dict.__doc__ = _doc.make_save_dict.__doc__

    def save_to_numpy(self, file: str, **kwargs) -> None:
        save_dict = self.make_save_dict()
        save_dict.update(kwargs)
        np.savez(file, **save_dict)
    
    save_to_numpy.__doc__ = _doc.save_to_numpy.__doc__

    @classmethod
    def from_xml(cls): pass
    # TODO-PJ linmk to func

    @classmethod
    def from_npz(
            cls,
            data,
            shared=None,
            prefix: str = "",
            **kwargs
            ) -> DemandVectorSP:
        # load np data
        if isinstance(data, str):
            data = np.load(data)
        
        # make empty edge object and fill with data
        dv = cls.__new__(cls)
        if len(data[prefix + "rates"]) == 0:
            dv._node_labels = np.empty((0, 2), str)
            dv._node_ids = np.empty((0, 2), int)
            dv._rates = np.empty(0)
        else:
            dv._node_labels = data[prefix + "node_labels"]
            dv._node_ids = data[prefix + "node_ids"]
            dv._rates = data[prefix + "rates"]
        dv._s = shared
        dv.cache = Cache()
        
        return dv
    
    from_npz.__func__.__doc__ = _doc.from_npz_shared.__doc__

    @property
    def all_single(self) -> bool:
        return True
    
    all_single.__doc__ = DemandVector.all_single.__doc__

    @property
    def rate(self) -> np.ndarray:
        """ndarray (k, ) of flow: rates."""
        return self._rates

    @property
    def source_lbl(self) -> np.ndarray:
        """ndarray (k, ) of str: source labels."""
        return self._node_labels[:, 0]

    @property
    def sink_lbl(self) -> np.ndarray:
        """ndarray (k, ) of str: sink labels."""
        return self._node_labels[:, 1]

    @property
    def source_id(self) -> np.ndarray:
        """ndarray (k, ) of int: source indices."""
        return self._node_ids[:, 0]

    @property
    def sink_id(self) -> np.ndarray:
        """ndarray (k, ) of int: sink indices."""
        return self._node_ids[:, 1]

    @property
    def total_rate(self) -> np.ndarray:
        return self.rate

    total_rate.__doc__ = DemandVector.total_rate.__doc__


def read_comm_from_xml(data):
    """Read commodity data from XML.
    
    Parameters
    ----------
    data : str, xml.etree.ElementTree.ElementTree, or xml.etree.ElementTree.Element
        XML file, tree or root itself.
    
    Returns
    -------
    com_data : list, optional
        Commodity data, one commodity per element in list. A tuple
        ``(source, target, rate) `` corresponds to a commodity with a
        one source and one target. Otherwise a dict with the out/inflow
        is appended, e.g., {node1: -100, node2: 50, node3: 50}.
    """
    if isinstance(data, et.Element) is False:
        data = xml_find_root(data)
    if data.tag != TAG_COMMODITIES:
        data = data.find(TAG_COMMODITIES)
    if data is None:
        return None
    return _comm_from_xml_node(data)


def _comm_from_xml_node(xml_commodity_node: et.Element) -> list:
    com_data = []
    for c in xml_commodity_node:
        if all(k in c.attrib for k in ("from", "to", "rate")):
            data = (c.attrib["from"], c.attrib["to"], float(c.attrib["rate"]))
            com_data.append(data)
        else:
            c_info = dict()
            b_nodes = c.findall("b")
            for b in b_nodes:
                c_info[b.attrib['node']] = float(b.attrib['value'])
            com_data.append(c_info)
    
    return com_data


def demand_vector(
        data,
        shared=None,
        dtype_int=None,
        dtype_float=None,
        single_pairs: bool = False,
        **kwargs
        ) -> DemandVector | DemandVectorSP:
    """Setup demand vector based on ``data``.
    
    Parameters
    ----------
    data : DemandVector, DemandVectorSP, spmatrix, ndarray, or iterable
    
        ``DemandVector or DemandVectorSP``: (a copy) of data will be
        returned.
        
        ``spmatrix or ndarray``: a DemandVector will be inferred.
        
        ``iterable and single_pairs == False`` : DemandVector.
        
        ``iterable and single_pairs == True`` : DemandVectorSP.
    
    shared : Shared
        Shared object for all network objects.
    single_pairs : bool, default=False,
        If True, infer DemandVectorSP, else DemandVector.
    kwargs : keyword arguments, optional
        Further keyword arguments passed to constructor.
    
    Returns
    -------
    DemandVector or DemandVectorSP
        Demand vector.
    
    See Also
    --------
    paminco.net.demand.DemandVector
    paminco.net.demand.DemandVectorSP
    """
    # Parse from xml
    if isinstance(data, str):
        data = read_comm_from_xml(data)
    
    if isinstance(data, DemandVector):
        return DemandVector(
            data,
            shared=shared,
            dtype_int=dtype_int,
            dtype_float=dtype_float,
            **kwargs
        )
    elif isinstance(data, DemandVectorSP):
        return DemandVectorSP(
            data,
            shared=shared,
            dtype_int=dtype_int,
            dtype_float=dtype_float,
            **kwargs
        )
    elif isinstance(data, np.ndarray) or sps.isspmatrix(data):
        return DemandVector(
            data,
            shared=shared,
            dtype_int=dtype_int,
            dtype_float=dtype_float,
            **kwargs
        )
    else:
        if ((isinstance(data, tuple) and len(data) == 3) or
                isinstance(data, dict)):
            data = [data]
        
        if single_pairs is True:
            return DemandVectorSP(
                data,
                shared=shared,
                dtype_int=dtype_int,
                dtype_float=dtype_float,
                **kwargs
            )
        else:
            return DemandVector(
                data,
                shared=shared,
                dtype_int=dtype_int,
                dtype_float=dtype_float,
                **kwargs
            )


class DemandFunction(abc.ABC):
    """Abstract class representing a demand function.
    
    Parameters
    ----------
    shared : Shared, optional
        Shared network object used for label <-> id mappings.
        
    Attributes
    ----------
    cache : Cache
        A cache.
    shared
    all_single_pairs
    shared
    demand_vectors
    is_single_commodity
    is_multi_commodity
    value_at_1
    unique_sources
    """

    def __init__(
            self,
            shared=None,
            dtype_int=None,
            dtype_float=None,
            ) -> None:
        self._shared = shared
        self._dtype_int = dtype_int
        self._dtype_float = dtype_float
        self.cache = Cache()

    def __call__(self, param: float = 1) -> sps.csc_matrix:
        return self.value(param)

    def __eq__(self, other) -> bool:
        assert len(self) == len(other), \
            "DemandFunctions must have same number of commodities."
        for (dv1, dv2) in list(zip(self.demand_vectors, other.demand_vectors)):
            if dv1 != dv2:
                return False
        return True

    def __len__(self) -> int:
        """Number of commodities modelled by the demand function."""
        return len(self.b)

    def reset_cache(self, hard: bool = False) -> None:
        self.cache = Cache()
    
    reset_cache.__doc__ = _doc.reset_cache.__doc__

    def inflow(self, param: float) -> np.ndarray:
        """Get the total inflow of all commodities for demand(param).
        
        Parameters
        ----------
        param : float
        
        Returns
        -------
        np.ndarray
            Total inflow of all commodities.
        """
        b = self.value(param)
        return np.array(b.multiply(b > 0).sum(axis=0), copy=False).ravel()

    def ddx_inflow(self, param: float) -> np.ndarray:
        """Get the total inflow of all commodities for demand'(param).
        
        Parameters
        ----------
        param : float
        
        Returns
        -------
        np.ndarray
            Total inflow of all commodities.
        """
        b = self.ddx(param)
        return np.array(b.multiply(b > 0).sum(axis=0), copy=False).ravel()

    def max_inflow(
            self,
            max_param: float,
            min_param: float = 0,
            ) -> np.ndarray:
        """Get maxmimum inflow for every commodity.
        
        Parameters
        ----------
        max_param : float
        
        min_param : float, default=0
        
        Returns
        -------
        ndarray
            Ndarray of shape (k, ): max inflow for every commodity.
        """
        b_min = self.value(min_param)
        b_max = self.value(max_param)
        inflow_min = np.array(b_min.multiply(b_min > 0).sum(axis=0), copy=False).ravel()
        inflow_max = np.array(b_max.multiply(b_max > 0).sum(axis=0), copy=False).ravel()
        return np.max(np.vstack((inflow_min, inflow_max)).T, axis=1)

    def delete_nodes(self, *args, **kwargs) -> int:
        len_before = len(self)
        for dv in self.demand_vectors.values():
            dv.delete_nodes(*args, **kwargs)
        return len_before - len(self)
    
    delete_nodes.__doc__ = DemandVector.delete_nodes.__doc__

    def map_node_label_to_id(self) -> None:
        for dv in self.demand_vectors.values():
            dv.map_node_label_to_id()
    
    map_node_label_to_id.__doc__ = DemandVector.map_node_label_to_id.__doc__

    def map_node_id_to_label(self) -> None:
        for dv in self.demand_vectors.values():
            dv.map_node_id_to_label()
    
    map_node_id_to_label.__doc__ = DemandVector.map_node_id_to_label.__doc__

    def to_single_pairs(self, **kwargs) -> None:
        """Decompose all commodities in demand vectors."""
        for k in self._demand_vectors:
            v = getattr(self, k)
            setattr(self, k, v.to_single_pairs(**kwargs))

    def add_to_etree(
            self,
            root: et.Element,
            overwrite: bool = True,
            ) -> None:
        """Add demand data to xml.etree.ElementTree.Element.
        
        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Element to which 'commodities' will be appended to.
        overwrite : bool, default=True
            If True, existing 'commodities' Element in root will be
            deleted. If False, edge data will be appended to the
            existing data.
        """
        if overwrite is True:
            for c in root.findall(TAG_COMMODITIES):
                root.remove(c)
                
        # Write type of demand func to metadata
        metadata = xml_add_element(root, "metadata")
        demand_func = xml_add_element(metadata, "demand_func")
        demand_func.text = self.__class__.__name__
                
        # Write all demand vectors of demand function
        for dv_name, dv in self.demand_vectors.items():
            dv.add_to_etree(root, name=dv_name)

    @classmethod
    def from_xml(
            self,
            data,
            shared=None,
            **kw,
            ) -> LinearDemandFunction | AffineDemandFunction:
        # Get root of XML
        root = xml_find_root(data)
        
        # Determine which type of demand function to readin
        demand_func = root.find("metadata/demand_func")
        
        if (demand_func is None
                or demand_func.text == "LinearDemandFunction"):
            return LinearDemandFunction.from_xml(root, shared=shared, **kw)
        elif demand_func.text == "AffineDemandFunction":
            return AffineDemandFunction.from_xml(root, shared=shared, **kw)
        raise ValueError(f"Invalid demand_func in XML: {demand_func.text}.")

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        
        save_dict["demand_type"] = self.__class__.__name__
            
        for (k, dv) in self.demand_vectors.items():
            save_dict = dv.make_save_dict(
                prefix=prefix + k + "_",
                save_dict=save_dict,
            )
        
        return save_dict

    make_save_dict.__doc__ = _doc.make_save_dict.__doc__

    def save_to_numpy(self, file: str, **kwargs) -> None:
        save_dict = self.make_save_dict()
        save_dict.update(kwargs)
        np.savez(file, **save_dict)
    
    save_to_numpy.__doc__ = _doc.save_to_numpy.__doc__

    @classmethod
    def from_npz(
            cls,
            data,
            shared=None,
            prefix: str = "",
            **kwargs
            ) -> LinearDemandFunction | AffineDemandFunction:
        # load np data
        if isinstance(data, str):
            data = np.load(data)
        
        if ("demand_type" not in data or
                str(data["demand_type"]) == "LinearDemandFunction"):
            return LinearDemandFunction.from_npz(data, shared=shared, prefix=prefix)
        elif str(data["demand_type"]) == "AffineDemandFunction":
            return AffineDemandFunction.from_npz(data, shared=shared, prefix=prefix)
        raise ValueError(f"invalid demand type: {str(data['demand_type'])}.")

    from_npz.__func__.__doc__ = _doc.from_npz_shared.__doc__

    @property
    def _dv0(self):
        return getattr(self, self._demand_vectors[0])

    @property
    def all_single_pairs(self) -> bool:
        """bool: whether all commodities have single source/sink."""
        return bool((self(1).getnnz(axis=0) == 2).all())
    
    @property
    def n(self) -> int:
        if self.shared is not None:
            return self.shared.n
        elif self._dv0.n is not None:
            return self._dv0.n
        raise AttributeError(
            "Number of nodes has not been set: Neither shared object nor "
            "number of nodes specified."
        )
    
    @n.setter
    def n(self, val: int) -> None:
        for dv in self.demand_vectors.values():
            dv.n = val
    
    @property
    def shared(self):
        """Shared object for network objects.
        
        See Also
        --------
        paminco.net.shared.Shared
        """
        return self._dv0.shared
    
    @shared.setter
    def shared(self, s: Shared) -> None:
        for dv in self.demand_vectors.values():
            dv.shared = s
    
    shared.__doc__ = _DV.shared.__doc__
    
    @property
    def dtype_int(self):
        return self._dv0.dtype_int
    
    @dtype_int.setter
    def dtype_int(self, val) -> None:
        for dv in self.demand_vectors.values():
            dv.dtype_int = val
    
    @property
    def dtype_float(self):
        return self._dv0.dtype_float
    
    @dtype_float.setter
    def dtype_float(self, val) -> None:
        for dv in self.demand_vectors.values():
            dv.dtype_float = val
    
    @property
    def demand_vectors(self):
        """Get names and values for all vectors making up demand func."""
        return {k: getattr(self, k) for k in self._demand_vectors}

    @property
    def is_single_commodity(self) -> bool:
        """Whether demand function consists of single commodity only."""
        return (self.sparse().shape[1] == 1)

    @property
    def is_multi_commodity(self) -> bool:
        """Whether demand function consists of more than one commodity."""
        return (self.sparse().shape[1] > 1)

    @property
    def value_at_1(self) -> sps.spmatrix:
        """Get value of demand function h(1)."""
        if self.cache.is_valid("value_at_1") is False:
            v = self.cache["value_at_1"] = self(1)
            return v
        return self.cache["value_at_1"]

    @property
    def unique_sources(self) -> np.ndarray:
        """Get all unique sources."""
        if self.cache.is_valid("unique_sources") is False:
            u_sources = np.empty(0, dtype=int)
            for dv in self.demand_vectors.values():
                u_sources = np.union1d(u_sources, np.where((dv() < 0).getnnz(axis=1))[0])
            v = self.cache["unique_sources"] = u_sources
            return v
        return self.cache["unique_sources"]

    @abc.abstractmethod
    def ddx(self) -> sps.csc_matrix: ...
    @abc.abstractmethod
    def value(self) -> sps.csc_matrix: ...
    @abc.abstractmethod
    def get_source_sink_rate(self): ...


class LinearDemandFunction(DemandFunction):
    """A Linear demand function.
    
    Modelling the function::
    
        demand(param) = param * b.
    
    Parameters
    ----------
    b : DemandVector, DemandVectorSP, spmatrix or iterable
        Demand direction, scales linearly with param.
    shared : Shared, optional
        Shared object for all network objects.
    
    Attributes
    ----------
    b : DemandVector or DemandVectorSP
        Demand direction, scales linearly with param.
    cache : Cache
        A cache.
    shared
    all_single_pairs
    shared
    demand_vectors
    is_single_commodity
    is_multi_commodity
    value_at_1
    unique_sources
    
    See Also
    --------
    demand_vector : Initialization of demand vectors.
    """
    _demand_vectors = ["b"]

    def __init__(
            self,
            b,
            shared=None,
            **kwargs
            ) -> None:
        super().__init__(shared)
        self.b = demand_vector(b, shared=shared, **kwargs)

    def max_inflow(
            self,
            max_param: float,
            min_param: float = 0,
            ) -> np.ndarray:
        "Inflow rate at ``max_param`` for all commodities."
        return max_param * self.b.total_rate

    def value(self, param: float = 1, **kwargs) -> sps.csc_matrix:
        """Value of demand function: demand(param).
        
        Parameters
        ----------
        param: float, default=1
        
        Returns
        -------
        csc_matrix
            demand'(param).
        
        See Also
        --------
        DemandVector.sparse : details on the structure of the return
            matrix.
        DemandVectorSP.sparse : details on the structure of the return
            matrix.
        """
        return param * self.b.sparse(**kwargs)

    def ddx(self, param: float, **kwargs) -> sps.csc_matrix:
        """Derivative of demand function: demand'(param).

        Parameters
        ----------
        param: float, default=1

        Returns
        -------
        csc_matrix
            demand'(param).

        See Also
        --------
        DemandVector.sparse : details on the structure of the return
            matrix.
        DemandVectorSP.sparse : details on the structure of the return
            matrix.
        """
        return self.b.sparse(**kwargs)

    def get_source_sink_rate(self, param: float):
        """Get indices of source and sink, and rates for all commodities.
        
        Parameters
        ----------
        param : float
            Scale rates by param.
        
        Returns
        -------
        source_indices : ndarray
            Indices of sources, ndarray (k', ) of int, where k' is the
            total amount of commodities making up this demand function.
            k' is summed up over all demand vectors.
        sink_indices : ndarray
            Indices of sinks, ndarray (k', ) of int.
        rates : ndarray
            Rates, ndarray (k', ) of float.
        """
        return self.b.get_source_sink_rate(scale_by=param)

    @classmethod
    def from_npz(
            cls,
            data,
            shared=None,
            prefix: str = "",
            **kwargs
            ) -> LinearDemandFunction:
        prefix = prefix + cls._demand_vectors[0] + "_"
        b = _DV.from_npz(data, shared=shared, prefix=prefix, **kwargs)
        return cls(b, shared=shared, **kwargs)

    @classmethod
    def from_xml(
            cls,
            data,
            shared=None,
            **kw,
            ) -> LinearDemandFunction:
        root = xml_find_root(data)
        b = _comm_from_xml_node(root.find(TAG_COMMODITIES))
        return cls(b, shared=shared, **kw)

    from_xml.__func__.__doc__ = _doc.from_xml_shared.__doc__


class AffineDemandFunction(DemandFunction):
    """An affine demand function.
    
    Modelling the function::
    
        demand(param) = b0 + param * b
    
    where b0 and b are demand vectors modelling an equal amount of
    commodities.
    
    Parameters
    ----------
    b0 : DemandVector, DemandVectorSP, spmatrix or iterable
        Base demand, unaffected by param.
    b : DemandVector, DemandVectorSP, spmatrix or iterable
        Demand direction, scales linearly with param.
    shared : Shared, optional
        Shared object for all network objects.
    
    Attributes
    ----------
    b0 : DemandVector or DemandVectorSP
        Base demand, unaffected by param.
    b : DemandVector or DemandVectorSP
        Demand direction, scales linearly with param.
    cache : Cache
        A cache.
    shared
    all_single_pairs
    shared
    demand_vectors
    is_single_commodity
    is_multi_commodity
    value_at_1
    unique_sources
    
    See Also
    --------
    demand_vector : Initialization of demand vectors.
    """
    _demand_vectors = ["b", "b0"]

    def __init__(
            self,
            b0,
            b,
            shared=None,
            **kwargs
            ) -> None:
        super().__init__(shared)
        
        # init bection and base_demand
        self.b0 = demand_vector(b0, shared=shared, **kwargs)
        self.b = demand_vector(b, shared=shared, **kwargs)
        
        if len(self.b) != len(self.b0):
            raise ValueError("'b' and 'b0' must have the same number of commodities.")

    def value(self, param: float = 1, **kwargs) -> sps.csc_matrix:
        return self.b0.sparse(**kwargs) + param * self.b.sparse(**kwargs)
    
    value.__doc__ = LinearDemandFunction.value.__doc__

    def ddx(self, param: float, **kwargs) -> sps.csc_matrix:
        return self.b.sparse(**kwargs)
    
    ddx.__doc__ = LinearDemandFunction.ddx.__doc__

    def get_source_sink_rate(self, param: float):
        s0, t0, r0 = self.b0.get_source_sink_rate(1)
        s, t, r = self.b.get_source_sink_rate(scale_by=param)
        s = np.hstack((s0, s))
        t = np.hstack((t0, t))
        r = np.hstack((r0, r))
        return s, t, r
    
    get_source_sink_rate.__doc__ = LinearDemandFunction.get_source_sink_rate.__doc__

    @classmethod
    def from_npz(
            cls,
            data,
            shared=None,
            prefix: str = "",
            **kwargs
            ) -> LinearDemandFunction:
        b = _DV.from_npz(
            data,
            shared=shared,
            prefix=prefix + cls._demand_vectors[0] + "_",
            **kwargs
        )
        b0 = _DV.from_npz(
            data,
            shared=shared,
            prefix=prefix + cls._demand_vectors[1] + "_",
            **kwargs
        )
        return cls(b0, b, shared=shared, **kwargs)

    @classmethod
    def from_xml(
            cls,
            data,
            shared=None,
            **kw,
            ) -> AffineDemandFunction:
        root = xml_find_root(data)
        commodities = root.findall(TAG_COMMODITIES)
        for comm in commodities:
            if comm.attrib["name"] == "b":
                b = _comm_from_xml_node(comm)
            elif comm.attrib["name"] == "b0":
                b0 = _comm_from_xml_node(comm)
        return cls(b0, b, shared=shared, **kw)
