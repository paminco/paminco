from __future__ import annotations

import abc
import copy
import xml.etree.ElementTree as et
from numpy.lib.shape_base import vsplit
import psutil

import numpy as np
import numexpr as ne
import pandas as pd
import multiprocessing as mp
from itertools import zip_longest

from .shared import Shared
from paminco.utils.typing import is_int
from paminco.utils.misc import Cache
from paminco.utils.readin import (
    xml_find_root,
    xml_add_element,
    parse_polynomial,
    parse_number,
)
import paminco._doc as _doc


XML_TAG_EDGE_COST = "cost"
XML_POLYCOST_TAG = "polynomial"
XML_POLYCOST_TAG_COEFF = "coefficient"
XML_POLYCOST_ATTRIB_SIGNED = "signed"
XML_PWQ_TAG = "piecewisequadratic"
XML_PWQ_TAG_FUNPART = "functionpart"
XML_SYMCOST_TAG = "symbolic"
XML_SYMCOST_TAG_FUNCDEF = "costfuncs"

DEFAULT_DTYPE_FLOAT = np.float64
DEFAULT_DTYPE_INT = np.int32


class NetworkCost(abc.ABC):
    """Abstract class representing all cost functions of a network."""

    def __init__(self, shared: Shared = None) -> None:
        """Initialize NetworkCost."""
        # Default dtypes (used if shared is None or if not overwritten
        # by subclasses)
        if shared is not None and not isinstance(shared, Shared):
            raise TypeError("Shared is not of type shared!")
        self._s = shared
        self._dtype_float = None
        self._dtype_int = None

    def __call__(self, *args, **kwargs):
        """Map to the value() function."""
        return self.value(*args, **kwargs)

    def __len__(self) -> int:
        """Return number of cost functions (=number of edges)."""
        return self._s.m

    def __getitem__(self, edge: int) -> EdgeCost:
        """Return the edge cost of the edge with index edge. Wrapper for self.edge_cost(edge)"""
        return self.edge_cost(edge)

    def ddx(self, *args, **kwargs):
        """Return value of the first derivative.
        
        Wrapper for ``value(*args, d=1, **kwargs)``.
        """
        return self.value(*args, d=1, **kwargs)

    def F(self, x, dtype=None) -> np.ndarray:
        """Cost of edge.
        
        Parameters
        ----------
        x : ndarray
            Edge flow, ndarray of float of shape (m, ).
            
        Returns
        -------
        ndarray
            ``F(x)``, ndarray of float of shape (m, ).
        """
        # return self.value(x, d=0, dtype=dtype)
        return self.value(x, d=0)
    
    def f(self, x, dtype=None) -> np.ndarray:
        """Marginal edge cost.
        
        Parameters
        ----------
        x : ndarray
            Edge flow, ndarray of float of shape (m, ).
            
        Returns
        -------
        ndarray
            ``f(x)``, ndarray of float of shape (m, ).
        """
        # return self.value(x, d=1, dtype=dtype)
        return self.value(x, d=1)
    
    def f1(self, x, dtype=None) -> np.ndarray:
        """First derivative of marginal cost.
        
        Parameters
        ----------
        x : ndarray
            Edge flow, ndarray of float of shape (m, ).
            
        Returns
        -------
        ndarray
            ``f'(x)``, ndarray of float of shape (m, ).
        """
        # return self.value(x, d=2, dtype=dtype)
        return self.value(x, d=2)
    
    def f2(self, x, dtype=None) -> np.ndarray:
        """Second derivative of marginal cost.
        
        Parameters
        ----------
        x : ndarray
            Edge flow, ndarray of float of shape (m, ).
            
        Returns
        -------
        ndarray
            ``f''(x)``, ndarray of float of shape (m, ).
        """
        # return self.value(x, d=3, dtype=dtype)
        return self.value(x, d=3)

    def edge_value(
            self,
            index: int,
            x,
            *args,
            d: int = 0,
            **kwargs
            ):
        """Return value of edge cost function.

        Parameters
        ----------
        index : int
            The index of the edge.
        x : float, np.array
            The value where the cost functions are evaluated.
            Can either be an array with values for every edge or a single
            value that is used for all edges.
        d : int, default=0
            The derivative order. d=0 returns the function value.

        Returns
        -------
        ndarray
            Costs of all edges, ndarray of float of shape (m, )
        """
        flow = np.zeros(len(self))
        flow[index] = x
        flow_cost = self.value(flow, *args, d=d, **kwargs)
        return flow_cost[index]

    def laplace_weights(self, x) -> np.ndarray:
        """Returns the Laplace weights for all edges at x.

        The Laplace weight of an edge e is defined as ``1 / c''(x_e)``,
        where c'' is the second derivative of the edge costs.

        Parameters
        ----------
        x : float or ndarray
            Flow on edges. If float, value is broadcasted for all
            edges.

        Returns
        -------
        weights : ndarray
            An array containg the Laplace weights for all edges.

        See also
        --------
        paminco.network.network.Network.L : Laplacian matrix.
        """
        weights = self.value(x, d=2)
        finite_vals = np.isfinite(weights)
        weights[finite_vals] = 1 / weights[finite_vals]
        weights[np.logical_not(finite_vals)] = 0
        return weights

    def is_continuous(self) -> bool:
        """Whether all edge costs are continuous."""
        return self.is_smooth(k=0)

    @staticmethod
    def from_xml(
            xml,
            shared: Shared = None,
            **kwargs
            ) -> NetworkCost:
        data = xml_find_root(xml)
        ioe = IOError("Network has inconsistent cost types. "
                      "The cost functions of all edges should be of "
                      "the same type.")
        cls = None
        if data.find(f"edges/edge/{XML_TAG_EDGE_COST}/{XML_POLYCOST_TAG}") is not None:
            if cls is not None:
                raise ioe
            cls = PolynomialCost
        if data.find(f"edges/edge/{XML_TAG_EDGE_COST}/{XML_PWQ_TAG}") is not None:
            if cls is not None:
                raise ioe
            cls = PiecewiseQuadraticCost
        if data.find(f"edges/edge/{XML_TAG_EDGE_COST}/{XML_SYMCOST_TAG}") is not None:
            if cls is not None:
                raise ioe
            return SymbolicCost.from_xml(data, shared=shared, **kwargs)
        if cls is None:
            return None
        return cls.from_xml(data, shared=shared, **kwargs)

    from_xml.__func__.__doc__ = _doc.from_xml_shared.__doc__

    @staticmethod
    def from_npz(
            data,
            prefix: str = "",
            shared: Shared = None,
            **kwargs
            ) -> PolynomialCost | PiecewiseQuadraticCost | SymbolicCost:
        if isinstance(data, str):
            data = np.load(data)
        if ("cost_type" not in data
                or str(data["cost_type"]) == "PolynomialCost"):
            return PolynomialCost.from_npz(data, prefix=prefix, shared=shared, **kwargs)
        elif str(data["cost_type"]) == "PiecewiseQuadraticCost":
            return PiecewiseQuadraticCost.from_npz(data, prefix=prefix, shared=shared, **kwargs)
        elif str(data["cost_type"]) == "SymbolicCost":
            return SymbolicCost.from_npz(data, prefix=prefix, shared=shared, **kwargs)
        else:
            raise ValueError(f"invalid cost type: {str(data['cost_type'])}")
    
    from_npz.__func__.__doc__ = _doc.from_npz_shared.__doc__

    @property
    def dtype_float(self):
        if self._dtype_float is not None:
            return self._dtype_float
        if self._s is not None:
            return self._s.dtype_float
        return DEFAULT_DTYPE_FLOAT
   
    @dtype_float.setter
    def dtype_float(self, dtype):
        self._dtype_float = dtype
    
    @property
    def dtype_int(self):
        if self._dtype_int is not None:
            return self._dtype_int
        if self._s is not None:
            return self._s.dtype_int
        return DEFAULT_DTYPE_INT

    @dtype_int.setter
    def dtype_int(self, dtype):
        self._dtype_int = dtype

    @property
    def m(self) -> int:
        """Number of cost functions (= number of edges)."""
        return len(self)

    @property
    def shared(self):
        """Shared object for network objects.

        See Also
        --------
        paminco.net.shared.Shared
        """
        return self._s

    @abc.abstractmethod
    def edge_cost(self, edge: int) -> EdgeCost:
        """Return the edge cost of the edge with index edge."""
        ...
    @abc.abstractmethod
    def differentiate(self) -> None: 
        """Return a new NetworkCost object with derivative cost functions."""
        ...
    @abc.abstractmethod
    def integrate(self) -> None:
        """Return a new NetworkCost object with integrated cost functions."""
        ...
    @abc.abstractmethod
    def is_smooth(self, k=1) -> bool:
        """Return `True` if the function is `k`-times differentiable."""
        ...
    @abc.abstractmethod
    def delete_edges(self) -> None: ...
    @abc.abstractmethod
    def save_to_numpy(self) -> None: ...
    @abc.abstractmethod
    def value(self, *args, **kwargs):
        """Abstract Method for function evaluation."""
        ...


class EdgeCost(abc.ABC):
    """Abstract class that represent the cost of a single edge."""

    def __init__(self, edge: int = None):
        """Initialize EdgeCost of the edge with index edge"""
        self.edge = edge

    def __call__(self, x: float, d: int = 0) -> float:
        """Return the value of the d-th derivative of this function at x."""
        return self.ddx(x, d=d)

    def __str__(self):
        if self.edge is not None:
            return f"EdgeCost for edge {self.edge}"
        else:
            return "EdgeCost object"

    def value(self, x: float) -> float:
        "Return the value of this function at x. Wrapper for __call__(x)"
        return self(x)

    @abc.abstractmethod
    def ddx(self, x: float, d: int = 1) -> float:
        """Return the value of the d-th derivative of this function at x."""
        ...


class SymbolicCost(NetworkCost):
    dtofunc = {0: "F", 1: "f", 2: "f1", 3: "f2"}

    def __init__(
            self,
            coeffs: dict,
            F=None,
            f=None,
            f1=None,
            f2=None,
            shared=None,
            m=None,
            ):
        super().__init__(shared)
        
        # Convert coefficients to 1D arrays
        for coeffname, c in coeffs.items():
            coeffs[coeffname] = np.array(c).flatten()
        
        # Determine auto m (max length of coefficients)
        if m is None:
            m = max([len(v) for v in coeffs.values()])
        self._m = m
        
        # Build funcs
        self.funcs = {
            "F": F,
            "f": f,
            "f1": f1,
            "f2": f2,
        }
        for fname, f in self.funcs.items():
            if f is not None:
                self.funcs[fname] = f.replace("\"", "")
            
        if m is not None and m != 1:
            for coeffname, c in coeffs.items():
                if len(c) != m:
                    c = np.full(m, c)
                coeffs[coeffname] = c
        if m == 1:
            for coeffname, c in coeffs.items():
                coeffs[coeffname] = c[0]
        
        self.coeffs = coeffs
        
    def __len__(self) -> int:
        return self.m
    
    def __repr__(self) -> str:
        out = super().__repr__()
        out += "\n"
        for (k, v) in self.funcs.items():
            out += f"{k}: \t: {v}\n"
        return out
        
    def edge_cost(self, edge: int):
        # TODO: Create seperate class for cost functions of a single edge inheriting from EdgeCost
        coeffs = {k: v[edge] for k, v in self.coeffs.items()}
        return SymbolicCost(coeffs, **self.funcs, m=1)
    
    def value(self, x, d: int = 0, dtype=None):
        if dtype is None:
            dtype = self.dtype_float
        if len(self) > 1 and isinstance(x, (int, float)):
            # x = np.full(self.m, x, dtype=dtype)
            x = np.full(self.m, x)
        local_dict = dict(self.coeffs)
        local_dict["x"] = x
        f = self.funcs[self.dtofunc[d]]
        val = ne.evaluate(f, local_dict)
        if val.size == 1 and self.m > 1:
            return np.full(self.m, val, dtype=dtype)
        return val
    
    def delete_edges(self, edges) -> None:
        if len(edges) > 0:
            for c in self.coeffs.values():
                np.delete(c, edges)
    
    def add_to_etree(
            self,
            edge_node: et.Element,
            idx: int,
            overwrite: bool = True
            ) -> None:
        cost_node = xml_add_element(edge_node, XML_TAG_EDGE_COST)
        symcost = xml_add_element(cost_node, XML_SYMCOST_TAG, overwrite=overwrite)
        for k, v in self[idx].coeffs.items():
            symbol = et.SubElement(symcost, k)
            symbol.text = str(v)
    
    def costfuncs_to_metadata(self, metadata: et.Element) -> None:
        costfuncnode = xml_add_element(metadata, XML_SYMCOST_TAG_FUNCDEF)
        for fname, f in self.funcs.items():
            if f is not None:
                fnode = xml_add_element(costfuncnode, fname)
                fnode.text = f

    @classmethod
    def from_xml(
            cls,
            data,
            F=None,
            f=None,
            f1=None,
            f2=None,
            shared=None,
            **kw,
            ):
        # Retrieve coefficients from xml
        coeffs = {}
        root = xml_find_root(data)
        for e in root.find("edges"):
            for symbol in e.find(f"{XML_TAG_EDGE_COST}/{XML_SYMCOST_TAG}"):
                name = symbol.tag
                value = float(symbol.text)
                if name in coeffs:
                    coeffs[name].append(value)
                else:
                    coeffs[name] = [value]
        for c in coeffs.keys():
            coeffs[c] = np.array(coeffs[c])
        
        # Retrieve funcs from xml if not specified as parameters
        def _get_func(name: str):
            func = root.find(f"metadata/{XML_SYMCOST_TAG_FUNCDEF}/{name}")
            if func is not None:
                func = func.text
            return func
        
        if F is None:
            F = _get_func("F")
        if f is None:
            f = _get_func("f")
        if f1 is None:
            f1 = _get_func("f1")
        if f2 is None:
            f2 = _get_func("f2")
            
        return cls(coeffs, F=F, f=f, f1=f1, f2=f2, shared=shared, **kw)
    
    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        
        save_dict["cost_type"] = self.__class__.__name__
        
        for k, v in self.coeffs.items():
            save_dict[prefix + f"coeffs_{k}"] = v
            
        for k, v in self.funcs.items():
            if v is not None:
                save_dict[prefix + f"costfuncs_{k}"] = v
        
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
            prefix: str = "",
            F=None,
            f=None,
            f1=None,
            f2=None,
            shared=None,
            **kw,
            ):
        # load np data
        if isinstance(data, str):
            data = np.load(data)
            
        # Retrieve coefficients from data
        coeffs = {}
        for k, v in data.items():
            if k.startswith(prefix + "coeffs"):
                coeffs[k.split("coeffs_")[-1]] = v
        
        # Retrieve funcs from xml if not specified as parameters
        def _get_func(name: str):
            ret = data.get(f"{prefix}costfuncs_{name}", None)
            if ret is not None:
                ret = str(ret)
            return ret
        
        if F is None:
            F = _get_func("F")
        if f is None:
            f = _get_func("f")
        if f1 is None:
            f1 = _get_func("f1")
        if f2 is None:
            f2 = _get_func("f2")
            
        return cls(coeffs, F=F, f=f, f1=f1, f2=f2, shared=shared, **kw)
    
    @property
    def m(self) -> int:
        if self._m is not None:
            return self._m
        if self.shared is not None:
            return self.shared.m
        raise ValueError("number of edges (m) not set.")
            
    def is_smooth(self, k=1) -> bool:
        raise NotImplementedError()

    def times_X(self) -> None:
        raise NotImplementedError()

    def differentiate(self) -> None:
        raise NotImplementedError()

    def integrate(self) -> None:
        raise NotImplementedError()


class SimplePolynomial(EdgeCost):
    r"""Class representing a simple polynomial cost function.
    
    Represents either a function

    .. math::
        p(x) = \sum_{k=0}^n a_k x^k

    if `signed = False` or the function
    
    .. math::
        p(x) = a_0 + \sum_{k=1}^n a_k |x| x^{k-1}
    
    if `signed = True`. The polynomial is represented by its coefficients
    :math:`a_0, \dotsc, a_n`.

    Parameters
    ----------
    coeff: iterable
        The list of coefficients :math:`a_0, \dotsc, a_n`.
    signed: bool
        If true, all terms for :math:`k \geq 0` of the polynomial are
        multiplied by the sign of x.
    edge: int, default=None
        Optional edge index to indicate to which edge this cost function
        belongs.

    Attributes
    ----------
    degree: int
        The degree of this polynomial (= number of coefficients - 1)
    """

    def __init__(self, coeff, signed=False, edge=None):
        super().__init__(edge)
        if len(coeff) == 0:
            self.coeff = np.array([0])
        self.coeff = coeff
        self.signed = signed
        self.n = len(coeff)

    def __len__(self) -> int:
        """Return the number of coefficients (degree + 1) of this polynomial"""
        return self.n

    def __repr__(self) -> str:
        def var_to_str(deg=0, signed=None):
            if signed is None:
                signed = self.signed
            if deg == 0:
                return "|x|" if signed else "x"
            else:
                if not signed:
                    return "x^{:d}".format(deg)
                else:
                    return var_to_str(0, True) + " " + var_to_str(deg - 1, False)

        out = ""
        for i, c in enumerate(self.coeff):
            if c != 0:
                if len(out) > 0 or c < 0:
                    out += " + " if c > 0 else " - "
                out += "{:.6f}".format(abs(c)) + " * " + var_to_str(i)
        if self.edge is not None:
            out = f"EdgeCost for edge {self.edge}: {out}"
        return out

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def derivative_factors(cls, k, d) -> int:
        r"""Return the factors of the k-th term in the d-th derivative.
        
        The (k, d) factor is computed as :math:`\frac{k!}{(k-d)!}`
        """
        out = 1
        for i in range(k - d + 1, k + 1):
            out *= i
        return out

    def ddx(self, x, d=1) -> float:
        out = 0
        for k in range(d, self.n):
            try:
                if self.coeff[k] != 0:
                    val_k = (SimplePolynomial.derivative_factors(k, d) * x ** (k - d)) * self.coeff[k]
                else:
                    val_k = 0
            except OverflowError:
                raise RuntimeError(
                    f"Overflow error in Polynomial in term {self.coeff[k]} * "
                    f"{x}^{k-d}"
                )
            if self.signed and (k - d > 0) and x < 0:
                val_k *= -1
            out += val_k
        return out

    @property
    def degree(self):
        return self.n - 1


class PolynomialCost(NetworkCost):
    r"""Polynomial network costs.

    Represents polynomial network costs of the form
    
    .. math::
        F_e (x) = \sum_{k=0}^n a_{e, k} x^k
        \text{ or } F_e (x) = a_{e, 0} + \sum_{k=1}^n a_{e, k} |x| x^{k-1}

    Every edge cost can either be unsigned (first type) or signed (second type).
    The polynomials are represented by their ``coefficients`` and the booleans
    in ``signed`` indicating the type of the polynomial

    Parameters
    ----------
    shared : Shared
        Shared network object.
    data : PolynomialCost, ndarray, tuple
        The data of the polynomial cost. Can be of type

        * PolynomialCost
            The new cost will be a copy of the PolynomialCost
            object in ``data``
        * ndarray or list
            An array or list of coefficents. The `i`-th entry of this
            array contains the coefficients :math:`a_{e_i, k}, k \in \{0,
            \dotsc d\}` where :math:`d` is the maximum degree of all
            cost functions of all edges.
        * tuple ``(coefficients, signed)``
            A tuple containing an array of coefficients (as described
            above), and an array of booleans for all edges indicating
            whether the respective polynomial function of the edge
            is signed or not.
    dtype_float : dtype, optional
        Dataytpe for coefficient value.
    dtype_int : dtype, optional
        ``Unused``, consistency to PiecewiseQuadraticCost.
    copy : bool, default=True
        Whether to copy ``data``.

    Attributes
    ----------
    degree
    m
    coefficients : ndarray
        Coefficients of polynomial edge costs, ndarray of shape (m, degree).
    signed : ndarray
        Array of booleans, indicating wheter the polynomials of the respective
        edges are signed. In a signed polynomial, every term x^k is replaced
        by sgn(x) * x^k.
    """

    def __init__(
            self,
            data,
            shared: Shared = None,
            dtype_float=None,
            dtype_int=None,  # consistency with pwqc
            copy: bool = True,
            ) -> None:

        kwargs = {"dtype_float": dtype_float, "copy": copy}

        if isinstance(data, PolynomialCost):
            # PolynomialCost given
            return self.__init__((data.coefficients, data.signed), shared=shared, **kwargs)
        elif isinstance(data, (list, np.ndarray)):
            # Only coeff given -> all polynomials are not signed
            return self.__init__((data, np.full(len(data), False)), shared=shared, **kwargs)
        elif isinstance(data, tuple) and len(data) == 2:
            # (coeff, abs_val) given
            super().__init__(shared=shared)
            # Update dtypes (types are ignored if None)
            self.dtype_float = dtype_float
            self.dtype_int = dtype_int
            coeff, signed = data
            if len(coeff) != len(signed):
                raise ValueError(
                    "Length of coefficent array does not match length "
                    f"of sign array: {len(coeff)} != {len(signed)}."
                )

            # Init coeffs
            self.coefficients = np.array(coeff, copy=True, dtype=self.dtype_float)

            # Init signed flags
            self.signed = signed
        else:
            raise ValueError(f"incorrect input for data: {data}")

        self._cache = Cache()

    def __len__(self) -> int:
        return len(self.coefficients)

    def __eq__(self, other) -> bool:
        if isinstance(other, PolynomialCost) is False:
            return False
        if not np.array_equal(self.coefficients, other.coefficients):
            return False
        if not np.array_equal(self.signed, other.signed):
            return False
        return True

    def times_x(self, inplace: bool = True):
        return self._shift(d=1, inplace=inplace)

    def _shift(self, d: int = 1, inplace: bool = False, factor=None):
        if factor is None:
            factor = 1
        
        if d < 0:
            coeff = self.coefficients[:, -d:] * factor
            signed = self.signed[:]
        else:
            coeff = np.zeros((len(self), 1 + self.degree + d))
            coeff[:, d:(1 + d + self.degree)] = self.coefficients * factor
            signed = self.signed[:]
        
        if inplace is True:
            self.__init__((coeff, signed), shared=self.shared)
        else:
            return PolynomialCost((coeff, signed), shared=self.shared)

    def integrate(self, d: int = 1, inplace: bool = False):
        f = (1 + np.arange(self.degree + 1))
        f = np.r_[[1 / (f + d_) for d_ in range(d)]].prod(axis=0)
        return self._shift(d=d, inplace=inplace, factor=f)

    def differentiate(self, d: int = 1, inplace: bool = False):
        f = np.arange(self.degree + 1)[d:]
        f = np.r_[[f - d_ for d_ in range(d)]].prod(axis=0)
        return self._shift(d=-d, inplace=inplace, factor=f)

    def edge_cost(self, edge: int) -> SimplePolynomial:
        """Get cost of edge.

        Parameters
        ----------
        edge : int
            Indice of edge to get cost for.

        Returns
        -------
        SimplePolynomial
            Polynomial edge cost.
        """
        idx = "ecost_" + str(edge)
        if self._cache.is_valid(idx) is False:
            c = self.coefficients[edge]
            s = self.signed[edge]
            self._cache[idx] = SimplePolynomial(c, s)
        return self._cache[idx]

    def value(self, x, d: int = 0) -> np.ndarray:
        """Return the value of all polynomials functions.

        Parameters
        ----------
        x : float or ndarry
            Network flow. If float, value is broadcasted for all edges.
        d : int, default=0
            Derivative order. E.g., ``d=0`` return f(x) and
            ``d=1`` returns f'(x).

        Returns
        -------
        ndarray
            Cost or derivative of order d given network flow x.

        Examples
        --------
        >>> net = paminco.net.load_sioux()
        >>> net.cost.value(100000 + np.zeros(net.m), d=0)[:5]
        array([4.6000e+06, 4.4000e+06, 4.6000e+06, 2.4825e+09, 4.4000e+06])

        Derivative of cost for small flows is mainly influenced by free
        flow time for traffic networks.
        >>> net = paminco.net.load_sioux()
        >>> net.cost.value(1000, d=1)[:5]
        array([6.000002, 4.000002, 6.000002, 5.001241, 4.000002])
        """
        if d > self.degree:
            return np.zeros(len(self))

        if isinstance(x, (int, float)):
            x = np.full(len(self), x, dtype=self.shared.dtype_float)

        out = 0
        sgns = None
        # If at least one signed polynomial exists
        if sum(self.signed) > 0:
            sgns = np.array([-1 if self.signed[e] and v < 0 else 1 for e, v in enumerate(x)])

        for k in range(d, self.degree + 1):
            fac = SimplePolynomial.derivative_factors(k, d)
            val_k = fac * self.coefficients[:, k] * (x ** (k - d))
            if sgns is not None and k - d > 0:
                val_k *= sgns
            out += val_k

        return out

    def interpolate(self, rule: InterpolationRule, x_max=None, max_breakpoints_per_edge=1e5) -> PiecewiseQuadraticCost:
        """Interpolate this polynomial cost function with a piecewise quadratic cost function.
        
        Parameters
        ----------
        rule: InterpolationRule
            The :class:`InterpolationRule` for the breakpoint computation in the interpolation
        x_max: float, default=None
            The upper bound for the interpolation on every edge (on undirected edges
            the interpolation runs from -x_max to x_max)
            If set to None, x_max is infered from the interpolation rule if possible
            and otherwise set to a default value defined in :class:`NetworkCostInterpolation`
        max_breakpoints_per_edge: int, default=1e05
            The maximum number of breakpoints on every edge

        See also
        --------
        :class:`NetworkCostInterpolation` : NetworkCostInterpolation.
        """
        interpolator = NetworkCostInterpolation(self, rule, x_max, max_breakpoints_per_edge)
        return interpolator.interpolate()
    
    def _coeff_to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.coefficients)
        df.columns = [f"x^{i}" for i in range(df.shape[1])]
        return df

    def delete_edges(self, edges) -> None:
        if len(edges) > 0:
            # print(len(self.coefficients), len(self.signed))
            self.coefficients = np.delete(self.coefficients, edges, axis=0)
            self.signed = np.delete(self.signed, edges, axis=0)
            self._cache.reset()
            assert len(self.coefficients) == len(self.signed)

    def is_smooth(self, k=1):
        """Return True if all functions are k-times differentiable."""
        return True

    def add_to_etree(
            self,
            edge_node: et.Element,
            idx: int,
            overwrite: bool = True
            ):
        coeffs = self.coefficients[idx]
        signed = "True" if self.signed[idx] else "False"

        cost_node = edge_node.find("cost")
        if cost_node is None:
            cost_node = et.SubElement(edge_node, "cost")

        if overwrite is True and cost_node.find("polynomial") is not None:
            cost_node.remove(cost_node.find("polynomial"))
        pcost_node = cost_node.find("polynomial")
        if pcost_node is None:
            pcost_node = et.SubElement(cost_node, "polynomial")
            pcost_node.attrib['signed'] = signed

        coeff_dict = zip(np.arange(len(coeffs)).astype(str), coeffs)

        for deg, val in coeff_dict:
            coe_node = et.SubElement(pcost_node, 'coefficient')
            coe_node.attrib['i'] = deg
            coe_node.text = str(val)

    @classmethod
    def from_xml(
            cls,
            data,
            shared: Shared = None,
            default_edge_cost=True,
            **kwargs
            ) -> PolynomialCost:
        """Create a PolynomialCost object from xml data.
        
        Parameters
        ----------
        data : str, xml.etree.ElementTree.ElementTree, or xml.etree.ElementTree.Element
            XML file, tree or root itself.
        shared : Shared, default=None
            The shared object of the network associated with these cost.
        default_edge_cost : boolean, tuple, default=True
            The default edge cost to be used if an edge has no polynomial cost
            data. Can take the following values

            * `True`
                Edges without cost get default cost of 0
            * `False` or `None`
                A ValueError is raised if an edge without cost information
                is found
            * tuple ``(coefficients, signed)``
                A tuple containing the coefficients and the signed parameter
                of the default edge cost function to be used for
                edges without cost information
            
        **kwargs : keyword arguments
            Further keyword arguments for the PolynomialCost object
        
        Returns
        -------
        polycost : PolynomialCost
            The polynomial network cost object read from data.
        """
        data = xml_find_root(data)
        
        edges = data.find("edges")
        if edges is None:
            raise ValueError("'edges' not found in 'data'.")
        
        coefficients = []
        signed = []
        for e in edges:
            c, s = cls._edge_cost_from_xml(
                e,
                default_edge_cost=default_edge_cost
            )
            coefficients.append(c)
            signed.append(s)

        coefficients = np.array(list(zip_longest(*coefficients, fillvalue=0))).T
        signed = np.array(signed)
        
        return cls((coefficients, signed), shared=shared, **kwargs)

    @staticmethod
    def _edge_cost_from_xml(edge_node, default_edge_cost=True):
        """Read the cost of one edge from xml node ``edge_node``.
        
        Helper function for readin function for from_xml."""
        cost_node = edge_node.find("cost")
        poly_node = cost_node.find("polynomial")
        if poly_node is None:
            # how to deal with non existing polynomial costs
            if default_edge_cost is True:
                return [0], True
            if default_edge_cost is None or default_edge_cost is False:
                raise ValueError("no polynomial cost for edge from {} to {}".
                                 format(edge_node.attrib["from"],
                                        edge_node.attrib["to"]))
            return default_edge_cost
        else:
            signed = (poly_node.attrib.get("signed", '').strip().lower() == 'true')
            fct_literal = poly_node.attrib.get("function", None)
            if fct_literal is not None:
                # readin from function
                coeff = parse_polynomial(fct_literal)
            else:
                # readin from polynomial coefficients
                fct_data = {int(c.get("i")): float(c.text) for c in poly_node}
                deg = max(fct_data.keys())
                coeff = [fct_data.get(i, 0) for i in range(deg + 1)]
        
        return coeff, signed

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        save_att = {
            "coefficients": "coefficients",
            "signed": "signed",
        }
        save_dict["cost_type"] = self.__class__.__name__
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
    def from_npz(
            cls,
            data,
            prefix: str = "",
            shared: Shared = None,
            **kwargs
            ) -> PolynomialCost:
        if isinstance(data, str):
            data = np.load(data)
        poly_data = (data[prefix + "coefficients"],
                     data[prefix + "signed"])
        return cls(poly_data, shared=shared, **kwargs)

    from_npz.__func__.__doc__ = _doc.from_npz_shared.__doc__

    @property
    def degree(self):
        """Return the maximum degree of all polynomial functions."""
        return self.coefficients.shape[1] - 1

    @property
    def m(self):
        """Return the number of cost functions (= number of edges)."""
        return len(self.coefficients)


class PiecewiseQuadraticCoefficients:
    r"""A class representing coefficients of piecewise quadratic functions.

    This class manages the coefficients of network cost (i.e., of cost
    functions for all edge cost functions) of the form
    
    .. math::
        F_e(x) = a_{e,i} x^2 + b_{e,i} x + \mathrm{offset}_{e,i} \quad
        \text{for } x \in [\tau_{e,i}, \tau_{e,i+1})

    The coefficients are stored in an array with columns ``a, b, offset,
    tau, lap_weight, d, sig_l, sig_u``.
    A row of this array contains the coefficients of a function part and
    the associated lower breakpoint :math:`\tau_{e, i}`.
    Additionally to the coefficients and breakpoints ``a, b, offset, tau``
    of the function this object holds precomputed values for the
    Laplace matrix weights ``lap_weight`` defined as
    :math:`c = \frac{1}{2 a}`, the values ``d`` defined as
    :math:`d = \frac{b}{2 a}` and the derivative values at the lower and upper
    breakpoints ``sig_l`` and ``sig_u`` defined as :math:`\sigma_l :=
    2 a \tau + b` and :math:`\sigma_u := 2 a \tau_{+} + b`, where
    :math:`\tau_+` is the next breakpoint.

    The first breakpoint for every edge cost function should always be
    ``-np.inf``. Lower bounds can be modeled by setting the function to
    positive infinity for values less than the lower bound, using the
    values ``[-np.inf, -np.inf, np.inf]`` for ``[a, b, offset]`` for the
    first function part for the breakpoint ``-np.inf``. Then starting at the
    lower bound as the first breakpoint.
    Similarly, for edges with upper bounds, add a function part with
    coefficient values ``[np.inf, np.inf, np.inf]`` for a breakpoint
    corresponding to the upper bound.

    There are two main ways to index breakpoints / coefficients.

    * Region based indexing.
        A region refers to a vector indices
        :math:`t_{e}, e \in E` where every index :math:`t_e` corresponds to an
        index of some function part/breakpoint of the function :math:`F_e(x)`.

    * Position based indexing.
        In this case, the index of the breakpoint
        (and the coefficients) refers to the position in the internal array.
    
    The attributes first_pos and last_pos contain arrays of the first/last
    position indices of the first/last breakpoint of every edge. The functions
    :func:`region_to_pos` and :func:`pos_to_region` can be used to convert
    indices of one type to the other.
    
    Parameters
    ----------
    data : PiecewiseQuadraticCoefficients, tuple
        The data to initialize the coefficients. Either contained in another
        PiecewiseQuadraticCoefficent object or the
        raw data consisting of a tuple ``(coefficients, edge_indices)``. In the
        latter case, the coefficients ``a, b, offset, tau`` should be contained
        in the ndarray coefficients as described above. The one-dimensional
        array ``edge_indices`` contains the index of the edge that the
        coeffiecent/breakpoint data in the same row in ``coefficients`` belongs
        to.
    dtype_float : default=None
        The float data type for the ndarray objects. If set to None, the type
        is inferred from the data.
    dtype_int : default=None
        The int data type for the ndarray objects. If set to None, the type
        is inferred from the data.
    copy : boolean, default=True
        If true, all data arrays are copied.
    
    Attributes
    ----------
    edge : 1D-array of int
    a : 1D-array of float
    b : 1D-array of float
    offset : 1D-array of float
    tau : 1D-array of float
    lap_weight : 1D-array of float
    d : 1D-array of float
    sig_l : 1D-array of float
    sig_u : 1D-array of float
    """

    PWC_COEFFICIENT_COLS = ['a', 'b', 'offset', 'tau', 'lap_weight', 'd',
                            'sig_l', 'sig_u']

    def __init__(
            self,
            data,
            dtype_float=None,
            dtype_int=None,
            copy: bool = True,
            ):
        arr_kw = {
            "dtype_float": dtype_float,
            "dtype_int": dtype_int,
            "copy": copy,
        }
        if isinstance(data, PiecewiseQuadraticCoefficients):
            return self.__init__((data.coefficients, data._edge_indices), **arr_kw)

        coefficients, edge_indices = data
        edge_indices = edge_indices.ravel()

        if len(coefficients) != len(edge_indices):
            raise ValueError(
                "Number of Coefficient and edge indices do not "
                "match."
            )
        if coefficients.shape[1] not in [4, len(self.PWC_COEFFICIENT_COLS)]:
            raise ValueError(
                "Invalid shape of coefficient array for "
                "piecewise quadratic function."
            )

        if coefficients.shape[1] == 4:
            # data is coefficients of the piecewise functions that needs
            # further initialization
            self._init_coefficients(coefficients, edge_indices, copy=copy)
        else:
            # data is array of all additionally precomputed coefficents
            self.coefficients = np.array(coefficients, copy=copy)
            self._edge_indices = np.array(edge_indices, copy=copy)

        self._init_position_offsets()

        if dtype_float is not None:
            self.coefficients = self.coefficients.astype(dtype_float)
        if dtype_int is None:
            dtype_int = np.int32
        self._edge_indices = self._edge_indices.astype(dtype_int)

    def __repr__(self) -> str:
        out = "Coefficients of Piecewise Quadratic Function\n"
        out += "=" * (len(out) - 1) + "\n"
        out += self.to_df().__repr__()
        return out

    def __getitem__(self, idx) -> PiecewiseQuadraticCoefficients:
        idx = np.ravel(idx)
        return PiecewiseQuadraticCoefficients((self.coefficients[idx, :],
                                               self._edge_indices[idx]))

    def __len__(self) -> int:
        return len(self.coefficients)

    def __eq__(self, other) -> bool:
        if np.array_equal(self.coefficients, other.coefficients) is False:
            return False
        if np.array_equal(self._edge_indices, other._edge_indices) is False:
            return False
        return True

    def _init_coefficients(
            self,
            coefficients: np.array,
            edge_indices: np.array,
            copy: bool = True
            ) -> None:
        # Create (copy of) coefficent array and store (copy of) edge index array
        self.coefficients = np.array(coefficients, copy=copy)
        self._edge_indices = np.array(edge_indices, copy=copy)
        self._finite_rows = np.isfinite(coefficients[:, 0])

        self._init_lap_weights()
        self._init_d()
        self._init_sigma()

    def _init_lap_weights(self):
        c = self.coefficients
        # Compute lapweights = 1 / (2 * a) (0 weights if a is infinite)
        lap_weights = np.full((len(c), 1), 0.0)
        lap_weights[self._finite_rows, 0] = 0.5 / c[self._finite_rows, 0]
        # -> Append lapweights to coefficient array
        c = np.append(c, lap_weights, axis=1)
        self.coefficients = c

    def _init_d(self) -> None:
        c = self.coefficients
        d = np.full((len(c), 1), 0.0)
        # Compute d = b / (2 * a) [ for finite regions ]
        d[self._finite_rows, 0] = \
            0.5 * c[self._finite_rows, 1] / c[self._finite_rows, 0]

        # Case: Infinite Rows with a = -inf
        l_infinite_rows = ~np.logical_or(self._finite_rows, c[:, 0] > 0)
        # -> Goto next rows
        bp_rows = np.insert(l_infinite_rows, 0, False)[:-1]

        d[l_infinite_rows] = np.reshape(-c[bp_rows, 3], (-1, 1))

        # Case: Infinite Rows with a = + inf
        u_infinite_rows = ~np.logical_or(self._finite_rows, c[:, 0] < 0)
        d[u_infinite_rows] = np.reshape(-c[u_infinite_rows, 3], (-1, 1))

        # -> Append to coefficient array
        c = np.append(c, d, axis=1)
        self.coefficients = c

    def _init_sigma(self) -> None:
        c = self.coefficients

        # Compute values of marginal cost at breakpoints
        # -> 2 * a * tau + b
        sig_l = np.array(c[:, 5])
        fr = self._finite_rows
        sig_l[fr] = 2 * c[fr, 0] * c[fr, 3] + c[fr, 1]
        # -> At infinite rows, use the b value
        sig_l[~self._finite_rows] = c[~self._finite_rows, 1]

        # Values at upper breakpoints (tau_{i+1}) are just the shifted
        # values at lower breakpoints
        sig_u = np.append(sig_l[1:], np.inf)
        # Last upper breakpoints for all edges are always infinity
        # -> Identify last function parts
        last_fct_part = (np.append(self._edge_indices[1:], [-1])
                         - self._edge_indices) != 0
        sig_u[last_fct_part.flatten()] = np.inf

        c = np.c_[c, sig_l, sig_u]

        self.coefficients = c

    def _init_position_offsets(self) -> None:
        first_pos = np.concatenate([[0], 1 + np.diff(self.edge).nonzero()[0]])
        last_pos = np.hstack((first_pos[1:] - 1, len(self) - 1))
        self._first_pos = first_pos.astype(int)
        self._last_pos = last_pos.astype(int)

    def get(self, attribute, at=None) -> np.ndarray:
        """Return all attribute values or attribute values at positions.
        
        Parameters
        ----------
        attribute : list, str
            The name(s) of the attributes
        at : list, int, default=None
            The position indices of the rows that should be returned.
            If set to None (default), all rows are returned.

        Returns
        -------
        attribute_list : numpy.ndarray
            An array with attribute(s) value(s)
        """
        if isinstance(attribute, list) and isinstance(attribute[0], str):
            ret = np.array([getattr(self, a) for a in attribute]).T
            if at is not None:
                return ret[at, :]
            return ret
        elif isinstance(attribute, str):
            ret = getattr(self, attribute)
            if at is not None:
                return ret[at]
            return ret
        else:
            raise ValueError("'attribute' must be (list of) strings.")

    def to_df(self) -> pd.DataFrame:
        """Get piecewise coefficients as DataFrame."""
        df = pd.DataFrame(
            self.coefficients,
            columns=PiecewiseQuadraticCoefficients.PWC_COEFFICIENT_COLS
        )
        df['edge'] = self._edge_indices
        return df

    def region_to_pos(self, region: np.ndarray) -> np.ndarray:
        """Convert a region index array to a position index array"""
        return self._first_pos + region

    def pos_to_region(self, pos: np.ndarray) -> np.ndarray:
        """Convert a position index array to a region index array"""
        return pos - self._first_pos

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        if save_dict is None:
            save_dict = {}

        save_att = {
            "coefficients": "coefficients",
            "edge_indices": "_edge_indices",
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
    def from_npz(
            cls,
            data,
            prefix: str = "",
            **kwargs
            ) -> PiecewiseQuadraticCoefficients:
        if isinstance(data, str):
            data = np.load(data)
        pwqc_data = (data[prefix + "coefficients"],
                     data[prefix + "edge_indices"])
        return cls(pwqc_data, **kwargs)

    from_npz.__func__.__doc__ = _doc.from_npz.__doc__

    @property
    def a(self) -> np.ndarray:
        r"""All coefficients :math:`a_{e, t}`."""
        return self.coefficients[:, 0]

    @property
    def b(self) -> np.ndarray:
        r"""All coefficients :math:`b_{e, t}`."""
        return self.coefficients[:, 1]

    @property
    def d(self) -> np.ndarray:
        r"""All values :math:`d_{e, t} = \frac{b_{e,t}}{2 a_{e,t}}`."""
        return self.coefficients[:, 5]

    @property
    def edge(self) -> np.ndarray:
        """A list of edge indices."""
        return self._edge_indices

    @property
    def lap_weight(self) -> np.ndarray:
        """All laplace weights."""
        return self.coefficients[:, 4]
        
    @property
    def offset(self) -> np.ndarray:
        r"""All coefficients :math:`\mathrm{offset}_{e, t}`."""
        return self.coefficients[:, 2]

    @property
    def first_pos(self) -> np.ndarray:
        """An array containing the first position index of all edges."""
        return self._first_pos
    
    @property
    def last_pos(self) -> np.ndarray:
        """An array containing the last position index of all edges."""
        return self._last_pos

    @property
    def sig_l(self) -> np.ndarray:
        """All derivative values at lower breakpoints."""
        return self.coefficients[:, 6]

    @property
    def sig_u(self) -> np.ndarray:
        """All derivative values at upper breakpoints."""
        return self.coefficients[:, 7]

    @property
    def tau(self) -> np.ndarray:
        r"""All breakpoints :math:`\tau_{e,t}` of all edges"""
        return self.coefficients[:, 3]

    @property
    def m(self) -> np.ndarray:
        """The total number of edges."""
        return self.edge[-1]

    @property
    def dtype_float(self):
        """Datatype of coefficients."""
        return self.coefficients.dtype

    @property
    def dtype_int(self):
        """Datatype of edge indices."""
        return self._edge_indices.dtype


class PiecewiseQuadraticFunction(EdgeCost):
    r"""A class representing the piecewise quadratic function of a single edge.
    
    The cost function defined as

    .. math::
        F(x) = a_i x^2 + b_i x + \mathrm{offset}_i \quad \text{for } x \in [\tau_i, \tau_{i+1})

    with breakpoints :math:`\tau_i`.
    
    Parameters
    ----------
    a: 1D-array of float
        List of the coefficients :math:`a_i` of the quadratic terms
    b: 1D-array of float
        List of the coefficients :math:`b_i` of the linear terms
    offset: 1D-array of float
        List of the offsets :math:`\mathrm{offset}_i`
    tau: 1D-array of float
        List of the breakpoints :math:`\tau_i`
    edge: int, default=None
        The index to which this cost functions belongs to
    """

    def __init__(self, a, b, offset, tau, edge=None):
        super().__init__(edge)
        self.a = a
        self.b = b
        self.offset = offset
        self.tau = tau

    def ddx(self, x, d=1):
        if d > 2:
            return 0

        for i, t in enumerate(self.tau):
            if x >= t:
                pos = i
        
        if d == 2:
            return 2 * self.a[pos]
        if d == 1:
            return 2 * self.a[pos] * x + self.b[pos]
        if d == 0:
            return self.a[pos] * x * x + self.b[pos] * x + self.offset[pos]
        
        raise ValueError(f"Invalid derivative order {d}.")


class PiecewiseQuadraticCost(NetworkCost):
    r"""Piecewise quadratic network costs.

    Represents piecewise quadratic network costs of all edges of the form
    
    .. math::
        F_e(x) = a_{e,i} x^2 + b_{e,i} x + \mathrm{offset}_{e,i} \quad
        \text{for } x \in [\tau_{e,i}, \tau_{e,i+1})

    where :math:`\tau_{e,i}` are the breakpoints of the cost functions.

    The cost functions are represented by their coefficients and breakpoints
    and can be index by either `position` or `region` indices.
    The region indices correspond to the index :math:`i` of the respective
    breakpoint, the position indices correspond to the position in the
    internal array and are the region indices offset by a certain value.

    For more information see :class:`~PiecewiseQuadraticCoefficients`.

    Parameters
    ----------
    shared : Shared
        Shared network object.
    data : :class:`~paminco.net.cost.PiecewiseQuadraticCoefficients`, tuple
        The raw data of these cost function.
        Can be of type ``PiecewiseQuadraticCoefficients`` containing
        the coefficients and breakpoints, or a tuple ``(coefficients, indices)``
        containing an array of coefficients and an array of corresponing edge
        indices.
    dtype_float : dtype, optional
        Dataytpe for coefficient value.
    dtype_int : dtype, optional
        Datatype for edge index value.
    copy : bool, default=True
        Whether to copy ``data``.

    Attributes
    ----------
    m : int
    coefficients : :class:`~paminco.net.cost.PiecewiseQuadraticCoefficients`
    first_pos : numpy.ndarray
    last_pos : numpy.ndarray
    rounding_margins : numpy.ndarray

    See also
    --------
    PiecewiseQuadraticCoefficients
    PiecewiseQuadraticFunction
    """

    def __init__(
            self,
            data: PolynomialCost | PiecewiseQuadraticCoefficients | tuple,
            shared: Shared = None,
            dtype_float=None,
            dtype_int=None
            ) -> None:
        super().__init__(shared=shared)
        ve_msg = (
            "data must be PolynomialCost, (np.array, np.array),"
            "PiecewiseQuadraticCoefficients"
            "(PolynomialCost, PiecewiseQuadraticCoefficients) or "
            "(PolynomialCost, (np.array, np.array))."
        )
        # Update dtypes (ignored if None)
        self.dtype_int = dtype_int
        self.dtype_float = dtype_float
        arr_kw = {
            "dtype_float": self.dtype_float,
            "dtype_int": self.dtype_int,
            "copy": copy,
        }

        # cast to tuple
        if isinstance(data, tuple) is False:
            data = tuple([data])

        if len(data) == 1:
            if isinstance(data[0], PiecewiseQuadraticCoefficients):
                self._ec = data[0]
            else:
                raise ValueError(ve_msg)
        elif len(data) == 2:
            # (coefficients, edge_indices)
            self._ec = PiecewiseQuadraticCoefficients(data, **arr_kw)
        else:
            raise ValueError(ve_msg)

        self._set_rounding_margins()

    def __eq__(self, other):
        if self.coefficients != other.coefficients:
            return False
        if np.array_equal(self.first_pos, other.first_pos) is False:
            return False
        if np.array_equal(self.last_pos, other.last_pos) is False:
            return False
        return True

    def __repr__(self):
        out = "Piecewise Quadratic Cost Function"
        out += "\n" + ("=" * len(out)) + "\n"
        out += self._ec.to_df().__repr__()
        return out

    def __len__(self):
        return self._ec.edge[-1] + 1

    def differentiate(self):
        raise NotImplementedError("not implemented for pwqc")

    def integrate(self):
        raise NotImplementedError("not implemented for pwqc")

    def _set_rounding_margins(self):
        # references for convenience
        e = self._ec.edge
        a = self._ec.a

        # get finite a values and edge indices
        fin = np.isfinite(a)
        a = a[fin]
        e = e[fin]
        idx = np.concatenate([[0], 1 + np.diff(e).nonzero()[0]])

        # calculate slopes
        np.abs(a, out=a)
        np.log2(a, out=a)
        a = np.floor(a).astype(int)

        # get minimum for all edges
        self._rounding_margins = np.minimum.reduceat(a, idx)

    def edge_cost(self, edge: int) -> PiecewiseQuadraticFunction:
        """Return the piecewise quadratic function of edge ``edge``."""
        at = np.arange(self.first_pos[edge], self.last_pos[edge] + 1)
        a = self._ec.get('a', at)
        b = self._ec.get('b', at)
        offset = self._ec.get('offset', at)
        tau = self._ec.get('tau', at)
        return PiecewiseQuadraticFunction(a, b, offset, tau, edge)

    def value(
            self,
            x: np.ndarray,
            d: int = 0,
            at=None,
            is_region: bool = False
            ):
        r"""Return the values of all edge cost functions for the flow ``x``.

        Compute the vector :math:`\big(F_{e_1} (x_{e_1}), \dotsc,
        F_{e_m} (x_{e_m}) \big)^{\top}` given the flow vector
        :math:`\big( x_{e_1}, \dotsc, x_{e_m} \big)^\top`.
        If the derivative order ``d`` is set, the function :math:`F(x)` is
        replaced by its ``d``-th derivative, i.e., :math:`F^{(d)} (x)`.
        
        Parameters
        ----------
        x: np.ndarray
            An array containing flow values :math:`x_e` for all edges.
        d: int, default=0
            The derivative order.
        at:
            A vector of position or region indices.
        is_region: bool, default=False
            Indicates, whether ``at`` contains position indices (default case)
            or region indices.

        Returns
        -------
        cost_vector: np.ndarray
            The vector of the cost value of all edges.
        """
        if d < 0:
            raise ValueError("Invalid dervative order 'd' " + str(d))
        if d > 2:
            return np.zeros(len(x))

        # determine proper position in edge coefficients
        if at is None:
            # retrieve position by comparing flow and eta for edges
            at = self.position_of(x)
        elif is_region is True:
            # position is base pos + region
            at = self._ec.region_to_pos(at)

        return self._val_from_pos(flow=x, d=d, pos=at)

    def _val_from_pos(
            self,
            flow,
            d: int,
            pos
            ) -> np.ndarray:
        a, b, off = self._ec.get(["a", "b", "offset"], at=pos).T
        x = flow

        # determine output dtype
        dummy = np.array(1.22, dtype=self.dtype_float)
        dtype = (dummy * flow[0]).dtype

        if d == 0:
            # ignore a + b coeffs if offset is infinite to avoid nan
            fin = np.isfinite(off)
            out = np.zeros(len(flow), dtype=dtype)
            out[fin] = off[fin] + b[fin] * x[fin] + a[fin] * x[fin]**2
            return out
            # return off + (b*nfo_rows)*x + (a*nfo_rows)*x**2
        elif d == 1:
            # ignore a coeffs if b is infinite to avoid nan
            fin = np.isfinite(b)
            out = np.zeros(len(flow), dtype=dtype)
            out[fin] = b[fin] + 2 * a[fin] * x[fin]
            # return b + 2*(a*nfb_rows)*x
            return out
        elif d == 2:
            return 2 * a
        raise ValueError("Invalid dervative order " + str(d))

    def get(self,
            attribute,
            at=None,
            is_region: bool = True
            ) -> np.ndarray:
        """Return all attribute values or attribute values at positions.
        
        Parameters
        ----------
        attribute : list, str
            The name(s) of the attributes
        at : list, int, default=None
            The position indices of the rows that should be returned.
            If set to None (default), all rows are returned.
            If ``is_region`` is set to True, at is assumend to be a
            vector of region indices
        is_region : bool, default=True
            If true, at is assumed to be a region index vector.
            Otherwise is is assumed to be a position index vector.

        Returns
        -------
        attribute_list : numpy.ndarray
            An array with attribute(s) value(s)
        """
        if at is not None:
            if is_region is True:
                at = self._ec.region_to_pos(at)
        return self._ec.get(attribute, at)

    def get_coefficients(
            self,
            at: np.ndarray,
            is_region: bool = False
            ) -> PiecewiseQuadraticCoefficients:
        """Return PiecewiseQuadraticCoefficients for given positions.
        
        Parameters
        ----------
        at : list, int
            The position index or indices of the coefficient rows
            that should be included in the new coefficient object.
            If ``is_region`` is set to True, it is assumend that
            pos contains region index or indices
        is_region : bool, default=False
            If set to True, at is assumed to contain region indices
            rather than position indices.
        
        Returns
        -------
        coefficients : PiecewiseQuadraticCoefficients
            The new coefficient object with coefficients from the
            specified lines.
        """
        if is_region is True:
            at = self._ec.region_to_pos(at)

        return self._ec[at]

    def delete_edges(
            self,
            edges
            ) -> None:

        # delete from coefficients
        # 1) identify rows to keep
        edges = np.array(edges).reshape(1, -1)
        to_keep = ~((self._ec.edge.reshape(-1, 1) - edges) == 0).any(axis=1)
        self._ec = self._ec[to_keep]

        # 2) reset indexing of edge in coefficients
        d = np.diff(self._ec.edge)
        d[d.nonzero()] = 1
        self._ec._edge_indices = np.concatenate([[0], np.cumsum(d)])
        self._ec._init_position_offsets()
        self._set_rounding_margins()

    def c(self, region):
        r"""Return the vector of all c values for the given region.
        
        This is an alias for the function `laplace_weights`.

        Returns the vector :math:`\big(c_{e_1, t_1}, \dotsc
        c_{e_m, t_m})\big)` containing all values
        :math:`c_{e, t_e} = \frac{1}{2 a_{e, t_e}}` of all edges
        for the given ``region`` represented by a region vector of the form
        :math:`\big(t_1, \dotsc t_m\big)`.

        See also
        --------
        laplace_weights :
            Returns the same values.
        """
        return self.laplace_weights(region=region)

    def delta_c(
            self,
            region,
            step,
            edge=None
            ):
        r"""Return the difference in c values for region and region + step.

        Computes :math:`\Delta c := c_{e, t_e + s} - c_{e, t_e}` where
        :math:`t` corresponds to the parameter ``region`` and :math:`s`
        corresponds to the parameter ``step``.
        Computes this either for a single edge or for all edges, depending
        on the parameter ``edge``.

        Note that :math:`c_{e, i} := \frac{1}{2 a_{e,i}}`.

        Parameters
        ----------
        region : int or iterable
            The region. Can be a region vector t or, if used together with
            keyword edge, an int t_{e}
        step : int or iterable
            The relative offset of the second region, i.e.,
            region = region + step.
            If given as int, the same step is used for all edges.
        edge : int, optional
            The edge for which delta_c is requested.
            Default: None - return a vector of delta_c for all edges.

        Returns
        -------
        delta_c : float or np.ndarray
            The (vector of) delta_c.
        """
        if is_int(edge) is True:
            if is_int(step) is False or is_int(region) is False:
                raise ValueError(
                    "If edge is given, region and step must be of type int."
                )
            c = self._ec.lap_weight
            pos_e = self.first_pos[edge]
            return c[pos_e + region + step] - c[pos_e + region]
        return (self.get("lap_weight", at=region, is_region=True)
                - self.get("lap_weight", at=region + step, is_region=True))

    def laplace_weights(
            self,
            flow=None,
            region=None
            ):
        r"""Returns the Laplace weights for all edges at x.

        Returns the vector :math:`\big(c_{e_1, t_1}, \dotsc
        c_{e_m, t_m})\big)` containing all values
        :math:`c_{e, t_e} = \frac{1}{2 a_{e, t_e}}` of all edges
        for the given ``region`` represented by a region vector of the form
        :math:`\big(t_1, \dotsc t_m\big)`.

        Parameters
        ----------
        region : ndarray, optional
            Regions of edges. If not given, laplace weights are
            determined using ``flow``.
        flow : ndarray, optional
            Flow on edges. Use flow to calculate laplace weights if
            ``region`` is not given.

        Returns
        -------
        weights : ndarray
            An array containg the Laplace weights for all edges.

        See Also
        --------
        paminco.net.cost.NetworkCost.laplace_weights :
            Use second derivative of cost functions to determine
            Laplace weights.
        paminco.network.network.Network.L : Laplacian matrix.
        c : Alias for this function.
        """
        if region is None:
            return super().laplace_weights(flow)

        return self.get('lap_weight', region)

    def position_of(
            self,
            flow: np.ndarray,
            up_bias: float = 1e-05,
            ) -> np.ndarray:
        # references for convenience
        edge = self.get("edge").ravel()
        tau = self.get("tau").ravel()

        # map flows to edges in data
        repeats = np.concatenate([[0],
                                  1 + np.diff(edge).nonzero()[0],
                                  [len(edge)]])
        repeats = np.diff(repeats)
        flows = np.repeat(flow, repeats) + up_bias

        # determine the first positions where df.tau <= df.flow
        tau_smaller_flow = (tau <= flows) * (edge + 1)
        index = np.hstack(((np.diff(tau_smaller_flow) < 0), False))

        out = np.copy(self.last_pos)
        out[edge[index]] = np.where(index)[0]
        return out

    def position_of_potential(self, potential: np.ndarray) -> np.ndarray:
        pot_diff = potential * self._s.Gamma()
        pos = np.zeros(self._s.m, dtype=self._s.dtype_int)
        for e in range(self._s.m):
            for i in range(self.first_pos[e], self.last_pos[e] + 1):
                if self._ec.sigl[i] <= pot_diff[e] and \
                   pot_diff[e] < self._ec.sigu[i]:
                    pos[e] = i
                    break
        return pos

    def region_of(
            self,
            flow: np.ndarray
            ) -> np.ndarray:
        """Find the region (see value function for details) of the flow."""
        pos = self.position_of(flow)
        return pos - self.first_pos

    def region_of_potential(self, potential: np.ndarray) -> np.ndarray:
        pos = self.position_of_potential(potential)
        return pos - self.first_pos

    def is_continuous(self, margin=0) -> bool:
        return self.is_smooth(k=0, margin=margin)

    def is_smooth(self, k=1, margin=0) -> bool:
        def _comp_with_margin(x, y):
            if margin == 0:
                return x == y
            return y - margin <= x and x <= y + margin

        # Test higher derivatives first
        if k > 0:
            if self.is_smooth(k - 1) is False:
                return False
        # Derivative is constant 0 for k > 2
        if k > 2:
            return True

        # Iterate through all coefficients
        for i in range(len(self._ec.coefficients) - 1):
            # If two adjacent coefficient rows belong to the same edge,
            # test smoothness
            if self._ec.edge[i + 1] == self._ec.edge[i]:
                # unpack the coefficients of the two coefficient rows
                c1 = self._ec.coefficients[i, 0:3]
                c2 = self._ec.coefficients[i + 1, 0:4]
                a1, b1, off1 = c1
                a2, b2, off2, tau = c2
                # If all coefficients are finite, compare values at breakpoint
                if np.alltrue(np.isfinite(c1)) and np.alltrue(np.isfinite(c2)):
                    # Case: Function -> compare a tau^2 + b * tau + offset
                    if k == 0:
                        x = a1 * tau ** 2 + b1 * tau + off1
                        y = a2 * tau ** 2 + b2 * tau + off2
                    # Case: First derivative -> compare 2 * a * tau + offset
                    if k == 1:
                        x = 2 * a1 * tau + b1
                        y = 2 * a2 * tau + b2
                    # Case: second derivative -> compare [2 * ] a
                    if k == 2:
                        x, y = a1, a2
                    if _comp_with_margin(x, y) == False:
                        return False
                # If there are infinite coefficients
                else:
                    lp = self._ec.last_pos[self._ec.edge[i]]
                    # If the first coefficient row is infinite _and_ the
                    # breakpoint IS finite, then the function must be
                    # discondinuous
                    t1 = self._ec.coefficients[i, 3]
                    if (not np.alltrue(np.isfinite(c1)) and np.isfinite(t1)):
                        return False
                    # If the last coefficient row is infinite _and_ the
                    # breakpoint IS finite and it is not the last breakpoint,
                    # then the function must be discontinuous
                    if (
                        not np.alltrue(np.isfinite(c2))
                        and np.isfinite(tau)
                        and lp != i + 1
                    ):
                        return False

        return True
    
    def add_to_etree(
            self,
            edge_node: et.Element,
            idx: int,
            overwrite: bool = True
            ) -> None:
        """Add cost for edge with index ``idx`` to xml Element.

        Parameters
        ----------
        edge_node : et.Element
            Xml element to append cost to.
        idx : int
            Index of edge to append cost for.
        overwrite : bool, default=True
            If ``True``, existing cost data in edge_node will be
            overwritten.
        """
        # Get edge coefficients for edge
        ec = self._ec[(self._ec.edge == idx)]

        cost_node = edge_node.find("cost")
        if cost_node is None:
            cost_node = et.SubElement(edge_node, "cost")
        if overwrite is True and cost_node.find("piecewisequadratic") is not None:
            cost_node.remove(cost_node.find("piecewisequadratic"))
        pcost_node = cost_node.find("piecewisequadratic")
        if pcost_node is None:
            pcost_node = et.SubElement(cost_node, "piecewisequadratic")

        for i in range(len(ec)):
            fp = et.SubElement(pcost_node, "functionpart")
            fp.attrib['a'] = str(ec.a[i])
            fp.attrib['b'] = str(ec.b[i])
            fp.attrib['c'] = str(ec.offset[i])
            fp.attrib['tau'] = str(ec.tau[i])

    @classmethod
    def from_xml(
            cls,
            data,
            shared: Shared = None,
            default_edge_cost: bool = True,
            **kwargs
            ) -> PiecewiseQuadraticCost:
        data = xml_find_root(data)
        
        edges = data.find("edges")
        if edges is None:
            raise ValueError("'edges' not found in 'data'.")
        
        coefficients = []
        edge_indices = []
        for i, e in enumerate(edges):
            c = cls._edge_cost_from_xml(
                e,
                default_edge_cost=default_edge_cost
            )
            coefficients.append(c)
            edge_indices.append(np.full(len(c), i))
        
        coefficients = np.vstack(coefficients)
        edge_indices = np.concatenate(edge_indices)
        
        return cls((coefficients, edge_indices), shared=shared, **kwargs)

    @staticmethod
    def _edge_cost_from_xml(edge_node, default_edge_cost: bool = True):
        cost_node = edge_node.find("cost")
        if cost_node is not None:
            pwq_node = cost_node.find("piecewisequadratic")
        else:
            pwq_node = None
        if pwq_node is None:
            if default_edge_cost is True:
                return np.array([0, 0, 0, -np.inf])
            elif default_edge_cost is False or default_edge_cost is None:
                estr = f"from {edge_node.attrib['from']} to " + \
                       f"{edge_node.attrib['to']}"
                raise ValueError("no polynomial cost for edge " + estr)
            else:
                return default_edge_cost
                
        part_nodes = pwq_node.findall("functionpart")
        coeff = []
        for part_node in part_nodes:
            a = parse_number(part_node.attrib.get("a", 0.0))
            b = parse_number(part_node.attrib.get("b", 0.0))
            c = parse_number(part_node.attrib.get("c", 0.0))
            tau = parse_number(part_node.attrib.get("tau", 0.0))
            coeff.append([a, b, c, tau])

        return np.array(coeff)

    from_xml.__func__.__doc__ = _doc.from_xml_shared.__doc__

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        
        save_dict["cost_type"] = self.__class__.__name__

        save_dict = self._ec.make_save_dict(prefix=prefix + "ec_",
                                            save_dict=save_dict)

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
            shared: Shared = None,
            prefix: str = "",
            **kwargs
            ) -> PiecewiseQuadraticCost:
        if isinstance(data, str):
            data = np.load(data)
        ec = PiecewiseQuadraticCoefficients.from_npz(data, prefix=prefix + "ec_")
        if any([k.startswith(prefix + "poly_") for k in data.keys()]):
            polycost = PolynomialCost.from_npz(data,
                                               shared=shared,
                                               prefix=prefix + "poly_")
            return cls((polycost, ec), **kwargs, shared=shared)
        return cls(ec, **kwargs, shared=shared)

    from_npz.__func__.__doc__ = _doc.from_npz_shared.__doc__

    @property
    def coefficients(self) -> PiecewiseQuadraticCoefficients:
        """The coefficients of the piecewise quadratic cost functions.
        
        Returns the :class:`~paminco.net.cost.PiecewiseQuadraticCoefficients` object
        containing all coefficients and breakpoints of the piecewise edge
        cost functions."""
        return self._ec

    @property
    def m(self) -> int:
        """Number of cost functions (= number of edges)."""
        return len(self._pos)

    @property
    def first_pos(self) -> np.ndarray:
        """Return an array containing the first position index of all edges."""
        return self._ec.first_pos

    @property
    def last_pos(self) -> np.ndarray:
        """Return an array containing the first position index of all edges."""
        return self._ec.last_pos

    @property
    def rounding_margins(self) -> np.ndarray:
        """ndarray (m, ) of float: minimal log2 of all slopes."""
        return self._rounding_margins


class InterpolationRule(abc.ABC):
    """Abstract class for a breakpoint computation rule for the interpolation."""

    @abc.abstractmethod
    def step(self, edge_cost, edge: int, x: float):
        """Compute the next breakpoint :math:`x_{i+1}`."""
        ...


class BreakpointsInterpolationRule(InterpolationRule):
    """Interpolation rule based on a given list of breakpoints.
    
    Parameters
    ----------
        The list of fixed breakpoints
    """
    
    def __init__(self, breakpoints):
        self.breakpoints = breakpoints

    def step(self, edge_cost, edge: int, x: float) -> float:
        """Return the closest breakpoint to ``x`` in ``self.breakpoints``"""
        for bp in self.breakpoints:
            if bp > x:
                return bp - x
        return np.inf


class EquidistantInterpolationRule(InterpolationRule):
    r"""Equidistant breakpoint rule for interpolation.
    
    This rule computes the next breakpoint as :math:`x_{i+1} = x_{i} + \Delta x`
    where :math:`\Delta x` is the equidistant step size

    Parameters
    ----------
    delta_x: float
        The equidistant interpolation step size :math:`delta_x`
    """

    def __init__(self, delta_x: float):
        self.delta_x = delta_x
    
    def step(self, edge_cost, edge: int, x: float):
        r"""Compute the next breakpoint :math:`x_{i+1} = x_{i} + \Delta x`.
        
        Parameters ``edge_cost`` and ``edge`` are not used."""
        return self.delta_x


class NetworkCostInterpolation():
    """A class managing the piecewise quadratic interpolation of the NetworkCost.
    
    Manages piecwise quadratic interpolation of the NetworkCost ``cost`` with
    the breakpoint rule ``rule``.

    Parameters
    ----------
    cost: NetworkCost
        The :class:`NetworkCost` to be interpolated
    rule: InterpolationRule
            The :class:`InterpolationRule` for the breakpoint computation in the interpolation
    x_max: float, default=None
        The upper bound for the interpolation on every edge (on undirected edges
        the interpolation runs from -x_max to x_max)
        If set to None, x_max is infered from the interpolation rule if possible
        and otherwise set to a default value defined in :class:`NetworkCostInterpolation`
    max_breakpoints_per_edge: int, default=1e05
        The maximum number of breakpoints on every edge
        
    """

    def __init__(
            self,
            cost: NetworkCost,
            rule: InterpolationRule,
            x_max=None,
            max_breakpoints_per_edge: int = 1e5
            ) -> None:
        self.cost = cost
        self.shared = cost.shared
        if cost.shared is None:
            raise RuntimeError(
                "Cost cannot be interpolated without being conencted "
                "to a network via shared object."
                )
        self.rule = rule
        if x_max is None:
            if hasattr(rule, "x_max"):
                x_max = rule.x_max
            else:
                x_max = 1000
        self.x_max = x_max
        self.max_breakpoints_per_edge = max_breakpoints_per_edge
    
    def interpolate(
            self,
            multiprocessing: bool = False
            ) -> PiecewiseQuadraticCost:
        """Interpolate network cost.
        
        Parameters
        ----------
        multiprocessing : bool, default=False
            Whether to interpolate egdes in parallel. Creates overhead
            and thus should be used for larger networks (in terms of
            links) only.
        
        Returns
        -------
        :class:`PiecewiseQuadraticCost`
            The piecewise quadratic network cost that interpolate the network cost
        """
        if multiprocessing is True:
            num_cpus = psutil.cpu_count(logical=False) - 1
            pool = mp.Pool(num_cpus)
            edge_idx = np.array_split(np.arange(self.shared.m), self.shared.m)
            data = pool.map(self._interpolate_edge, edge_idx)
            data = np.vstack([*data])
        else:
            data = []
            for edge in range(self.shared.m):
                ec = self._interpolate_edge(edge)
                data.append(ec)
            data = np.vstack(data)
        
        dtype_float = self.shared.dtype_float
        dtype_int = self.shared.dtype_int
        
        coeff = PiecewiseQuadraticCoefficients((data[:, 1:], data[:, 0]),
                                               dtype_float=dtype_float,
                                               dtype_int=dtype_int)

        return PiecewiseQuadraticCost(coeff, shared=self.cost.shared)

    def _interpolate_edge(self, edge):
        lb = self.shared.edges.lb[edge]
        ub = self.shared.edges.ub[edge]
        ei = EdgeCostInterpolation(edge,
                                   self.cost[edge],
                                   self.rule,
                                   self.x_max,
                                   lb,
                                   ub,
                                   self.max_breakpoints_per_edge)
        coefficients = ei.interpolate()
        index_col = np.full(len(coefficients), edge).reshape(len(coefficients), -1)
        return np.hstack([index_col, coefficients])


class EdgeCostInterpolation():
    """A class managing the piecwise quadratic interpolation of the cost function of a single edge.
    
    Objects of this class are typically create within :class:`NetworkCostInterpolation`
    in order to interpolate the edge costs.

    Parameters
    ----------
    edge: int
        The index of the edge that is interpolated.
    edge_cost: EdgeCost
        The :class:`EdgeCost` function to interpolate
    rule: InterpolationRule
            The :class:`InterpolationRule` for the breakpoint computation in the interpolation
    x_max: float, default=None
        The upper bound for the interpolation on every edge (on undirected edges
        the interpolation runs from -x_max to x_max)
        If set to None, x_max is infered from the interpolation rule if possible
        and otherwise set to a default value defined in :class:`NetworkCostInterpolation`
    lb: float, default=-np.inf
        The lower flow bound of this edge
    ub: float, default=np.inf
        The upper flow bound of this edge
    max_breakpoints_per_edge: int, default=1e05
        The maximum number of breakpoints on every edge
    """

    def __init__(
            self,
            edge: int,
            edge_cost: EdgeCost,
            rule: InterpolationRule,
            x_max: float,
            lb: float = -np.inf,
            ub: float = np.inf,
            max_breakpoints: int = 1e5
            ) -> None:
        self.edge = edge
        self.cost = edge_cost
        self.rule = rule
        self.x_max = x_max
        self.lb = lb
        self.ub = ub
        self.max_breakpoints = max_breakpoints

        self.F = lambda x: edge_cost(x, d=0)
        self.f = lambda x: edge_cost(x, d=1)

    def interpolate(self) -> np.ndarray:
        """Interpolate the Edge cost.
        
        Returns
        -------
        pwqd: ndarray
            Ndarray of shape (-1, 4) containing the piecewise coefficients.
        """
        lb = self.lb
        ub = self.ub

        x_lo, x_up = max(-self.x_max, lb), min(self.x_max, ub)
        
        pwqd = []
        xi = x_lo
        breakpoint_counter = 0
        
        while xi < x_up:
            delta = self.rule.step(self.cost, self.edge, xi)
            
            if delta is None:
                raise RuntimeError(
                    "Could not compute next interpolation "
                    f"breakpoint{self.edge}. Last breakpoint was {xi}."
                )
            
            next_xi = xi + delta

            # If the step size is infinite, the function is at most quadratic
            # => The coefficients are computed correctly for any next_xi > xi
            if delta == np.inf:
                next_xi = xi + 1
            piece = self._compute_piecewise_coeff(xi, next_xi)
            
            # Prepend additional function parts at the beginning
            if breakpoint_counter == 0:
                if lb > -np.inf:
                    pwqd.append([-np.inf, -np.inf, np.inf, -np.inf])
                if x_lo > lb:
                    aux_piece = copy.deepcopy(piece)
                    aux_piece[-1] = lb
                    pwqd.append(aux_piece)
            
            pwqd.append(piece)
            
            xi += delta
            breakpoint_counter += 1
            
            if breakpoint_counter > self.max_breakpoints:
                raise RuntimeError(
                    f"Interpolation breakpoint limit exceeded{self.edge}."
                )
        
        # Append additional function parts at the end
        if xi < ub:
            aux_piece = copy.deepcopy(piece)
            aux_piece[-1] = xi
            pwqd.append(aux_piece)
        if ub < np.inf:
            pwqd.append([np.inf, np.inf, np.inf, ub])
        
        return np.array(pwqd)

    def _compute_piecewise_coeff(self, x, y) -> list:
        fx = self.F(x)
        dfx, dfy = self.f(x), self.f(y)
        a = (dfy - dfx) / (y - x) / 2
        b = dfx - 2 * a * x
        # TODO: offset lead to discontinuous function
        offset = fx - a * x ** 2 - b * x
        return [a, b, offset, x]
