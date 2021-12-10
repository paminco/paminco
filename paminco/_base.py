from __future__ import annotations

from collections.abc import MutableSequence
import abc
from copy import deepcopy
from time import time, localtime, strftime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as spip

from . import _doc
from .callback import CallBackFlag, SimpleTimer
from .net import Network
from .utils.misc import callback_to_list


class Base(abc.ABC):
    """Abstract solver class.
    
    Parameters
    ----------
    network : Network
        A network.
    name : str, optional
        Name of solver. If None (default), an automatic name will be
        generated.
    callback : (list of callable), optional
        Will be called during initialization and run() with flags indicating
        the status of the algorithm.
    use_simple_timer : bool, default=True
        Whether timestamps will be collected during initialization and run.
    copy_network : bool, default=True
        Whether to work on a copy of ``network``.
    
    Attributes
    ----------
    callbacks : list
        List of callbacks.
    network
    name
    
    """

    def __init__(
            self,
            network: Network,
            name=None,
            callback=None,
            use_simple_timer: bool = True,
            copy_network: bool = True
            ) -> None:
        self._init_timestamp = time()
        self._name = name
        
        # Handle callbacks
        if use_simple_timer is True:
            self.timer = SimpleTimer()
            self.callbacks = [self.timer] + callback_to_list(callback)
        else:
            self.callbacks = callback_to_list(callback)
        
        self.callback(CallBackFlag.INIT_START)
        
        # Init network
        if copy_network is True:
            self._net = deepcopy(network)
        else:
            self._net = network

    def callback(self, flag: CallBackFlag, aux_callback=None):
        if aux_callback:
            all_cb = self.callbacks + callback_to_list(aux_callback)
        else:
            all_cb = self.callbacks
        
        for cb in all_cb:
            cb(self, flag)

    @property
    def network(self) -> Network:
        """Get network obj."""
        return self._net

    @property
    def name(self) -> str:
        """Get name of object.
        
        If ``name`` is passed as argument in __init__, it will be
        returned. Otherwise, an automatically created 'name' based on
        the timestamp will be returned.
        
        Returns
        -------
        str
            Object name.
        """
        if self._name is not None:
            return self._name
        return self.__class__.__name__ + "_" + strftime("%y%m%d-%H:%M", localtime(self._init_timestamp))
    
    @name.setter
    def name(self, value: str) -> None:
        self._name = value


class Config:
    """Base class for configurations."""

    def __repr__(self) -> str:
        out = ""
        for (k, v) in sorted(self.__dict__.items()):
            out += str(k).ljust(30) + str(v) + "\n"
        return out


class BreakpointSolution:
    """Class that stores breakpoint solutions.
    
    Parameters
    ----------
    param : float
        Value of demand multiplier at breakpoint.
    flow : ndarray
        Minimum cost at breakpoint.
    cost : float, optional
        Total cost of ``flow``, calculated by ``network.cost(flow).sum().``
    potential : ndarray, optional
        Optimal node potential vector that induces ``flow``.
    
    Attributes
    ----------
    param : float
        Value of demand multiplier at breakpoint.
    flow : ndarray
        Minimum cost at breakpoint.
    cost : float, optional
        Total cost of ``flow``, calculated by ``network.cost(flow).sum().``
    potential : ndarray, optional
        Optimal node potential vector that induces ``flow``.
    """

    def __init__(
            self,
            param: float,
            flow: np.ndarray,
            cost=None,
            potential=None,
            ) -> None:
        self.param = param
        self.flow = flow
        self.cost = cost
        self.potential = potential


class TypedList(MutableSequence):
    """List that allows only certain types.
    
    https://stackoverflow.com/questions/3487434 @Alex Martelli
    """

    def __init__(self, oktypes, *args) -> None:
        self.oktypes = oktypes
        self.list = list()
        self.extend(list(args))

    def check(self, v) -> None:
        if not isinstance(v, self.oktypes):
            raise TypeError("Invalid list element.")

    def __len__(self) -> int:
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i) -> None:
        del self.list[i]

    def __setitem__(self, i, v) -> None:
        self.check(v)
        self.list[i] = v

    def insert(self, i, v) -> None:
        self.check(v)
        self.list.insert(i, v)

    def __str__(self) -> str:
        return str(self.list)


class ParametricSolution(TypedList):
    """List that stores `B` BreakpointSolutions.
    
    Parameters
    ----------
    breakpoints : sequence of BreakpointSolutions
    
    Attributes
    ----------
    flow_ip : scipy.interpolate.interpolate.interp1d, optional
        Interpolator of edge flows.
    pot_ip : scipy.interpolate.interpolate.interp1d, optional
        Interpolator of node potentials.
    has_potentials
    has_costs
    arr_param
    arr_flow
    arr_cost
    arr_potential
    dflow
    dpi
    """

    def __init__(self, *args) -> None:
        super().__init__(BreakpointSolution, *args)

    def __call__(self, param) -> np.ndarray:
        return self.flow_at(param)

    def all_params(
            self,
            filter_same: bool = True,
            return_indices: bool = False
            ):
        """Get parameters of all parametric solutions.
        
        Parameters
        ----------
        filter_same : bool, default=True
            Whether to remove equal parameters. E.g., [1, 2, 3, 3, 5]
            will be reduced to [1, 2, 3, 5]
        return_indices : bool, default=False
            Whether to return indices of filtered params if
            ``filter_same`` is True.
        
        Returns
        -------
        params : np.ndarray
            Parameters of (filtered) parametric solutions.
        idx : np.ndarray, optional
            Indices of filtered params, if ``filter_same`` is True and
            ``return_indices`` is True.
        """
        params = np.array([s.param for s in self])
        if filter_same is True:
            idx = np.concatenate([[0], 1 + np.diff(params).nonzero()[0]])
            params = params[idx]
            if return_indices is True:
                return params, idx
        return params

    def flow_at(self, param) -> np.ndarray:
        """Interpolate flow at param.
        
        Parameters
        ----------
        param : array_like
            1D array of real values to interpolate flow at.
        
        Returns
        -------
        ndarray
            Flow at param, 2D array of shape (m, d), where d is number of
            elements in ``param``.
        
        See Also
        --------
        scipy.interpolate.interp1d
        """
        if len(self) == 1 and self.dflow is not None:
            p, flow = self[0].param, self[0].flow
            return flow + (param - p) * self.dflow
            
        if hasattr(self, "flow_ip") is False:
            raise RuntimeError("Flows have not been interpolated.")
        
        flow_ip = self.flow_ip
        if self.dflow is None:
            return flow_ip(param)
        else:
            param = np.array(param).reshape(-1)
            if len(param) > 1 and (np.diff(param) < 0).any():
                raise ValueError("Params must increase monotonically.")
            
            # Valid to interpolate
            param_ip = param[param <= flow_ip.x.max()]
            res = flow_ip(param_ip)
            
            if len(param_ip) != len(param):
                res_non_ip = []
                for p in param[param > flow_ip.x.max()]:
                    flow_at_p = self[-1].flow + (p - self[-1].param) * self.dflow
                    res_non_ip.append(flow_at_p)
                res_non_ip = np.array(res_non_ip).T
                
                # Stack solutions
                res = np.hstack((res, res_non_ip))
            
            return res.squeeze()

    def potential_at(self, param) -> np.ndarray:
        """Interpolate potential at param.
        
        Parameters
        ----------
        param : array_like
            1D array of real values to interpolate potential at.
        
        Returns
        -------
        ndarray
            Potential at param, 2D array of shape (n, d), where d is number of
            elements in ``param``.
        
        See Also
        --------
        scipy.interpolate.interp1d
        """
        if len(self) == 1 and self.dpi is not None:
            p, potential = self[0].param, self[0].potential
            return potential + (param - p) * self.dpi
        
        if hasattr(self, "pot_ip") is False:
            raise RuntimeError("Potentials have not been interpolated.")
        
        pot_ip = self.pot_ip
        if self.dpi is None:
            return pot_ip(param)
        else:
            param = np.array(param).reshape(-1)
            if len(param) > 1 and (np.diff(param) < 0).any():
                raise ValueError("Params must increase monotonically.")
            
            # Valid to interpolate
            param_ip = param[param <= pot_ip.x.max()]
            res = pot_ip(param_ip)
            
            if len(param_ip) != len(param):
                res_non_ip = []
                for p in param[param > pot_ip.x.max()]:
                    pot_at_p = self[-1].potential + (p - self[-1].param) * self.dpi
                    res_non_ip.append(pot_at_p)
                res_non_ip = np.array(res_non_ip).T
                
                # Stack solutions
                res = np.hstack((res, res_non_ip))
            
            return res.squeeze()

    def plot_flow_on_edge(
            self,
            edge: int,
            x=None,
            ax=None,
            **kwargs,
            ) -> None:
        """Plot flow on edge.
        
        Parameters
        ----------
        edge : int
            Index of edge to plot flow.
        x : ndarray, optional
            Plot flow for demand factor x. If None, flow will be plotted for 
            linspace(min_param, max_param, 100).
        
        Returns
        -------
        fig : Figure, optional
            Figure that created ax belongs to if `ax` is None.
        ax : AxesSubplot, optional
            Axis where flow is plotted on, retured if None given.
        """
        return_figax = False
        if ax is None:
            fig, ax = plt.subplots()
            return_figax = True
        
        if x is None:
            x = np.linspace(self[0].param, self[-1].param, 100)
        ax.plot(x, self.flow_at(x)[edge], **kwargs)
        
        if return_figax:
            return fig, ax

    def set_interpolators(self, dflow=None, dpi=None) -> None:
        """Set interpolator between breakpoints.
        
        Parameters
        ----------
        dflow : array_like (m, ), optional
            Slope of flow after last breakpoint-
        dpi : array_like (n, ), optional
            Slope of potential after last breakpoint-
        """
        self._set_flow_interpolator(dflow=dflow)
        self._set_potential_interpolator(dpi=dpi)

    def _set_flow_interpolator(self, dflow=None) -> None:
        # Setup scipy flow interpolator between breakpoints
        p, idx = self.all_params(filter_same=True, return_indices=True)
        f = np.array([s.flow for s in self])[idx, :]
        if len(p) > 1:
            self.flow_ip = spip.interp1d(p, f.T)
        
        if dflow is not None:
            self._dflow = np.array(dflow)

    def _set_potential_interpolator(self, dpi=None) -> None:
        # Setup scipy potential interpolator between breakpoints
        if self.has_potentials is False:
            return
        
        p, idx = self.all_params(filter_same=True, return_indices=True)
        pot = np.array([s.potential for s in self])[idx, :]
        if len(p) > 1:
            self.pot_ip = spip.interp1d(p, pot.T)
        
        if dpi is not None:
            self._dpi = np.array(dpi)

    def make_save_dict(
            self,
            prefix: str = "",
            save_dict=None,
            **kwargs
            ) -> dict:
        if save_dict is None:
            save_dict = {}
        
        save_dict["arr_param"] = self.arr_param
        save_dict["arr_flow"] = self.arr_flow
        
        if self.has_costs is True:
            save_dict["arr_cost"] = self.arr_cost
        if self.has_potentials is True:
            save_dict["arr_potential"] = self.arr_potential
        if self.dflow is not None:
            save_dict["dflow"] = self.dflow
        if self.dpi is not None:
            save_dict["dpi"] = self.dpi
        
        save_dict.update(**kwargs)
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

    def to_df(
            self,
            prefix_flow: str = "edge",
            prefix_potential: str = "node",
            add_delta_row: bool = False,
            ) -> pd.DataFrame:
        """Get ParametricSolution as dataframe.
        
        Parameters
        ----------
        prefix_flow : str, default="edge"
            Prefix of flow columns.
        prefix_potential : str, default="node"
            Prefix of potential columns.
        add_delta_row : bool, default=False
            If True, a row will be appended that contains the slope values
            of flow (and potential).
        """
        # Build index
        if self.dflow is None:
            add_delta_row = False
        index = np.arange(len(self))
        if add_delta_row:
            index = [*index.astype(str)] + ["delta"]
        
        # Build flow df
        if add_delta_row is True and self.dflow is not None:
            df = pd.DataFrame(np.vstack((self.arr_flow, self.dflow)),
                              index=index)
        else:
            df = pd.DataFrame(self.arr_flow, index=index)
        df.columns = [f"{prefix_flow}_{e}" for e in df.columns]
        
        # Build potential df
        if self.has_potentials is True:
            if add_delta_row is True and self.dpi is not None:
                df_pot = pd.DataFrame(np.vstack((self.arr_potential, self.dpi)),
                                      index=index)
            else:
                df_pot = pd.DataFrame(self.arr_potential, index=index)
            df_pot.columns = [f"{prefix_potential}_{e}" for e in df_pot.columns]
            
            # Merge flow and pot frames
            df = df.join(df_pot)
        
        # Set cost in merged frame
        if self.has_costs is True:
            try:
                df["cost"] = [*self.arr_cost] + [np.nan]
            except ValueError:
                df["cost"] = self.arr_cost
            # Bring cost column to front
            cols = [df.columns[-1]] + df.columns[:-1].tolist()
            df = df[cols]
            
        # Add param column
        if "delta" in df.index:
            df["param"] = [*self.arr_param] + [np.nan]
        else:
            df["param"] = self.arr_param
        
        # Bring param column to front
        cols = [df.columns[-1]] + df.columns[:-1].tolist()
        df = df[cols]
        
        return df

    @classmethod
    def from_arrays(
            cls,
            param,
            flow,
            potential=None,
            cost=None,
            dflow=None,
            dpi=None,
            ) -> ParametricSolution:
        """Build ParametricSolution from arrays.
        
        Parameters
        ----------
        param : array_like (B, )
            Parameters at breakpoints.
        flow : array_like (B, m)
            Edge flow at ``param``.
        potential : array_like (B, n), optional
            Node potential at ``param``.
        cost : array_like (B, ), optional
            Total cost of edge ``flow`` at ``param``.
        dflow : array_like (B, ), optional
            Slope of flow after last breakpoint.
        dpi : array_like (B, ), optional
            Slope of potential after last breakpoint.
            
        Returns
        -------
        ParametricSolution
        """
        param = np.array(param)
        flow = np.array(flow)
        if potential is not None:
            potential = np.array(potential)
        else:
            potential = [None] * len(param)
        if cost is not None:
            cost = np.array(cost)
        else:
            cost = [None] * len(param)
        
        # Build class and fill with breakpoint solutions
        ps = cls()
        for para, f, pot, c in list(zip(param, flow, potential, cost)):
            bp = BreakpointSolution(float(para), f, c, pot)
            ps.append(bp)
        
        # Cleanup
        ps.set_interpolators(dflow=dflow, dpi=dpi)
        
        return ps

    @classmethod
    def from_df(
            cls,
            df: pd.DataFrame,
            prefix_flow: str = "edge",
            prefix_potential: str = "node",
            ) -> ParametricSolution:
        """Build ParametricSolution from pandas.DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe used to build ParametricSolution.
        prefix_flow : str, default="edge"
            Prefix of columns in ``df`` that hold edge flows. Column names
            are expected to be strings of the form ``f"{prefix_flow}_{idx}"``.
        prefix_potential : str, default="node"
            Prefix of columns that hold node potentials. Same column naming
            convention as for prefix_flow applies.
        
        Returns
        -------
        ParametricSolution
        """
        # Get flow and potential indices
        edge_cols = {int(c.split("_")[1]): i
                     for i, c in enumerate(df.columns)
                     if c.startswith(prefix_flow)}
        edge_cols = np.array(list(edge_cols.values()))[np.array(list(edge_cols.keys()))]
        node_cols = {int(c.split("_")[1]): i
                     for i, c in enumerate(df.columns)
                     if c.startswith(prefix_potential)}
        node_cols = np.array(list(node_cols.values()))[np.array(list(node_cols.keys()))]
        
        # Get interpolation values for x after last breakpoint (if available)
        dflow = None
        dpi = None
        if "delta" in df.index:
            dflow = df.loc["delta"][edge_cols].values
            if len(node_cols) > 0:
                dpi = df.loc["delta"][node_cols].values
            df.drop("delta", inplace=True)
        
        # Get params
        if "param" in df:
            # Param is column
            param = df.param.values
        else:
            # Params are in index
            param = [i.split("_")[1] for i in df.index]
        
        # Get flow
        arr_flow = df.values[:, edge_cols]
        
        # Get potentials
        arr_pot = None
        if len(node_cols) > 0:
            # Potential columns exist
            arr_pot = df.values[:, node_cols]
        arr_cost = None
        
        # Get cost
        if "cost" in df:
            arr_cost = df.cost.values
        
        return cls.from_arrays(param, arr_flow, arr_pot, arr_cost, dflow, dpi)

    def to_csv(
            self,
            file,
            prefix_flow: str = "edge",
            prefix_potential: str = "node",
            add_delta_row: bool = True,
            ) -> None:
        """Save ParametricSolution to ``CSV``.
        
        Parameters
        ----------
        file : str or file handle
            File to save ParametricSolution to, passed to
            :func:`pandas.DataFrame.to_csv`.
        prefix_flow : str, default="edge"
            Prefix of flow columns.
        prefix_potential : str, default="node"
            Prefix of potential columns.
        add_delta_row : bool, default=False
            If True, a row will be appended that contains the slope values
            of flow (and potential).
        """
        df = self.to_df(prefix_flow=prefix_flow,
                        prefix_potential=prefix_potential,
                        add_delta_row=add_delta_row)
        return df.to_csv(file, index_label="index")
    
    @classmethod
    def from_csv(
            cls,
            file,
            prefix_flow: str = "edge",
            prefix_potential: str = "node",
            ) -> ParametricSolution:
        """Build ParametricSolution from ``CSV``.
        
        Parameters
        ----------
        file : str, path object or file-like object
            File passed to :func:`pandas.read_csv`.
        prefix_flow : str, default="edge"
            Prefix of columns in ``df`` that hold edge flows. Column names
            are expected to be strings of the form ``f"{prefix_flow}_{idx}"``.
        prefix_potential : str, default="node"
            Prefix of columns that hold node potentials. Same column naming
            convention as for prefix_flow applies.
        
        Returns
        -------
        ParametricSolution
        """
        df = pd.read_csv(file, index_col="index")
        return cls.from_df(
            df,
            prefix_flow=prefix_flow,
            prefix_potential=prefix_potential
        )

    @classmethod
    def from_npz(
            cls,
            data,
            prefix: str = "",
            ) -> ParametricSolution:
        if isinstance(data, str):
            data = np.load(data)
        
        return cls.from_arrays(
            param=data[prefix + "arr_param"],
            flow=data[prefix + "arr_flow"],
            potential=data.get(prefix + "arr_potential", None),
            cost=data.get(prefix + "arr_cost", None),
            dflow=data.get(prefix + "dflow", None),
            dpi=data.get(prefix + "dpi", None),
        )

    from_npz.__func__.__doc__ = _doc.from_npz.__doc__

    @property
    def has_potentials(self) -> bool:
        """Whether node potentials have been set in breakpoint solutions."""
        return (self[0].potential is not None)
    
    @property
    def has_costs(self) -> bool:
        """Whether cost have been set in breakpoint solutions."""
        return (self[0].cost is not None)

    @property
    def arr_param(self) -> np.ndarray:
        """1D array (B, ) of float: parameters of breakpoint solutions."""
        return np.array([s.param for s in self])

    @property
    def arr_flow(self) -> np.ndarray:
        """2D array (B, m) of float: edge flows of breakpoint solutions."""
        return np.array([s.flow for s in self])

    @property
    def arr_cost(self) -> np.ndarray:
        """1D array (B, ) of float: total cost of breakpoint solutions."""
        return np.array([s.cost for s in self])

    @property
    def arr_potential(self) -> np.ndarray:
        """2D array (B, n) of float: node potential of breakpoint solutions."""
        return np.array([s.potential for s in self])

    @property
    def dflow(self):
        """Get slope of flow after last breakpoint."""
        if hasattr(self, "_dflow") is False:
            return None
        return self._dflow
    
    @property
    def dpi(self):
        """Get slope of potential after last breakpoint."""
        if hasattr(self, "_dpi") is False:
            return None
        return self._dpi


class ParametricSolver(Base):
    """Solver that calculate parametric min cost flows.

    Attributes
    ----------
    param_solution : ParametricSolution.
        List that stores `B` breakpoint solutions. Holds all data to to
        interpolate flows, potentials.
    
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, param) -> np.ndarray:
        return self.flow_at(param)

    def flow_at(self, param) -> np.ndarray:
        return self.param_solution.flow_at(param)

    flow_at.__doc__ = ParametricSolution.flow_at.__doc__
    
    def potential_at(self, param) -> np.ndarray:
        return self.param_solution.potential_at(param)

    potential_at.__doc__ = ParametricSolution.potential_at.__doc__

    def cost_at(self, param):
        """Interpolate cost at param.
        
        Parameters
        ----------
        param : array_like
            1D array of real values to interpolate cost at.
        
        Returns
        -------
        float or ndarray
            Cost at param, 1D array.
        
        See Also
        --------
        scipy.interpolate.interp1d
        """
        if isinstance(param, (int, float)):
            return self.network.cost(self.flow_at(param)).sum()
        else:
            return np.array([self.network.cost(f).sum()
                             for f in self.flow_at(param).T])

    def all_params(self, *args, **kwargs):
        return self.param_solution.all_params(*args, **kwargs)
    
    all_params.__doc__ = ParametricSolution.all_params.__doc__

    def plot_flow_on_edge(self, *args, **kwargs):
        return self.param_solution.plot_flow_on_edge(*args, **kwargs)
    
    plot_flow_on_edge.__doc__ = ParametricSolution.plot_flow_on_edge.__doc__

    def _add_param_solution(self, *args, **kwargs) -> None:
        self.param_solution.append(BreakpointSolution(*args, **kwargs))

    def _set_cost(self) -> None:
        # calculate cost at all solution breakpoints
        for ps in self.param_solution:
            ps.cost = self.network.cost(ps.flow).sum()

    def close_run(self, dflow=None, dpi=None) -> None:
        """Close run of solver."""
        self.param_solution.set_interpolators(dflow=dflow, dpi=dpi)

    @property
    def param_solution(self) -> ParametricSolution:
        """Get ParametricSolution object.
        
        See Also
        --------
        ParametricSolution
        """
        return self._param_solution

    @property
    def config(self):
        """Get configuration of solver."""
        return self._c


class AlphaBetaApproximativeSolver(ParametricSolver):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
