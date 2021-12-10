"""Module defining basic callback flags and simple callbacks."""

import time
import copy

import numpy as np

from paminco.utils.typing import IntEnum2


class CallBackFlag(IntEnum2):
    """Enum defining callback flags for callbacks."""

    INIT_START = 0
    """Start of initialization."""

    INIT_END = 1
    """End of initialization."""

    RUN_START = 2
    """Start of run."""

    RUN_END = 3
    """End of run."""

    ITER_PRE = 4
    """Before iteration begin."""

    ITER_START = 5
    """Start of an iteration."""

    ITER_END = 6
    """End of an iteration."""


class SimpleTimer:
    """Callback to generate basic timing infos for iterative solver.
    
    Attributes
    ----------
    iteration
    initialization
    time_run
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._iters_start = []
        self._iters_end = []

    def __repr__(self) -> str:
        out = super().__repr__()
        
        if hasattr(self, "_name"):
            out += "\n  model timed: " + self._name
        
        try:
            out += ("\n  total time: {:.3f}s"
                    .format((self._init_end - self._init_start)
                            + (self._run_end - self._run_start)))
        except:
            pass
        
        try:
            out += ("\n    - initialization: {:.3f}s"
                    .format(self._init_end - self._init_start))
        except:
            pass
        
        try:
            out += "\n    - run: {:.3f}s".format(self._run_end - self._run_start)
            try: 
                out += " (avg: {:.3f}s)".format(self.iteration.mean())
            except:
                pass
        except:
            pass
            
        return out

    def __call__(self, model, status_flag: CallBackFlag):
        if status_flag == CallBackFlag.INIT_START:
            self._init_start = time.time()
            self._name = model.name
        elif status_flag == CallBackFlag.INIT_END:
            self._init_end = time.time()
        elif status_flag == CallBackFlag.RUN_START:
            self._run_start = time.time()
        elif status_flag == CallBackFlag.ITER_PRE:
            self._iter_pre = time.time()
        elif status_flag == CallBackFlag.RUN_END:
            self._run_end = time.time()
        elif status_flag == CallBackFlag.ITER_START:
            self._iters_start.append(time.time())
        elif status_flag == CallBackFlag.ITER_END:
            self._iters_end.append(time.time())
        else:
            raise ValueError(
                f"status flag not recognized by {self.__class__.__name__}."
            )
    
    @property
    def iteration(self) -> np.ndarray:
        """ndarray of float: time of iterations."""
        return np.array(self._iters_end) - np.array(self._iters_start)

    @property
    def initialization(self) -> float:
        """float: time for initialization."""
        return self._init_end - self._init_start

    @property
    def time_run(self) -> float:
        """float: total time for run."""
        return self._run_end - self._run_start

    @property
    def total_time(self) -> float:
        """float: total time including run and initialization time"""
        return self.initialization + self.time_run


class SimpleDebuggerIterativeModel:
    """Simple callback that stores copies of the model during run."""

    def __init__(self) -> None:
        self._debug_copies = {}
    
    def __getitem__(self, idx: int):
        return self.get(idx)

    def __call__(self, model, status_flag: CallBackFlag) -> None:
        if status_flag == CallBackFlag.RUN_START:
            self._make_debug_copy(model, -1)
        elif status_flag == CallBackFlag.ITER_PRE:
            self._make_debug_copy(model, 0)
        elif status_flag == CallBackFlag.ITER_END:
            self._make_debug_copy(model, model.i)

    def get(self, i: int):
        """Get debug copy of for iteration ``i``.

        Parameters
        ----------
        i : int
            Iteration for which to retrieve debug copy of model.
            
            * ``i == -1`` -> after initialization / before run.
            * ``i ==  0`` -> before first iteration in run loop.
            * ``i ==  1`` -> after first iteration in run loop.
        
        Returns
        -------
        Any
        """
        return self._debug_copies[i]

    def _make_debug_copy(self, model, i: int) -> None:
        # customizable
        self._debug_copies[i] = copy.deepcopy(model)
