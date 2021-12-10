import abc

import numpy as np

from paminco._base import Base


class FlowOptimizer(Base):

    def __init__(
            self,
            network,
            name=None,
            callback=None,
            use_simple_timer: bool = True,
            copy_network: bool = True,
            **kw
            ) -> None:
        super().__init__(network,
                         name=name,
                         callback=callback,
                         use_simple_timer=use_simple_timer,
                         copy_network=copy_network)

    @abc.abstractmethod
    def run(self): ...
    @property
    @abc.abstractmethod
    def flow(self): ...


class LinearWarmstart:
    def __init__(
            self,
            flow: np.ndarray,
            param: float,
            copy_flow: bool = True
            ) -> None:
        self.param = param
        if copy_flow is True:
            flow = np.copy(flow)
        self.flow = flow

    def __call__(self, fac: float) -> np.ndarray:
        if np.isclose(fac, 0) or np.isclose(self.param, 0):
            return np.zeros_like(self.flow)
        return fac / self.param * np.nan_to_num(self.flow)
