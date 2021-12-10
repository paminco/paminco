# flake8: noqa

from . import cost
from . import demand
from . import network
from . import path
from . import shared

from .network import Network
from .shared import Shared, Nodes, Edges
from .cost import NetworkCost, PolynomialCost, PiecewiseQuadraticCost, SymbolicCost
from .demand import demand_vector, DemandFunction, LinearDemandFunction, AffineDemandFunction
from ._data import DATA_BRAESS
from ._data_sioux import TNTP_SIOUX_NET, TNTP_SIOUX_NODE, TNTP_SIOUX_TRIPS
from ._data_gas import temporary_gas_files
from paminco.utils.readin import FileLineIterator


def load_sioux(**kw):
    tntp_net = FileLineIterator(TNTP_SIOUX_NET, is_string=True).dump()
    tntp_node = FileLineIterator(TNTP_SIOUX_NODE, is_string=True).dump()
    tntp_trips = FileLineIterator(TNTP_SIOUX_TRIPS, is_string=True).dump()
    return Network.from_tntp(tntp_net, tntp_trips, tntp_node, **kw)


def load_braess(**kw):
    return Network(**DATA_BRAESS, **kw)


def load_gas(name: str, **kw) -> Network:
    with temporary_gas_files(name) as files:
        return Network.from_gaslib(*files, **kw)


__all__ = [s for s in dir() if not s.startswith("_")]
