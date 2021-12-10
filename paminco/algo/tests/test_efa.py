import pytest
import numpy as np

from paminco.net.network import Network
from paminco.algo.efa import EFA
from paminco.net._data_examples import (NET_ELECTRICAL_BRAESS,
                                       NET_ELECTRICAL_PIECEWISE,
                                       NET_DISCONTINUOUS_COST)


@pytest.fixture
def simple_electrical_network():
    return Network.from_xml(NET_ELECTRICAL_BRAESS)


@pytest.fixture
def ambiguous_network():
    return Network.from_xml(NET_DISCONTINUOUS_COST)


@pytest.fixture
def piecewise_electrical_network():
    return Network.from_xml(NET_ELECTRICAL_PIECEWISE)


def basic_efa_test(
        net,
        target_breakpoints,
        target_flows_at_breakpoints,
        aux_lambdas=None,
        target_flows_at_aux_lambdas=None,
        lambda_max=np.inf
        ):
    efa = EFA(net, lambda_max=lambda_max, round_lambda=8)
    efa.run()

    breakpoints = efa.all_params(filter_same=False)
    assert np.allclose(breakpoints, target_breakpoints, rtol=1e-05, atol=1e-08)

    flows_at_breakpoints = efa.flow_at(breakpoints)
    assert np.allclose(flows_at_breakpoints, target_flows_at_breakpoints,
                       rtol=1e-05, atol=1e-08)

    if aux_lambdas is not None:
        test_flows = efa.flow_at(aux_lambdas)
        assert np.allclose(test_flows, target_flows_at_aux_lambdas,
                           rtol=1e-05, atol=1e-08)

    edge_data = np.array([[ "s", "v1"],
                          [ "s", "v2"],
                          ["v1", "v2"],
                          ["v1",  "t"],
                          ["v2",  "t"]])
    poly_cost = np.array([[0, 0, 1],
                          [0, 3, 0.5],
                          [0, 0, 0.5],
                          [0, 3, 0.5],
                          [0, 0, 1]])
    demand_data = (("s", "t", 1))
    d = {"s": 0, "v1": 1, "v2": 2, "t": 3}
    net = Network(edge_data,
                  cost_data=poly_cost,
                  demand_data=demand_data,
                  kw_edge={"map_labels_to_indices": d})
    # TODO: fix the interpolation


@pytest.mark.e2e
def test_simple_electrical_network(simple_electrical_network):
    net = simple_electrical_network
    breakpoints = [0, 3.0, 3.0, 8.0]
    flows_at_breakpoints = np.array([[0., 0., 0., 0., 0.],
                                     [2., 1., 1., 1., 2.],
                                     [2., 1., 1., 1., 2.],
                                     [4., 4., 0., 4., 4.]]).T
    aux_lambdas = [3, 8, 20, 32]
    target_flows_at_aux_lambdas = ([[ 2. ,  4. ,  9.5, 15. ],
                                    [ 1. ,  4. , 10.5, 17. ],
                                    [ 1. , -0. , -1. , -2. ],
                                    [ 1. ,  4. , 10.5, 17. ],
                                    [ 2. ,  4. ,  9.5, 15. ]])
    basic_efa_test(net, breakpoints, flows_at_breakpoints, aux_lambdas,
                   target_flows_at_aux_lambdas)


@pytest.mark.e2e
def test_ambiguous_network(ambiguous_network):
    net = ambiguous_network
    breakpoints = [0, 3.0, 4.0, 13/3, 17/3, 6]
    flows_at_breakpoints = np.array([[0., 0., 0.],
                                     [1., 1., 2.],
                                     [1., 1., 3.],
                                     [4/3, 4/3, 3],
                                     [2., 2., 11/3],
                                     [2., 2., 4.]]).T
    basic_efa_test(net, breakpoints, flows_at_breakpoints)


@pytest.mark.e2e
def test_piecewise_electrical_network(piecewise_electrical_network):
    breakpoints = [0, 3/2, 15/4, 23/4]
    flows_at_breakpoints = np.array([[0., 0., 0.],
                                     [.5, .5, 1],
                                     [2., 2., 7/4],
                                     [3., 3., 11/4]]).T
    basic_efa_test(
        piecewise_electrical_network,
        breakpoints,
        flows_at_breakpoints
    )


@pytest.mark.e2e
def test_start_in_ambiguous_region():
    xml = """
    <network>
    <edges>
        <edge from="s" to="v">
        <cost>
            <piecewisequadratic>
            <functionpart a="-inf" b="-inf" c="inf" tau="-inf"/>
            <functionpart a=".5" b="0" c="0" tau="0"/>
            </piecewisequadratic>
        </cost>
        </edge>
        <edge from="v" to="w">
        <cost>
            <piecewisequadratic>
            <functionpart a="-inf" b="-inf" c="inf" tau="-inf"/>
            <functionpart a=".5" b="0" c="0" tau="0"/>
            <functionpart a=".5" b="0" c="0" tau=".5"/>
            </piecewisequadratic>
        </cost>
        </edge>
        <edge from="w" to="t">
        <cost>
            <piecewisequadratic>
            <functionpart a="-inf" b="-inf" c="inf" tau="0"/>
            <functionpart a=".5" b="0" c="0" tau="-inf"/>
            </piecewisequadratic>
        </cost>
        </edge>
    </edges>
    <commodities>
        <commodity from="v" to="t" rate="1.0" />
    </commodities>
    <nodes>
        <node node="s" x="0" y="0" />
        <node node="v" x="1" y="0" />
        <node node="w" x="2" y="0" />
        <node node="t" x="3" y="0" />
    </nodes>
    </network>
    """
    net = Network.from_xml(xml)
    efa = EFA(net)
    assert np.array_equal(np.array([0, 1, 1]), np.array(efa._initial_region_linear()))
    efa.run()
    breakpoints = efa.all_params(filter_same=False)
    target_breakpoints = np.array([0, 0.5])
    assert np.allclose(breakpoints, target_breakpoints, rtol=1e-05, atol=1e-08)