import pytest
import numpy as np

from paminco.algo.mca import MCAInterpolationRule
from paminco.net import load_sioux
from paminco.net.shared import ID_UNMAPPED, LBL_UNMAPPED, FlowDirection, Edges
from paminco.net.network import Network
from paminco.net.demand import LinearDemandFunction, AffineDemandFunction
from paminco.utils.testing import assert_raises


@pytest.fixture
def net_sioux():
    return load_sioux()


class TestNodes:
    def test_df(self, net_sioux):
        df = net_sioux.nodes.to_df()
        assert list(df.columns) == ["label", "zone", "x", "y"]


class TestEdges:
    def test_init(self):
        data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 1)]
        
        e = Edges(data, directed_flow=True)
        assert np.array_equal(e.indices, np.array(data))
        
        # Check bounds
        e = Edges(data, directed_flow=True)
        b = np.array([[ 0., np.inf],
                    [ 0., np.inf],
                    [ 0., np.inf],
                    [ 0., np.inf],
                    [ 0., np.inf]])
        assert np.array_equal(e.bounds, b)
        assert e.flow_dir == FlowDirection.DIRECTED
        e = Edges(data, directed_flow=False)
        b = np.array([[ -np.inf, np.inf],
                    [ -np.inf, np.inf],
                    [ -np.inf, np.inf],
                    [ -np.inf, np.inf],
                    [ -np.inf, np.inf]])
        assert np.array_equal(e.bounds, b)
        assert e.flow_dir == FlowDirection.UNDIRECTED
        e = Edges((data, (-4, 5)))
        b = np.array([[ -4, 5],
                      [ -4, 5],
                      [ -4, 5],
                      [ -4, 5],
                      [ -4, 5]])
        assert np.array_equal(e.bounds, b)
        assert e.flow_dir == FlowDirection.UNDIRECTED
        
        # Index to label mappings
        e = Edges(data, map_indices_to_labels=None)
        assert len(np.unique(e.labels)) == 1
        assert e.labels[0][0] == LBL_UNMAPPED
        e = Edges(data, map_indices_to_labels=False)
        assert len(np.unique(e.labels)) == 1
        assert e.labels[0][0] == LBL_UNMAPPED
        e = Edges(data, map_indices_to_labels=True)
        assert np.array_equal(e.labels, np.array(data).astype(str))
        e = Edges(data, map_indices_to_labels={0: "A", 1: "A", 2: "A", 3: "B", 4: "B"})
        lbl = np.array([['A', 'A'],
            ['A', 'A'],
            ['A', 'B'],
            ['B', 'B'],
            ['B', 'A']], dtype='<U1')
        assert np.array_equal(e.labels, lbl)
        def lbl_callable(index):
            d = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B"}
            return np.vectorize(d.__getitem__)(index)
        e = Edges(data, map_indices_to_labels=lbl_callable)
        assert np.array_equal(e.labels, lbl)
        
        # Label to index mappings
        data_lbl = (np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 1)]) + 1).astype(str)
        
        e = Edges(data_lbl, map_labels_to_indices=None)
        assert len(np.unique(e.indices)) == 1
        assert e.indices[0][0] == ID_UNMAPPED
        e = Edges(data_lbl, map_labels_to_indices=False)
        assert len(np.unique(e.indices)) == 1
        assert e.indices[0][0] == ID_UNMAPPED
        
        e = Edges(data_lbl, map_labels_to_indices=True)
        idx = np.array([[0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]])
        assert np.array_equal(e.indices, idx)
    
    def test_df(self, net_sioux):
        df = net_sioux.edges.to_df()
        assert list(df.columns) == ["source_lbl", "target_lbl", "s", "t", "lb", "ub"]


class TestShared:
    @pytest.mark.parametrize("rng", [1, 42, 123])
    def test_sioux(self, net_sioux, rng):
        # shuffle node label mapping randomly
        rng = np.random.default_rng(rng)
        rng.shuffle(net_sioux.nodes.labels)
        net_sioux.shared.update()
        
        s = net_sioux.shared
        assert net_sioux.shared.m == 76
        assert net_sioux.shared.n == 24
        
        # check node mapping in edge
        for (source, target) in [("10", "16"), ("1", "2"), ("18", "7")]:
            s, t = net_sioux.shared.get_node_id([source, target])
            e = net_sioux.shared.nodes2edge[(s, t)]
            assert net_sioux.shared.edges.source_lbl[e] == source
            assert net_sioux.shared.edges.target_lbl[e] == target
        
        # check gamma
        for (source, target) in [("10", "16"), ("1", "2"), ("18", "7")]:
            s, t = net_sioux.shared.get_node_id([source, target])
            e = net_sioux.shared.nodes2edge[(s, t)]
            gam = net_sioux.shared.Gamma()[:, e].todok()
            assert gam[(s, 0)] == -1
            assert gam[(t, 0)] == 1
        
        net = load_sioux()
        net.shared.delete_edges([1, 2, 3])
        assert net.shared.Gamma().shape == (24, 73)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtyping_float(dtype):
    net_sioux = load_sioux(dtype_float=dtype)
    assert net_sioux.shared.dtype_float == dtype
    assert net_sioux.shared.dtype_float == dtype
    assert net_sioux.shared.nodes.x.dtype == dtype
    assert net_sioux.demand().dtype == dtype
    
    # check costs
    assert net_sioux.cost.coefficients.dtype == dtype
    # assert net_sioux.cost.value(np.ones(net_sioux.m)).dtype == np.float64
    assert net_sioux.cost.value(np.ones(net_sioux.m, dtype=dtype)).dtype == dtype
    # assert net_sioux.cost.ddx(np.ones(net_sioux.m)).dtype == np.float64
    assert net_sioux.cost.ddx(np.ones(net_sioux.m, dtype=dtype)).dtype == dtype
    # DOES NOT CONVERGE FOR np.float16 !
    i_rule = MCAInterpolationRule(1.01, 1, net_sioux.m, 1000)
    net_sioux._c = net_sioux.cost.interpolate(i_rule)
    assert net_sioux.cost._ec.dtype_float == dtype
    # assert net_sioux.cost.value(np.ones(net_sioux.m)).dtype == np.float64
    assert net_sioux.cost.value(np.ones(net_sioux.m, dtype=dtype)).dtype == dtype
    # assert net_sioux.cost.ddx(np.ones(net_sioux.m)).dtype == np.float64
    assert net_sioux.cost.ddx(np.ones(net_sioux.m, dtype=dtype)).dtype == dtype


@pytest.mark.parametrize("del_nodes",
    [
        ["2", "5", "11", "13"],
        ["8", "10", "13", "14"]
    ])
def test_is_connected(del_nodes):
    net_sioux = load_sioux()
    net_sioux.delete_nodes(del_nodes, is_label=True)
    assert net_sioux.is_connected() is False


def test_set_demand():
    net = load_sioux()
    
    s1, t1 = "1", "2"
    net.set_demand((s1, t1, 100), mode="linear")
    assert isinstance(net.demand, LinearDemandFunction)
    assert net.demand().shape == (24, 1)
    assert net.demand().data.tolist() == [-100, 100]
    net.shared.get_node_label(net.demand().indices) == [s1, t1]
    
    s1id, t1id = net.shared.get_node_id([s1, t1])
    net.set_demand((s1id, t1id, 100), mode="linear", is_label=False)
    assert isinstance(net.demand, LinearDemandFunction)
    assert net.demand().shape == (24, 1)
    assert net.demand().data.tolist() == [-100, 100]
    net.shared.get_node_label(net.demand().indices) == [s1, t1]
    
    s2, t2 = "7", "4"
    net.set_demand((s2, t2, 66), mode="to_affine")
    assert isinstance(net.demand, AffineDemandFunction)
    assert net.demand(0).data.tolist() == [66, -66]
    assert net.demand(1).data.tolist() == [-100, 100, 66, -66]
    
    net.set_demand(mode="to_linear")
    assert isinstance(net.demand, LinearDemandFunction)
    assert net.demand().shape == (24, 1)
    assert net.demand().data.tolist() == [-100, 100]
    net.shared.get_node_label(net.demand().indices) == [s1, t1]
    
    net.set_demand(((s2, t2, 66), (s1, t1, 100)), mode="affine")
    assert isinstance(net.demand, AffineDemandFunction)
    assert net.demand(0).data.tolist() == [66, -66]
    assert net.demand(1).data.tolist() == [-100, 100, 66, -66]
    
    assert_raises(ValueError, net.set_demand, mode="t_linar")
    assert_raises(TypeError, net.set_demand, (12, 3, 77))
    assert_raises(TypeError, net.set_demand, ("12", 3, 77))
