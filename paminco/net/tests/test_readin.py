import tempfile

import pytest
import numpy as np

from paminco.net import load_sioux
from paminco.net.network import Network
from paminco.utils.readin import (parse_polynomial,
                                 xml_find_root)
from paminco.net.cost import (
    NetworkCost,
    SymbolicCost,
    PolynomialCost,
    PiecewiseQuadraticCost,
)
from paminco.utils.typing import is_int
from paminco.utils.testing import assert_raises

from paminco.net._data_gas import temporary_gas_files
from paminco.net._data_examples import NET_ELECTRICAL_BRAESS


@pytest.fixture
def simple_electrical_network():
    return Network.from_xml(NET_ELECTRICAL_BRAESS)
@pytest.fixture
def net_sioux():
    return load_sioux()


@pytest.mark.parametrize("polystr, target", 
    [
        ("3 + 8x + 16x^2 - 99.2x^3 + e^2", [3, 8, 16, -99.2]),
        ("3 + 8x + 16x^2 - 99.2x^3 + e^2x", [3, 10, 16, -99.2]),
        ("36x - 12 + 22 - 34x^3", [10, 36, 0, -34]),
    ])
def test_parse_polynomial(polystr, target):
    assert parse_polynomial(polystr) == target


@pytest.mark.parametrize("filename, target_type", 
    [
        (NET_ELECTRICAL_BRAESS, PiecewiseQuadraticCost)
    ])
def test_autodetect_cost(filename, target_type):
    cdata = NetworkCost.from_xml(filename)
    assert isinstance(cdata, target_type)


class TestSioux:
    def test_readin_nodes(self, net_sioux):
        nod = net_sioux.nodes
        assert len(nod) == 24
        assert np.array_equal(nod.labels, (np.arange(24) + 1).astype(str))
        assert np.array_equal(nod.indices, np.arange(24))
        m = net_sioux.shared.get_node_id
        assert np.isclose(nod.x[m("1")], -96.77041974)
        assert np.isclose(nod.x[m("4")], -96.74716843)
        assert np.isclose(nod.y[m("6")], 43.58758553)
        assert np.isclose(nod.y[m("24")], 43.50316422)
        assert nod.zone.tolist() == [False]*24
        assert nod.has_zones == False
    
    def test_readin_edges(self, net_sioux):
        edge = net_sioux.edges
        for a in ["lb", "ub", "s", "t", "source_lbl", "target_lbl"]:
            e = getattr(edge, a)
            assert isinstance(e, np.ndarray)
            assert e.shape == (76, )
        assert is_int(edge.s[0])
        assert is_int(edge.t[0])
        assert isinstance(edge.source_lbl[0], str)
        assert isinstance(edge.target_lbl[0], str)
    
    @pytest.mark.parametrize("single_pairs", [False, True])
    def test_readin_demand(self, single_pairs):
        net_sioux = load_sioux(kw_demand={"single_pairs": single_pairs})
        pos = (net_sioux.demand(1) > 0)
        assert net_sioux.demand(1)[pos].sum() == 360600.0
        d = net_sioux.demand()[:, 0].toarray().ravel()
        assert np.where(d == -100)[0] == net_sioux.shared.get_node_id("1")
        assert np.where(d == 100)[0] == net_sioux.shared.get_node_id("2")
        d = net_sioux.demand()[:, 25].toarray().ravel()
        assert np.where(d == -200)[0] == net_sioux.shared.get_node_id("2")
        assert np.where(d == 200)[0] == net_sioux.shared.get_node_id("4")


class TestSimpleElectrical:
    @pytest.mark.parametrize("flow, target", 
        [
            (np.array([1, 1, 1, 1, 1]), np.array([2, 5, 1, 5, 2])),
            (np.array([2, 2, 2, 2, 2]), np.array([4, 6, 2, 6, 4])), 
            (np.array([-1, -1, -1, -1, -1]), np.array([-2, -5, -9/2, -5, -2])),
        ])
    def test_cost_values(self, simple_electrical_network, flow, target):
        net = simple_electrical_network
        assert np.array_equal(net.cost.ddx(flow), target)
        
    @pytest.mark.parametrize("coeff, target", 
        [
            ("a", [1, 2.5, .5, 9/4, 1/2, 2.5, .5, 1]),
            ("b", [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0]),
            ("tau", [-np.inf, -np.inf, 1, -np.inf, 0, -np.inf, 1, -np.inf]),
            ("lap_weight", [1/2, 1/5, 1, 2/9, 1, 1/5, 1, 1/2]),
        ])
    def test_cost_coefficients(self, simple_electrical_network, coeff, target):
        assert np.array_equal(getattr(simple_electrical_network.cost.coefficients, coeff), target)
    

class TestErrorHandling:
    def test_polycost_not_in_file(self):
        net = load_sioux()
        f = tempfile.mkstemp(".xml")[1]
        net.to_xml(f, prettify=True)
        root = xml_find_root(f)
        
        # remove polycost from some edge
        edges = root.find("edges")
        del_idx = 5
        ecost = edges[del_idx].find("cost")
        polycost = ecost.find("polynomial")
        ecost.remove(polycost)

        pc = PolynomialCost.from_xml(root, default_edge_cost=True)
        assert pc.coefficients[del_idx].sum() == 0

        assert_raises(ValueError, lambda r: PolynomialCost.from_xml(r, default_edge_cost=False), root)


class TestXML:
    def test_symcost(self):
        data = \
        """<?xml version="1.0" ?>
        <network>
            <metadata>
                <costfuncs>
                    <F>2*b*x + a</F>
                </costfuncs>
            </metadata>
        <edges>
            <edge from="1" to="2">
            <cost>
                <symbolic>
                    <a>12</a>
                    <b>0.1</b>
                    <c>0.5</c>
                </symbolic>
            </cost>
            </edge>
            <edge from="1" to="3">
            <cost>
                <symbolic>
                    <a>17</a>
                    <c>0.3</c>
                </symbolic>
            </cost>
            </edge>
            </edges>
        </network>
        """
        sc = SymbolicCost.from_xml(data)
        assert (sc.coeffs["a"] == [12., 17.]).all()
        assert (sc.coeffs["b"] == [0.1, 0.1]).all()
        assert (sc.coeffs["c"] == [0.5, 0.3]).all()
        
        data2 = \
        """<?xml version="1.0" ?>
        <network>
        <edges>
            <edge from="1" to="2">
            </edge>
            <edge from="1" to="3">
            </edge>
            </edges>
        </network>
        """
        root = xml_find_root(data2)
        names = []
        vals = []
        for i, e in enumerate(root.findall("edges/edge")):
            sc.add_to_etree(e, i)
            for s in e.find("cost/symbolic"):
                names.append(s.tag)
                vals.append(s.text)
        assert names == ['a', 'b', 'c', 'a', 'b', 'c']
        assert vals == ['12.0', '0.1', '0.5', '17.0', '0.1', '0.3']
        
        root = xml_find_root(data)
        names = []
        vals = []
        for i, e in enumerate(root.findall("edges/edge")):
            sc.add_to_etree(e, i, overwrite=False)
            for s in e.find("cost/symbolic"):
                names.append(s.tag)
                vals.append(s.text)
        assert names == ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'c', 'a', 'b', 'c']
        assert vals == ['12', '0.1', '0.5', '12.0', '0.1', '0.5', '17', '0.3', '17.0', '0.1', '0.3']
        
        root = xml_find_root(data)
        names = []
        vals = []
        for i, e in enumerate(root.findall("edges/edge")):
            sc.add_to_etree(e, i, overwrite=True)
            for s in e.find("cost/symbolic"):
                names.append(s.tag)
                vals.append(s.text)
        assert names == ['a', 'b', 'c', 'a', 'b', 'c']
        assert vals == ['12.0', '0.1', '0.5', '17.0', '0.1', '0.3']


@pytest.mark.parametrize(
    "gasnet,n,m,node_labels",
    [
        ("gas11", 8, 8, ['N02', 'N04', 'entry01', 'entry02', 'entry03', 'exit01', 'exit02', 'exit03']),
        ("gas24", 18, 19, ['entry03', 'entry01', 'exit01', 'exit02', 'exit03', 'exit04', 'exit05',
                           'N101', 'N04', 'N05a', 'N05b', 'N05c', 'N05d', 'N06', 'N08', 'N10', 'N11', 'N13'])
    ]
    )
def test_gas_readin(gasnet, n, m, node_labels):
    with temporary_gas_files(gasnet) as filenames:
        net_file, scn_file = filenames
        net = Network.from_gaslib(net_file, scn_file)
    
    assert net.n == n
    assert net.m == m
    for nl in net.nodes.labels:
        assert nl in node_labels
    
