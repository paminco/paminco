import tempfile
import copy

import pytest
import numpy as np

from paminco.net import load_sioux
from paminco.net.network import Network
from paminco.utils.testing import assert_raises
from paminco.algo.mca import MCAInterpolationRule


@pytest.fixture
def net_sioux():
    return load_sioux()


class TestEdges:
    @pytest.mark.parametrize("del_idx",
        [
            2,
            [0, 1, 2],
            [2, 11, 24, 55]
        ])
    def test_delete_no_update(self, del_idx):
        net = load_sioux()
        net.shared.delete_edges(del_idx, update=False)
        assert net.shared.get_edge_id((1, 0)) == 2

        s, t = (1, 5)
        e_id = net.shared.get_edge_id((s, t))
        assert e_id == 3
        e = net.edges[e_id]
        assert (e["s"] != s or e["t"] != t)

    @pytest.mark.parametrize("del_idx, target_id",
            [
                (2, 2),
                ([0, 1, 2], 0),
                ([1, 2, 11, 24, 55], 1),
            ])
    def test_delete_update(self, del_idx, target_id):
        net = load_sioux()
        net.shared.delete_edges(del_idx, update=True)
        assert_raises(KeyError, net.shared.get_edge_id, (1, 0))

        s, t = (1, 5)
        e_id = net.shared.get_edge_id((s, t))
        assert e_id == target_id
        e = net.edges[e_id]
        assert (e["s"] == s and e["t"] == t)

    @pytest.mark.parametrize("del_nodes, is_label",
            [
                ([0, 1, 2, 6, 7], False),
                (["1", "2", "3", "7", "8"], True),
            ])
    def test_delete_nodes(self, del_nodes, is_label):
        net = load_sioux()
        if is_label is True:
            del_nodes_id = net.shared.get_node_id(del_nodes)
        else:
            del_nodes_id = del_nodes
        net.shared.delete_nodes_in_edges(del_nodes, is_label=is_label, update=False)
        for d in del_nodes_id:
            assert np.count_nonzero(net.edges.indices - d) == np.count_nonzero(net.edges.indices)
        assert len(net.shared.nodes2edge) != net.shared.m

        net = load_sioux()
        net.shared.delete_nodes_in_edges(del_nodes, is_label=is_label, update=True)
        assert len(net.shared.nodes2edge) == net.shared.m


class TestNodes:
    @pytest.mark.parametrize("del_nodes",
        [
            [0, 1, 2, 6, 7],
            ["1", "2", "3", "7", "8"]
        ])
    def test_delete(self, del_nodes):
        # no update of mapping
        net = load_sioux()
        net.shared.delete_nodes(del_nodes, is_label=(isinstance(del_nodes[0], str)), update=False)
        assert net.shared.get_node_id("4") == 3
        assert net.shared.get_node_id("24") == 23

        # update of mapping
        net = load_sioux()
        net.shared.delete_nodes(del_nodes, is_label=(isinstance(del_nodes[0], str)), update=True)
        assert net.shared.get_node_id("4") == 0
        assert net.shared.get_node_id("24") == 18


class TestCost:
    @pytest.mark.slow
    @pytest.mark.parametrize("rng", [42])
    def test_delete_edges_poly(self, rng):
        rng = np.random.default_rng(rng)
        net = load_sioux()
        net._remove_zones()
        del_idx = rng.choice(net.n, 20, replace=False)
        net2 = copy.deepcopy(net)

        m = net.m

        i_rule = MCAInterpolationRule(1.01, 1, m, 1000)
        net._c = net.cost.interpolate(i_rule)
        net.delete_edges(del_idx)

        net2.delete_edges(del_idx)
        m2 = net2.m
        # Beta must be adjusted since interpolation formula depends on m
        i_rule2 = MCAInterpolationRule(1.01, 1*m2/m, m2, 1000)
        net2._c = net2.cost.interpolate(i_rule2)

        assert net2.cost == net.cost


class TestCleanSioux:
    def test_del_node(self):
        net_sioux = load_sioux()
        net_sioux.shared.delete_nodes(["1", "4", "8"])
        assert net_sioux.shared.node2id["2"] == 0
        assert net_sioux.shared.node2id["5"] == 2
        assert net_sioux.shared.get_node_id(["2", "5"]) == [0, 2]

        net_sioux = load_sioux()
        net_sioux.shared.delete_nodes([0, 1, 2, 4, 5, 11, 12, 18, 22, 23], is_label=False)
        assert net_sioux.shared.get_node_id(["4", "7", "22"]) == [0, 1, 13]

    @pytest.mark.parametrize("del_nodes, expected_removed",
        [
            (["12", "24"], ["13"]),
            (["11", "15", "23"], ["14"]),
            (["21", "22", "19", "18"], ["20"]),
            (["12", "24", "21", "22", "19", "18"], ["13", "20"]),
        ])
    def test_remove_isolated_nodes(self, net_sioux, del_nodes, expected_removed):
        # isolate some nodes
        net_sioux.delete_nodes(del_nodes, is_label=True)

        net_sioux.clean(remove_zones=False,
                        remove_parallel_edges=False,
                        remove_zero_cost_edges=False,
                        remove_isolated_nodes=True,
                        remove_unreachable_nodes=False)

        intersect1d = np.intersect1d(net_sioux.nodes.labels, expected_removed)
        assert len(intersect1d) == 0, \
            "Not all isolated nodes have been removed: {}".format(intersect1d)

    # @pytest.mark.parametrize("duplicate_idx",
    #     [
    #         [12, 2, 18],
    #         [0, 2, 44],
    #         [1, 4, 37]
    #     ])
    # def test_remove_parallel_edges(self, net_sioux, duplicate_idx):
    #     # TODO-SiouxXML
    #     net_dict_sioux = Network.from_xml(
    #         "data/traffic/SiouxFalls.xml",
    #          return_dict=True
    #     )

    #     # duplicate edges for edge data
    #     (st, bounds) = net_dict_sioux["edge_data"]
    #     st = np.array(st)
    #     bounds = np.array(bounds)
    #     st = np.vstack((st, st[duplicate_idx, :]))
    #     bounds = np.vstack((bounds, bounds[duplicate_idx, :]))
    #     net_dict_sioux["edge_data"] = (st, bounds)

    #     # duplicate edges for cost data
    #     polycostdata = net_sioux.cost
    #     c, s = polycostdata.coefficients, polycostdata.signed
    #     new_coeff = np.vstack((c, c[duplicate_idx, :]))
    #     new_signed = np.hstack((s, s[duplicate_idx]))

    #     net_dict_sioux["cost_data"] = (new_coeff, new_signed)
    #     print(polycostdata)

    #     net_dirty = Network(**net_dict_sioux)
    #     net_dirty.clean(remove_zones=False,
    #                     remove_parallel_edges=True,
    #                     remove_zero_cost_edges=False,
    #                     remove_isolated_nodes=False,
    #                     remove_unreachable_nodes=False)

    #     assert np.array_equal(net_sioux._c.coefficients, net_dirty._c.coefficients), \
    #         "Cost data not equivalent."

    #     assert net_sioux.edges == net_dirty.edges, \
    #         "Edge data not equivalent."

    @pytest.mark.parametrize("start_node_unreachables,target_nodes",
        [
            ("1", ['1', '2', '4', '5', '6', '7', '8', '9',
                   '10', '16', '17', '18', '19']),
            ("13", ['12', '13', '14', '21', '22', '23', '24']),
        ])
    def test_remove_unreachaeble_nodes(
            self,
            net_sioux,
            start_node_unreachables,
            target_nodes
        ):
        # start from bigger connected components
        net_sioux.delete_nodes(["3", "11", "15", "20"], is_label=True)
        net_sioux.clean(remove_zones=False,
                        remove_parallel_edges=False,
                        remove_zero_cost_edges=False,
                        remove_isolated_nodes=False,
                        remove_unreachable_nodes=True,
                        start_node_unreachables=start_node_unreachables)

        assert net_sioux.nodes.labels.tolist() == target_nodes, \
            "Incorrect nodes have been deleted."

    def test_remove_zero_cost_edges(self, net_sioux):
        # set some edges to zero cost
        zero_cost_idx = np.random.randint(low=0, high=net_sioux.m - 1, size=10)
        net_sioux.cost.coefficients[zero_cost_idx, :] = 0

        net_sioux.clean(remove_zones=False,
                        remove_parallel_edges=False,
                        remove_zero_cost_edges=True,
                        remove_isolated_nodes=False,
                        remove_unreachable_nodes=False)

        assert len(net_sioux.shared.edges) == len(net_sioux.cost), \
            "Cost and edgelist do not have same size."

        assert np.all(abs(net_sioux.cost.coefficients).sum(axis=1) > 0) == True, \
            "Not all zero cost edges have been deleted."

    def test_remove_zones(self, net_sioux):
        # make some articical zones
        zone_idx = np.random.randint(low=0, high=net_sioux.n - 1, size=3)
        zone_labels = net_sioux.shared.get_node_label(zone_idx)
        net_sioux.nodes.zone[zone_idx] = True

        net_sioux.clean(remove_zones=True,
                        remove_parallel_edges=False,
                        remove_zero_cost_edges=False,
                        remove_isolated_nodes=False,
                        remove_unreachable_nodes=False)

        assert len(np.intersect1d(zone_labels, net_sioux.nodes.labels)) == 0, \
            "Zones not removed from nodes."

        edge_nodes = np.unique(net_sioux.shared.edges.labels)
        assert len(np.intersect1d(zone_labels, edge_nodes)) == 0, \
            "Zones not removed from edgelist."

        assert len(net_sioux.shared.edges) == len(net_sioux.cost.coefficients), \
            "Cost and edgelist do not have same size."
