import tempfile

import pytest
import numpy as np

from paminco.net.shared import Shared, ID_UNMAPPED, LBL_UNMAPPED
from paminco.net.demand import (Commodity, CommoditySingleSourceSink, CommodityMultiSourceSink,
                                   DemandVector, DemandVectorSP,
                                   LinearDemandFunction, AffineDemandFunction)
from paminco.utils.testing import assert_raises


class TestCommodity:
    def test_from_data(self):
        c = Commodity.from_data(("1", "2", 3))
        assert isinstance(c, CommoditySingleSourceSink)
        c2 = Commodity.from_data({"1": -10, "2": 5, "3": 5})
        assert isinstance(c2, CommodityMultiSourceSink)
        assert_raises(TypeError, c.__eq__, c2)
        assert_raises(TypeError, c2.__eq__, c)
        assert_raises(TypeError, Commodity.from_data, ["1", "2", 3])
        c3 = Commodity.from_data(("1", "4", 3))
        assert (c == c3) is False


class TestCommoditySingleSourceSink:
    def test_init(self):
        assert_raises(AssertionError, CommoditySingleSourceSink.from_tuple, (1, 2, 3, 4))
        assert_raises(TypeError, CommoditySingleSourceSink, 1, 2, 3, is_label=True)
        assert_raises(TypeError, CommoditySingleSourceSink, "1", 2, 3, is_label=True)
        assert_raises(TypeError, CommoditySingleSourceSink, 1, "2", 3, is_label=False)
        assert_raises(TypeError, CommoditySingleSourceSink, "1", "2", 3, is_label=False)
        
    def test_setup_label(self):
        c = CommoditySingleSourceSink("2", "20", 12, is_label=True)

        assert c.source_lbl == "2"
        assert c.source_id == ID_UNMAPPED
        assert c.sink_lbl == "20"
        assert c.sink_id == ID_UNMAPPED
        assert c.rate == 12
        assert c.total_rate == 12
        assert c.has_valid_indices == False
        assert len(c) == 2
        assert c.node_in_comm("2") == True
        assert c.node_in_comm(2, is_label=False) == False
        assert c.node_in_comm("20") == True
        assert c.node_in_comm("4") == False

        c2 = c.get_scaled_copy(3)
        assert c2.source_lbl == "2"
        assert c2.source_id == ID_UNMAPPED
        assert c2.sink_lbl == "20"
        assert c2.sink_id == ID_UNMAPPED
        assert c2.rate == 36
        assert c2.total_rate == 36
        assert c2.has_valid_indices == False

        mapper = {"2": 0, "20": 6}
        c.map_node_labels(mapper)
        assert c.source_lbl == "2"
        assert c.source_id == 0
        assert c.sink_lbl == "20"
        assert c.sink_id == 6
        assert c.node_in_comm(2, is_label=False) is False
        assert c.node_in_comm(6, is_label=False) is True
        assert c.to_tuple(as_label=True) == ("2", "20", 12)
        assert c.to_tuple(as_label=False) == (0, 6, 12)
        
    def test_setup_id(self):
        c = CommoditySingleSourceSink(0, 6, 12, is_label=False)

        assert c.source_lbl == LBL_UNMAPPED
        assert c.source_id == 0
        assert c.sink_lbl == LBL_UNMAPPED
        assert c.sink_id == 6
        assert c.rate == 12
        assert c.total_rate == 12
        assert c.has_valid_indices == True
        assert len(c) == 2
        assert c.node_in_comm("2") == False
        assert c.node_in_comm(2, is_label=False) == False
        assert c.node_in_comm(6, is_label=False) == True
        assert c.node_in_comm(1, is_label=False) == False

        c2 = c.get_scaled_copy(3)
        assert c2.source_lbl == LBL_UNMAPPED
        assert c2.source_id == 0
        assert c2.sink_lbl == LBL_UNMAPPED
        assert c2.sink_id == 6
        assert c2.rate == 36
        assert c2.total_rate == 36
        assert c2.has_valid_indices == True

        mapper = {0: "2", 6: "20"}
        c.map_node_ids(mapper)
        assert c.source_lbl == "2"
        assert c.source_id == 0
        assert c.sink_lbl == "20"
        assert c.sink_id == 6
        assert c.node_in_comm("1", is_label=True) == False
        assert c.node_in_comm("20", is_label=True) == True
        assert c.to_tuple(as_label=True) == ("2", "20", 12)
        assert c.to_tuple(as_label=False) == (0, 6, 12)


class TestCommodityMultiSourceSink:
    def test_setup_label(self):
        data = {"1": -100, "2": -50, "5": 75, "6": 50, "8": 25}
        c = CommodityMultiSourceSink(data)

        assert c.is_single is False
        assert c.has_valid_indices is False
        assert np.array_equal(c.source_lbl, 
                            np.array(['1', '2']))
        assert np.array_equal(c.source_id, 
                            np.array([ID_UNMAPPED, ID_UNMAPPED]))
        assert np.array_equal(c.sink_lbl, 
                            np.array(['5', '6', '8']))
        assert np.array_equal(c.sink_id, 
                            np.array([ID_UNMAPPED]*3))
        assert np.array_equal(c.rate, 
                            np.array([-100.,  -50.,   75.,   50.,   25.]))
        assert np.array_equal(c.node_ids, 
                            np.array([ID_UNMAPPED]*5))
        assert np.array_equal(c.is_sink, 
                            np.array([False, False, True, True, True]))
        assert c.total_rate == 150
        assert c.node_in_comm(["1", "4444"]) == True
        assert c.node_in_comm("0") == False
        assert c.node_in_comm("1") == True
        assert sum([cs[2] for cs in c.decompose()]) == 150

        c2 = c.get_scaled_copy(3)
        assert c2.total_rate == 450
        assert sum([cs[2] for cs in c2.decompose()]) == 450
        assert np.array_equal(c2.rate, 
                            np.array([-100.,  -50.,   75.,   50.,   25.])*3)

        mapper = {"1": 12, "2": 8, "5": 1, "6": 2, "8": 4}
        c.map_node_labels(mapper)
        assert c.has_valid_indices is True
        assert np.array_equal(c.node_ids, 
                            np.array(list(mapper.values())))
        assert np.array_equal(c.source_id, 
                            np.array([12, 8]))
        assert np.array_equal(c.sink_id, 
                            np.array([1, 2, 4]))
        assert c.node_in_comm(44) == False
        assert c.node_in_comm([12, 8, 777]) == True

        for (k, v) in data.items():
            assert c.to_dict()[k] == v

    def test_setup_id(self):
        data = {12: -100, 8: -50, 1: 75, 2: 50, 4: 25}
        c = CommodityMultiSourceSink(data, is_label=False)
        assert c.total_rate == 150
        assert np.array_equal(c.source_id, 
                            np.array([12, 8]))
        assert np.array_equal(c.sink_id, 
                            np.array([1, 2, 4]))
        assert np.array_equal(c.source_lbl, 
                            np.array([LBL_UNMAPPED]*2))
        assert np.array_equal(c.sink_lbl, 
                            np.array([LBL_UNMAPPED]*3))
        assert np.array_equal(c.rate, 
                            np.array([-100.,  -50.,   75.,   50.,   25.]))
        assert np.array_equal(c.is_sink, 
                            np.array([False, False, True, True, True]))

        mapper = {"1": 12, "2": 8, "5": 1, "6": 2, "8": 4}
        mapper = {v:k for (k,v) in mapper.items()}
        c.map_node_ids(mapper)
        assert np.array_equal(c.source_lbl, 
                            np.array(['1', '2']))
        assert np.array_equal(c.sink_lbl, 
                            np.array(['5', '6', '8']))


def make_single_com_data(
        lbls, 
        total_rate: float = 100, 
        rng=None):
    rng = np.random.default_rng(rng)
    nodes = rng.choice(lbls, 2, replace=False)
    return (*nodes, total_rate)


def make_multi_com_data(
        lbls, 
        n_sources: int = 3, 
        n_sinks: int = 3, 
        total_rate = 100, 
        rng=None):
    rng = np.random.default_rng(rng)
    source_sink = rng.choice(lbls, n_sources + n_sinks, replace=False)
    rate = np.hstack((-np.repeat(total_rate/n_sources, n_sources), np.repeat(total_rate/n_sinks, n_sinks)))
    return dict(list(zip(source_sink, rate)))


def make_rnd_labels(N: int = 100, rng=None):
    rng = np.random.default_rng(rng)
    lbls = rng.choice(np.arange(5*N), size=N, replace=False).astype(str)
    return lbls


def make_shared(labels, rng=None):
    rng = np.random.default_rng(rng)
    N = len(labels)
    zone = rng.integers(0, 2, N).astype(bool)

    # make shared and node objects
    source_target = np.array(labels).reshape(-1, 2)  # irrelevant how many edges to test demand
    
    return Shared((source_target, (0, np.inf)), (labels, None, zone))


def make_demand_data(lbls, N_single, N_multi, rng=None, return_rate: bool = False):
    rng = np.random.default_rng(rng)
    idx = np.arange(N_single + N_multi)
    idx_single = rng.choice(idx, N_single, replace=False)
    idx_multi = np.setdiff1d(idx, idx_single)
    idx_all = np.hstack((idx_single, idx_multi))
    rate = rng.random(N_single + N_multi) * 100
    
    def rndi(high = 100000):
        return rng.integers(1, high, 1)
    max_ss = int(min(10, len(lbls)/5))
    
    data_mixed = []
    for s in idx_single:
        ds = make_single_com_data(lbls, total_rate=rate[s], rng=rndi())
        data_mixed.append(ds)
    for m in idx_multi:
        dm = make_multi_com_data(lbls, n_sources=rndi(max_ss), n_sinks=rndi(max_ss), total_rate=rate[m], rng=rndi())
        data_mixed.append(dm)
        
    data_mixed = [c for idx, c in sorted(zip(idx_all, data_mixed))]
    
    if return_rate is True:
        return (data_mixed, rate)
    return data_mixed


LBLS = make_rnd_labels(N=100, rng=42)
SHARED = make_shared(LBLS, rng=42)


class TestDemandVector:
    @pytest.mark.parametrize("rng,N_s,N_m", 
        [
            (0, 1, 2), 
            (1, 7, 2), 
            (2, 2, 4),
        ])
    def test_setup(self, rng, N_s, N_m):
        data_mixed, total_rate = make_demand_data(LBLS, N_s, N_m, rng, return_rate=True)
        dv = DemandVector(data_mixed, shared=SHARED, copy=True)
        assert id(dv.shared) == id(SHARED)
        assert np.isclose(total_rate, dv.total_rate).all() == True
        assert len(dv) == len(data_mixed)
        assert np.mean(dv._comm[0].source_id) == ID_UNMAPPED
        
    def test_property(self):
        s = make_shared(["racoon", "hummingbird", "elefant", "lion", 
                         "giraffe", "racoon", "bird", "godzilla"])

        demand_data_lbl = [("racoon", "hummingbird", 14),
                           {"bird": -100, "elefant": -50, "lion": 10, 
                            "giraffe": 90, "racoon": 50},
                           ("hummingbird", "godzilla", 8000),
                           ("elefant", "giraffe", 2)
                          ]

        dv = DemandVector(demand_data_lbl, shared=s, is_label=True)

        assert [len(c) for c in dv.commodities] == [2, 5, 2, 2]

        assert dv.source_lbl[0] == "racoon"
        assert dv.source_lbl[1].tolist() == ["bird", "elefant"]
        assert dv.source_lbl[2] == "hummingbird"
        assert dv.source_lbl[3] == "elefant"

        assert dv.sink_lbl[0] == "hummingbird"
        assert dv.sink_lbl[1].tolist() == ["lion", "giraffe", "racoon"]
        assert dv.sink_lbl[2] == "godzilla"
        assert dv.sink_lbl[3] == "giraffe"

        assert dv.rate[0] == 14
        assert dv.rate[1].tolist() == [-100.,  -50.,   10.,   90.,   50.]
        assert dv.rate[2] == 8000
        assert dv.rate[3] == 2

        assert dv.total_rate.tolist() == [14, 150, 8000, 2]

        assert dv.source_id[0] == -9999
        assert dv.source_id[1].tolist() == [-9999]*2

        dv.map_node_label_to_id()
        assert dv.sink_id[0] == s.get_node_id("hummingbird")
        dv.reset_cache()
        assert dv.source_id[0] == s.get_node_id("racoon")
        assert dv.source_id[1].tolist() == [s.get_node_id("bird"), s.get_node_id("elefant")]
    
    @pytest.mark.parametrize("rng,N_s,N_m", 
        [
            (440, 1, 2), 
            (12311, 7, 2), 
            (1232, 2, 4),
        ])
    def test_sparse(self, rng, N_s, N_m):
        data_mixed = make_demand_data(LBLS, N_s, N_m, rng)
        dv = DemandVector(data_mixed, shared=SHARED, copy=True)
        dv.map_node_label_to_id()
        s = dv.sparse().toarray()
        assert s.shape == (SHARED.n, N_s + N_m)
        assert np.isclose(s.sum(), 0)
        for i in range(N_s + N_m):
            assert len(s[:, i].nonzero()[0]) == len(dv._comm[i])
            
        dv2 = DemandVector(dv.sparse(), shared=SHARED)
        assert (dv2.sparse() != dv.sparse()).nnz == 0
            
    @pytest.mark.parametrize("rng,N_s,N_m", 
        [
            (30, 1, 2), 
            (12311, 7, 2), 
            (8, 2, 4),
        ])
    def test_save_load(selfself, rng, N_s, N_m):
        data_mixed = make_demand_data(LBLS, N_s, N_m, rng)
        dv = DemandVector(data_mixed, shared=SHARED, copy=True)
        f = tempfile.mkstemp(suffix=".npz")[1]
        dv.save_to_numpy(f)
        dv2 = DemandVector.from_npz(f, shared=SHARED)
        assert dv == dv2
        
    @pytest.mark.parametrize("rng,N_s,N_m", 
        [
            (21, 1, 2), 
            (19, 7, 2), 
            (42, 2, 4),
        ])
    def test_decompose(self, rng, N_s, N_m):
        data_mixed = make_demand_data(LBLS, N_s, N_m, rng)
        dv = DemandVector(data_mixed, shared=SHARED, copy=True)
        dv2 = dv.to_single_pairs()
        assert np.isclose(dv.total_rate.sum(), dv2.total_rate.sum())
        assert len(dv2) > len(dv)
        
        
class TestDemandVectorSP:
    @pytest.mark.parametrize("rng,N_s", 
        [
            (10, 4), 
            (13, 8), 
            (27, 22),
        ])
    def test_setup(self, rng, N_s):
        # TODO-PJ -> was will ich mit der next line testen???
        # assert_raises(TypeError, DemandVectorSP, make_demand_data(LBLS, 10, 4, rng))
        
        data_mixed, total_rate = make_demand_data(LBLS, N_s, 0, rng, return_rate=True)
        dv = DemandVectorSP(data_mixed, shared=SHARED, copy=True)
        assert id(dv.shared) == id(SHARED)
        assert np.isclose(total_rate, dv.total_rate).all() == True
        assert len(dv) == len(data_mixed)
        assert np.mean(dv.source_id[0]) == ID_UNMAPPED
        
        #TODO-12131: DemandVectorSP init by sparse 
        # dv.map_node_label_to_id()
        # dv2 = DemandVectorSP(SHARED, dv.sparse())
        # assert (dv2.sparse() != dv.sparse()).nnz == 0
        
    def test_delete_node(self):
        s = make_shared(["racoon", "hummingbird", "elefant", "lion", 
                         "giraffe", "racoon", "bird", "godzilla"])

        demand_data_lbl = [("racoon", "hummingbird", 14),
                           ("hummingbird", "godzilla", 8000),
                           ("elefant", "giraffe", 2)
                           ]

        dv = DemandVectorSP(demand_data_lbl, shared=s, is_label=True)
        dv.map_node_label_to_id()
        dv.delete_nodes(7)
        dv2 = DemandVectorSP([demand_data_lbl[0], demand_data_lbl[2]], shared=s, is_label=True)
        dv2.map_node_label_to_id()
        assert dv == dv2

        dv = DemandVectorSP(demand_data_lbl, shared=s, is_label=True)
        dv.map_node_label_to_id()
        dv.delete_nodes(1)
        dv2 = DemandVectorSP([demand_data_lbl[2]], shared=s, is_label=True)
        dv2.map_node_label_to_id()
        assert dv == dv2
    
    def test_property(self):
        s = make_shared(["racoon", "hummingbird", "elefant", "lion", 
                         "giraffe", "racoon", "bird", "godzilla"])

        demand_data_lbl = [("racoon", "hummingbird", 14),
                           ("hummingbird", "godzilla", 8000),
                           ("elefant", "giraffe", 2)
                          ]

        dv = DemandVectorSP(demand_data_lbl, shared=s, is_label=True)

        assert dv.source_lbl[0] == "racoon"
        assert dv.source_lbl[1] == "hummingbird"
        assert dv.source_lbl[2] == "elefant"

        assert dv.sink_lbl[0] == "hummingbird"
        assert dv.sink_lbl[1] == "godzilla"
        assert dv.sink_lbl[2] == "giraffe"

        assert dv.rate[0] == 14
        assert dv.rate[1] == 8000
        assert dv.rate[2] == 2

        assert dv.total_rate.tolist() == [14, 8000, 2]

        assert dv.source_id[0] == -9999
        assert dv.source_id[1] == -9999
        assert dv.source_id[2] == -9999
        
        dv.map_node_label_to_id()
        assert dv.sink_id[0] == s.get_node_id("hummingbird")
        assert dv.source_id[2] == s.get_node_id("elefant")
    
    @pytest.mark.parametrize("rng,N_s", 
        [
            (1012, 4), 
            (1113, 8), 
            (272222, 22),
        ])
    def test_save_load(self, rng, N_s):
        data_mixed = make_demand_data(LBLS, N_s, 0, rng)
        dv = DemandVectorSP(data_mixed, shared=SHARED, copy=True)
        f = tempfile.mkstemp(suffix=".npz")[1]
        dv.save_to_numpy(f)
        dv2 = DemandVectorSP.from_npz(f, shared=SHARED)
        assert dv == dv2
        
class TestCompareDemandVector:
    @pytest.mark.parametrize("rng,N_s", 
        [
            (10, 4), 
            (13, 8), 
            (27, 22),
        ])
    def test_sparse(self, rng, N_s):
        data_mixed = make_demand_data(LBLS, N_s, 0, rng)
        dv = DemandVector(data_mixed, shared=SHARED, copy=True)
        dv.map_node_label_to_id()
        dvsp = DemandVectorSP(data_mixed, shared=SHARED, copy=True)
        dvsp.map_node_label_to_id()
        assert (dvsp.sparse() != dv.sparse()).nnz == 0
        

class TestDemandFunctionLinear:
    def test_simple(self):
        lbls = ["a", "b", "c", "d"]
        demand_data_id = ([(0, 1, 1), (2, 3, 3)])
        demand_data_lbl = ([("a", "b", 1), ("c", "d", 3)])
        s = make_shared(lbls)
        
        ldf = LinearDemandFunction(demand_data_lbl, shared=s, is_label=True)
        ldf.map_node_label_to_id()
        ldf1 = LinearDemandFunction(demand_data_id, shared=s, is_label=False)
        ldf1.map_node_id_to_label()
        ldf2 = LinearDemandFunction(demand_data_lbl, shared=s, is_label=True, single_pairs=True)
        ldf2.map_node_label_to_id()
        ldf3 = LinearDemandFunction(demand_data_id, shared=s, is_label=False, single_pairs=True)
        ldf3.map_node_id_to_label()

        for df in [ldf, ldf1, ldf2, ldf3]:
            # check for param = 0
            assert np.array_equal(df.value(0).toarray()[:, 0],
                                  np.array([0, 0, 0, 0]))
            assert np.array_equal(df.value(0).toarray()[:, 1],
                                  np.array([0, 0, 0, 0]))

            # check for param = 10
            assert np.array_equal(df.value(10).toarray()[:, 0],
                                  np.array([-10, 10, 0, 0]))
            assert np.array_equal(df.value(10).toarray()[:, 1],
                                  np.array([0, 0, -30, 30]))
            
            # check for param = 30
            assert np.array_equal(df.value(30).toarray()[:, 0],
                                  np.array([-30, 30, 0, 0]))
            assert np.array_equal(df.value(30).toarray()[:, 1],                                  
                                  np.array([0, 0, -90, 90]))
            
            # check __call__
            assert np.array_equal(df(30).toarray()[:, 0], df.value(30).toarray()[:, 0])
            
        ldf = LinearDemandFunction(demand_data_lbl*2, shared=s, is_label=True)
        ldf.map_node_label_to_id()
        ldf.delete_nodes(s.get_node_id("a"))
        assert np.array_equal(ldf.value(10).toarray()[:, 0],
                              np.array([0, 0, -30, 30]))
        assert np.array_equal(ldf.value(10).toarray()[:, 1],
                              np.array([0, 0, -30, 30]))
        
    def test_save_load(self ):
        lbls = ["a", "b", "c", "d"]
        demand_data_id = ([(0, 1, 1), (2, 3, 3)])
        demand_data_lbl = ([("a", "b", 1), ("c", "d", 3)])
        s = make_shared(lbls)
        
        ldf = LinearDemandFunction(demand_data_lbl, shared=s, is_label=True)
        ldf.map_node_label_to_id()
        ldf1 = LinearDemandFunction(demand_data_id, shared=s, is_label=False)
        ldf1.map_node_id_to_label()
        ldf2 = LinearDemandFunction(demand_data_lbl, shared=s, is_label=True, single_pairs=True)
        ldf2.map_node_label_to_id()
        ldf3 = LinearDemandFunction(demand_data_id, shared=s, is_label=False, single_pairs=True)
        ldf3.map_node_id_to_label()

        for df in [ldf, ldf1, ldf2, ldf3]:
            f = tempfile.mkstemp(suffix=".npz")[1]
            df.save_to_numpy(f)
            df2 = LinearDemandFunction.from_npz(f, shared=s)
            assert df == df2


class TestDemandFunctionAffine:
    def test_simple(self):
        lbls = ["a", "b", "c", "d", "e", "f", "g", "h"]
        demand_data_id = ([(0, 1, 1), (2, 3, 3)], 
                          [(4, 5, 2), (6, 7, 3)])
        demand_data_lbl = ([("a", "b", 1), ("c", "d", 3)], 
                           [("e", "f", 2), ("g", "h", 3)])
        s = make_shared(lbls)
        
        adf = AffineDemandFunction(*demand_data_lbl, shared=s, is_label=True)
        adf.map_node_label_to_id()
        adf1 = AffineDemandFunction(*demand_data_id, shared=s, is_label=False)
        adf1.map_node_id_to_label()
        adf2 = AffineDemandFunction(*demand_data_lbl, shared=s, is_label=True, single_pairs=True)
        adf2.map_node_label_to_id()
        adf3 = AffineDemandFunction(*demand_data_id, shared=s, is_label=False, single_pairs=True)
        adf3.map_node_id_to_label()

        for df in [adf, adf1, adf2, adf3]:
            # check for param = 0
            assert np.array_equal(df.value(0).toarray()[:, 0],
                                  np.array([-1, 1, 0, 0, 0, 0, 0, 0]))
            assert np.array_equal(df.value(0).toarray()[:, 1],
                                  np.array([0, 0, -3, 3, 0, 0, 0, 0]))

            # check for param = 10
            assert np.array_equal(df.value(10).toarray()[:, 0],
                                  np.array([-1, 1, 0, 0, -20, 20, 0, 0]))
            assert np.array_equal(df.value(10).toarray()[:, 1],
                                  np.array([0, 0, -3, 3, 0, 0, -30, 30]))
            
            # check for param = 30
            assert np.array_equal(df.value(30).toarray()[:, 0],
                                  np.array([-1, 1, 0, 0, -60, 60, 0, 0]))
            assert np.array_equal(df.value(30).toarray()[:, 1],
                                  np.array([0, 0, -3, 3, 0, 0, -90, 90]))
            
            # check __call__
            assert np.array_equal(df(30).toarray()[:, 0], df.value(30).toarray()[:, 0])
            
        adf = AffineDemandFunction(*demand_data_lbl, s, is_label=True)
        adf.map_node_label_to_id()
        adf.delete_nodes([s.get_node_id("c"), s.get_node_id("g")])
        assert np.array_equal(adf.value(10).toarray()[:, 0],
                              np.array([-1, 1, 0, 0, -20, 20, 0, 0]))
        
        adf = AffineDemandFunction(*demand_data_lbl, shared=s, is_label=True)
        adf.map_node_label_to_id()
        adf.delete_nodes([s.get_node_id("a"), s.get_node_id("e")])
        assert len(adf) == 1
        
        adf = AffineDemandFunction(*demand_data_lbl, shared=s, is_label=True)
        adf.map_node_label_to_id()
        adf.delete_nodes([s.get_node_id("a")])
        assert_raises(ValueError, adf.value)

    def test_save_load(self):
        lbls = ["a", "b", "c", "d", "e", "f", "g", "h"]
        demand_data_id = ([(0, 1, 1), (2, 3, 3)], 
                          [(4, 5, 2), (6, 7, 3)])
        demand_data_lbl = ([("a", "b", 1), ("c", "d", 3)], 
                           [("e", "f", 2), ("g", "h", 3)])
        s = make_shared(lbls)
        
        adf = AffineDemandFunction(*demand_data_lbl, shared=s, is_label=True)
        adf.map_node_label_to_id()
        adf1 = AffineDemandFunction(*demand_data_id, shared=s, is_label=False)
        adf1.map_node_id_to_label()
        adf2 = AffineDemandFunction(*demand_data_lbl, shared=s, is_label=True, single_pairs=True)
        adf2.map_node_label_to_id()
        adf3 = AffineDemandFunction(*demand_data_id, shared=s, is_label=False, single_pairs=True)
        adf3.map_node_id_to_label()

        for df in [adf, adf1, adf2, adf3]:
            f = tempfile.mkstemp(suffix=".npz")[1]
            df.save_to_numpy(f)
            df2 = AffineDemandFunction.from_npz(f, shared=s)
            assert df == df2
