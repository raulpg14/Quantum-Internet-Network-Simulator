import pytest
import numpy as np
from qcn.engine.network import Network
from qcn.engine.simulation import run_simulation
from qcn.engine.config import (
    NETWORK_TYPE_OFBQI, NETWORK_TYPE_SBQI,
    SIM_MODE_DISTRIBUTION, SIM_MODE_EVOLUTION,
    DEFAULT_RADIUS_DISTRIBUTION,
)


class TestNetwork:
    def test_add_nodes_count(self):
        net = Network()
        net.add_nodes(100, 500, seed=0)
        assert len(net.nodes) == 100

    def test_add_nodes_reproducible(self):
        net1 = Network()
        net1.add_nodes(100, 500, seed=42)
        net2 = Network()
        net2.add_nodes(100, 500, seed=42)
        np.testing.assert_array_equal(net1.pos_matrix, net2.pos_matrix)

    def test_add_nodes_different_seeds(self):
        net1 = Network()
        net1.add_nodes(100, 500, seed=1)
        net2 = Network()
        net2.add_nodes(100, 500, seed=2)
        assert not np.array_equal(net1.pos_matrix, net2.pos_matrix)

    def test_connect_ofbqi_produces_edges(self):
        net = Network()
        net.add_nodes(100, 1261, seed=0)
        net.connect_nodes(NETWORK_TYPE_OFBQI)
        assert len(net.edges) > 0

    def test_connect_sbqi_produces_edges(self):
        net = Network()
        net.add_nodes(100, 1261, seed=0)
        net.connect_nodes(NETWORK_TYPE_SBQI)
        assert len(net.edges) > 0

    def test_to_networkx_node_count(self):
        net = Network()
        net.add_nodes(50, 500, seed=7)
        net.connect_nodes(NETWORK_TYPE_OFBQI)
        G, pos = net.to_networkx()
        assert G.number_of_nodes() == 50
        assert len(pos) == 50


class TestSimulation:
    def test_distribution_ofbqi(self):
        res = run_simulation({
            "nodes": 100,
            "radius": DEFAULT_RADIUS_DISTRIBUTION,
            "type": NETWORK_TYPE_OFBQI,
            "mc_iter": 2,
            "sim_mode": SIM_MODE_DISTRIBUTION,
            "seed": 0,
        })
        assert res.get("success") is True
        assert res["mode"] == "distribution"
        assert len(res["dist_x"]) > 0
        assert len(res["dist_y"]) > 0

    def test_distribution_sbqi(self):
        res = run_simulation({
            "nodes": 100,
            "radius": DEFAULT_RADIUS_DISTRIBUTION,
            "type": NETWORK_TYPE_SBQI,
            "mc_iter": 2,
            "sim_mode": SIM_MODE_DISTRIBUTION,
            "seed": 0,
        })
        assert res.get("success") is True

    def test_evolution_returns_correct_keys(self):
        res = run_simulation({
            "nodes": 100,
            "radius": 2,
            "type": NETWORK_TYPE_OFBQI,
            "mc_iter": 1,
            "nets_per_mc": 3,
            "rad_incr": 50,
            "sim_mode": SIM_MODE_EVOLUTION,
            "seed": 0,
        })
        assert res.get("success") is True
        assert "x_nodes" in res
        assert "y_path" in res
        assert "y_diameter" in res

    def test_seed_reproducibility_in_simulation(self):
        params = {
            "nodes": 80,
            "radius": DEFAULT_RADIUS_DISTRIBUTION,
            "type": NETWORK_TYPE_OFBQI,
            "mc_iter": 1,
            "sim_mode": SIM_MODE_DISTRIBUTION,
            "seed": 99,
        }
        res1 = run_simulation(params)
        res2 = run_simulation(params)
        np.testing.assert_array_equal(res1["dist_x"], res2["dist_x"])
        np.testing.assert_array_equal(res1["dist_y"], res2["dist_y"])