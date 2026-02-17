"""Comparison unit tests for Chord, Pastry, and Kademlia.

Maps Section 6.2 system-level metrics to DHT-level proxies:

    System Metric            DHT Proxy Measured Here
    ────────────────────     ─────────────────────────────
    TTFT Reduction           Lookup hop count
    Cache Hit Rate (>85%)    Routing correctness
    System Scalability       Hop count vs. N
    Parallel Fetch Speedup   (protocol-level, qualitative)
    Generation Quality       N/A at DHT layer

Run:  pytest tests/ -v -s
"""

import math
import random
import statistics
import time

import pytest

from dht_comparison.base import NetworkSimulator, generate_node_ids, generate_keys
from dht_comparison.chord import ChordNode
from dht_comparison.kademlia import KademliaNode
from dht_comparison.pastry import PastryNode

ID_BITS = 16

PROTOCOLS = [
    pytest.param("Chord", ChordNode, ChordNode.ground_truth, id="Chord"),
    pytest.param("Kademlia", KademliaNode, KademliaNode.ground_truth, id="Kademlia"),
    pytest.param("Pastry", PastryNode, PastryNode.ground_truth, id="Pastry"),
]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def build_network(node_cls, num_nodes, id_bits=ID_BITS,
                  stabilize_rounds=None):
    """Create a DHT of *num_nodes* and stabilize it."""
    if stabilize_rounds is None:
        stabilize_rounds = max(num_nodes * 3, 30)

    network = NetworkSimulator()
    node_ids = generate_node_ids(num_nodes, id_bits)
    nodes = []

    for nid in node_ids:
        node = node_cls(nid, network, id_bits)
        bootstrap = nodes[0].node_id if nodes else None
        node.join(bootstrap)
        nodes.append(node)

    for _ in range(stabilize_rounds):
        for node in nodes:
            node.stabilize()

    return network, nodes


def _ground_truth(name, gt_fn, key, node_ids):
    if name == "Pastry":
        return gt_fn(key, node_ids, 2 ** ID_BITS)
    return gt_fn(key, node_ids)


# ──────────────────────────────────────────────────────────────────────
# 1. Routing Correctness  (proxy → Cache Hit Rate target >85%)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_routing_correctness(name, NodeClass, gt_fn):
    """Every lookup must resolve to the correct responsible node."""
    _, nodes = build_network(NodeClass, 10)
    node_ids = sorted(n.node_id for n in nodes)
    keys = generate_keys(200)

    correct = 0
    for key in keys:
        result = nodes[0].lookup(key)
        assert result.success, f"{name}: lookup failed for key {key}"
        expected = _ground_truth(name, gt_fn, key, node_ids)
        if result.responsible_node == expected:
            correct += 1

    accuracy = correct / len(keys)
    print(f"\n  {name} correctness: {accuracy:.1%}  ({correct}/{len(keys)})")
    assert accuracy >= 0.95, (
        f"{name}: accuracy {accuracy:.1%} is below the 95% threshold")


@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_lookup_from_every_node(name, NodeClass, gt_fn):
    """Lookups must succeed regardless of the initiating node."""
    _, nodes = build_network(NodeClass, 10)
    keys = generate_keys(50)

    for node in nodes:
        for key in keys:
            result = node.lookup(key)
            assert result.success, (
                f"{name}: lookup from {node.node_id} failed for key {key}")


# ──────────────────────────────────────────────────────────────────────
# 2. Hop Count — O(log N)  (proxy → TTFT Reduction)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_hop_count_is_olog_n(name, NodeClass, gt_fn):
    """Mean hop count must stay within a generous O(log N) bound."""
    num_nodes = 20
    _, nodes = build_network(NodeClass, num_nodes)
    keys = generate_keys(500)

    hops = []
    for key in keys:
        node = random.Random(key).choice(nodes)
        result = node.lookup(key)
        if result.success:
            hops.append(result.hop_count)

    mean_hops = statistics.mean(hops)
    upper_bound = math.log2(num_nodes) * 2  # generous
    print(f"\n  {name}  N={num_nodes}:  mean={mean_hops:.2f}  "
          f"median={statistics.median(hops):.1f}  max={max(hops)}  "
          f"bound<={upper_bound:.1f}")
    assert mean_hops <= upper_bound, (
        f"{name}: mean hops {mean_hops:.2f} exceeds 2·log₂(N)={upper_bound:.1f}")


# ──────────────────────────────────────────────────────────────────────
# 3. Scalability  (proxy → linear throughput up to 20 nodes)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_scalability(name, NodeClass, gt_fn):
    """Hop count should grow sub-linearly (logarithmically) with N."""
    keys = generate_keys(200)
    results = {}

    for n in [5, 10, 20]:
        _, nodes = build_network(NodeClass, n)
        hops = []
        for key in keys:
            node = random.Random(key).choice(nodes)
            r = node.lookup(key)
            if r.success:
                hops.append(r.hop_count)
        sorted_h = sorted(hops)
        results[n] = {
            "mean": statistics.mean(hops),
            "median": statistics.median(hops),
            "p95": sorted_h[int(len(sorted_h) * 0.95)],
        }

    print(f"\n  {name} scalability:")
    for n, s in results.items():
        print(f"    N={n:3d}:  mean={s['mean']:.2f}  "
              f"median={s['median']:.1f}  p95={s['p95']}")

    ratio = results[20]["mean"] / max(results[5]["mean"], 0.01)
    assert ratio < 5.0, (
        f"{name}: hop ratio N=5→20 is {ratio:.1f}x (expected < 5x for O(log N))")


# ──────────────────────────────────────────────────────────────────────
# 4. Join Cost
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_join_cost(name, NodeClass, gt_fn):
    """Measure the number of messages required for a node to join."""
    _, nodes = build_network(NodeClass, 10)

    join_costs = []
    for i in range(5):
        extra_ids = generate_node_ids(1, ID_BITS, seed=1000 + i)
        new_node = NodeClass(extra_ids[0], nodes[0].network, ID_BITS)
        cost = new_node.join(nodes[0].node_id)
        join_costs.append(cost)
        nodes.append(new_node)
        for _ in range(10):
            for node in nodes:
                node.stabilize()

    print(f"\n  {name} join cost:  mean={statistics.mean(join_costs):.1f}  "
          f"range=[{min(join_costs)}, {max(join_costs)}]")


# ──────────────────────────────────────────────────────────────────────
# 5. State Overhead  (routing table size)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_state_overhead(name, NodeClass, gt_fn):
    """Routing-table size per node should be manageable."""
    _, nodes = build_network(NodeClass, 20)
    sizes = [n.routing_table_size() for n in nodes]
    print(f"\n  {name} routing-table size (N=20):  "
          f"mean={statistics.mean(sizes):.1f}  "
          f"range=[{min(sizes)}, {max(sizes)}]")


# ──────────────────────────────────────────────────────────────────────
# 6. Churn Resilience  (correctness after node failures)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_churn_resilience(name, NodeClass, gt_fn):
    """Routing must stay correct after graceful node departures."""
    _, nodes = build_network(NodeClass, 20)

    rng = random.Random(42)
    to_remove = rng.sample(nodes, 3)
    for node in to_remove:
        node.leave()
        nodes.remove(node)

    for _ in range(30):
        for node in nodes:
            node.stabilize()

    remaining_ids = sorted(n.node_id for n in nodes)
    keys = generate_keys(200)

    correct = 0
    for key in keys:
        result = random.Random(key).choice(nodes).lookup(key)
        if result.success:
            expected = _ground_truth(name, gt_fn, key, remaining_ids)
            if result.responsible_node == expected:
                correct += 1

    accuracy = correct / len(keys)
    print(f"\n  {name} post-churn correctness: {accuracy:.1%}")
    assert accuracy >= 0.85, (
        f"{name}: post-churn accuracy {accuracy:.1%} is below 85%")


# ──────────────────────────────────────────────────────────────────────
# 7. Throughput  (lookups / second)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_lookup_throughput(name, NodeClass, gt_fn):
    """Measure raw in-process lookup throughput (no simulated latency)."""
    _, nodes = build_network(NodeClass, 20)
    keys = generate_keys(1000)

    start = time.perf_counter()
    for key in keys:
        random.Random(key).choice(nodes).lookup(key)
    elapsed = time.perf_counter() - start

    throughput = len(keys) / elapsed
    print(f"\n  {name} throughput (N=20):  {throughput:,.0f} lookups/sec  "
          f"({elapsed:.3f}s for {len(keys)} lookups)")


# ──────────────────────────────────────────────────────────────────────
# 8. Store / Retrieve round-trip
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,NodeClass,gt_fn", PROTOCOLS)
def test_store_and_retrieve(name, NodeClass, gt_fn):
    """Values stored via the DHT must be retrievable."""
    _, nodes = build_network(NodeClass, 10)

    for i in range(50):
        key = generate_keys(1, ID_BITS, seed=2000 + i)[0]
        value = f"payload-{i}"
        ok = nodes[0].store(key, value)
        assert ok, f"{name}: store failed for key {key}"

        got = nodes[0].retrieve(key)
        assert got == value, (
            f"{name}: retrieve mismatch for key {key}: {got!r} != {value!r}")
