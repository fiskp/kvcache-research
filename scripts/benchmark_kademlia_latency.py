"""Kademlia-only benchmark with virtual time (simulated latency).

Builds Kademlia networks at increasing N, runs lookups, and advances
virtual time by one unit per lookup round (hop). Asserts that mean
simulated latency scales as O(log N).

Run:  python -m pytest tests/benchmark_kademlia_latency.py -v -s
"""

import math
import random
import statistics

import pytest

from dht_comparison.base import (
    NetworkSimulator,
    generate_node_ids,
    generate_keys,
)
from dht_comparison.kademlia import KademliaNode

ID_BITS = 16
RTT_PER_HOP = 1.0  # virtual time units per lookup round (1 RTT per round)
LOOKUPS_PER_N = 200
# O(log N) bound: mean_latency <= COEF * log2(N) * RTT_PER_HOP
COEF = 2.5


def build_kademlia_network(num_nodes: int, id_bits: int = ID_BITS,
                           per_hop_delay: float = RTT_PER_HOP,
                           stabilize_rounds: int | None = None):
    """Create a Kademlia network with virtual time enabled."""
    if stabilize_rounds is None:
        stabilize_rounds = max(num_nodes * 3, 30)

    network = NetworkSimulator(per_hop_delay=per_hop_delay)
    node_ids = generate_node_ids(num_nodes, id_bits)
    nodes = []

    for nid in node_ids:
        node = KademliaNode(nid, network, id_bits)
        bootstrap = nodes[0].node_id if nodes else None
        node.join(bootstrap)
        nodes.append(node)

    for _ in range(stabilize_rounds):
        for node in nodes:
            node.stabilize()

    return network, nodes


@pytest.mark.parametrize("n", [10, 20, 50, 100, 200])
def test_kademlia_simulated_latency_olog_n(n: int):
    """Mean simulated latency (virtual time) must scale as O(log N)."""
    network, nodes = build_kademlia_network(n)
    keys = generate_keys(LOOKUPS_PER_N)

    latencies = []
    for key in keys:
        initiator = random.Random(key).choice(nodes)
        t0 = network.virtual_time
        result = initiator.lookup(key)
        t1 = network.virtual_time
        if result.success:
            latencies.append(t1 - t0)

    assert len(latencies) == len(keys), "all lookups should succeed"
    mean_latency = statistics.mean(latencies)
    upper_bound = COEF * math.log2(n) * RTT_PER_HOP

    print(f"\n  N={n:3d}:  mean_latency={mean_latency:.2f}  "
          f"log2(N)={math.log2(n):.2f}  bound={upper_bound:.1f}  "
          f"median={statistics.median(latencies):.2f}  max={max(latencies):.0f}")

    assert mean_latency <= upper_bound, (
        f"N={n}: mean simulated latency {mean_latency:.2f} "
        f"exceeds {COEF}*log2(N)*RTT = {upper_bound:.1f}"
    )


def test_kademlia_latency_scales_sublinearly():
    """Simulated latency ratio (large N / small N) should be sublinear in N."""
    keys = generate_keys(LOOKUPS_PER_N)
    results = {}

    for n in [10, 20, 50, 100, 200]:
        network, nodes = build_kademlia_network(n)
        latencies = []
        for key in keys:
            initiator = random.Random(key).choice(nodes)
            t0 = network.virtual_time
            result = initiator.lookup(key)
            t1 = network.virtual_time
            if result.success:
                latencies.append(t1 - t0)
        results[n] = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
        }

    print("\n  Kademlia simulated latency vs N (virtual time units):")
    for n in [10, 20, 50, 100, 200]:
        r = results[n]
        print(f"    N={n:3d}:  mean={r['mean']:.2f}  median={r['median']:.2f}  "
              f"log2(N)={math.log2(n):.2f}")

    # From N=10 to N=200, mean latency should grow by a factor < 2.5 * log2(200)/log2(10)
    ratio = results[200]["mean"] / max(results[10]["mean"], 0.01)
    log_ratio = math.log2(200) / math.log2(10)
    assert ratio < COEF * log_ratio, (
        f"Latency ratio N=200/N=10 = {ratio:.2f}x should be O(log N) "
        f"(log2(200)/log2(10) ≈ {log_ratio:.2f})"
    )
