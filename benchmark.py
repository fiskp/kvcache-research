"""DHT Protocol Comparison Benchmark.

Runs a head-to-head comparison of Chord, Pastry, and Kademlia and
prints a formatted report mapped to Section 6.2 metrics.

Usage
-----
    python benchmark.py
"""

import math
import random
import statistics
import time

from dht_comparison.base import NetworkSimulator, generate_node_ids, generate_keys
from dht_comparison.chord import ChordNode
from dht_comparison.kademlia import KademliaNode
from dht_comparison.pastry import PastryNode

ID_BITS = 16

PROTOCOLS = {
    "Chord":    (ChordNode,    ChordNode.ground_truth),
    "Kademlia": (KademliaNode, KademliaNode.ground_truth),
    "Pastry":   (PastryNode,   PastryNode.ground_truth),
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def build_network(node_cls, num_nodes, id_bits=ID_BITS):
    network = NetworkSimulator()
    node_ids = generate_node_ids(num_nodes, id_bits)
    nodes = []
    join_costs = []

    for nid in node_ids:
        node = node_cls(nid, network, id_bits)
        bootstrap = nodes[0].node_id if nodes else None
        cost = node.join(bootstrap)
        join_costs.append(cost)
        nodes.append(node)

    rounds = max(num_nodes * 3, 30)
    for _ in range(rounds):
        for node in nodes:
            node.stabilize()

    return network, nodes, join_costs


def _gt(name, gt_fn, key, node_ids):
    if name == "Pastry":
        return gt_fn(key, node_ids, 2 ** ID_BITS)
    return gt_fn(key, node_ids)


def benchmark_lookups(nodes, keys, gt_fn, name):
    hops_list = []
    correct = 0
    total = 0
    node_ids = sorted(n.node_id for n in nodes)

    for key in keys:
        node = random.Random(key).choice(nodes)
        result = node.lookup(key)
        if result.success:
            hops_list.append(result.hop_count)
            expected = _gt(name, gt_fn, key, node_ids)
            if result.responsible_node == expected:
                correct += 1
            total += 1

    if not hops_list:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0,
                "max": 0, "correctness": 0}

    sh = sorted(hops_list)
    n = len(sh)
    return {
        "mean":   statistics.mean(hops_list),
        "median": statistics.median(hops_list),
        "p95":    sh[int(n * 0.95)] if n > 1 else sh[0],
        "p99":    sh[int(n * 0.99)] if n > 1 else sh[0],
        "max":    max(hops_list),
        "correctness": correct / total if total else 0,
    }


def print_table(header, rows, widths=None):
    if widths is None:
        widths = [max(len(str(row[i])) for row in [header] + rows) + 2
                  for i in range(len(header))]
    print("  " + "".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("  " + "-" * sum(widths))
    for row in rows:
        print("  " + "".join(str(v).ljust(w) for v, w in zip(row, widths)))


# ──────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DHT PROTOCOL COMPARISON BENCHMARK")
    print("  Section 6.2 metric proxies for P2P-RAGCache  (Milestone 1)")
    print(sep)

    keys_200 = generate_keys(200)
    keys_1000 = generate_keys(1000)

    # ── 1. Scalability: hop count vs N ────────────────────────────────
    print("\n  1. SCALABILITY -- Lookup Hop Count vs Network Size")
    print("     (Proxy for TTFT Reduction & System Scalability)\n")

    scale = {}
    for name, (cls, gt) in PROTOCOLS.items():
        scale[name] = {}
        for n in [5, 10, 20, 50]:
            _, nodes, _ = build_network(cls, n)
            scale[name][n] = benchmark_lookups(nodes, keys_200, gt, name)

    for n in [5, 10, 20, 50]:
        print(f"  N = {n}")
        header = ["Protocol", "Mean", "Median", "P95", "P99",
                  "Max", "Correct"]
        rows = []
        for name in PROTOCOLS:
            s = scale[name][n]
            rows.append([
                name,
                f"{s['mean']:.2f}",
                f"{s['median']:.1f}",
                str(s["p95"]),
                str(s["p99"]),
                str(s["max"]),
                f"{s['correctness']:.1%}",
            ])
        print_table(header, rows, [12, 8, 8, 6, 6, 6, 10])
        print()

    # ── 2. Join cost ──────────────────────────────────────────────────
    print("  2. JOIN COST (messages per join)\n")
    header = ["Protocol", "Mean", "Min", "Max"]
    rows = []
    for name, (cls, _) in PROTOCOLS.items():
        _, _, jc = build_network(cls, 15)
        costs = jc[1:]  # first node has 0 cost
        rows.append([
            name,
            f"{statistics.mean(costs):.1f}",
            str(min(costs)),
            str(max(costs)),
        ])
    print_table(header, rows, [12, 10, 8, 8])

    # ── 3. State overhead ─────────────────────────────────────────────
    print("\n  3. STATE OVERHEAD -- routing-table entries per node (N=20)\n")
    header = ["Protocol", "Mean", "Min", "Max"]
    rows = []
    for name, (cls, _) in PROTOCOLS.items():
        _, nodes, _ = build_network(cls, 20)
        sizes = [nd.routing_table_size() for nd in nodes]
        rows.append([
            name,
            f"{statistics.mean(sizes):.1f}",
            str(min(sizes)),
            str(max(sizes)),
        ])
    print_table(header, rows, [12, 10, 8, 8])

    # ── 4. Churn resilience ───────────────────────────────────────────
    print("\n  4. CHURN RESILIENCE -- correctness after 3/20 node departures")
    print("     (Proxy for Cache Hit Rate under failures)\n")
    header = ["Protocol", "Post-Churn Correctness"]
    rows = []
    for name, (cls, gt) in PROTOCOLS.items():
        _, nodes, _ = build_network(cls, 20)
        rng = random.Random(42)
        to_remove = rng.sample(nodes, 3)
        for nd in to_remove:
            nd.leave()
            nodes.remove(nd)
        for _ in range(30):
            for nd in nodes:
                nd.stabilize()
        rem_ids = sorted(nd.node_id for nd in nodes)
        ok = 0
        for key in keys_200:
            r = random.Random(key).choice(nodes).lookup(key)
            if r.success and r.responsible_node == _gt(name, gt, key, rem_ids):
                ok += 1
        rows.append([name, f"{ok / len(keys_200):.1%}"])
    print_table(header, rows, [12, 25])

    # ── 5. Throughput ─────────────────────────────────────────────────
    print("\n  5. LOOKUP THROUGHPUT (N=20, 1 000 lookups)\n")
    header = ["Protocol", "Lookups/sec", "Elapsed"]
    rows = []
    for name, (cls, _) in PROTOCOLS.items():
        _, nodes, _ = build_network(cls, 20)
        t0 = time.perf_counter()
        for key in keys_1000:
            random.Random(key).choice(nodes).lookup(key)
        dt = time.perf_counter() - t0
        rows.append([name, f"{len(keys_1000)/dt:,.0f}", f"{dt:.3f}s"])
    print_table(header, rows, [12, 15, 10])

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SUMMARY  (at target scale N=20)")
    print(sep)

    s20 = {name: scale[name][20] for name in PROTOCOLS}
    log_n = math.log2(20)
    print(f"\n  {'Protocol':<12} {'Mean Hops':<11} {'Correct':<11} "
          f"vs log2(20)={log_n:.1f}")
    print(f"  {'-'*50}")
    for name in PROTOCOLS:
        r = s20[name]
        ok = "PASS" if r["mean"] <= log_n * 2 else "HIGH"
        print(f"  {name:<12} {r['mean']:<11.2f} {r['correctness']:<11.1%} {ok}")

    print()


if __name__ == "__main__":
    main()
