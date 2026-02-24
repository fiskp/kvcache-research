"""Kademlia latency visualization tool.

Runs the Kademlia benchmark with virtual time, collects lookup traces,
and produces static plots: latency vs N, latency/hop distributions,
single-lookup path view, and k-bucket routing table snapshot.

Run from project root:
    python scripts/viz_kademlia_latency.py

Output: figures/*.png (created in ./figures/)
"""

import math
import os
import random
import statistics
import sys

# Run from project root so dht_comparison is on path
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
except ImportError:
    pass

from dht_comparison.base import (
    NetworkSimulator,
    generate_node_ids,
    generate_keys,
)
from dht_comparison.kademlia import KademliaNode

# ---------------------------------------------------------------------------
# Config (aligned with tests/benchmark_kademlia_latency.py)
# ---------------------------------------------------------------------------
ID_BITS = 16
RTT_PER_HOP = 1.0
LOOKUPS_PER_N = 200
COEF = 2.5  # O(log N) reference: mean_latency <= COEF * log2(N) * RTT
N_VALUES = [10, 20, 50, 100, 200]
FIGURES_DIR = "figures"


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


def run_benchmark():
    """Run benchmark for each N; return list of trace dicts and per-N bucket snapshot."""
    traces: list[dict] = []
    bucket_snapshots: dict[int, list[list[int]]] = {}  # N -> buckets of first node

    keys = generate_keys(LOOKUPS_PER_N)

    for n in N_VALUES:
        network, nodes = build_kademlia_network(n)
        # Snapshot routing table of first node for k-bucket viz
        bucket_snapshots[n] = [list(b) for b in nodes[0].buckets]

        for key in keys:
            initiator = random.Random(key).choice(nodes)
            t0 = network.virtual_time
            result = initiator.lookup(key)
            t1 = network.virtual_time
            if result.success:
                traces.append({
                    "N": n,
                    "id_bits": ID_BITS,
                    "key": key,
                    "initiator_id": initiator.node_id,
                    "latency": t1 - t0,
                    "hop_count": result.hop_count,
                    "path": list(result.path),
                })
    return traces, bucket_snapshots


def aggregate_by_n(traces: list[dict]):
    """Group by N; return dict N -> {mean_latency, median_latency, p95_latency, mean_hops, ...}."""
    by_n: dict[int, list[dict]] = {}
    for t in traces:
        by_n.setdefault(t["N"], []).append(t)
    agg = {}
    for n, group in by_n.items():
        latencies = [t["latency"] for t in group]
        hops = [t["hop_count"] for t in group]
        sl = sorted(latencies)
        sh = sorted(hops)
        nn = len(sl)
        agg[n] = {
            "mean_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": sl[int(nn * 0.95)] if nn > 1 else sl[0],
            "mean_hops": statistics.mean(hops),
            "median_hops": statistics.median(hops),
            "p95_hops": sh[int(nn * 0.95)] if nn > 1 else sh[0],
            "traces": group,
        }
    return agg


# ---------------------------------------------------------------------------
# Plot 3.1: Latency vs N (scaling) + O(log N) reference
# ---------------------------------------------------------------------------
def plot_latency_scaling(agg: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = sorted(agg.keys())
    mean_latencies = [agg[n]["mean_latency"] for n in ns]
    x_curve = np.linspace(min(ns), max(ns), 200)
    y_ref = COEF * np.log2(x_curve) * RTT_PER_HOP

    ax.plot(ns, mean_latencies, "o-", color="C0", linewidth=2, markersize=8, label="Mean simulated latency")
    ax.plot(x_curve, y_ref, "--", color="gray", linewidth=1.5, label=rf"$C \cdot \log_2 N$ (C={COEF})")
    ax.set_xlabel("Network size N")
    ax.set_ylabel("Mean latency (virtual time units)")
    ax.set_title("Kademlia: Simulated lookup latency vs N")
    ax.legend()
    ax.set_xticks(ns)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3.2: Latency and hop count distributions per N (boxplots)
# ---------------------------------------------------------------------------
def plot_distributions(agg: dict, out_path_latency: str, out_path_hops: str):
    ns = sorted(agg.keys())
    latency_by_n = [[t["latency"] for t in agg[n]["traces"]] for n in ns]
    hops_by_n = [[t["hop_count"] for t in agg[n]["traces"]] for n in ns]

    # Latency boxplot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.boxplot(latency_by_n, tick_labels=ns)
    ax1.set_ylabel("Latency (virtual time units)")
    ax1.set_xlabel("Network size N")
    ax1.set_title("Lookup latency distribution by N")
    fig1.tight_layout()
    fig1.savefig(out_path_latency, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Hop count boxplot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.boxplot(hops_by_n, tick_labels=ns)
    ax2.set_ylabel("Hop count")
    ax2.set_xlabel("Network size N")
    ax2.set_title("Lookup hop count distribution by N")
    fig2.tight_layout()
    fig2.savefig(out_path_hops, dpi=150, bbox_inches="tight")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Plot 3.3: Single-lookup path (one path per N, nodes on ID line, arrows by hop)
# ---------------------------------------------------------------------------
def plot_paths(agg: dict, out_path: str):
    id_space = 2 ** ID_BITS
    n_plots = len(N_VALUES)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.colormaps["viridis"].resampled(12)

    for idx, n in enumerate(N_VALUES):
        ax = axes[idx]
        group = agg[n]["traces"]
        # Representative: median latency lookup (middle-ish)
        sorted_by_lat = sorted(group, key=lambda t: t["latency"])
        mid = len(sorted_by_lat) // 2
        trace = sorted_by_lat[mid]
        path = trace["path"]
        if len(path) < 2:
            ax.text(0.5, 0.5, f"N={n}\n1-hop path", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, id_space)
            ax.set_ylim(-0.2, 0.2)
            continue

        # Nodes on a line by ID (x = node_id, y = 0)
        node_x = {nid: nid for nid in path}
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            xa, xb = node_x[a], node_x[b]
            color = cmap(i / max(len(path) - 1, 1))
            ax.annotate(
                "", xy=(xb, 0), xytext=(xa, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
            )
            mid_x = (xa + xb) / 2
            ax.text(mid_x, 0.08, f"t={i+1}", fontsize=8, ha="center", color=color)
        for nid in path:
            ax.plot(nid, 0, "ko", markersize=6)
        ax.set_xlim(-id_space * 0.02, id_space * 1.02)
        ax.set_ylim(-0.25, 0.25)
        ax.set_yticks([])
        ax.set_xlabel("Node ID (key space)")
        ax.set_title(f"N={n}, key={trace['key']}, latency={trace['latency']:.0f}, hops={trace['hop_count']}")
        ax.set_aspect("auto")

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Kademlia lookup path (representative median-latency lookup per N)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3.4: k-bucket view for one node (bucket index vs size; optional sample IDs)
# ---------------------------------------------------------------------------
def plot_kbuckets(bucket_snapshots: dict[int, list[list[int]]], out_path: str,
                  chosen_n: int = 50):
    """Bar chart of bucket sizes for a chosen N (first node)."""
    buckets = bucket_snapshots.get(chosen_n)
    if not buckets:
        return
    sizes = [len(b) for b in buckets]
    indices = list(range(len(buckets)))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(indices, sizes, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Bucket index i (XOR distance in [2^i, 2^(i+1)))")
    ax.set_ylabel("Bucket size")
    ax.set_title(f"Kademlia k-bucket routing table (one node, N={chosen_n})")
    ax.set_xticks(indices)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Running Kademlia benchmark (virtual time)...")
    traces, bucket_snapshots = run_benchmark()
    print(f"  Collected {len(traces)} lookup traces.")
    agg = aggregate_by_n(traces)
    print("  Aggregated by N.")

    base = os.path.join(FIGURES_DIR, "kademlia")
    print("Generating plots...")
    plot_latency_scaling(agg, f"{base}_latency_scaling.png")
    print(f"  Saved {base}_latency_scaling.png")
    plot_distributions(agg, f"{base}_latency_distributions.png", f"{base}_hop_distributions.png")
    print(f"  Saved {base}_latency_distributions.png, {base}_hop_distributions.png")
    plot_paths(agg, f"{base}_path_example.png")
    print(f"  Saved {base}_path_example.png")
    plot_kbuckets(bucket_snapshots, f"{base}_kbuckets.png", chosen_n=50)
    print(f"  Saved {base}_kbuckets.png")
    print("Done.")


if __name__ == "__main__":
    main()
