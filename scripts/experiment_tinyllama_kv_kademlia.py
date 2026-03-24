import argparse
import csv
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dht_comparison.base import DHTNode, NetworkSimulator, generate_node_ids
from dht_comparison.kademlia import KademliaNode


ID_BITS = 16
K_BUCKET = 8
ALPHA = 3
PER_HOP_DELAY_MS = 1.0
BASE_TRANSFER_MS = 0.2
BANDWIDTH_MBPS = 100.0

STRATEGY_TOKEN_LAYER = "token_layer_group"
STRATEGY_TOKEN_ONLY = "token_only"
STRATEGY_LAYER_ONLY = "layer_only"
STRATEGIES = [STRATEGY_TOKEN_LAYER, STRATEGY_TOKEN_ONLY, STRATEGY_LAYER_ONLY]


@dataclass
class ExperimentConfig:
    run_id: str
    strategy: str
    seq_len: int
    num_peers: int
    replication: int
    churn_rate: float
    seed: int
    token_block: Optional[int] = None
    layer_group_size: Optional[int] = None
    layer_block_size: Optional[int] = None
    decode_steps: int = 5


@dataclass
class ChunkRecord:
    chunk_id: str
    chunk_key: int
    seq_len: int
    token_start: int
    token_end: int
    layer_start: int
    layer_end: int
    chunk_total_bytes: int
    strategy: str


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return data[low]
    weight = rank - low
    return data[low] * (1 - weight) + data[high] * weight


def load_tinyllama_profile(repo_root: Path) -> Dict[str, Any]:
    summary_path = repo_root / "phase2_outputs" / "tinyllama_kv_stats.csv"
    layers_path = repo_root / "phase2_outputs" / "tinyllama_kv_layer_stats.csv"
    if not summary_path.exists() or not layers_path.exists():
        raise FileNotFoundError(
            "Missing Phase 2 outputs. Run scripts/kv_cache_phase2_tinyllama.py first."
        )

    summary_by_len: Dict[int, Dict[str, float]] = {}
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_len = int(row["seq_len"])
            summary_by_len[seq_len] = {
                "total_kv_bytes": float(row["total_kv_bytes_llama"]),
                "kv_bytes_per_token": float(row["kv_bytes_per_token_llama"]),
                "kv_bytes_per_layer": float(row["kv_bytes_per_layer_llama"]),
            }

    layer_bytes_by_len: Dict[int, List[int]] = {}
    with layers_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_len = int(row["seq_len"])
            layer_total = int(row["total_layer_bytes"])
            layer_bytes_by_len.setdefault(seq_len, []).append(layer_total)

    num_layers = len(next(iter(layer_bytes_by_len.values())))
    return {
        "summary_by_len": summary_by_len,
        "layer_bytes_by_len": layer_bytes_by_len,
        "num_layers": num_layers,
    }


def build_chunks(
    cfg: ExperimentConfig,
    profile: Dict[str, Any],
) -> List[ChunkRecord]:
    layer_bytes = profile["layer_bytes_by_len"][cfg.seq_len]
    num_layers = profile["num_layers"]

    chunks: List[ChunkRecord] = []
    if cfg.strategy == STRATEGY_TOKEN_LAYER:
        if cfg.token_block is None or cfg.layer_group_size is None:
            raise ValueError("token_layer_group requires token_block and layer_group_size.")
        for token_start in range(0, cfg.seq_len, cfg.token_block):
            token_end = min(cfg.seq_len, token_start + cfg.token_block)
            token_scale = (token_end - token_start) / cfg.seq_len
            for layer_start in range(0, num_layers, cfg.layer_group_size):
                layer_end = min(num_layers, layer_start + cfg.layer_group_size)
                bytes_total = int(sum(layer_bytes[layer_start:layer_end]) * token_scale)
                cid = (
                    f"m=tinyllama|l={cfg.seq_len}|s={cfg.strategy}|tb={cfg.token_block}|"
                    f"lg={cfg.layer_group_size}|ts={token_start}|te={token_end}|"
                    f"ls={layer_start}|le={layer_end}"
                )
                chunks.append(
                    ChunkRecord(
                        chunk_id=cid,
                        chunk_key=DHTNode.hash_key(cid, ID_BITS),
                        seq_len=cfg.seq_len,
                        token_start=token_start,
                        token_end=token_end,
                        layer_start=layer_start,
                        layer_end=layer_end,
                        chunk_total_bytes=max(bytes_total, 1),
                        strategy=cfg.strategy,
                    )
                )
    elif cfg.strategy == STRATEGY_TOKEN_ONLY:
        if cfg.token_block is None:
            raise ValueError("token_only requires token_block.")
        for token_start in range(0, cfg.seq_len, cfg.token_block):
            token_end = min(cfg.seq_len, token_start + cfg.token_block)
            token_scale = (token_end - token_start) / cfg.seq_len
            bytes_total = int(sum(layer_bytes) * token_scale)
            cid = (
                f"m=tinyllama|l={cfg.seq_len}|s={cfg.strategy}|tb={cfg.token_block}|"
                f"ts={token_start}|te={token_end}|ls=0|le={num_layers}"
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=cid,
                    chunk_key=DHTNode.hash_key(cid, ID_BITS),
                    seq_len=cfg.seq_len,
                    token_start=token_start,
                    token_end=token_end,
                    layer_start=0,
                    layer_end=num_layers,
                    chunk_total_bytes=max(bytes_total, 1),
                    strategy=cfg.strategy,
                )
            )
    elif cfg.strategy == STRATEGY_LAYER_ONLY:
        if cfg.layer_block_size is None:
            raise ValueError("layer_only requires layer_block_size.")
        for layer_start in range(0, num_layers, cfg.layer_block_size):
            layer_end = min(num_layers, layer_start + cfg.layer_block_size)
            bytes_total = int(sum(layer_bytes[layer_start:layer_end]))
            cid = (
                f"m=tinyllama|l={cfg.seq_len}|s={cfg.strategy}|lb={cfg.layer_block_size}|"
                f"ts=0|te={cfg.seq_len}|ls={layer_start}|le={layer_end}"
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=cid,
                    chunk_key=DHTNode.hash_key(cid, ID_BITS),
                    seq_len=cfg.seq_len,
                    token_start=0,
                    token_end=cfg.seq_len,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    chunk_total_bytes=max(bytes_total, 1),
                    strategy=cfg.strategy,
                )
            )
    else:
        raise ValueError(f"Unsupported strategy: {cfg.strategy}")
    return chunks


def build_network(num_peers: int, seed: int) -> tuple[NetworkSimulator, List[KademliaNode]]:
    network = NetworkSimulator(per_hop_delay=PER_HOP_DELAY_MS)
    node_ids = generate_node_ids(num_peers, id_bits=ID_BITS, seed=seed)
    nodes: List[KademliaNode] = []
    for nid in node_ids:
        node = KademliaNode(nid, network, id_bits=ID_BITS, k=K_BUCKET, alpha=ALPHA)
        bootstrap = nodes[0].node_id if nodes else None
        node.join(bootstrap)
        nodes.append(node)
    for _ in range(max(30, num_peers * 2)):
        for node in nodes:
            if network.get_node(node.node_id) is not None:
                node.stabilize()
    return network, nodes


def alive_nodes(nodes: List[KademliaNode], network: NetworkSimulator) -> List[KademliaNode]:
    return [n for n in nodes if network.get_node(n.node_id) is not None]


def transfer_delay_ms(chunk_bytes: int) -> float:
    bytes_per_ms = (BANDWIDTH_MBPS * 1024 * 1024) / 1000.0
    return BASE_TRANSFER_MS + (chunk_bytes / max(bytes_per_ms, 1.0))


def place_replicas(
    cfg: ExperimentConfig,
    chunks: List[ChunkRecord],
    nodes: List[KademliaNode],
    network: NetworkSimulator,
    placement_rows: List[Dict[str, Any]],
    used_bytes: Dict[int, int],
) -> Dict[str, List[int]]:
    rng = random.Random(cfg.seed)
    replicas: Dict[str, List[int]] = {}
    for chunk in chunks:
        replica_keys: List[int] = []
        for replica_idx in range(cfg.replication):
            rkey = DHTNode.hash_key(f"{chunk.chunk_id}|r={replica_idx}", ID_BITS)
            initiator = rng.choice(alive_nodes(nodes, network))
            ok = initiator.store(rkey, {"chunk_id": chunk.chunk_id, "bytes": chunk.chunk_total_bytes})
            if ok:
                lookup = initiator.lookup(rkey)
                owner = lookup.responsible_node
                before = used_bytes.get(owner, 0)
                after = before + chunk.chunk_total_bytes
                used_bytes[owner] = after
                placement_rows.append(
                    {
                        "run_id": cfg.run_id,
                        "chunk_id": chunk.chunk_id,
                        "replica_index": replica_idx,
                        "replica_key": rkey,
                        "peer_id": owner,
                        "is_primary": replica_idx == 0,
                        "placement_epoch": 0,
                        "peer_used_kv_bytes_before": before,
                        "peer_used_kv_bytes_after": after,
                        "placement_reason": "initial",
                    }
                )
            replica_keys.append(rkey)
        replicas[chunk.chunk_id] = replica_keys
    return replicas


def fail_nodes_for_churn(
    cfg: ExperimentConfig, nodes: List[KademliaNode], network: NetworkSimulator, rng: random.Random
) -> List[int]:
    if cfg.churn_rate <= 0.0:
        return []
    active = alive_nodes(nodes, network)
    fail_count = int(len(active) * cfg.churn_rate)
    if fail_count <= 0:
        return []
    fail_count = min(fail_count, max(0, len(active) - 1))
    to_fail = rng.sample(active, fail_count)
    failed_ids = [n.node_id for n in to_fail]
    for node in to_fail:
        network.unregister(node.node_id)
    for _ in range(2):
        for node in alive_nodes(nodes, network):
            node.stabilize()
    return failed_ids


def repair_missing_replicas(
    cfg: ExperimentConfig,
    chunks: List[ChunkRecord],
    replicas: Dict[str, List[int]],
    nodes: List[KademliaNode],
    network: NetworkSimulator,
    used_bytes: Dict[int, int],
    placement_rows: List[Dict[str, Any]],
    epoch: int,
    rng: random.Random,
) -> int:
    repairs = 0
    live_nodes = alive_nodes(nodes, network)
    if not live_nodes:
        return repairs

    chunk_by_id = {c.chunk_id: c for c in chunks}
    for chunk_id, replica_keys in replicas.items():
        chunk = chunk_by_id[chunk_id]
        for idx, rkey in enumerate(replica_keys):
            has_live_copy = False
            for node in live_nodes:
                if rkey in node.data:
                    has_live_copy = True
                    break
            if has_live_copy:
                continue
            initiator = rng.choice(live_nodes)
            ok = initiator.store(rkey, {"chunk_id": chunk_id, "bytes": chunk.chunk_total_bytes})
            if ok:
                lookup = initiator.lookup(rkey)
                owner = lookup.responsible_node
                before = used_bytes.get(owner, 0)
                after = before + chunk.chunk_total_bytes
                used_bytes[owner] = after
                placement_rows.append(
                    {
                        "run_id": cfg.run_id,
                        "chunk_id": chunk_id,
                        "replica_index": idx,
                        "replica_key": rkey,
                        "peer_id": owner,
                        "is_primary": idx == 0,
                        "placement_epoch": epoch,
                        "peer_used_kv_bytes_before": before,
                        "peer_used_kv_bytes_after": after,
                        "placement_reason": "recovery",
                    }
                )
                repairs += 1
    return repairs


def run_single_experiment(
    cfg: ExperimentConfig, profile: Dict[str, Any], out_dir: Path
) -> Dict[str, Any]:
    rng = random.Random(cfg.seed)
    network, nodes = build_network(cfg.num_peers, cfg.seed)
    chunks = build_chunks(cfg, profile)
    used_bytes: Dict[int, int] = {}

    chunk_rows: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk_rows.append(
            {
                "run_id": cfg.run_id,
                "chunk_id": chunk.chunk_id,
                "chunk_key": chunk.chunk_key,
                "seq_len": chunk.seq_len,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
                "chunk_len_tokens": chunk.token_end - chunk.token_start,
                "layer_start": chunk.layer_start,
                "layer_end": chunk.layer_end,
                "chunk_total_bytes": chunk.chunk_total_bytes,
                "split_strategy": chunk.strategy,
            }
        )

    placement_rows: List[Dict[str, Any]] = []
    replicas = place_replicas(cfg, chunks, nodes, network, placement_rows, used_bytes)

    latencies_ms: List[float] = []
    chunk_lookup_ms: List[float] = []
    hop_counts: List[int] = []
    transfer_bytes_total = 0
    useful_bytes_total = 0
    chunk_success = 0
    chunk_total = 0
    decode_step_success = 0
    timeout_count = 0
    repair_events = 0
    access_rows: List[Dict[str, Any]] = []

    for step in range(cfg.decode_steps):
        failed_ids = fail_nodes_for_churn(cfg, nodes, network, rng)
        repair_events += repair_missing_replicas(
            cfg, chunks, replicas, nodes, network, used_bytes, placement_rows, step + 1, rng
        )
        active = alive_nodes(nodes, network)
        if not active:
            break
        requester = rng.choice(active)
        step_ok = True

        for chunk in chunks:
            chunk_total += 1
            useful_bytes_total += chunk.chunk_total_bytes
            event_base = {
                "run_id": cfg.run_id,
                "step": step,
                "requesting_peer_id": requester.node_id,
                "chunk_id": chunk.chunk_id,
                "transfer_bytes": chunk.chunk_total_bytes,
                "failed_node_ids": ",".join(str(x) for x in failed_ids),
            }

            got_chunk = False
            for replica_key in replicas[chunk.chunk_id]:
                t0 = network.virtual_time
                lookup = requester.lookup(replica_key)
                t1 = network.virtual_time
                lookup_ms = t1 - t0
                hop_counts.append(lookup.hop_count)
                chunk_lookup_ms.append(lookup_ms)
                target = network.get_node(lookup.responsible_node)
                hit = bool(target is not None and replica_key in target.data)
                if hit:
                    xfer = transfer_delay_ms(chunk.chunk_total_bytes)
                    network.advance_time(xfer)
                    total_latency = lookup_ms + xfer
                    latencies_ms.append(total_latency)
                    transfer_bytes_total += chunk.chunk_total_bytes
                    chunk_success += 1
                    got_chunk = True
                    access_rows.append(
                        {
                            **event_base,
                            "operation": "GET",
                            "replica_key": replica_key,
                            "serving_peer_id": lookup.responsible_node,
                            "hop_count": lookup.hop_count,
                            "lookup_latency_ms": round(lookup_ms, 4),
                            "total_latency_ms": round(total_latency, 4),
                            "result": "hit",
                        }
                    )
                    break
                access_rows.append(
                    {
                        **event_base,
                        "operation": "GET",
                        "replica_key": replica_key,
                        "serving_peer_id": lookup.responsible_node,
                        "hop_count": lookup.hop_count,
                        "lookup_latency_ms": round(lookup_ms, 4),
                        "total_latency_ms": round(lookup_ms, 4),
                        "result": "miss",
                    }
                )
            if not got_chunk:
                timeout_count += 1
                step_ok = False
        if step_ok:
            decode_step_success += 1

    chunk_success_rate = (chunk_success / chunk_total) if chunk_total else 0.0
    decode_step_success_rate = (decode_step_success / cfg.decode_steps) if cfg.decode_steps else 0.0
    overhead_ratio = (transfer_bytes_total / useful_bytes_total) if useful_bytes_total else 0.0
    source_bytes = sum(c.chunk_total_bytes for c in chunks)
    storage_amplification = (
        (cfg.replication * source_bytes) / source_bytes if source_bytes else 0.0
    )

    summary = {
        "run_id": cfg.run_id,
        "strategy": cfg.strategy,
        "seq_len": cfg.seq_len,
        "num_peers": cfg.num_peers,
        "replication": cfg.replication,
        "churn_rate": cfg.churn_rate,
        "seed": cfg.seed,
        "token_block": cfg.token_block or "",
        "layer_group_size": cfg.layer_group_size or "",
        "layer_block_size": cfg.layer_block_size or "",
        "decode_steps": cfg.decode_steps,
        "chunks_per_step": len(chunks),
        "decode_step_latency_ms_p50": round(percentile(latencies_ms, 0.50), 4),
        "decode_step_latency_ms_p95": round(percentile(latencies_ms, 0.95), 4),
        "decode_step_latency_ms_p99": round(percentile(latencies_ms, 0.99), 4),
        "chunk_get_latency_ms_p95": round(percentile(chunk_lookup_ms, 0.95), 4),
        "chunk_success_rate": round(chunk_success_rate, 6),
        "decode_step_success_rate": round(decode_step_success_rate, 6),
        "timeout_rate": round((timeout_count / max(chunk_total, 1)), 6),
        "avg_hops_per_chunk": round(statistics.mean(hop_counts), 4) if hop_counts else 0.0,
        "bytes_transferred_total": transfer_bytes_total,
        "useful_bytes_total": useful_bytes_total,
        "bytes_overhead_ratio": round(overhead_ratio, 6),
        "storage_amplification": round(storage_amplification, 4),
        "peer_utilization_p50": round(percentile(list(used_bytes.values()), 0.50), 2)
        if used_bytes
        else 0.0,
        "peer_utilization_p95": round(percentile(list(used_bytes.values()), 0.95), 2)
        if used_bytes
        else 0.0,
        "eviction_events_count": 0,
        "recovery_repair_events_count": repair_events,
    }

    return {
        "summary": summary,
        "chunk_rows": chunk_rows,
        "placement_rows": placement_rows,
        "access_rows": access_rows,
    }


def csv_write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_strategy(summary_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(row["strategy"], []).append(row)

    result: Dict[str, Dict[str, float]] = {}
    for strategy, rows in grouped.items():
        result[strategy] = {
            "p95_latency": statistics.mean(float(r["decode_step_latency_ms_p95"]) for r in rows),
            "availability": statistics.mean(float(r["chunk_success_rate"]) for r in rows),
            "overhead": statistics.mean(float(r["bytes_overhead_ratio"]) for r in rows),
        }
    return result


def normalized_score(values: Dict[str, float], lower_is_better: bool) -> Dict[str, float]:
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if math.isclose(vmin, vmax):
        return {k: 1.0 for k in values}
    scores = {}
    for k, v in values.items():
        x = (v - vmin) / (vmax - vmin)
        scores[k] = 1.0 - x if lower_is_better else x
    return scores


def build_weighted_ranking(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg = aggregate_by_strategy(summary_rows)
    p95 = {k: v["p95_latency"] for k, v in agg.items()}
    avail = {k: v["availability"] for k, v in agg.items()}
    overhead = {k: v["overhead"] for k, v in agg.items()}

    p95_score = normalized_score(p95, lower_is_better=True)
    avail_score = normalized_score(avail, lower_is_better=False)
    overhead_score = normalized_score(overhead, lower_is_better=True)

    rows = []
    for strategy in agg:
        total = 0.60 * p95_score[strategy] + 0.25 * avail_score[strategy] + 0.15 * overhead_score[strategy]
        rows.append(
            {
                "strategy": strategy,
                "avg_p95_latency_ms": round(agg[strategy]["p95_latency"], 4),
                "avg_chunk_success_rate": round(agg[strategy]["availability"], 6),
                "avg_overhead_ratio": round(agg[strategy]["overhead"], 6),
                "weighted_score": round(total, 6),
            }
        )
    rows.sort(key=lambda r: r["weighted_score"], reverse=True)
    return rows


def plot_results(summary_rows: List[Dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    by_strategy = aggregate_by_strategy(summary_rows)
    strategies = list(by_strategy.keys())

    p95 = [by_strategy[s]["p95_latency"] for s in strategies]
    plt.figure(figsize=(8, 5))
    plt.bar(strategies, p95)
    plt.title("p95 Decode-Step Latency by Strategy")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Strategy")
    plt.tight_layout()
    plt.savefig(figures_dir / "latency_p95_by_strategy.png", dpi=220)
    plt.close()

    churn_groups: Dict[float, List[float]] = {}
    for row in summary_rows:
        churn_groups.setdefault(float(row["churn_rate"]), []).append(float(row["decode_step_latency_ms_p95"]))
    churn_x = sorted(churn_groups.keys())
    churn_y = [statistics.mean(churn_groups[c]) for c in churn_x]
    plt.figure(figsize=(8, 5))
    plt.plot(churn_x, churn_y, marker="o")
    plt.title("Latency vs Churn")
    plt.xlabel("Churn rate")
    plt.ylabel("Mean p95 latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "latency_vs_churn.png", dpi=220)
    plt.close()

    avail_groups: Dict[float, List[float]] = {}
    for row in summary_rows:
        avail_groups.setdefault(float(row["churn_rate"]), []).append(float(row["chunk_success_rate"]))
    avail_y = [statistics.mean(avail_groups[c]) for c in churn_x]
    plt.figure(figsize=(8, 5))
    plt.plot(churn_x, avail_y, marker="o")
    plt.title("Availability vs Churn")
    plt.xlabel("Churn rate")
    plt.ylabel("Chunk success rate")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "availability_vs_churn.png", dpi=220)
    plt.close()

    rep_groups: Dict[int, List[float]] = {}
    for row in summary_rows:
        rep_groups.setdefault(int(row["replication"]), []).append(float(row["bytes_overhead_ratio"]))
    rep_x = sorted(rep_groups.keys())
    rep_y = [statistics.mean(rep_groups[r]) for r in rep_x]
    plt.figure(figsize=(8, 5))
    plt.plot(rep_x, rep_y, marker="o")
    plt.title("Overhead vs Replication")
    plt.xlabel("Replication factor")
    plt.ylabel("Mean overhead ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "overhead_vs_replication.png", dpi=220)
    plt.close()


def build_report(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    ranking_rows: List[Dict[str, Any]],
    run_config: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# TinyLLaMA KV Splitting with Kademlia - Experiment Report\n")
    lines.append("## Run configuration")
    lines.append(f"- Stage: `{run_config['stage']}`")
    lines.append(f"- Total runs: `{len(summary_rows)}`")
    lines.append(f"- Strategies: `{', '.join(run_config['strategies'])}`")
    lines.append(f"- Network sizes: `{run_config['num_peers']}`")
    lines.append(f"- Replication factors: `{run_config['replication']}`")
    lines.append(f"- Churn rates: `{run_config['churn_rates']}`")
    lines.append(f"- Sequence lengths: `{run_config['seq_lens']}`")
    lines.append(f"- Seeds: `{run_config['seeds']}`\n")

    lines.append("## Weighted ranking (60% latency, 25% availability, 15% overhead)")
    lines.append("| strategy | avg_p95_latency_ms | avg_chunk_success_rate | avg_overhead_ratio | weighted_score |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in ranking_rows:
        lines.append(
            f"| {row['strategy']} | {row['avg_p95_latency_ms']:.4f} | "
            f"{row['avg_chunk_success_rate']:.6f} | {row['avg_overhead_ratio']:.6f} | "
            f"{row['weighted_score']:.6f} |"
        )
    lines.append("")

    if ranking_rows:
        winner = ranking_rows[0]["strategy"]
        lines.append("## Recommended strategy")
        lines.append(f"- Winner: `{winner}` based on configured weighted scoring.\n")

    lines.append("## Artifacts")
    lines.append("- `chunk_catalog.csv`")
    lines.append("- `placement.csv`")
    lines.append("- `access_trace.csv`")
    lines.append("- `summary_metrics.csv`")
    lines.append("- `weighted_ranking.csv`")
    lines.append("- `figures/latency_p95_by_strategy.png`")
    lines.append("- `figures/latency_vs_churn.png`")
    lines.append("- `figures/availability_vs_churn.png`")
    lines.append("- `figures/overhead_vs_replication.png`")
    lines.append("")

    (output_dir / "REPORT.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def strategy_variants(strategy: str) -> List[Dict[str, Optional[int]]]:
    if strategy == STRATEGY_TOKEN_LAYER:
        return [
            {"token_block": tb, "layer_group_size": lg, "layer_block_size": None}
            for tb in (32, 64, 128)
            for lg in (2, 4)
        ]
    if strategy == STRATEGY_TOKEN_ONLY:
        return [
            {"token_block": tb, "layer_group_size": None, "layer_block_size": None}
            for tb in (32, 64, 128)
        ]
    if strategy == STRATEGY_LAYER_ONLY:
        return [
            {"token_block": None, "layer_group_size": None, "layer_block_size": lb}
            for lb in (1, 2, 4)
        ]
    raise ValueError(f"Unknown strategy: {strategy}")


def build_matrix(stage: str, seeds: int) -> Dict[str, Any]:
    if stage == "stage_a":
        run_cfg = {
            "stage": stage,
            "strategies": STRATEGIES,
            "num_peers": [64],
            "replication": [2],
            "churn_rates": [0.0, 0.05],
            "seq_lens": [512],
            "seeds": list(range(seeds)),
            "decode_steps": 5,
        }
    elif stage == "stage_b":
        run_cfg = {
            "stage": stage,
            "strategies": STRATEGIES,
            "num_peers": [32, 64, 128],
            "replication": [1, 2, 3],
            "churn_rates": [0.0, 0.05, 0.15],
            "seq_lens": [128, 512, 1024],
            "seeds": list(range(seeds)),
            "decode_steps": 5,
        }
    elif stage == "full":
        run_cfg = {
            "stage": stage,
            "strategies": STRATEGIES,
            "num_peers": [32, 64, 128],
            "replication": [1, 2, 3],
            "churn_rates": [0.0, 0.05, 0.15],
            "seq_lens": [128, 512, 1024],
            "seeds": list(range(max(10, seeds))),
            "decode_steps": 8,
        }
    else:
        raise ValueError("stage must be one of: stage_a, stage_b, full")
    return run_cfg


def iter_experiment_configs(run_cfg: Dict[str, Any], max_runs: Optional[int]) -> Iterable[ExperimentConfig]:
    idx = 0
    for strategy in run_cfg["strategies"]:
        for variant in strategy_variants(strategy):
            for n in run_cfg["num_peers"]:
                for r in run_cfg["replication"]:
                    for churn in run_cfg["churn_rates"]:
                        for seq_len in run_cfg["seq_lens"]:
                            for seed in run_cfg["seeds"]:
                                cfg = ExperimentConfig(
                                    run_id=f"run_{idx:05d}",
                                    strategy=strategy,
                                    seq_len=seq_len,
                                    num_peers=n,
                                    replication=r,
                                    churn_rate=churn,
                                    seed=seed,
                                    token_block=variant["token_block"],
                                    layer_group_size=variant["layer_group_size"],
                                    layer_block_size=variant["layer_block_size"],
                                    decode_steps=run_cfg["decode_steps"],
                                )
                                yield cfg
                                idx += 1
                                if max_runs is not None and idx >= max_runs:
                                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TinyLLaMA KV cache splitting experiments over Kademlia."
    )
    parser.add_argument(
        "--stage",
        choices=["stage_a", "stage_b", "full"],
        default="stage_a",
        help="Experiment stage preset.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds per config (full forces at least 10).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to kv_kademlia_experiments/<timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    profile = load_tinyllama_profile(repo_root)
    run_cfg = build_matrix(args.stage, args.seeds)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "kv_kademlia_experiments" / f"{args.stage}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    chunk_rows_all: List[Dict[str, Any]] = []
    placement_rows_all: List[Dict[str, Any]] = []
    access_rows_all: List[Dict[str, Any]] = []

    configs = list(iter_experiment_configs(run_cfg, args.max_runs))
    total = len(configs)
    for i, cfg in enumerate(configs, start=1):
        print(
            f"[{i}/{total}] {cfg.run_id} strategy={cfg.strategy} "
            f"L={cfg.seq_len} N={cfg.num_peers} R={cfg.replication} "
            f"churn={cfg.churn_rate} seed={cfg.seed}"
        )
        result = run_single_experiment(cfg, profile, output_dir)
        summary_rows.append(result["summary"])
        chunk_rows_all.extend(result["chunk_rows"])
        placement_rows_all.extend(result["placement_rows"])
        access_rows_all.extend(result["access_rows"])

    ranking_rows = build_weighted_ranking(summary_rows)

    csv_write(output_dir / "chunk_catalog.csv", chunk_rows_all)
    csv_write(output_dir / "placement.csv", placement_rows_all)
    csv_write(output_dir / "access_trace.csv", access_rows_all)
    csv_write(output_dir / "summary_metrics.csv", summary_rows)
    csv_write(output_dir / "weighted_ranking.csv", ranking_rows)
    plot_results(summary_rows, output_dir / "figures")
    build_report(output_dir, summary_rows, ranking_rows, run_cfg)

    config_out = {
        **run_cfg,
        "max_runs": args.max_runs,
        "effective_total_runs": len(summary_rows),
        "constants": {
            "id_bits": ID_BITS,
            "k_bucket": K_BUCKET,
            "alpha": ALPHA,
            "per_hop_delay_ms": PER_HOP_DELAY_MS,
            "base_transfer_ms": BASE_TRANSFER_MS,
            "bandwidth_mbps": BANDWIDTH_MBPS,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")

    print(f"\nCompleted {len(summary_rows)} runs.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

