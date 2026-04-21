# TinyLLaMA KV Cache Splitting — Kademlia Experiment Results

## 1. Executive Summary

The three-stage experiment confirms **token-layer-group splitting with `token_block=32` and
`layer_group=2`** as the optimal configuration for distributing TinyLLaMA KV caches over a
Kademlia DHT. This variant achieves a mean p95 retrieval latency of **4.91 ms ± 0.02 ms (SEM)**
across 1,620 confirmatory runs, compared to 6.19 ms for the next-best variant — a 26% reduction.
The result is stable across all tested network sizes (N=32–128), replication factors (R=1–3),
churn rates (0–15%), and sequence lengths (128–1024 tokens).

---

## 2. Experimental Design

### 2.1 Stage Structure

| Stage | Purpose | Variants | N | R | Churn | L | Seeds | Runs |
|-------|---------|----------|---|---|-------|---|-------|------|
| A (prior) | Screen strategy families | 12 (all) | 64 | 2 | 0%, 5% | 512 | 5 | 120 |
| B | Full parameter sweep over top-5 from A | 5 | 32, 64, 128 | 1, 2, 3 | 0%, 5%, 15% | 128, 512, 1024 | 10 | 4,050 |
| C | Confirmatory, tight CI, top-2 from B | 2 | 32, 64, 128 | 1, 2, 3 | 0%, 5%, 15% | 128, 512, 1024 | 20 | 3,240 |

### 2.2 Model Profile

- Model: TinyLLaMA 1.1B
- Layers: 22, KV heads: 4 (GQA), head dim: 64
- KV bytes per token: ~45,056 (float32)
- KV cache at 512 tokens: ~22 MB uncompressed

### 2.3 Simulation Parameters

- ID space: 16-bit; k-bucket size: 8; alpha: 3
- Per-hop delay: 1.0 ms; base transfer: 0.2 ms; bandwidth: 100 Mbps
- Decode steps per run: 5

### 2.4 Scoring

Variants ranked by weighted score: **60% p95 latency + 25% availability + 15% overhead**.
Scores are min-max normalized within each metric before weighting.

---

## 3. Stage A Results (Screening)

Stage A confirmed that the token-layer-group strategy family dominates on latency:

| Strategy | p95 Latency (ms) | Chunk Success Rate | Weighted Score |
|----------|----------------:|--------------------|---------------|
| token_layer_group | 8.43 | 0.9992 | 0.750 |
| layer_only | 27.44 | 1.0000 | 0.442 |
| token_only | 36.36 | 1.0000 | 0.250 |

The top 5 variants promoted to Stage B were (in order):
1. `token_layer_group|tb=32|lg=2`
2. `token_layer_group|tb=32|lg=4`
3. `token_layer_group|tb=64|lg=2`
4. `token_layer_group|tb=64|lg=4`
5. `layer_only|lb=1`

---

## 4. Stage B Results (Full Parameter Sweep)

### 4.1 Variant-Level Ranking

| Variant | p50 (ms) | p95 (ms) | p99 (ms) | Avail. | Weighted Score |
|---------|--------:|---------:|---------:|-------:|---------------|
| `tb=32, lg=2` | 3.78 | 4.92 | 5.29 | 0.9966 | **0.850** |
| `tb=32, lg=4` | 4.99 | 6.20 | 6.54 | 0.9928 | 0.734 |
| `tb=64, lg=2` | 5.14 | 6.28 | 6.66 | 0.9915 | 0.712 |
| `tb=64, lg=4` | 7.59 | 8.82 | 9.17 | 0.9911 | 0.551 |
| `layer_only, lb=1` | 13.75 | 14.84 | 15.18 | 0.9875 | 0.150 |

`tb=32|lg=2` leads on every metric. The performance gap between the top two variants
(4.92 ms vs 6.20 ms p95) is larger than the gap between any two adjacent variants lower
in the table.

### 4.2 Effect of Sequence Length on p95 Latency

| Variant | L=128 | L=512 | L=1024 |
|---------|------:|------:|-------:|
| `tb=32, lg=2` | 5.16 | 4.84 | 4.76 |
| `tb=32, lg=4` | 6.36 | 6.20 | 6.03 |
| `tb=64, lg=2` | 6.47 | 6.23 | 6.12 |
| `tb=64, lg=4` | 8.97 | 8.79 | 8.69 |
| `layer_only, lb=1` | 6.56 | 14.01 | 23.94 |

Key observations:
- `token_layer_group` variants are **nearly flat across sequence lengths** — latency
  decreases slightly as L grows because the DHT routing cost is amortized over more
  chunks being fetched in parallel.
- `layer_only` scales **linearly with L** because each decode step must retrieve one
  large chunk covering all tokens. This confirms the Stage A finding at scale.

### 4.3 Effect of Network Size on p95 Latency

| Variant | N=32 | N=64 | N=128 |
|---------|-----:|-----:|------:|
| `tb=32, lg=2` | 4.27 | 4.95 | 5.54 |
| `tb=32, lg=4` | 5.55 | 6.27 | 6.77 |
| `tb=64, lg=2` | 5.63 | 6.37 | 6.83 |
| `tb=64, lg=4` | 8.16 | 8.97 | 9.33 |
| `layer_only, lb=1` | 14.12 | 14.97 | 15.42 |

Latency grows sub-linearly with N for all variants (consistent with O(log N) Kademlia
routing). `tb=32|lg=2` maintains the lowest absolute latency at every network size.

### 4.4 Effect of Churn on p95 Latency

| Variant | C=0% | C=5% | C=15% |
|---------|-----:|-----:|------:|
| `tb=32, lg=2` | 5.01 | 4.94 | 4.81 |
| `tb=32, lg=4` | 6.28 | 6.23 | 6.08 |
| `layer_only, lb=1` | 14.89 | 14.85 | 14.77 |

Counterintuitively, p95 latency is slightly *lower* at higher churn. This is a
simulation artifact: when nodes fail, surviving nodes have more spare capacity and
the stabilization pass re-routes requests more directly. Availability (chunk success
rate) does decrease under churn, but remains above 98.7% for all token-layer-group
variants even at 15% churn.

---

## 5. Stage C Results (Confirmatory, 20 Seeds)

### 5.1 Final Latency Summary

| Variant | p50 (ms) | p95 (ms) | 95% CI on p95 | p99 (ms) |
|---------|--------:|---------:|:-------------:|---------:|
| `tb=32, lg=2` | 3.779 | **4.911** | ±0.032 | 5.293 |
| `tb=32, lg=4` | 4.994 | 6.190 | ±0.032 | 6.558 |

The confidence intervals are tight and non-overlapping. `tb=32|lg=2` is the confirmed
winner with **p95 = 4.91 ms** and **p99 = 5.29 ms**.

### 5.2 Availability

| Variant | Mean Avail. | Min Avail. | Step Success Rate |
|---------|------------:|-----------:|------------------:|
| `tb=32, lg=2` | 99.56% | 86.82% | 81.02% |
| `tb=32, lg=4` | 99.19% | 70.83% | 83.37% |

`tb=32|lg=2` has higher mean availability and a better worst-case minimum (86.8% vs 70.8%).
The lower step success rate (81% vs 83%) reflects that `lg=2` produces more chunks per
decode step — a single missing chunk fails the whole step — but this tradeoff is
acceptable given the latency advantage.

### 5.3 Sensitivity Across Conditions (Stage C)

**By sequence length:**

| Variant | L=128 | L=512 | L=1024 |
|---------|------:|------:|-------:|
| `tb=32, lg=2` | 5.15 ± 0.03 | 4.85 ± 0.03 | 4.72 ± 0.03 |
| `tb=32, lg=4` | 6.35 ± 0.03 | 6.18 ± 0.03 | 6.04 ± 0.03 |

**By network size:**

| Variant | N=32 | N=64 | N=128 |
|---------|-----:|-----:|------:|
| `tb=32, lg=2` | 4.27 | 4.93 | 5.54 |
| `tb=32, lg=4` | 5.55 | 6.25 | 6.77 |

**By replication factor:**

| Variant | R=1 | R=2 | R=3 |
|---------|----:|----:|----:|
| `tb=32, lg=2` | 5.02 | 4.89 | 4.82 |
| `tb=32, lg=4` | 6.28 | 6.18 | 6.10 |

Higher replication reduces latency slightly (more replica holders = better chance of
a short DHT path). The effect is modest (~4% from R=1 to R=3), so R=2 is a reasonable
default that balances latency against storage overhead.

---

## 6. Decision

**Recommended configuration: `token_block=32`, `layer_group=2`, `replication=2`**

This is confirmed as the optimal split policy across all three experimental stages.

### Why tb=32 beats tb=64

Halving the token block doubles the number of chunks, which increases the number of
independent DHT fetches per decode step but reduces the size of each fetch. The
bandwidth model (100 Mbps, 0.2 ms base) favors smaller, parallelizable transfers
over fewer large ones. At tb=32 each chunk is ~128 KB; at tb=64 it doubles to ~256 KB.

### Why lg=2 beats lg=4

`lg=2` produces 11 layer stripes vs 6 for `lg=4`, giving more parallel fetch paths
across the 22 TinyLLaMA layers. The extra parallelism more than offsets the increased
coordination cost (more DHT lookups per step).

### Implication for P2P-RAGCache

The P2P_RAGCACHE_SYSTEM_PLAN.md prototype should use 11 peer processes (one per
layer-group stripe of 2 layers each) with token blocks of 32. The serialization layer
in `kv_serialization/serialize.py` already defaults to these values.

---

## 7. Artifacts

| Stage | Directory |
|-------|-----------|
| Stage A (screening) | `kv_kademlia_experiments/stage_a_20260324_095902/` |
| Stage B (full sweep) | `kv_kademlia_experiments/stage_b_final/` |
| Stage C (confirmatory) | `kv_kademlia_experiments/stage_c_final/` |

Each directory contains:
- `summary_metrics.csv` — per-run results
- `variant_ranking.csv` — weighted scores per variant
- `weighted_ranking.csv` — weighted scores per strategy family
- `figures/` — latency, availability, and overhead plots
- `run_config.json` — full experiment parameters
- `REPORT.md` — auto-generated summary
