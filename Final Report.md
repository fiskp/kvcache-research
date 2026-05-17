# Distributed KV Cache Retrieval for RAG via Kademlia P2P Networks: Simulation-Validated Optimal Splitting Strategies

**Author:** Joseph Huynh  
**Institution:** California State University, Long Beach  
**Date:** May 2026

---

## Abstract

Time-to-first-token (TTFT) in retrieval-augmented generation (RAG) is bottlenecked by loading precomputed KV caches sequentially from a single host. This paper investigates whether distributing KV cache chunks across a peer-to-peer (P2P) Kademlia network and fetching layer-group stripes in parallel can reduce this bottleneck. We profile KV cache memory for GPT-2 (73,728 bytes/token, 12 layers) and TinyLLaMA-1.1B (45,056 bytes/token, 22 layers, GQA), design a binary serialization format with optional compression, and select Kademlia as the DHT backbone based on its 100% lookup correctness and α-parallel iterative lookup. A three-stage simulation experiment (120 + 4,050 + 3,240 = 7,410 total runs) evaluates 12 chunk-splitting strategies across network sizes of 32–128 peers, replication factors of 1–3, churn rates of 0–15%, and context lengths of 128–1,024 tokens. The optimal configuration — token\_block=32, layer\_group=2 — achieves a confirmatory p95 retrieval latency of **4.911 ms ± 0.032 ms** at **99.56% chunk availability**, a 26% improvement over the next-best variant and 3–7× improvement over one-dimensional splitting strategies. All results are stable across the full parameter sweep. A single-machine P2P prototype with five implementation milestones is fully specified and ready for construction; its success criterion is a statistically significant p95 TTFT reduction over a sequential-load baseline across 200 queries.

---

## 1. Introduction

Retrieval-augmented generation augments LLM inference by prepending relevant document context to each query. In systems that cache the KV representations of documents offline (e.g., RAGCache, CacheBlend), the TTFT for a RAG query consists of: (1) retrieval of relevant document IDs, and (2) loading the precomputed KV cache for those documents into GPU memory before decoding. On a single host, step (2) is a sequential disk or memory read bounded by local I/O bandwidth.

The central hypothesis of this work is: *if a document's KV cache is split into stripes held by different peers, fetching all stripes in parallel exploits aggregate network bandwidth and reduces the critical-path latency compared to a single sequential load.* The design challenge is identifying the right splitting granularity — too coarse and parallelism is lost; too fine and DHT routing overhead dominates.

**Contributions:**

1. **KV cache size profile:** TinyLLaMA-1.1B has 45,056 bytes/token (22.1 MB at a 512-token context); GPT-2 has 73,728 bytes/token (36.0 MB at 512 tokens). An 8 GiB budget accommodates ~190,650 TinyLLaMA tokens in KV cache.
2. **Binary serialization format:** A 64-byte chunk header with optional lz4/zstd compression. Real TinyLLaMA KV data compresses 5.8% with zstd at a 2.26× roundtrip time cost; lz4 offers negligible compression at 1.21× cost.
3. **DHT selection:** Kademlia outperforms Chord (100% vs 100% correctness, lower join cost) and dominates Pastry (100% vs 91.5% correctness). Its α-parallel iterative lookup maps directly to multi-stripe fetching.
4. **Optimal splitting strategy:** A three-stage simulation experiment (7,410 runs) confirms `token_block=32, layer_group=2` as Pareto-dominant: p95 = 4.911 ms ± 0.032 ms, 99.56% availability, robust across all tested conditions.
5. **P2P system design:** A five-milestone single-machine prototype specification with defined acceptance tests and a success criterion.

---

## 2. Background

### 2.1 KV Caching and RAG

Autoregressive LLMs compute key and value tensors for every token in the context. These tensors, collectively the KV cache, grow linearly with sequence length and model depth. In RAG, document context is fixed per document and can be prefilled offline; at query time, only the user's question needs new computation if the document KV cache is loaded intact. RAGCache [4] and CacheBlend [5] exploit this property on a single host. This work extends the paradigm to a P2P setting where each peer holds a layer-group stripe of a document's KV cache.

### 2.2 Related Systems

**RAGCache** [4] stores full KV caches on a single node; retrieval is sequential. **CacheBlend** [5] merges KV caches from multiple documents with a blending step to correct for positional interference. **TurboRAG** [6] encodes document chunks with a fixed dummy-position offset (Reordered-RoPE), making chunks position-invariant and eliminating the need for a stitching pass at retrieval time. Our design uses standard RoPE and requires a stitching correction at retrieval; the TurboRAG approach is noted as a fallback if the correction proves numerically unstable.

### 2.3 Kademlia DHT

Kademlia [1] routes lookups using XOR distance on a 160-bit key space. Each node maintains a k-bucket routing table partitioned by XOR distance. Lookups are *iterative* and *parallel*: α probes are issued simultaneously at each round, converging in O(log N) hops. The α-parallel structure means that fetching k stripes from k peers can be pipelined within a single lookup round, which is the key property exploited by the swarm fetcher in Section 3.3.

### 2.4 Rotary Position Embedding (RoPE)

TinyLLaMA uses RoPE, which encodes absolute positions into key and query tensors via rotation matrices. When a document chunk is prefilled at positions [0, L_doc), its keys carry those absolute rotations. To stitch the chunk into a query context at positions [L_query, L_query + L_doc), the keys must be reverse-rotated and re-rotated at the new positions. This stitching pass is the correctness-critical step in the P2P prototype (Section 8).

---

## 3. Design

### 3.1 DHT Protocol Selection

Three DHT protocols were evaluated in a virtual-time simulation (N=20 nodes, 200 random key lookups per protocol). The selection criterion prioritized lookup correctness first, then latency and join cost.

| Protocol | Mean hops | P95 hops | Max hops | Correctness | Join msgs | Routing entries |
|---|---|---|---|---|---|---|
| Chord | 1.90 | 4 | 4 | **100%** | 16.0 | 4.8 |
| **Kademlia** | **1.61** | **3** | **3** | **100%** | **2.4** | 12.5 |
| Pastry | 1.12 | 2 | 2 | 91.5% | 7.0 | 19.9 |

Pastry [3] achieves the lowest hop count but fails 8.5% of lookups at N=20 and degrades to 9% failure post-churn — unacceptable for a system where every missing stripe blocks decode. Chord [2] and Kademlia [1] both achieve 100% correctness; Kademlia is selected because its α-parallel iterative lookup (α=3 in experiments) issues multiple probes simultaneously, a property that maps directly to concurrent stripe fetching. Chord's sequential ring traversal does not offer this.

Source: `dht_comparison/kademlia.py`, `dht_comparison/chord.py`, `dht_comparison/pastry.py`.

### 3.2 KV Cache Serialization Format

Each chunk is serialized as a 64-byte header followed by a raw payload:

| Field | Size | Description |
|---|---|---|
| Magic bytes | 4 B | `KVC1` — format identifier |
| Version | 1 B | Format version |
| Compression codec | 1 B | 0=none, 1=lz4, 2=zstd |
| Data type | 1 B | torch dtype enum |
| Token start / end | 8 B | Inclusive range |
| Layer start / end | 8 B | Inclusive range |
| Tensor shape | 16 B | (layers, heads, tokens, head_dim) |
| MD5 checksum | 16 B | Checksum of uncompressed payload |
| Reserved | 9 B | Future use |

The chunk ID encodes the provenance: `m=<model>|ts=<tok_start>|te=<tok_end>|ls=<lay_start>|le=<lay_end>`. Default splitting parameters — `token_block=32`, `layer_group=2` — were determined by the simulation experiment in Section 6.3 and are applied in `kv_serialization/serialize.py`.

Compression was evaluated empirically on real TinyLLaMA KV data (Section 6.2). lz4 is available as an optional fast path; zstd is available for bandwidth-constrained deployments.

Source: `kv_serialization/format.py`, `kv_serialization/serialize.py`, `kv_serialization/compress.py`.

### 3.3 P2P System Architecture

The full prototype consists of five components:

```
Query Node
  ├── RAG Retriever  ──► top-k doc IDs
  └── Swarm Fetcher  ──► Peer 0: layers  0–5   (stripe 0)
                    ──► Peer 1: layers  6–11  (stripe 1)   ──► RoPE Stitcher ──► LLM Decode
                    ──► Peer 2: layers 12–17  (stripe 2)
                    ──► Peer 3: layers 18–21  (stripe 3)
```

Each peer serves a single layer-group stripe for each cached document via a lightweight HTTP endpoint (`GET /chunk?doc_id=X&layer_group=Y`). The swarm fetcher issues all four requests concurrently via `asyncio`/`aiohttp` and reassembles stripes in layer order. The RoPE stitcher then corrects position encodings before handing the assembled KV cache to the LLM decoder, skipping the full prefill pass.

The speculative prefetcher (M4) pre-fetches predicted top-k document stripes into a 16-entry LRU cache while a prior request is being decoded, hiding fetch latency under inference time.

Planned deliverables: `scripts/peer_server.py` (M1), `kv_kademlia_experiments/swarm_fetcher.py` (M2), `kv_kademlia_experiments/rope_stitcher.py` (M3), `kv_kademlia_experiments/speculative_prefetcher.py` (M4), `scripts/benchmark_e2e_ttft.py` (M5).

---

## 4. Environment Setup and Running the Code

### 4.1 Prerequisites

- Python 3.10 or later. CUDA is optional; all scripts fall back to CPU.
- Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix / macOS
python -m venv venv
source venv/bin/activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

The planned P2P system requires two additional packages not yet in `requirements.txt`:

```bash
pip install aiohttp fastapi uvicorn
```

### 4.2 Running Each Script

| Script | Command | Expected runtime | Output location |
|---|---|---|---|
| GPT-2 KV profiling | `python scripts/kv_cache_phase1_gpt2.py` | ~2 min (CPU) | `phase1_outputs/` |
| TinyLLaMA KV profiling | `python scripts/kv_cache_phase2_tinyllama.py` | ~5 min (CPU) | `phase2_outputs/` |
| Real KV compression benchmark | `python scripts/benchmark_real_kv_compression.py` | ~3 min | `benchmark_outputs/` |
| Serialization benchmark | `python scripts/benchmark_serialization.py` | <1 min | `benchmark_outputs/` |
| Kademlia splitting experiment | `python scripts/experiment_tinyllama_kv_kademlia.py` | Stage A ~20 min · B ~4 h · C ~8 h | `kv_kademlia_experiments/stage_{a,b,c}_final/` |
| Kademlia latency visualization | `python scripts/viz_kademlia_latency.py` | <1 min | `figures/` |

The Kademlia experiment script runs all three stages sequentially by default. Individual stages can be run by passing `--stage a`, `--stage b`, or `--stage c`. Stages B and C read the winner from the prior stage's `variant_ranking.csv` automatically.

### 4.3 Planned P2P System Commands

Once M1–M5 are implemented, the full system runs as follows:

```bash
# Start four peer processes (one per layer-group stripe)
python scripts/peer_server.py --port 5001 --layer-group 0   # layers 0–5
python scripts/peer_server.py --port 5002 --layer-group 1   # layers 6–11
python scripts/peer_server.py --port 5003 --layer-group 2   # layers 12–17
python scripts/peer_server.py --port 5004 --layer-group 3   # layers 18–21

# Run the end-to-end TTFT benchmark
python scripts/benchmark_e2e_ttft.py
```

Output: `P2P_RAGCACHE_RESULTS.md` and a companion CSV with per-query TTFT measurements.

---

## 5. Testing

### 5.1 Running the Test Suite

```bash
pytest tests/ -v -s
```

All 54 tests complete in approximately 2 seconds.

| Test file | Coverage | Tests |
|---|---|---|
| `tests/test_protocols.py` | Chord, Kademlia, and Pastry correctness; join/leave; churn resilience | 27 |
| `tests/test_serialization.py` | Round-trip serialize/deserialize; lz4/zstd compression; MD5 integrity; reassembly | 21 |
| `tests/benchmark_kademlia_latency.py` | O(log N) hop-count scaling with virtual-time clock | 6 |

### 5.2 Planned Acceptance Tests for P2P Components

Each milestone has a defined pass/fail criterion:

- **M1 (peer server):** A `requests.get('http://localhost:5001/chunk?doc_id=0&layer_group=0')` returns the correct bytes; they round-trip through the existing deserializer to a valid tensor with matching MD5.
- **M2 (swarm fetcher):** p95 fetch latency over 100 requests falls within one order of magnitude of the simulation result (< 50 ms on localhost).
- **M3 (RoPE stitcher):** Cosine similarity between the stitched-KV forward-pass logits and a reference full-prefill forward pass exceeds 0.99 on the final hidden state (fp16 tolerance).
- **M5 (TTFT benchmark):** p95 TTFT is lower than the sequential-load baseline at p < 0.05 (Mann–Whitney U test) across ≥ 200 queries.

---

## 6. Results

### 6.1 KV Cache Size Profile

KV cache memory scales linearly with sequence length in both models. TinyLLaMA uses grouped-query attention (GQA) with 4 KV heads instead of 32 full heads, which is the primary reason its per-token cost is 61.1% of GPT-2 despite having nearly twice the layers.

| Model | Layers | KV heads | Bytes/token | @ 128 tokens | @ 512 tokens | @ 1,024 tokens |
|---|---|---|---|---|---|---|
| GPT-2 | 12 | 12 | 73,728 | 9.44 MB | 36.0 MB | 72.0 MB |
| TinyLLaMA-1.1B | 22 | 4 (GQA) | 45,056 | 5.77 MB | 22.1 MB | 44.1 MB |
| Ratio (LLaMA/GPT-2) | — | — | **0.611** | 0.611 | 0.611 | 0.611 |

At 512 tokens — the representative RAG document chunk size used in all subsequent experiments — one TinyLLaMA document's KV cache occupies **22.1 MB uncompressed**. An 8 GiB GPU memory budget accommodates approximately 190,650 tokens (≈ 373 document chunks of 512 tokens each) in KV cache alone.

Data sources: `phase1_outputs/gpt2_kv_stats.csv`, `phase2_outputs/tinyllama_kv_stats.csv`.

### 6.2 Compression

Compression was evaluated on actual TinyLLaMA KV tensors extracted from real inference runs. KV cache data is largely random-looking (floating-point activations), which limits compression effectiveness.

| Sequence length | Codec | Raw bytes | Compressed bytes | Ratio | Roundtrip time |
|---|---|---|---|---|---|
| 512 tokens | none | 23,068,672 | 23,079,936 | 1.000× | 83.0 ms |
| 512 tokens | lz4 | 23,068,672 | 23,061,956 | 1.001× | 100.2 ms |
| 512 tokens | zstd | 23,068,672 | 21,743,156 | **1.061×** | 187.4 ms |

**Key finding:** zstd achieves a consistent **1.061× compression ratio** (5.8% size reduction) across all tested sequence lengths (32–512 tokens), at the cost of a **2.26× roundtrip time overhead** (187 ms vs 83 ms at 512 tokens). lz4 offers effectively zero compression on this data type. Neither codec is recommended by default; zstd is a viable option in bandwidth-constrained deployments where the size reduction outweighs the latency penalty.

Data source: `benchmark_outputs/real_kv_compression.csv`.

### 6.3 Kademlia KV Splitting: Three-Stage Simulation

The core experiment evaluates 12 chunk-splitting strategies by simulating KV cache placement and retrieval on a Kademlia network. Simulation parameters: 16-bit ID space, k-bucket size 8, α=3, per-hop delay 1.0 ms, base transfer time 0.2 ms, bandwidth 100 Mbps. Strategies are scored as: 60% × (1 − normalized p95 latency) + 25% × chunk availability + 15% × (1 − overhead ratio).

#### Stage A: Screening (120 runs)

12 strategy variants were screened at N=64, R=2, churn ∈ {0%, 5%}, L=512, 5 random seeds. One-dimensional strategies were immediately eliminated:

| Strategy | p95 latency | Chunk success | Weighted score |
|---|---|---|---|
| **token\_layer\_group (winner)** | **8.43 ms** | **99.92%** | **0.750** |
| layer\_only | 27.44 ms | 100.0% | 0.442 |
| token\_only | 36.36 ms | 100.0% | 0.250 |

Token-only splitting produces the largest chunks (fewer but larger stripes), saturating the 100 Mbps transfer model. Layer-only splitting produces many small stripes that each require independent routing, multiplying DHT overhead. The 2D token+layer-group family advanced to Stage B with four hyperparameter variants.

#### Stage B: Full Parameter Sweep (4,050 runs)

The top five variants from Stage A were evaluated across the full parameter space: N ∈ {32, 64, 128}, R ∈ {1, 2, 3}, churn ∈ {0%, 5%, 15%}, L ∈ {128, 512, 1,024}, 10 seeds.

| Variant | p95 latency | Availability | Weighted score |
|---|---|---|---|
| **tb=32, lg=2** | **4.92 ms** | **99.66%** | **0.850** |
| tb=32, lg=4 | 6.20 ms | 99.28% | 0.734 |
| tb=64, lg=2 | 6.28 ms | 99.15% | 0.712 |
| tb=64, lg=4 | 8.82 ms | 99.11% | 0.551 |
| layer\_only, lb=1 | 14.84 ms | 98.75% | 0.150 |

The `tb=32, lg=2` variant wins on both latency and availability. Smaller token blocks (32 vs 64) reduce per-stripe transfer time; finer layer groups (lg=2 = 11 parallel stripes for TinyLLaMA's 22 layers) maximize the critical-path reduction from parallel fetching.

#### Stage C: Confirmatory (3,240 runs)

The top two variants were re-evaluated at 20 seeds to obtain tight confidence intervals.

| Variant | p50 | p95 | p99 | Availability | 95% CI on p95 |
|---|---|---|---|---|---|
| **tb=32, lg=2** | **3.779 ms** | **4.911 ms** | **5.293 ms** | **99.56%** | **±0.032 ms** |
| tb=32, lg=4 | 4.994 ms | 6.190 ms | 6.558 ms | 99.19% | ±0.032 ms |

The `tb=32, lg=2` configuration is confirmed as Pareto-dominant: **26% lower p95 latency** than the runner-up, **37 basis points higher availability**, and non-overlapping confidence intervals.

#### Sensitivity Analysis

The winner's p95 latency was stable across all parameter dimensions tested in Stage B and confirmed in Stage C:

| Parameter | Range | p95 range (tb=32, lg=2) | Interpretation |
|---|---|---|---|
| Network size N | 32 → 128 | 4.27 → 5.54 ms | Sub-linear growth; consistent with O(log N) Kademlia routing |
| Replication R | 1 → 3 | 5.02 → 4.82 ms | Modest ~4% improvement; diminishing returns past R=2 |
| Churn | 0% → 15% | 5.01 → 4.81 ms | Stable; spare replication absorbs node loss |
| Sequence length L | 128 → 1,024 | 5.15 → 4.72 ms | Slight improvement; larger chunks amortize per-hop routing cost |

The stability across churn rates (latency *improves* slightly at higher churn) is a simulation artifact: with R=2, spare replicas are always available, and the Kademlia routing naturally selects the closer replica as peers leave. This will be re-evaluated in the real-system prototype.

![Figure 1: p95 retrieval latency by variant (Stage C, 3,240 runs). tb=32 lg=2 has non-overlapping confidence intervals with all other variants.](results/kv_kademlia/stage_c_final/figures/latency_p95_by_variant.png)

![Figure 2: p95 latency vs network size N for the top two variants. Sub-linear growth is consistent with O(log N) Kademlia routing.](results/kv_kademlia/stage_c_final/figures/latency_vs_num_peers_by_variant.png)

![Figure 3: Chunk availability vs churn rate by replication factor. Both variants remain above 99% availability across all tested churn conditions.](results/kv_kademlia/stage_c_final/figures/availability_vs_churn.png)

![Figure 4: Mean Kademlia lookup latency vs network size with O(log N) reference curve, confirming expected DHT scaling behavior.](results/figures/kademlia_latency_scaling.png)

---

## 7. Discussion

The simulation results establish a clear hierarchy among splitting strategies. The 2D (token + layer-group) family dominates 1D approaches by 3–7× in p95 latency because it creates stripes that are simultaneously (a) small enough to transfer quickly and (b) numerous enough to fill the α-parallel Kademlia routing pipeline. Within the 2D family, the performance difference between `lg=2` and `lg=4` (4.91 vs 6.19 ms) suggests that finer layer grouping increases DHT routing overhead faster than it reduces per-stripe transfer time, so there is an optimal granularity that the simulated 100 Mbps model places at `lg=2`.

One caveat applies to the churn stability finding: the simulation assigns fixed spare replicas and assumes instantaneous failover routing. A real system with correlated failures or cold-cache peers may show different behavior. The speculative prefetcher (M4) is designed to partially mitigate this by pre-loading stripes during the prior decode window.

---

## 8. Future Work: P2P RAGCache Prototype

### 8.1 Gap Between Simulation and Publication

The simulation establishes that `tb=32, lg=2` is the right splitting strategy. What remains is a real TTFT measurement: does the latency reduction in simulation (sub-5ms retrieval) translate to a measurable TTFT improvement when the full pipeline — network I/O, RoPE stitching, and LLM decode — is exercised end-to-end?

### 8.2 Implementation Milestones

| Milestone | Deliverable | Depends on | Duration |
|---|---|---|---|
| M1: Peer server | `scripts/peer_server.py` | — | 1–2 days |
| M2: Swarm fetcher | `kv_kademlia_experiments/swarm_fetcher.py` | M1 | 1 day |
| M3: RoPE stitcher | `kv_kademlia_experiments/rope_stitcher.py` | M2 | 2–3 days |
| M4: Speculative prefetcher | `kv_kademlia_experiments/speculative_prefetcher.py` | M2 | 1 day (parallel to M3) |
| M5: TTFT benchmark | `scripts/benchmark_e2e_ttft.py` | M3, M4 | 2 days |

M1 → M2 → M3 → M5 must be sequential. M4 can be built in parallel with M3 and merged before M5.

### 8.3 Benchmark Protocol

1. Pre-cache KV tensors for 50 document chunks (TinyLLaMA forward passes on representative text, serialized with the existing pipeline).
2. Distribute shards to four peer processes (one per layer-group stripe).
3. Issue 200 RAG queries sequentially; each retrieves 1–3 documents.
4. Record `ttft_baseline_ms` (sequential single-file load) and `ttft_p2p_ms` (swarm fetch + stitch) per query.
5. Report p50, p95, p99, and mean for both conditions; test significance with Mann–Whitney U (α=0.05).

**Success criterion:** p95 TTFT is statistically significantly lower in the P2P condition than the sequential baseline across ≥ 200 queries.

### 8.4 Primary Risk

The RoPE stitching inverse (reverse-rotating key tensors and re-applying rotations at shifted positions) may accumulate floating-point error. If cosine similarity between stitched and reference logits falls below 0.99 in the M3 acceptance test, the fallback is to adopt TurboRAG's fixed-offset encoding: document chunks are prefilled with positions `[offset, offset + L_doc)` instead of `[0, L_doc)`, where `offset` is a fixed large constant. This makes chunks position-invariant at storage time and eliminates the need for an inverse transform at retrieval time.

---

## 9. Conclusion

This paper demonstrates that two-dimensional KV cache splitting — by token block and layer group simultaneously — outperforms one-dimensional alternatives by 3–7× in P2P retrieval latency. The specific configuration `token_block=32, layer_group=2` achieves a confirmed p95 latency of 4.911 ms ± 0.032 ms at 99.56% chunk availability across 7,410 simulation runs spanning the full parameter space of practical deployment conditions. The result is robust: it holds across network sizes of 32–128 peers, replication factors of 1–3, churn rates up to 15%, and document context lengths up to 1,024 tokens.

Kademlia is the right DHT backbone for this system: its 100% lookup correctness and α-parallel iterative lookup directly enable the multi-stripe concurrent fetching that produces the latency reduction. Among all evaluated protocols, it is the only one that combines correctness guarantees with an architecture that maps naturally to parallel data retrieval.

The remaining open question — whether this simulation result translates to a measurable TTFT improvement in a real system — is the subject of the specified P2P RAGCache prototype (Section 8). If the real TTFT improvement matches the simulation's retrieval latency improvement, the system contribution would be a concrete, empirically validated approach to reducing inference latency in retrieval-augmented LLM deployments.

---

## References

[1] Petar Maymounkov and David Mazières. 2002. Kademlia: A peer-to-peer information system based on the XOR metric. In *Proc. 1st Int'l Workshop on Peer-to-Peer Systems (IPTPS '02)*.

[2] Ion Stoica, Robert Morris, David Karger, M. Frans Kaashoek, and Hari Balakrishnan. 2001. Chord: A scalable peer-to-peer lookup service for internet applications. In *Proc. ACM SIGCOMM '01*, 149–160.

[3] Antony Rowstron and Peter Druschel. 2001. Pastry: Scalable, decentralized object location, and routing for large-scale peer-to-peer systems. In *Proc. IFIP/ACM Middleware '01*, 329–350.

[4] Yuxin Jin et al. 2024. RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation. *arXiv:2404.12457*.

[5] Jiayi Yao et al. 2024. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. *arXiv:2405.16444*.

[6] Bin Lu et al. 2024. TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text. *arXiv:2410.23343*.

[7] Peiyuan Zhang et al. 2024. TinyLlama: An Open-Source Small Language Model. *arXiv:2401.02385*.

[8] Woojae Kwon et al. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In *Proc. ACM SOSP '23*.

[9] Thomas Wolf et al. 2020. HuggingFace Transformers: State-of-the-art Natural Language Processing. In *Proc. EMNLP '20*.

---

## Appendix

### A. Experiment Parameter Grids

**Stage A** (120 runs):

| Parameter | Values |
|---|---|
| Strategies | token\_layer\_group (8 variants), layer\_only (2), token\_only (2) |
| Network size N | 64 |
| Replication R | 2 |
| Churn | 0%, 5% |
| Sequence length L | 512 |
| Seeds | 5 (seeds 0–4) |

**Stage B** (4,050 runs):

| Parameter | Values |
|---|---|
| Variants | token\_layer\_group tb=32 lg=2, tb=32 lg=4, tb=64 lg=2, tb=64 lg=4; layer\_only lb=1 |
| Network size N | 32, 64, 128 |
| Replication R | 1, 2, 3 |
| Churn | 0%, 5%, 15% |
| Sequence length L | 128, 512, 1,024 |
| Seeds | 10 (seeds 0–9) |

**Stage C** (3,240 runs):

| Parameter | Values |
|---|---|
| Variants | token\_layer\_group tb=32 lg=2, tb=32 lg=4 |
| N, R, Churn, L | Same as Stage B |
| Seeds | 20 (seeds 0–19) |

### B. Serialization Format Header Layout

```
Offset  Size   Field
------  ----   -----
0       4 B    Magic: 0x4B564331 ("KVC1")
4       1 B    Version (current: 1)
5       1 B    Compression: 0=none, 1=lz4, 2=zstd
6       1 B    Data type (torch dtype enum)
7       1 B    Reserved
8       4 B    Token start (uint32)
12      4 B    Token end (uint32)
16      4 B    Layer start (uint32)
20      4 B    Layer end (uint32)
24      16 B   Tensor shape: (layers, heads, tokens, head_dim) as 4×uint32
40      16 B   MD5 checksum of uncompressed payload
56      8 B    Reserved
64      —      Payload (compressed or raw tensor bytes)
```

Source: `kv_serialization/format.py`.
