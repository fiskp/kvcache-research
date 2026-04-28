# P2P-RAGCache System — Implementation Plan

## 1. Objective

This phase builds a working single-machine prototype of a decentralized RAG KV
cache system, directly applying findings from the splitting strategy experiments.
The core claim to validate is:

> Fetching a single document's KV cache by pulling disjoint layer-group stripes
> from multiple peers in parallel reduces TTFT compared to loading the full cache
> sequentially from a single local source.

The prototype establishes this empirically with measurable TTFT numbers, making
the simulation results from the Kademlia splitting experiment into a concrete
system contribution.

**Success condition:** End-to-end TTFT for a RAG query using the P2P cache is
lower than the sequential-load baseline by a statistically significant margin,
measured at p50, p95, and p99 across at least 200 queries.

---

## 2. Background and Motivation

Prior work established:

- **KV cache profile:** TinyLLaMA has 22 layers, 4 GQA KV heads, ~45 KB/token.
  At a 512-token context, one document chunk's KV cache is ~22 MB uncompressed,
  ~17 MB with zstd (from `benchmark_outputs/real_kv_compression.csv`).
- **Splitting winner:** Token + layer-group (S1) achieved 8.4 ms p95 retrieval
  latency vs. 27.4 ms (token-only) and 36.4 ms (layer-only) in the Kademlia
  simulation.
- **Serialization format:** The existing pipeline in `kv_serialization/`
  already chunks, compresses, and reassembles KV tensors.

The gap between the simulation and a publishable result is: real multi-process
peers, a correct RoPE-offset stitching pass, and an end-to-end TTFT measurement.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Query Node                                                      │
│  ┌──────────┐   top-k doc IDs   ┌───────────────────────────┐   │
│  │  RAG     │ ────────────────► │  Swarm Fetcher            │   │
│  │  Retriev.│                   │  (parallel stripe GETs)   │   │
│  └──────────┘                   └─────────┬─────────────────┘   │
│                                           │ stripe tensors       │
│                                           ▼                      │
│                                  ┌─────────────────┐            │
│                                  │  RoPE Stitcher  │            │
│                                  │  (offset apply) │            │
│                                  └────────┬────────┘            │
│                                           │ stitched KV          │
│                                           ▼                      │
│                                  ┌─────────────────┐            │
│                                  │  LLM Decode     │            │
│                                  │  (skip prefill) │            │
│                                  └─────────────────┘            │
└──────────────────────────────────────────────────────────────────┘

Peer processes (same machine, separate ports):
  Peer 0 — holds layers  0–5   for each cached document chunk
  Peer 1 — holds layers  6–11
  Peer 2 — holds layers 12–17
  Peer 3 — holds layers 18–21
```

Each peer is a lightweight Python process exposing a single `GET /chunk` HTTP
or socket endpoint. The swarm fetcher issues all four requests concurrently and
assembles them through the stitcher before handing the KV cache to the model.

---

## 4. Implementation Milestones

### M1 — Peer Server (1–2 days)

Build a minimal per-peer store. Each peer:
- Loads a pre-serialized chunk shard from disk on startup.
- Exposes a `GET /chunk?doc_id=X&layer_group=Y` endpoint.
- Returns the raw bytes (zstd-compressed) from `kv_serialization/`.

**Deliverable:** `scripts/peer_server.py` — runnable as
`python scripts/peer_server.py --port 5001 --layer-group 0`.

Acceptance test: a single `curl` or `requests.get` returns the correct bytes
and round-trips to a valid tensor via the existing deserializer.

---

### M2 — Swarm Fetcher (1 day)

A client that, given a `doc_id`, fires requests to all four peers concurrently
using `asyncio` + `aiohttp` (or `concurrent.futures.ThreadPoolExecutor` as a
simpler fallback), collects the stripe responses, and reassembles them in layer
order.

**Deliverable:** `kv_kademlia_experiments/swarm_fetcher.py`

Metrics to emit per fetch:
- `stripe_latency_ms` per peer (time from request send to last byte received)
- `total_fetch_ms` (wall time from first request to last byte, i.e. the
  critical path = max of stripe latencies)
- `bytes_received` per stripe

Acceptance test: fetch latency distribution from 100 requests matches the
8.4 ms p95 order of magnitude seen in simulation.

---

### M3 — RoPE Stitching Pass (2–3 days)

This is the correctness-critical component. When document chunks arrive from
peers they carry positions `[0, L_doc)`. The query context occupies positions
`[0, L_query)`. The stitched KV must be shifted so document positions begin at
`L_query`.

**Approach:**
1. For each layer's key tensor, extract the original positions embedded in the
   rotary encoding (reverse-apply RoPE to recover raw queries/keys, re-apply
   with shifted positions).
2. For TinyLLaMA, use the `LlamaRotaryEmbedding` implementation from
   HuggingFace as a reference for the inverse transform.
3. Implement as `kv_kademlia_experiments/rope_stitcher.py` with a function
   `stitch(kv_stripes: list[KVChunk], query_len: int) -> past_key_values`.

**Deliverable:** `kv_kademlia_experiments/rope_stitcher.py`

Acceptance test: run one forward pass with a stitched KV cache and confirm
that the model's next-token logits match a reference run that prefilled the
full concatenated sequence from scratch (within fp16 tolerance, e.g. cosine
similarity > 0.99 on the final hidden state).

**Note on TurboRAG:** TurboRAG's Reordered-RoPE sidesteps this by making
chunks position-invariant at encode time. If the stitching inverse is
numerically unstable, fall back to this approach: re-encode document chunks
with a fixed dummy-position offset so that no inverse is needed at retrieval
time. Document whichever path is taken.

---

### M4 — Speculative Prefetch (1 day)

While the query is waiting in the inference queue, begin fetching the predicted
top-k document KV stripes in the background.

**Approach:**
- After the retriever returns doc IDs, immediately submit swarm fetch tasks
  to a background thread pool.
- By the time the previous request finishes and this one reaches the GPU, the
  stripes may already be in host memory.
- Cache fetched stripes in a small LRU (e.g. 16 entries) keyed by `doc_id`.

**Deliverable:** `kv_kademlia_experiments/speculative_prefetcher.py`

Metric: "prefetch hit rate" — fraction of requests where all stripes were
already in the LRU when the stitcher was called.

---

### M5 — End-to-End TTFT Benchmark (2 days)

Wire everything together and measure TTFT.

**Baseline:** Load the full document KV cache sequentially from a single file
on disk (no peers, no parallelism). This represents the existing single-host
approach (RAGCache / CacheBlend style).

**System:** Swarm fetch → stitch → decode.

**Protocol:**
1. Pre-cache KV tensors for 50 document chunks (run TinyLLaMA forward passes
   on representative text, serialize with the existing pipeline).
2. Distribute shards to peer processes.
3. Issue 200 RAG queries in sequence, each retrieving 1–3 documents.
4. Record: `ttft_baseline_ms` and `ttft_p2p_ms` per query.
5. Report p50, p95, p99, and mean for both conditions.

**Deliverable:** `scripts/benchmark_e2e_ttft.py` + output CSV +
`P2P_RAGCACHE_RESULTS.md` with tables and figures.

**Figures to produce:**
- CDF of TTFT: baseline vs. P2P
- Stripe latency breakdown (per-peer contribution to critical path)
- Prefetch hit rate vs. queue depth

---

## 5. File Layout

```
Research/
├── kv_kademlia_experiments/
│   ├── swarm_fetcher.py          # M2
│   ├── rope_stitcher.py          # M3
│   └── speculative_prefetcher.py # M4
├── scripts/
│   ├── peer_server.py            # M1
│   └── benchmark_e2e_ttft.py    # M5
└── P2P_RAGCACHE_RESULTS.md       # M5 output
```

Existing infrastructure reused without modification:
- `kv_serialization/` — chunk format, compress/decompress
- `phase2_outputs/` — TinyLLaMA KV profile used for sizing
- `benchmark_outputs/` — compression ratio data

---

## 6. Dependencies

All already in `requirements.txt` except:

| Package | Purpose | Add? |
|---|---|---|
| `aiohttp` | Async HTTP for swarm fetcher | Yes |
| `fastapi` + `uvicorn` | Peer server endpoint | Yes |

No new model weights needed — reuse `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| RoPE inverse is numerically unstable | Fall back to TurboRAG fixed-offset encoding at chunk store time |
| Localhost socket latency collapses the parallelism benefit | Report both localhost and simulated-WAN (add artificial delay via `asyncio.sleep`) results |
| LRU prefetch cache causes memory pressure | Cap at 16 entries × ~22 MB = ~350 MB; increase if hardware allows |
| Stitched KV produces wrong logits | Acceptance test in M3 catches this before the TTFT benchmark |

---

## 8. Execution Order

```
M1 (peer server)
    └── M2 (swarm fetcher)
             └── M3 (stitcher) ──► M5 (TTFT benchmark)
                                        ▲
                    M4 (speculative) ───┘
```

M1 → M2 → M3 must be sequential (each depends on the prior).
M4 can be developed in parallel with M3 and merged before M5.

---

## 9. Presentation Narrative

1. **Problem:** TTFT in RAG is bottlenecked by loading KV cache from a single
   local source.
2. **Insight from simulation:** Token + layer-group splitting achieves 8.4 ms
   p95 in a Kademlia network because it maximizes parallelism across peers.
3. **System:** A four-peer swarm fetcher that stripes a 22-layer KV cache by
   layer group, reassembled via a RoPE stitching pass.
4. **Result:** [to be filled from M5 output] p95 TTFT reduction vs. baseline.
5. **Conclusion:** Aggregate peer bandwidth can exceed single-disk I/O for
   realistic RAG document sizes, and token + layer-group is the right stripe
   unit to exploit this.
