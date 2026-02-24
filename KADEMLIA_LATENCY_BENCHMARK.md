# Kademlia Virtual-Time Latency Benchmark

This document describes the virtual-time simulation added to verify O(log N) lookup performance for Kademlia, and reports the benchmark test results.

---

## 1. Update Summary

### 1.1 Virtual time in the simulator

**File: `dht_comparison/base.py`**

- **`NetworkSimulator(per_hop_delay=0.0)`** — Constructor now accepts an optional `per_hop_delay`. Default `0` preserves previous behavior (no virtual time).
- **`virtual_time`** — Float that accumulates simulated time. Starts at 0.
- **`advance_time(delta=None)`** — Advances `virtual_time` by `delta`; if `delta` is omitted, uses `per_hop_delay`.

When `per_hop_delay > 0`, protocols can call `advance_time()` on each logical hop so that the increase in `virtual_time` during a lookup represents **simulated latency** (e.g. one RTT per round).

### 1.2 Kademlia integration

**File: `dht_comparison/kademlia.py`**

- In **`lookup()`**, after each iterative round (`hops += 1`), the node calls `network.advance_time()` if the network has `per_hop_delay > 0`.
- Implemented with `getattr()` so that existing code using a plain `NetworkSimulator()` is unchanged.

### 1.3 New benchmark tests

**File: `tests/benchmark_kademlia_latency.py`**

- **Kademlia only.** Builds networks with `NetworkSimulator(per_hop_delay=1.0)`.
- **`test_kademlia_simulated_latency_olog_n`** — Parametrized over **N ∈ {10, 20, 50, 100, 200}**:
  - For each N, runs 200 lookups.
  - Simulated latency per lookup = `virtual_time` after lookup minus `virtual_time` before.
  - Asserts: **mean simulated latency ≤ 2.5 × log₂(N) × RTT** (RTT = 1 virtual time unit).
- **`test_kademlia_latency_scales_sublinearly`** — Runs the same N values and checks that the growth in mean latency from N=10 to N=200 is consistent with O(log N).

---

## 2. How to run

From the project root:

```bash
python -m pytest tests/benchmark_kademlia_latency.py -v -s
```

- `-v` — Verbose test names.
- `-s` — Show print output (mean latency, bounds, etc.).

---

## 3. Test results

Run environment: Windows, Python 3.13, pytest 9.0.2. All 6 tests passed in ~1.35 s.

### 3.1 Per-N bounds (test_kademlia_simulated_latency_olog_n)

| N   | Mean latency | log₂(N) | Bound (2.5×log₂(N)) | Median | Max |
|-----|--------------|---------|----------------------|--------|-----|
| 10  | 1.04         | 3.32    | 8.3                  | 1.00   | 2   |
| 20  | 1.61         | 4.32    | 10.8                 | 1.00   | 3   |
| 50  | 2.50         | 5.64    | 14.1                 | 2.50   | 5   |
| 100 | 2.90         | 6.64    | 16.6                 | 3.00   | 5   |
| 200 | 3.51         | 7.64    | 19.1                 | 4.00   | 6   |

*Latency is in virtual time units (1 unit = 1 RTT per lookup round).*

### 3.2 Scalability summary (test_kademlia_latency_scales_sublinearly)

```
Kademlia simulated latency vs N (virtual time units):
  N= 10:  mean=1.04  median=1.00  log2(N)=3.32
  N= 20:  mean=1.61  median=1.00  log2(N)=4.32
  N= 50:  mean=2.50  median=2.50  log2(N)=5.64
  N=100:  mean=2.90  median=3.00  log2(N)=6.64
  N=200:  mean=3.51  median=4.00  log2(N)=7.64
```

Mean latency grows sublinearly with N and stays below the O(log N) bound in all cases. The sublinearity test (ratio N=200 / N=10) also passed.

---

## 4. Conclusion

- Virtual time is implemented in the base simulator and used by Kademlia to simulate one RTT per lookup round.
- The Kademlia-only benchmark confirms that **mean simulated latency scales as O(log N)** for N = 10, 20, 50, 100, and 200, consistent with the protocol’s design.

Existing tests (`tests/test_protocols.py`) remain unchanged and pass with the default `per_hop_delay=0`.
