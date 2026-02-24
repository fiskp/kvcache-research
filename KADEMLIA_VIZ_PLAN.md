## Kademlia Latency Visualization Tool – Design Plan

### 1. Goals and Scope

- **Primary goal**: Provide an in-process visualization tool for the Kademlia DHT that makes the **virtual-time latency benchmark** intuitive:
  - Show that **simulated lookup latency scales as O(log N)**.
  - Reveal how Kademlia’s **routing structure and lookup paths** drive that behaviour.
- **Scope (v1)**:
  - **Kademlia-only**, using the existing `NetworkSimulator` with virtual time.
  - Read-only, built **on top of** the current implementation (no protocol changes).
  - Implemented as a **Python script or Jupyter notebook** that:
    - Runs the Kademlia benchmark across multiple N.
    - Collects detailed lookup traces.
    - Produces static plots (matplotlib / seaborn).
- **Out of scope (v1)**:
  - Chord/Pastry visualizations.
  - Real network stack, multi-process simulation, or interactive web UI.

### 2. Data to Collect

For each network size \(N\) (e.g. 10, 20, 50, 100, 200), the tool will:

- Build a Kademlia network with:
  - `NetworkSimulator(per_hop_delay=1.0)`
  - Stable state after the same stabilization rounds as in tests.
- For each lookup (e.g. 200 lookups per N), record:
  - **Global parameters**
    - `N` — network size.
    - `id_bits` — ID space size (16 by default).
  - **Per-lookup metrics**
    - `key` — lookup key (int in ID space).
    - `initiator_id` — node ID that started the lookup.
    - `latency` — `t1 - t0` using `network.virtual_time` before/after lookup.
    - `hop_count` — `LookupResult.hop_count`.
    - `path` — `LookupResult.path` (sequence of node IDs visited).
  - **Optional structural snapshot (per N or per sampled lookup)**
    - For a selected node:
      - `buckets` — sizes and contents of `node.buckets`.
      - Possibly a few example node IDs per bucket.

The tool may hold this in memory (list of dicts) or persist to JSON/CSV for offline analysis.

### 3. Visualizations

#### 3.1 Latency vs N (Scaling Plot)

- **What**: Show how **mean simulated latency** grows with N.
- **How**:
  - X-axis: `N` (10, 20, 50, 100, 200).
  - Y-axis: `mean_latency` (virtual time units).
  - Overlay reference curve \(C \cdot \log_2 N\) using the same coefficient as the benchmark (e.g. `COEF = 2.5`).
- **Purpose**: Visual confirmation that latency growth is consistent with O(log N).

#### 3.2 Latency / Hop Distributions per N

- **What**: Distribution of `latency` and `hop_count` for each N.
- **How**:
  - Histograms or boxplots:
    - N on the x-axis.
    - Latency or hop count on the y-axis.
  - One subplot per N or grouped boxplots.
- **Purpose**: Show spread and tails, not just the mean; surface outliers or heavy tails.

#### 3.3 Single-Lookup Path Visualization

- **What**: Visualize one (or a few) lookup paths as they traverse the network.
- **How**:
  - Choose a representative key and initiator for each N.
  - Use `path` and `hop_count` to reconstruct the sequence of nodes.
  - Lay nodes on a line (0 to \(2^{id\_bits}-1\)) or a circle by node ID.
  - Draw arrows from each node to the next in the path:
    - Color-coded by hop index.
    - Annotate with virtual time after that hop (round index).
- **Purpose**: Intuitively show how Kademlia “zooms in” on the key in a few RTTs.

#### 3.4 Routing Table (k-Bucket) View

- **What**: Snapshot of the routing table for a selected node.
- **How**:
  - For a chosen node, inspect `node.buckets`.
  - Visualize:
    - Bucket index \(i\) on the y-axis.
    - Either:
      - Bucket size as a bar chart, or
      - A heatmap/table with example node IDs per bucket.
- **Purpose**: Connect the theoretical k-bucket structure to its concrete instantiation and the observed latency.

### 4. Implementation Approach

#### 4.1 Technology Choices

- **Language**: Python (same as the DHT implementation).
- **Visualization**: matplotlib and/or seaborn.
- **Form factor**:
  - Option A: CLI script, e.g. `scripts/viz_kademlia_latency.py`.
  - Option B: Jupyter notebook, e.g. `notebooks/kademlia_latency_viz.ipynb`.

#### 4.2 High-Level Workflow

1. **Benchmark run**
   - For each N in a predefined list:
     - Build the network with `NetworkSimulator(per_hop_delay=1.0)`.
     - Run a fixed number of lookups.
     - Record metrics and structures as described in Section 2.
2. **Data aggregation**
   - Group by N, compute summary statistics:
     - Mean, median, p95 latency and hop count.
3. **Plotting**
   - Generate the visualizations from Section 3 into:
     - On-screen windows (interactive), and/or
     - Saved image files (e.g. `figures/kademlia_latency_N.png`).

### 5. Future Extensions (Optional)

- Add equivalent visualizations for **Chord** and **Pastry** for cross-protocol comparison.
- Introduce simple interactivity (e.g. sliders in a notebook) to:
  - Select N, key, and initiator for path visualization.
  - Toggle between viewing hops vs latency.
- Export data and figures to integrate into reports or papers.

