## KV Cache Inspection — Phase 1 (GPT-2 Baseline)

### 1. Objective

Phase 1 establishes a **clean, reproducible baseline** for KV cache behaviour using a
small, well-understood GPT-like model. The goals are:

- **G1 — Access**: Verify that we can reliably extract the model’s KV cache
(`past_key_values`) during inference.
- **G2 — Structure**: Characterise how the cache is split across layers/heads/tokens:
tensor shapes, dtypes, and device placement.
- **G3 — Size**: Quantify the memory footprint of the KV cache as a function of
sequence length, batch size, and model configuration.

All of this is done on a **simple GPT-2 model** to minimise distractions from modern
architecture details (e.g. grouped query attention) and to de-risk the methodology.

---

### 2. Model and Stack Choices

**Model:** `gpt2` (~124M parameters) via HuggingFace `transformers`.

**Why GPT-2 for Phase 1:**

- **Simplicity:** Classic decoder-only transformer with straightforward multi-head
attention and no grouped-query tricks. Easier to reason about shapes and scaling.
- **Lightweight:** Runs comfortably on 16 GB RAM / 8 GB VRAM, even in FP32, so the
experiment is not bottlenecked by hardware.
- **Mature tooling:** `transformers` exposes `past_key_values` cleanly and there is
extensive documentation and prior art for GPT-2 introspection.

**Software stack (minimum):**

- Python 3.10+
- `torch`
- `transformers`
- `numpy`
- `matplotlib` (optional, for plots)

---

### 3. Experiment Design

#### 3.1 Basic KV Access

1. Load tokenizer and model:
  - `AutoTokenizer.from_pretrained("gpt2")`
  - `AutoModelForCausalLM.from_pretrained("gpt2")`
2. Tokenise a simple prompt (e.g. `"Hello world, this is a test prompt."`).
3. Run a single forward pass with:
  - `use_cache=True`
  - `return_dict=True`
4. Extract `outputs.past_key_values` and inspect:
  - Number of layers.
  - For each layer: shapes, dtypes, and devices of key/value tensors.

**Success condition:** We can print a concise summary such as:

> Layer 3: key shape = [1, 12, L, 64], value shape = [1, 12, L, 64], dtype=float32

for a variety of prompt lengths.

#### 3.2 Systematic Shape and Size Study

We study how the KV cache grows with:

- **Sequence length** L
- **Batch size** B (optional)

Steps:

1. Choose a set of sequence lengths, e.g. L \in 8, 32, 128, 512, 1024.
2. For each L:
  - Construct a prompt of approximately that many tokens.
  - Run a forward pass and collect `past_key_values`.
  - For each layer and for K/V separately:
    - Compute `num_elements = ∏ shape`.
    - Compute `bytes = num_elements × bytes_per_element` (4 for float32, 2 for float16).
3. Optionally repeat for different batch sizes B \in 1, 2, 4 at a fixed L.

**Outputs:**

- Table or CSV: `L, total_kv_bytes, kv_bytes_per_token, kv_bytes_per_layer`.
- Simple plots:
  - KV total bytes vs `L` (should be linear).
  - KV bytes per token (essentially constant in `L`).

#### 3.3 Per-Token / Per-Layer Formula

From the shapes we derive a general formula for GPT-2:

- Let:
  - `L` = sequence length,
  - `B` = batch size,
  - `H` = number of heads,
  - `D` = head dimension,
  - `Lx` = number of layers,
  - `s` = bytes per element (2 or 4).

Then, assuming keys/values have shape `[B, H, L, D]`:

- **Per layer, per token (K+V):**
  - `size_per_token_per_layer = 2 × H × D × s` bytes.
- **Total KV per token (all layers):**
  - `size_per_token_total = 2 × Lx × H × D × s` bytes.
- **Total KV for (B, L):**
  - `size_total = B × L × 2 × Lx × H × D × s` bytes.

Phase 1 validates that these formulas match empirical measurements for GPT-2.

---

### 4. Real-World Interpretation

Although GPT-2 is small and older than modern deployment models, the **scaling law**
we observe here (KV memory is linear in `L`, `Lx`, `H`, `D`, and `B`) is the same one
governing KV cache behaviour in production systems.

Phase 1 delivers:

- A **reference implementation** for KV cache inspection.
- Concrete numbers for “bytes per token of KV cache” on a small model.
- A baseline for reasoning about how many tokens/contexts can be held in memory on
a single node, which maps directly to P2P-RAGCache’s design questions about cache
capacity and replication.

These tools and intuitions are then reused in Phase 2 on a more realistic LLaMA-style
model.

---

### 5. Deliverables

- **Code:** The Phase 1 implementation lives in **`scripts/kv_cache_phase1_gpt2.py`**. It:
  - Runs GPT-2 with `use_cache=True`.
  - Prints layer-wise KV shapes and sizes.
  - Emits a simple report (stdout and CSV-style table) of KV size vs sequence length.

  **How to run** (from the repo root):

  ```bash
  pip install -r requirements.txt
  python scripts/kv_cache_phase1_gpt2.py
  ```

- **Doc:** A short summary (1–2 pages or section) with:
  - The empirical numbers for GPT-2.
  - The derived formulas.
  - A brief discussion of how this informs KV cache sizing decisions.

