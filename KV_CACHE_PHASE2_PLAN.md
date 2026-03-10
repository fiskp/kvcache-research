## KV Cache Inspection — Phase 2 (TinyLLaMA / LLaMA-Style Model)

### 1. Objective

Phase 2 applies the KV cache inspection methodology from Phase 1 to a **small
LLaMA-style model**. The goals are:

- **G4 — Modern architecture:** Characterise KV cache structure and size for a
  contemporary decoder-only model with rotary embeddings and (potentially) grouped
  query attention (GQA).
- **G5 — Comparison:** Directly compare GPT-2 vs LLaMA-style KV footprints (per token,
  per layer) at the same sequence lengths.
- **G6 — Deployment relevance:** Translate the numbers into insights about KV cache
  capacity and sharding strategies in realistic LLM serving setups.

Phase 2 does *not* redesign the tooling; instead, it **reuses Phase 1’s pipeline** to
highlight what changes (and what does not) when we move to a more modern model.

---

### 2. Model and Stack Choices

**Candidate model:** A small, open, LLaMA-style model such as:

- `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (≈1.1B parameters)

**Why LLaMA-style for Phase 2:**

- **Closer to production:** Modern open and closed LLMs (LLaMA-2/3, Mistral, etc.) use
  similar architectures. KV cache behaviour here is more representative of what a real
  system will see.
- **Attention optimisations:** Many LLaMA-style models use GQA/MQA to reduce KV
  size; understanding this is crucial for realistic KV cache sizing and sharding.
- **Feasible on your hardware:** 1.1B parameters is manageable on 8 GB VRAM with
  half-precision (`torch.float16`) and/or partial CPU offload.

**Software stack:**

- Same as Phase 1, plus:
  - Potential use of `device_map="auto"` (for GPU/CPU placement) and
    `torch_dtype=torch.float16` to fit within 8 GB VRAM.

---

### 3. Experiment Design

Phase 2 mirrors Phase 1 as closely as possible to keep comparisons fair.

#### 3.1 Basic KV Access

1. Load tokenizer and model:
   - `AutoTokenizer.from_pretrained(<tiny-llama-model>)`
   - `AutoModelForCausalLM.from_pretrained(<tiny-llama-model>, torch_dtype=torch.float16, device_map="auto")`
2. Tokenise prompts similar to those used for GPT-2 (Phase 1).
3. Run a single forward pass with:
   - `use_cache=True`
   - `return_dict=True`
4. Extract `outputs.past_key_values` and inspect:
   - Number of layers.
   - For each layer: shapes, dtypes, devices of key/value tensors.
   - Any evidence of grouped query attention (e.g. fewer KV “groups” than heads).

**Success condition:** We obtain a layer-wise summary analogous to GPT-2, even if
the exact shapes differ (e.g., due to GQA).

#### 3.2 Shape and Size Study (Same Protocol as Phase 1)

Use the **same set of sequence lengths** as in Phase 1, e.g.
\(L \in \{8, 32, 128, 512, 1024\}\), and optionally the same batch sizes.

For each \(L\):

1. Construct and tokenise a prompt of approximately `L` tokens.
2. Run a forward pass, collect `past_key_values`.
3. For each layer and K/V tensor:
   - Record shapes and dtypes.
   - Compute `num_elements` and total bytes.
4. Aggregate:
   - Per-layer KV size vs `L`.
   - Total KV size (all layers, K+V) vs `L`.

**Outputs:**

- Table/CSV: `L, total_kv_bytes_llama, kv_bytes_per_token_llama, kv_bytes_per_layer_llama`.
- Plots comparable to Phase 1 (same x-axis, similar y-axis).

#### 3.3 GPT-2 vs TinyLLaMA Comparison

Using results from both phases:

1. Align on common `L` values (e.g. 128, 512, 1024).
2. For each `L`, compute:
   - `kv_bytes_per_token_gpt2` vs `kv_bytes_per_token_llama`.
   - `total_kv_bytes_gpt2` vs `total_kv_bytes_llama`.
3. Visualise:
   - Bar chart or table showing per-token KV size for both models.
   - Optional: normalise by parameter count (bytes of KV cache per parameter at a
     fixed sequence length).

Key questions to answer:

- How much larger (or smaller) is the per-token KV footprint for TinyLLaMA compared
  to GPT-2?
- Does GQA (if present) significantly reduce KV size relative to a naïve multi-head
  layout?

---

### 4. Real-World Interpretation

Phase 2 connects the introspection work to real deployment scenarios:

- **Serving capacity:** Given per-token KV bytes for a LLaMA-style model, estimate:
  - How many tokens of history can be kept per user session on a single GPU.
  - How many concurrent sessions fit within 8 GB VRAM for typical context lengths.
- **Sharding strategies:** KV cache is what gets **sharded or replicated** when:
  - Splitting a model across multiple GPUs (layer-wise or tensor-parallel).
  - Offloading KV to remote peers in a P2P network (P2P-RAGCache scenario).
- **Protocol implications:** The size and shape of the KV cache inform:
  - How large each “cache chunk” is when distributed over the DHT from Milestone 1.
  - Whether it is more efficient to shard by layers, heads, or contiguous token
    ranges when designing KV cache distribution schemes.

By comparing GPT-2 and TinyLLaMA, we can argue:

- GPT-2 results validate the **basic scaling law** and our measurement pipeline.
- TinyLLaMA results show what practitioners will actually face when deploying
  modern LLaMA-like models, making subsequent P2P KV cache experiments grounded
  in realistic numbers.

---

### 5. Deliverables

- **Code:** Reuse or lightly extend the Phase 1 script/notebook to:
  - Load the chosen LLaMA-style model.
  - Produce layer-wise KV size reports across the agreed sequence lengths.
  - Emit comparable tables/plots to those from Phase 1.
- **Doc:** A short comparative writeup including:
  - Empirical KV size tables for both GPT-2 and TinyLLaMA.
  - A distilled summary of architectural differences that matter for KV cache
    (layers, heads, head dim, GQA).
  - Concrete rules of thumb, e.g. “Model X uses ~Y KB of KV cache per token at
    1k-token context length,” tied back to P2P-RAGCache’s capacity and replication
    decisions.

