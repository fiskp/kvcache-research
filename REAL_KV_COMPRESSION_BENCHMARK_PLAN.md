# Plan: Benchmark Serialization with Real TinyLLaMA KV Cache

## Motivation

The current serialization benchmarks use `torch.randn()` to generate synthetic KV
cache data. This produces essentially random bytes, resulting in a poor compression
ratio (~1.08x with zstd). Real model activations have structure — correlated values
across heads, clustered exponent bits, near-zero activations — that compression
algorithms can exploit.

This task answers: **"How much does compression actually help with real inference data?"**

## Approach

1. Load `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via HuggingFace Transformers
2. Run forward passes on a real prompt at multiple sequence lengths
3. Feed the real `past_key_values` through the existing serialization pipeline
4. Measure compression ratios and timing for none/lz4/zstd
5. Compare against the synthetic data baseline

## Script: `scripts/benchmark_real_kv_compression.py`

### Inputs

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Prompt: A representative text passage (to produce realistic activations)
- Sequence lengths: 32, 64, 128, 256, 512
- Compression methods: none, lz4, zstd
- Chunking configs: token_block=32, layer_group=2 (default from serialization layer)

### Measurements per configuration

| Metric | Description |
|--------|-------------|
| compression_ratio | raw_bytes / compressed_bytes |
| serialize_ms | Time to serialize all chunks |
| deserialize_ms | Time to deserialize all chunks |
| roundtrip_ms | Total serialize + deserialize + reassemble |
| chunk_count | Number of chunks produced |
| total_raw_bytes | Uncompressed KV cache size |
| total_compressed_bytes | Compressed size (including headers) |

### Outputs

- `benchmark_outputs/real_kv_compression.csv` — per-configuration results
- `benchmark_outputs/compression_comparison.csv` — side-by-side real vs synthetic ratios
- Terminal summary report

## Expected Results

Real KV cache data should compress better than random data because:
- Attention head activations are correlated (not uniformly random)
- Many values cluster near zero
- Adjacent layers produce similar activation magnitudes
- IEEE 754 exponent bits are concentrated in a narrow range

We expect compression ratios of ~1.2–1.5x with zstd on real data, compared to
~1.08x on random data. LZ4 should also show improvement over its current ~1.00x.

## Dependencies

- `torch>=2.0` (already in requirements.txt)
- `transformers>=4.40` (already in requirements.txt)
- `lz4>=4.0` (already in requirements.txt)
- `zstandard>=0.20` (already in requirements.txt)

## Verification

```bash
python scripts/benchmark_real_kv_compression.py
```

Generates CSV files and prints a comparison table. No GPU required — TinyLLaMA
runs on CPU for benchmarking purposes (inference speed is not the focus, only the
KV cache data characteristics matter).

## Commit Plan

1. **`feat: add real KV cache compression benchmark script`** — `scripts/benchmark_real_kv_compression.py`
2. **`docs: add real KV compression benchmark results`** — output CSVs and summary
