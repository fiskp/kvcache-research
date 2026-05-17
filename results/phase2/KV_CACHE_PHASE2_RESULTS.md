# KV Cache Inspection — Phase 2 Results (TinyLLaMA)

## Setup
- Model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Device: `cpu`
- Compute dtype target: `torch.float32`
- Layers: `22`
- Attention heads: `32`
- KV heads: `4`
- Hidden size: `2048`
- Head dim: `64`

## TinyLLaMA KV Table
| seq_len | total_kv_bytes_llama | kv_bytes_per_token_llama | kv_bytes_per_layer_llama |
|---:|---:|---:|---:|
| 8 | 360448 | 45056.00 | 16384.00 |
| 32 | 1441792 | 45056.00 | 65536.00 |
| 128 | 5767168 | 45056.00 | 262144.00 |
| 512 | 23068672 | 45056.00 | 1048576.00 |
| 1024 | 46137344 | 45056.00 | 2097152.00 |

## GPT-2 KV Table (Phase 2 exact-length rerun)
| seq_len | total_kv_bytes_gpt2 | kv_bytes_per_token_gpt2 | kv_bytes_per_layer_gpt2 |
|---:|---:|---:|---:|
| 8 | 589824 | 73728.00 | 49152.00 |
| 32 | 2359296 | 73728.00 | 196608.00 |
| 128 | 9437184 | 73728.00 | 786432.00 |
| 512 | 37748736 | 73728.00 | 3145728.00 |
| 1024 | 75497472 | 73728.00 | 6291456.00 |

## GPT-2 vs TinyLLaMA (Common Sequence Lengths)
| seq_len | kv_bytes_per_token_gpt2 | kv_bytes_per_token_llama | ratio_llama_to_gpt2_per_token |
|---:|---:|---:|---:|
| 8 | 73728.00 | 45056.00 | 0.6111 |
| 32 | 73728.00 | 45056.00 | 0.6111 |
| 128 | 73728.00 | 45056.00 | 0.6111 |
| 512 | 73728.00 | 45056.00 | 0.6111 |
| 1024 | 73728.00 | 45056.00 | 0.6111 |

## Deployment Interpretation
- Estimated TinyLLaMA KV per token: `45056.00` bytes.
- Approx. tokens fitting in 8 GiB for KV-only budget: `190650` tokens.
- Practical serving capacity is lower due to model weights, activations, framework overhead, and allocator fragmentation.
- For P2P sharding, KV chunks can be partitioned by contiguous token ranges or by layer; this data provides chunk-size estimates.

## Artifacts
- TinyLLaMA summary CSV: `phase2_outputs\tinyllama_kv_stats.csv`
- GPT-2 summary CSV: `phase2_outputs\gpt2_kv_stats.csv`
- TinyLLaMA per-layer CSV: `phase2_outputs\tinyllama_kv_layer_stats.csv`
- GPT-2 vs TinyLLaMA comparison CSV: `phase2_outputs\gpt2_vs_tinyllama_kv_comparison.csv`
- Figures: `phase2_outputs\figures`
- Run log: `phase2_outputs\TinyLLaMA-output.txt`
