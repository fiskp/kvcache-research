# KV Cache Inspection — Phase 1 Results (GPT-2)

## Setup
- Model: `gpt2`
- Device: `cpu`
- Compute dtype: `torch.float32`
- Layers: `12`
- Sample K tensor shape: `(1, 12, 8, 64)`

## GPT-2 KV Table
| seq_len | total_kv_bytes | kv_bytes_per_token | kv_bytes_per_layer |
|---:|---:|---:|---:|
| 8 | 589824 | 73728.00 | 49152.00 |
| 32 | 2359296 | 73728.00 | 196608.00 |
| 122 | 8994816 | 73728.00 | 749568.00 |
| 485 | 35758080 | 73728.00 | 2979840.00 |
| 969 | 71442432 | 73728.00 | 5953536.00 |

## Derived Formula (Validated)
- Per layer, per token (K+V): `2 x H x D x s` bytes.
- Total per token across all layers: `2 x Lx x H x D x s` bytes.
- Total for batch and sequence: `B x L x 2 x Lx x H x D x s` bytes.

## Deployment Interpretation
- Measured GPT-2 KV bytes/token: `73728.00`.
- 8 GiB KV-only upper bound: `116508` tokens.
- Real serving limits are lower due to model weights and runtime overhead.

## Artifacts
- Run log: `phase1_outputs\GPT-2-output.txt`
- Summary CSV: `phase1_outputs\gpt2_kv_stats.csv`
- Per-layer CSV: `phase1_outputs\gpt2_kv_layer_stats.csv`
- Figures: `phase1_outputs\figures`
