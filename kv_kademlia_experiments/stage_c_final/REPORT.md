# TinyLLaMA KV Splitting with Kademlia - Experiment Report

## Run configuration
- Stage: `stage_c`
- Total runs: `3240`
- Strategies: `token_layer_group, token_only, layer_only`
- Network sizes: `[32, 64, 128]`
- Replication factors: `[1, 2, 3]`
- Churn rates: `[0.0, 0.05, 0.15]`
- Sequence lengths: `[128, 512, 1024]`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Variants selected from previous stage: `token_layer_group|tb=32|lg=2, token_layer_group|tb=32|lg=4`

## Strategy-level ranking (60% latency, 25% availability, 15% overhead)
| strategy | avg_p95_latency_ms | avg_chunk_success_rate | avg_overhead_ratio | weighted_score |
|---|---:|---:|---:|---:|
| token_layer_group | 5.5504 | 0.993768 | 0.993690 | 1.000000 |

## Variant-level ranking (60% latency, 25% availability, 15% overhead)
| variant | avg_p95_latency_ms | avg_chunk_success_rate | avg_overhead_ratio | weighted_score |
|---|---:|---:|---:|---:|
| token_layer_group|tb=32|lg=2 | 4.9109 | 0.995587 | 0.995587 | 0.850000 |
| token_layer_group|tb=32|lg=4 | 6.1900 | 0.991949 | 0.991794 | 0.150000 |

## Recommended variant
- Winner: `token_layer_group|tb=32|lg=2` based on configured weighted scoring.

## Artifacts
- `chunk_catalog.csv`
- `placement.csv`
- `access_trace.csv`
- `summary_metrics.csv`
- `weighted_ranking.csv`
- `variant_ranking.csv`
- `figures/latency_p95_by_strategy.png`
- `figures/latency_vs_churn.png`
- `figures/availability_vs_churn.png`
- `figures/overhead_vs_replication.png`
- `figures/latency_p95_by_variant.png` (Stage B/C only)
- `figures/latency_vs_seq_len_by_variant.png` (Stage B/C only)
- `figures/latency_vs_num_peers_by_variant.png` (Stage B/C only)
