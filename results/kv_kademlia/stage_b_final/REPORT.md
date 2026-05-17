# TinyLLaMA KV Splitting with Kademlia - Experiment Report

## Run configuration
- Stage: `stage_b`
- Total runs: `4050`
- Strategies: `token_layer_group, token_only, layer_only`
- Network sizes: `[32, 64, 128]`
- Replication factors: `[1, 2, 3]`
- Churn rates: `[0.0, 0.05, 0.15]`
- Sequence lengths: `[128, 512, 1024]`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Variants selected from previous stage: `token_layer_group|tb=32|lg=2, token_layer_group|tb=32|lg=4, token_layer_group|tb=64|lg=2, token_layer_group|tb=64|lg=4, layer_only|lb=1`

## Strategy-level ranking (60% latency, 25% availability, 15% overhead)
| strategy | avg_p95_latency_ms | avg_chunk_success_rate | avg_overhead_ratio | weighted_score |
|---|---:|---:|---:|---:|
| token_layer_group | 6.5526 | 0.993006 | 0.992993 | 0.850000 |
| layer_only | 14.8359 | 0.987486 | 0.987486 | 0.150000 |

## Variant-level ranking (60% latency, 25% availability, 15% overhead)
| variant | avg_p95_latency_ms | avg_chunk_success_rate | avg_overhead_ratio | weighted_score |
|---|---:|---:|---:|---:|
| token_layer_group|tb=32|lg=2 | 4.9195 | 0.996621 | 0.996621 | 0.850000 |
| token_layer_group|tb=32|lg=4 | 6.1968 | 0.992846 | 0.992672 | 0.734249 |
| token_layer_group|tb=64|lg=2 | 6.2752 | 0.991505 | 0.991505 | 0.711972 |
| token_layer_group|tb=64|lg=4 | 8.8190 | 0.991052 | 0.991176 | 0.551063 |
| layer_only|lb=1 | 14.8359 | 0.987486 | 0.987486 | 0.150000 |

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
