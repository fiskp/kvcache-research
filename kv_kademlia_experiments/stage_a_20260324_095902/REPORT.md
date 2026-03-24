# TinyLLaMA KV Splitting with Kademlia - Experiment Report

## Run configuration
- Stage: `stage_a`
- Total runs: `120`
- Strategies: `token_layer_group, token_only, layer_only`
- Network sizes: `[64]`
- Replication factors: `[2]`
- Churn rates: `[0.0, 0.05]`
- Sequence lengths: `[512]`
- Seeds: `[0, 1, 2, 3, 4]`

## Weighted ranking (60% latency, 25% availability, 15% overhead)
| strategy | avg_p95_latency_ms | avg_chunk_success_rate | avg_overhead_ratio | weighted_score |
|---|---:|---:|---:|---:|
| token_layer_group | 8.4258 | 0.999242 | 0.999242 | 0.750000 |
| layer_only | 27.4350 | 1.000000 | 1.000000 | 0.441676 |
| token_only | 36.3583 | 1.000000 | 1.000000 | 0.250000 |

## Recommended strategy
- Winner: `token_layer_group` based on configured weighted scoring.

## Artifacts
- `chunk_catalog.csv`
- `placement.csv`
- `access_trace.csv`
- `summary_metrics.csv`
- `weighted_ranking.csv`
- `figures/latency_p95_by_strategy.png`
- `figures/latency_vs_churn.png`
- `figures/availability_vs_churn.png`
- `figures/overhead_vs_replication.png`
