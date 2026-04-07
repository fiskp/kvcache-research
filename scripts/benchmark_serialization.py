"""Benchmark script for KV cache serialization.

Measures serialization/deserialization timing and compression ratios
across different configurations.

Run:  python scripts/benchmark_serialization.py
"""

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_serialization.format import COMPRESS_NONE, COMPRESS_LZ4, COMPRESS_ZSTD, HEADER_SIZE
from kv_serialization.serialize import (
    chunk_kv_cache,
    serialize_chunk,
    deserialize_chunk,
    reassemble_kv_cache,
)

OUTPUT_DIR = REPO_ROOT / "benchmark_outputs"


@dataclass
class BenchmarkResult:
    seq_len: int
    token_block: int
    layer_group: int
    compression: str
    num_chunks: int
    total_raw_bytes: int
    total_serialized_bytes: int
    compression_ratio: float
    serialize_ms: float
    deserialize_ms: float
    reassemble_ms: float
    roundtrip_ms: float


def _compression_name(code: int) -> str:
    return {COMPRESS_NONE: "none", COMPRESS_LZ4: "lz4", COMPRESS_ZSTD: "zstd"}[code]


def _has_compression(code: int) -> bool:
    if code == COMPRESS_NONE:
        return True
    if code == COMPRESS_LZ4:
        try:
            import lz4.frame
            return True
        except ImportError:
            return False
    if code == COMPRESS_ZSTD:
        try:
            import zstandard
            return True
        except ImportError:
            return False
    return False


def make_fake_kv_cache(num_layers=22, num_kv_heads=4, seq_len=64, head_dim=64):
    """Create a fake KV cache matching TinyLLaMA structure."""
    torch.manual_seed(42)
    past_kv = []
    for _ in range(num_layers):
        k = torch.randn(1, num_kv_heads, seq_len, head_dim)
        v = torch.randn(1, num_kv_heads, seq_len, head_dim)
        past_kv.append((k, v))
    return tuple(past_kv)


def benchmark_config(
    seq_len: int,
    token_block: int,
    layer_group: int,
    compression: int,
    num_layers: int = 22,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    warmup: int = 1,
    repeats: int = 3,
) -> BenchmarkResult:
    """Benchmark a single serialization configuration."""
    past_kv = make_fake_kv_cache(num_layers, num_kv_heads, seq_len, head_dim)

    # Warmup
    for _ in range(warmup):
        chunks = chunk_kv_cache(past_kv, seq_len, token_block, layer_group)
        for c in chunks:
            data = serialize_chunk(c, compression)
            deserialize_chunk(data, c.chunk_id)

    # Timed runs
    serialize_times = []
    deserialize_times = []
    reassemble_times = []
    total_serialized = 0

    for _ in range(repeats):
        chunks = chunk_kv_cache(past_kv, seq_len, token_block, layer_group)

        # Serialize
        t0 = time.perf_counter()
        serialized = [serialize_chunk(c, compression) for c in chunks]
        t1 = time.perf_counter()
        serialize_times.append((t1 - t0) * 1000)

        total_serialized = sum(len(s) for s in serialized)

        # Deserialize
        t0 = time.perf_counter()
        restored_chunks = [
            deserialize_chunk(s, c.chunk_id)
            for s, c in zip(serialized, chunks)
        ]
        t1 = time.perf_counter()
        deserialize_times.append((t1 - t0) * 1000)

        # Reassemble
        t0 = time.perf_counter()
        reassemble_kv_cache(
            restored_chunks, num_layers, seq_len, num_kv_heads, head_dim,
        )
        t1 = time.perf_counter()
        reassemble_times.append((t1 - t0) * 1000)

    # Compute raw size
    element_size = 4  # float32
    raw_bytes = num_layers * 2 * num_kv_heads * seq_len * head_dim * element_size

    avg_ser = sum(serialize_times) / len(serialize_times)
    avg_des = sum(deserialize_times) / len(deserialize_times)
    avg_rea = sum(reassemble_times) / len(reassemble_times)
    header_overhead = len(chunks) * HEADER_SIZE

    return BenchmarkResult(
        seq_len=seq_len,
        token_block=token_block,
        layer_group=layer_group,
        compression=_compression_name(compression),
        num_chunks=len(chunks),
        total_raw_bytes=raw_bytes,
        total_serialized_bytes=total_serialized,
        compression_ratio=raw_bytes / max(total_serialized - header_overhead, 1),
        serialize_ms=round(avg_ser, 3),
        deserialize_ms=round(avg_des, 3),
        reassemble_ms=round(avg_rea, 3),
        roundtrip_ms=round(avg_ser + avg_des + avg_rea, 3),
    )


def run_benchmarks() -> List[BenchmarkResult]:
    """Run the full benchmark suite."""
    results = []

    seq_lens = [32, 64, 128, 256, 512]
    token_blocks = [16, 32, 64]
    layer_groups = [1, 2, 4]
    compressions = [COMPRESS_NONE]

    # Add available compression methods
    if _has_compression(COMPRESS_LZ4):
        compressions.append(COMPRESS_LZ4)
    if _has_compression(COMPRESS_ZSTD):
        compressions.append(COMPRESS_ZSTD)

    total = len(seq_lens) * len(token_blocks) * len(layer_groups) * len(compressions)
    count = 0

    for seq_len in seq_lens:
        for tb in token_blocks:
            for lg in layer_groups:
                for comp in compressions:
                    count += 1
                    print(
                        f"  [{count}/{total}] seq_len={seq_len}, "
                        f"tb={tb}, lg={lg}, comp={_compression_name(comp)}"
                    )
                    results.append(benchmark_config(seq_len, tb, lg, comp))

    return results


def write_csv(results: List[BenchmarkResult], path: Path):
    """Write benchmark results to CSV."""
    fields = [
        "seq_len", "token_block", "layer_group", "compression",
        "num_chunks", "total_raw_bytes", "total_serialized_bytes",
        "compression_ratio", "serialize_ms", "deserialize_ms",
        "reassemble_ms", "roundtrip_ms",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "seq_len": r.seq_len,
                "token_block": r.token_block,
                "layer_group": r.layer_group,
                "compression": r.compression,
                "num_chunks": r.num_chunks,
                "total_raw_bytes": r.total_raw_bytes,
                "total_serialized_bytes": r.total_serialized_bytes,
                "compression_ratio": f"{r.compression_ratio:.4f}",
                "serialize_ms": r.serialize_ms,
                "deserialize_ms": r.deserialize_ms,
                "reassemble_ms": r.reassemble_ms,
                "roundtrip_ms": r.roundtrip_ms,
            })


def print_report(results: List[BenchmarkResult]):
    """Print a summary report to stdout."""
    print("\n" + "=" * 80)
    print("KV Cache Serialization Benchmark Report")
    print("=" * 80)
    print(f"{'SeqLen':>6} {'TB':>4} {'LG':>4} {'Comp':>5} {'Chunks':>6} "
          f"{'Raw KB':>8} {'Ser KB':>8} {'Ratio':>6} "
          f"{'Ser ms':>8} {'Des ms':>8} {'Rea ms':>8} {'Total ms':>9}")
    print("-" * 80)

    for r in results:
        print(
            f"{r.seq_len:>6} {r.token_block:>4} {r.layer_group:>4} "
            f"{r.compression:>5} {r.num_chunks:>6} "
            f"{r.total_raw_bytes / 1024:>8.1f} "
            f"{r.total_serialized_bytes / 1024:>8.1f} "
            f"{r.compression_ratio:>6.2f} "
            f"{r.serialize_ms:>8.3f} {r.deserialize_ms:>8.3f} "
            f"{r.reassemble_ms:>8.3f} {r.roundtrip_ms:>9.3f}"
        )


def main():
    print("Running KV cache serialization benchmarks...")
    print(f"TinyLLaMA config: 22 layers, 4 KV heads, 64 head_dim, float32")

    results = run_benchmarks()

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "serialization_benchmark.csv"
    write_csv(results, csv_path)
    print(f"\nResults written to {csv_path}")

    print_report(results)


if __name__ == "__main__":
    main()
