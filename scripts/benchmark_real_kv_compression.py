"""Benchmark real TinyLLaMA KV cache compression.

Runs real TinyLlama-1.1B-Chat inference, feeds actual past_key_values through
the serialization pipeline, and measures compression ratios vs synthetic data.

Run:  python scripts/benchmark_real_kv_compression.py
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
)

OUTPUT_DIR = REPO_ROOT / "benchmark_outputs"

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_LAYERS = 22
NUM_KV_HEADS = 4
HEAD_DIM = 64
TOKEN_BLOCK = 32
LAYER_GROUP = 2

# Representative prompt — produces realistic attention activations
PROMPT = (
    "The transformer architecture has become the dominant paradigm for large language "
    "models. It relies on self-attention mechanisms to capture long-range dependencies "
    "in sequences. During inference, the key-value cache stores intermediate activations "
    "to avoid recomputing them for previously seen tokens. This is particularly important "
    "for autoregressive generation, where each new token attends to all previous tokens. "
    "Modern systems use grouped-query attention to reduce the memory footprint of the "
    "KV cache while preserving most of the representational capacity of multi-head attention."
)


@dataclass
class CompressionResult:
    seq_len: int
    compression: str
    num_chunks: int
    total_raw_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    serialize_ms: float
    deserialize_ms: float
    roundtrip_ms: float


def _compression_name(code: int) -> str:
    return {COMPRESS_NONE: "none", COMPRESS_LZ4: "lz4", COMPRESS_ZSTD: "zstd"}[code]


def _available_compressions() -> List[int]:
    methods = [COMPRESS_NONE]
    for code, pkg in [(COMPRESS_LZ4, "lz4.frame"), (COMPRESS_ZSTD, "zstandard")]:
        try:
            __import__(pkg)
            methods.append(code)
        except ImportError:
            print(f"  Warning: {pkg} not available, skipping")
    return methods


def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    print("  Model loaded.")
    return model, tokenizer


def get_real_kv_cache(model, tokenizer, seq_len: int) -> tuple:
    """Run a forward pass and return past_key_values as a legacy tuple at the given seq_len."""
    tokens = tokenizer.encode(PROMPT, return_tensors="pt")
    if tokens.shape[1] < seq_len:
        repeats = (seq_len // tokens.shape[1]) + 2
        tokens = tokens.repeat(1, repeats)
    tokens = tokens[:, :seq_len]
    with torch.no_grad():
        output = model(tokens, use_cache=True)
    pkv = output.past_key_values
    # Newer transformers returns a DynamicCache; convert to legacy tuple-of-tuples
    if hasattr(pkv, "layers"):
        pkv = tuple((layer.keys, layer.values) for layer in pkv.layers)
    elif hasattr(pkv, "key_cache"):
        pkv = tuple(
            (k.unsqueeze(0) if k.dim() == 3 else k,
             v.unsqueeze(0) if v.dim() == 3 else v)
            for k, v in zip(pkv.key_cache, pkv.value_cache)
        )
    return pkv


def make_synthetic_kv(seq_len: int) -> tuple:
    """Random KV cache matching TinyLLaMA structure, for comparison baseline."""
    torch.manual_seed(42)
    return tuple(
        (torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM),
         torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM))
        for _ in range(NUM_LAYERS)
    )


def benchmark_config(
    past_kv: tuple,
    seq_len: int,
    compression: int,
    repeats: int = 3,
) -> CompressionResult:
    """Benchmark one (past_kv, seq_len, compression) combination."""
    # Warmup
    chunks = chunk_kv_cache(past_kv, seq_len, TOKEN_BLOCK, LAYER_GROUP)
    for c in chunks:
        deserialize_chunk(serialize_chunk(c, compression), c.chunk_id)

    serialize_times: List[float] = []
    deserialize_times: List[float] = []
    total_compressed = 0

    for _ in range(repeats):
        chunks = chunk_kv_cache(past_kv, seq_len, TOKEN_BLOCK, LAYER_GROUP)

        t0 = time.perf_counter()
        serialized = [serialize_chunk(c, compression) for c in chunks]
        serialize_times.append((time.perf_counter() - t0) * 1000)

        total_compressed = sum(len(s) for s in serialized)

        t0 = time.perf_counter()
        for s, c in zip(serialized, chunks):
            deserialize_chunk(s, c.chunk_id)
        deserialize_times.append((time.perf_counter() - t0) * 1000)

    raw_bytes = NUM_LAYERS * 2 * NUM_KV_HEADS * seq_len * HEAD_DIM * 4  # float32
    header_overhead = len(chunks) * HEADER_SIZE
    avg_ser = sum(serialize_times) / len(serialize_times)
    avg_des = sum(deserialize_times) / len(deserialize_times)

    return CompressionResult(
        seq_len=seq_len,
        compression=_compression_name(compression),
        num_chunks=len(chunks),
        total_raw_bytes=raw_bytes,
        total_compressed_bytes=total_compressed,
        compression_ratio=raw_bytes / max(total_compressed - header_overhead, 1),
        serialize_ms=round(avg_ser, 3),
        deserialize_ms=round(avg_des, 3),
        roundtrip_ms=round(avg_ser + avg_des, 3),
    )


def write_real_csv(results: List[CompressionResult], path: Path):
    fields = [
        "seq_len", "compression", "num_chunks", "total_raw_bytes",
        "total_compressed_bytes", "compression_ratio",
        "serialize_ms", "deserialize_ms", "roundtrip_ms",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "seq_len": r.seq_len,
                "compression": r.compression,
                "num_chunks": r.num_chunks,
                "total_raw_bytes": r.total_raw_bytes,
                "total_compressed_bytes": r.total_compressed_bytes,
                "compression_ratio": f"{r.compression_ratio:.4f}",
                "serialize_ms": r.serialize_ms,
                "deserialize_ms": r.deserialize_ms,
                "roundtrip_ms": r.roundtrip_ms,
            })


def write_comparison_csv(
    real_results: List[CompressionResult],
    synth_results: List[CompressionResult],
    path: Path,
):
    synth_lookup = {(r.seq_len, r.compression): r.compression_ratio for r in synth_results}
    fields = ["seq_len", "compression", "real_ratio", "synthetic_ratio", "improvement_factor"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in real_results:
            synth_ratio = synth_lookup.get((r.seq_len, r.compression), 1.0)
            improvement = r.compression_ratio / max(synth_ratio, 1e-6)
            writer.writerow({
                "seq_len": r.seq_len,
                "compression": r.compression,
                "real_ratio": f"{r.compression_ratio:.4f}",
                "synthetic_ratio": f"{synth_ratio:.4f}",
                "improvement_factor": f"{improvement:.4f}",
            })


def print_report(
    real_results: List[CompressionResult],
    synth_results: List[CompressionResult],
):
    synth_lookup = {(r.seq_len, r.compression): r.compression_ratio for r in synth_results}

    print("\n" + "=" * 78)
    print("Real vs Synthetic KV Cache Compression  (token_block=32, layer_group=2)")
    print("=" * 78)
    print(f"{'SeqLen':>6} {'Comp':>5} {'Real Ratio':>11} {'Synth Ratio':>12} "
          f"{'Improvement':>12} {'RT ms':>8}")
    print("-" * 78)
    for r in real_results:
        synth_ratio = synth_lookup.get((r.seq_len, r.compression), 1.0)
        improvement = r.compression_ratio / max(synth_ratio, 1e-6)
        print(
            f"{r.seq_len:>6} {r.compression:>5} "
            f"{r.compression_ratio:>11.4f} {synth_ratio:>12.4f} "
            f"{improvement:>12.4f} {r.roundtrip_ms:>8.3f}"
        )


def main():
    seq_lens = [32, 64, 128, 256, 512]
    compressions = _available_compressions()

    model, tokenizer = load_model()

    print(f"\nBenchmarking real KV cache compression...")
    print(f"  token_block={TOKEN_BLOCK}, layer_group={LAYER_GROUP}, repeats=3")

    real_results: List[CompressionResult] = []
    synth_results: List[CompressionResult] = []

    total = len(seq_lens) * len(compressions)
    count = 0

    for seq_len in seq_lens:
        print(f"\n  Running inference at seq_len={seq_len}...")
        real_kv = get_real_kv_cache(model, tokenizer, seq_len)
        synth_kv = make_synthetic_kv(seq_len)

        for comp in compressions:
            count += 1
            comp_name = _compression_name(comp)
            print(f"  [{count}/{total}] seq_len={seq_len}, comp={comp_name}")
            real_results.append(benchmark_config(real_kv, seq_len, comp))
            synth_results.append(benchmark_config(synth_kv, seq_len, comp))

    OUTPUT_DIR.mkdir(exist_ok=True)
    real_csv = OUTPUT_DIR / "real_kv_compression.csv"
    comp_csv = OUTPUT_DIR / "compression_comparison.csv"
    write_real_csv(real_results, real_csv)
    write_comparison_csv(real_results, synth_results, comp_csv)

    print(f"\nResults written to:")
    print(f"  {real_csv}")
    print(f"  {comp_csv}")

    print_report(real_results, synth_results)


if __name__ == "__main__":
    main()
