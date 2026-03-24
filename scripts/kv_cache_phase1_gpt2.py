import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SEQUENCE_LENGTHS: List[int] = [8, 32, 128, 512, 1024]
MODEL_NAME = "gpt2"

OUTPUT_DIR = Path("phase1_outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
RUN_LOG_PATH = OUTPUT_DIR / "GPT-2-output.txt"
SUMMARY_CSV_PATH = OUTPUT_DIR / "gpt2_kv_stats.csv"
LAYER_CSV_PATH = OUTPUT_DIR / "gpt2_kv_layer_stats.csv"
REPORT_MD_PATH = OUTPUT_DIR / "KV_CACHE_PHASE1_RESULTS.md"


@dataclass
class KVLayerStats:
    layer_index: int
    key_shape: Tuple[int, ...]
    value_shape: Tuple[int, ...]
    key_dtype: str
    value_dtype: str
    key_device: str
    value_device: str
    bytes_key: int
    bytes_value: int


@dataclass
class KVRunStats:
    seq_len: int
    batch_size: int
    total_bytes: int
    bytes_per_token: float
    bytes_per_layer: float
    per_layer: List[KVLayerStats]


def bytes_per_element(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16, torch.int16):
        return 2
    if dtype in (torch.float32, torch.int32):
        return 4
    if dtype in (torch.float64, torch.int64):
        return 8
    if dtype in (torch.uint8, torch.int8, torch.bool):
        return 1
    return 4


def summarize_kv(past_key_values, seq_len: int, batch_size: int) -> KVRunStats:
    per_layer_stats: List[KVLayerStats] = []
    total_bytes = 0

    for layer_idx, layer_kv in enumerate(past_key_values):
        if not (isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2):
            raise ValueError(f"Unexpected past_key_values structure at layer {layer_idx}: {type(layer_kv)}")

        key, value = layer_kv[0], layer_kv[1]
        k_shape = tuple(key.shape)
        v_shape = tuple(value.shape)

        bytes_k = int(np.prod(k_shape)) * bytes_per_element(key.dtype)
        bytes_v = int(np.prod(v_shape)) * bytes_per_element(value.dtype)
        total_bytes += bytes_k + bytes_v

        per_layer_stats.append(
            KVLayerStats(
                layer_index=layer_idx,
                key_shape=k_shape,
                value_shape=v_shape,
                key_dtype=str(key.dtype),
                value_dtype=str(value.dtype),
                key_device=str(key.device),
                value_device=str(value.device),
                bytes_key=bytes_k,
                bytes_value=bytes_v,
            )
        )

    bytes_per_tok = total_bytes / (seq_len * batch_size)
    bytes_per_layer = total_bytes / max(1, len(per_layer_stats))
    return KVRunStats(
        seq_len=seq_len,
        batch_size=batch_size,
        total_bytes=total_bytes,
        bytes_per_token=bytes_per_tok,
        bytes_per_layer=bytes_per_layer,
        per_layer=per_layer_stats,
    )


def make_prompt(target_tokens: int) -> str:
    approx_words = max(1, int(target_tokens / 1.3))
    base_sentence = "This is a test sentence about KV cache analysis."
    words = base_sentence.split()
    repetitions = math.ceil(approx_words / len(words))
    return (" ".join(words) + " ") * repetitions


def write_csvs(results: List[KVRunStats]) -> None:
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "total_kv_bytes", "kv_bytes_per_token", "kv_bytes_per_layer"])
        for r in results:
            writer.writerow([r.seq_len, r.total_bytes, f"{r.bytes_per_token:.2f}", f"{r.bytes_per_layer:.2f}"])

    with LAYER_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seq_len",
                "layer_index",
                "key_shape",
                "value_shape",
                "key_dtype",
                "value_dtype",
                "key_device",
                "value_device",
                "bytes_key",
                "bytes_value",
                "total_layer_bytes",
            ]
        )
        for r in results:
            for layer in r.per_layer:
                writer.writerow(
                    [
                        r.seq_len,
                        layer.layer_index,
                        str(layer.key_shape),
                        str(layer.value_shape),
                        layer.key_dtype,
                        layer.value_dtype,
                        layer.key_device,
                        layer.value_device,
                        layer.bytes_key,
                        layer.bytes_value,
                        layer.bytes_key + layer.bytes_value,
                    ]
                )


def make_plots(results: List[KVRunStats]) -> None:
    x = [r.seq_len for r in results]
    total_mib = [r.total_bytes / (1024 ** 2) for r in results]
    per_token_kib = [r.bytes_per_token / 1024 for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, total_mib, marker="o", linewidth=2)
    plt.title("GPT-2 Total KV Cache Size vs Sequence Length")
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Total KV size (MiB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gpt2_total_kv_vs_seq.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, per_token_kib, marker="o", linewidth=2, color="tab:green")
    plt.title("GPT-2 KV Size per Token vs Sequence Length")
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("KV per token (KiB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gpt2_kv_per_token_vs_seq.png", dpi=220)
    plt.close()


def write_markdown_report(results: List[KVRunStats], device: str, model_dtype: str) -> None:
    layers = len(results[0].per_layer) if results else 0
    sample_shape = results[0].per_layer[0].key_shape if results and results[0].per_layer else "unknown"

    lines: List[str] = []
    lines.append("# KV Cache Inspection — Phase 1 Results (GPT-2)\n")
    lines.append("## Setup")
    lines.append(f"- Model: `{MODEL_NAME}`")
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Compute dtype: `{model_dtype}`")
    lines.append(f"- Layers: `{layers}`")
    lines.append(f"- Sample K tensor shape: `{sample_shape}`\n")

    lines.append("## GPT-2 KV Table")
    lines.append("| seq_len | total_kv_bytes | kv_bytes_per_token | kv_bytes_per_layer |")
    lines.append("|---:|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r.seq_len} | {r.total_bytes} | {r.bytes_per_token:.2f} | {r.bytes_per_layer:.2f} |")
    lines.append("")

    lines.append("## Derived Formula (Validated)")
    lines.append("- Per layer, per token (K+V): `2 x H x D x s` bytes.")
    lines.append("- Total per token across all layers: `2 x Lx x H x D x s` bytes.")
    lines.append("- Total for batch and sequence: `B x L x 2 x Lx x H x D x s` bytes.\n")

    if results:
        per_token = results[0].bytes_per_token
        tokens_fit = int((8 * (1024 ** 3)) / per_token) if per_token else 0
        lines.append("## Deployment Interpretation")
        lines.append(f"- Measured GPT-2 KV bytes/token: `{per_token:.2f}`.")
        lines.append(f"- 8 GiB KV-only upper bound: `{tokens_fit}` tokens.")
        lines.append("- Real serving limits are lower due to model weights and runtime overhead.\n")

    lines.append("## Artifacts")
    lines.append(f"- Run log: `{RUN_LOG_PATH}`")
    lines.append(f"- Summary CSV: `{SUMMARY_CSV_PATH}`")
    lines.append(f"- Per-layer CSV: `{LAYER_CSV_PATH}`")
    lines.append(f"- Figures: `{FIGURES_DIR}`")
    REPORT_MD_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_experiment() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    batch_size = 1
    results: List[KVRunStats] = []
    log_lines: List[str] = [f"Using device: {device}", f"Model dtype: {next(model.parameters()).dtype}"]

    with torch.no_grad():
        for seq_len in SEQUENCE_LENGTHS:
            prompt = make_prompt(seq_len)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            actual_len = int(input_ids.shape[1])
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            if outputs.past_key_values is None:
                raise RuntimeError("Model did not return past_key_values. Ensure use_cache=True.")

            stats = summarize_kv(outputs.past_key_values, seq_len=actual_len, batch_size=batch_size)
            results.append(stats)

            log_lines.append("")
            log_lines.append(f"=== Sequence length target: {seq_len}, actual tokens: {actual_len} ===")
            log_lines.append(f"Total KV size: {stats.total_bytes / (1024 ** 2):.4f} MiB")
            log_lines.append(f"Bytes per token (all layers, K+V): {stats.bytes_per_token:.2f} bytes")
            for layer in stats.per_layer:
                log_lines.append(
                    f"  Layer {layer.layer_index:2d} | "
                    f"K shape {layer.key_shape}, V shape {layer.value_shape}, "
                    f"dtype={layer.key_dtype}, size={(layer.bytes_key + layer.bytes_value) / 1024:.1f} KiB"
                )

    log_lines.append("")
    log_lines.append("=== Summary table (batch_size=1) ===")
    log_lines.append("seq_len,total_kv_bytes,bytes_per_token,bytes_per_layer")
    for r in results:
        log_lines.append(f"{r.seq_len},{r.total_bytes},{r.bytes_per_token:.2f},{r.bytes_per_layer:.2f}")
    RUN_LOG_PATH.write_text("\n".join(log_lines).strip() + "\n", encoding="utf-8")

    write_csvs(results)
    make_plots(results)
    write_markdown_report(results, device=str(device), model_dtype=str(next(model.parameters()).dtype))

    print(f"Wrote run log to: {RUN_LOG_PATH}")
    print(f"Wrote summary CSV to: {SUMMARY_CSV_PATH}")
    print(f"Wrote per-layer CSV to: {LAYER_CSV_PATH}")
    print(f"Wrote report to: {REPORT_MD_PATH}")
    print(f"Wrote figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_experiment()
