import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SEQUENCE_LENGTHS: List[int] = [8, 32, 128, 512, 1024]
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
GPT2_MODEL = "gpt2"

OUTPUT_DIR = Path("phase2_outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
LLAMA_CSV_PATH = OUTPUT_DIR / "tinyllama_kv_stats.csv"
LLAMA_LAYER_CSV_PATH = OUTPUT_DIR / "tinyllama_kv_layer_stats.csv"
GPT2_CSV_PATH = OUTPUT_DIR / "gpt2_kv_stats.csv"
COMPARISON_CSV_PATH = OUTPUT_DIR / "gpt2_vs_tinyllama_kv_comparison.csv"
REPORT_MD_PATH = OUTPUT_DIR / "KV_CACHE_PHASE2_RESULTS.md"
RUN_LOG_PATH = OUTPUT_DIR / "TinyLLaMA-output.txt"


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


def build_exact_length_input(
    tokenizer: AutoTokenizer, target_len: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer.encode("hello world", add_special_tokens=False)
    if not token_ids:
        fallback = tokenizer.eos_token_id
        if fallback is None:
            raise RuntimeError("Tokenizer has no usable token ids for prompt construction.")
        token_ids = [fallback]

    repeats = math.ceil(target_len / len(token_ids))
    prompt_ids = (token_ids * repeats)[:target_len]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


def summarize_kv(past_key_values, seq_len: int, batch_size: int) -> KVRunStats:
    per_layer_stats: List[KVLayerStats] = []
    total_bytes = 0

    for layer_idx, layer_kv in enumerate(past_key_values):
        if not (isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2):
            raise ValueError(
                f"Unexpected past_key_values structure at layer {layer_idx}: {type(layer_kv)}"
            )

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


def write_llama_csvs(results: List[KVRunStats]) -> None:
    with LLAMA_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seq_len",
                "total_kv_bytes_llama",
                "kv_bytes_per_token_llama",
                "kv_bytes_per_layer_llama",
            ]
        )
        for stats in results:
            writer.writerow(
                [stats.seq_len, stats.total_bytes, f"{stats.bytes_per_token:.2f}", f"{stats.bytes_per_layer:.2f}"]
            )

    with LLAMA_LAYER_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
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
        for stats in results:
            for layer in stats.per_layer:
                writer.writerow(
                    [
                        stats.seq_len,
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


def write_gpt2_csv(results: List[KVRunStats]) -> None:
    with GPT2_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seq_len",
                "total_kv_bytes_gpt2",
                "kv_bytes_per_token_gpt2",
                "kv_bytes_per_layer_gpt2",
            ]
        )
        for stats in results:
            writer.writerow(
                [stats.seq_len, stats.total_bytes, f"{stats.bytes_per_token:.2f}", f"{stats.bytes_per_layer:.2f}"]
            )


def write_comparison_csv(
    results_llama: List[KVRunStats], results_gpt2: List[KVRunStats]
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    gpt2_index = {r.seq_len: r for r in results_gpt2}
    llama_index = {r.seq_len: r for r in results_llama}
    common_lengths = sorted(set(llama_index.keys()).intersection(gpt2_index.keys()))

    with COMPARISON_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seq_len",
                "total_kv_bytes_gpt2",
                "kv_bytes_per_token_gpt2",
                "total_kv_bytes_llama",
                "kv_bytes_per_token_llama",
                "ratio_llama_to_gpt2_per_token",
            ]
        )

        for seq_len in common_lengths:
            gpt2 = gpt2_index[seq_len]
            llama = llama_index[seq_len]
            ratio = llama.bytes_per_token / gpt2.bytes_per_token if gpt2.bytes_per_token else float("nan")
            writer.writerow(
                [
                    seq_len,
                    gpt2.total_bytes,
                    f"{gpt2.bytes_per_token:.2f}",
                    llama.total_bytes,
                    f"{llama.bytes_per_token:.2f}",
                    f"{ratio:.4f}",
                ]
            )
            rows.append(
                {
                    "seq_len": seq_len,
                    "total_kv_bytes_gpt2": float(gpt2.total_bytes),
                    "kv_bytes_per_token_gpt2": gpt2.bytes_per_token,
                    "total_kv_bytes_llama": float(llama.total_bytes),
                    "kv_bytes_per_token_llama": llama.bytes_per_token,
                    "ratio_llama_to_gpt2_per_token": ratio,
                }
            )
    return rows


def make_plots(results_llama: List[KVRunStats], comparison_rows: List[Dict[str, float]]) -> None:
    x = [r.seq_len for r in results_llama]
    total_mib = [r.total_bytes / (1024 ** 2) for r in results_llama]
    per_token_kib = [r.bytes_per_token / 1024 for r in results_llama]

    plt.figure(figsize=(8, 5))
    plt.plot(x, total_mib, marker="o", linewidth=2)
    plt.title("TinyLLaMA Total KV Cache Size vs Sequence Length")
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Total KV size (MiB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tinyllama_total_kv_vs_seq.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, per_token_kib, marker="o", linewidth=2, color="tab:green")
    plt.title("TinyLLaMA KV Size per Token vs Sequence Length")
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("KV per token (KiB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tinyllama_kv_per_token_vs_seq.png", dpi=220)
    plt.close()

    if comparison_rows:
        cx = [int(r["seq_len"]) for r in comparison_rows]
        gpt2 = [r["kv_bytes_per_token_gpt2"] / 1024 for r in comparison_rows]
        llama = [r["kv_bytes_per_token_llama"] / 1024 for r in comparison_rows]
        idx = np.arange(len(cx))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(idx - width / 2, gpt2, width, label="GPT-2")
        plt.bar(idx + width / 2, llama, width, label="TinyLLaMA")
        plt.title("KV Bytes per Token: GPT-2 vs TinyLLaMA")
        plt.xlabel("Sequence length (tokens)")
        plt.ylabel("KV per token (KiB)")
        plt.xticks(idx, [str(v) for v in cx])
        plt.legend()
        plt.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "gpt2_vs_tinyllama_kv_per_token.png", dpi=220)
        plt.close()


def infer_architecture_notes(model) -> Dict[str, str]:
    cfg = model.config
    notes = {
        "num_hidden_layers": str(getattr(cfg, "num_hidden_layers", "unknown")),
        "num_attention_heads": str(getattr(cfg, "num_attention_heads", "unknown")),
        "num_key_value_heads": str(getattr(cfg, "num_key_value_heads", "unknown")),
        "hidden_size": str(getattr(cfg, "hidden_size", "unknown")),
        "head_dim": str(getattr(cfg, "head_dim", "unknown")),
    }
    return notes


def run_model_experiment(
    model_name: str,
    dtype: torch.dtype,
    execution_device: str,
    add_device_map_auto: bool = False,
) -> Tuple[List[KVRunStats], Dict[str, str]]:
    model_kwargs = {"dtype": dtype}
    if add_device_map_auto:
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    if execution_device == "cpu":
        model.to("cpu")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    arch_notes = infer_architecture_notes(model)
    results: List[KVRunStats] = []

    with torch.no_grad():
        for seq_len in SEQUENCE_LENGTHS:
            input_ids, attention_mask = build_exact_length_input(tokenizer, seq_len, device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            if outputs.past_key_values is None:
                raise RuntimeError("Model did not return past_key_values. Ensure use_cache=True.")
            stats = summarize_kv(outputs.past_key_values, seq_len=seq_len, batch_size=1)
            results.append(stats)

    return results, arch_notes


def build_report(
    results_llama: List[KVRunStats],
    results_gpt2: List[KVRunStats],
    comparison_rows: List[Dict[str, float]],
    arch_notes: Dict[str, str],
    execution_device: str,
    model_dtype: str,
) -> None:
    lines: List[str] = []
    lines.append("# KV Cache Inspection — Phase 2 Results (TinyLLaMA)\n")
    lines.append("## Setup")
    lines.append(f"- Model: `{TINYLLAMA_MODEL}`")
    lines.append(f"- Device: `{execution_device}`")
    lines.append(f"- Compute dtype target: `{model_dtype}`")
    lines.append(f"- Layers: `{arch_notes['num_hidden_layers']}`")
    lines.append(f"- Attention heads: `{arch_notes['num_attention_heads']}`")
    lines.append(f"- KV heads: `{arch_notes['num_key_value_heads']}`")
    lines.append(f"- Hidden size: `{arch_notes['hidden_size']}`")
    lines.append(f"- Head dim: `{arch_notes['head_dim']}`\n")

    lines.append("## TinyLLaMA KV Table")
    lines.append("| seq_len | total_kv_bytes_llama | kv_bytes_per_token_llama | kv_bytes_per_layer_llama |")
    lines.append("|---:|---:|---:|---:|")
    for r in results_llama:
        lines.append(
            f"| {r.seq_len} | {r.total_bytes} | {r.bytes_per_token:.2f} | {r.bytes_per_layer:.2f} |"
        )
    lines.append("")

    lines.append("## GPT-2 KV Table (Phase 2 exact-length rerun)")
    lines.append("| seq_len | total_kv_bytes_gpt2 | kv_bytes_per_token_gpt2 | kv_bytes_per_layer_gpt2 |")
    lines.append("|---:|---:|---:|---:|")
    for r in results_gpt2:
        lines.append(
            f"| {r.seq_len} | {r.total_bytes} | {r.bytes_per_token:.2f} | {r.bytes_per_layer:.2f} |"
        )
    lines.append("")

    lines.append("## GPT-2 vs TinyLLaMA (Common Sequence Lengths)")
    if comparison_rows:
        lines.append(
            "| seq_len | kv_bytes_per_token_gpt2 | kv_bytes_per_token_llama | ratio_llama_to_gpt2_per_token |"
        )
        lines.append("|---:|---:|---:|---:|")
        for row in comparison_rows:
            lines.append(
                f"| {int(row['seq_len'])} | {row['kv_bytes_per_token_gpt2']:.2f} | "
                f"{row['kv_bytes_per_token_llama']:.2f} | {row['ratio_llama_to_gpt2_per_token']:.4f} |"
            )
    else:
        lines.append("- No overlap between TinyLLaMA sequence lengths and parsed GPT-2 summary.")
    lines.append("")

    lines.append("## Deployment Interpretation")
    if results_llama:
        # Use the largest length as a practical rule-of-thumb for this run.
        max_row = max(results_llama, key=lambda r: r.seq_len)
        per_token = max_row.bytes_per_token
        # 8 GiB device budget
        vram_budget = 8 * (1024 ** 3)
        tokens_fit = int(vram_budget / per_token) if per_token else 0
        lines.append(f"- Estimated TinyLLaMA KV per token: `{per_token:.2f}` bytes.")
        lines.append(f"- Approx. tokens fitting in 8 GiB for KV-only budget: `{tokens_fit}` tokens.")
        lines.append(
            "- Practical serving capacity is lower due to model weights, activations, framework overhead, and allocator fragmentation."
        )
        lines.append(
            "- For P2P sharding, KV chunks can be partitioned by contiguous token ranges or by layer; this data provides chunk-size estimates."
        )
    lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- TinyLLaMA summary CSV: `{LLAMA_CSV_PATH}`")
    lines.append(f"- GPT-2 summary CSV: `{GPT2_CSV_PATH}`")
    lines.append(f"- TinyLLaMA per-layer CSV: `{LLAMA_LAYER_CSV_PATH}`")
    lines.append(f"- GPT-2 vs TinyLLaMA comparison CSV: `{COMPARISON_CSV_PATH}`")
    lines.append(f"- Figures: `{FIGURES_DIR}`")
    lines.append(f"- Run log: `{RUN_LOG_PATH}`")

    REPORT_MD_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_phase2() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        dtype = torch.float16
        model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        exec_device = "cuda (device_map=auto)"
    else:
        dtype = torch.float32
        model_kwargs = {"torch_dtype": dtype}
        exec_device = "cpu"

    execution_device = "cpu" if not torch.cuda.is_available() else "cuda"
    results, arch_notes = run_model_experiment(
        model_name=TINYLLAMA_MODEL,
        dtype=dtype,
        execution_device=execution_device,
        add_device_map_auto=torch.cuda.is_available(),
    )
    gpt2_results, _ = run_model_experiment(
        model_name=GPT2_MODEL,
        dtype=torch.float32,
        execution_device=execution_device,
        add_device_map_auto=False,
    )

    log_lines: List[str] = []

    log_lines.append(f"Using execution device: {exec_device}")
    log_lines.append(f"Model dtype target: {dtype}")
    log_lines.append(
        "Architecture: "
        f"layers={arch_notes['num_hidden_layers']}, "
        f"heads={arch_notes['num_attention_heads']}, "
        f"kv_heads={arch_notes['num_key_value_heads']}, "
        f"head_dim={arch_notes['head_dim']}"
    )

    for stats in results:
        log_lines.append("")
        log_lines.append(f"=== Sequence length: {stats.seq_len} ===")
        log_lines.append(f"Total KV size: {stats.total_bytes / (1024 ** 2):.4f} MiB")
        log_lines.append(f"Bytes per token (all layers, K+V): {stats.bytes_per_token:.2f}")
        for layer in stats.per_layer:
            log_lines.append(
                f"  Layer {layer.layer_index:2d} | "
                f"K {layer.key_shape} ({layer.key_dtype}, {layer.key_device}) | "
                f"V {layer.value_shape} ({layer.value_dtype}, {layer.value_device}) | "
                f"size={(layer.bytes_key + layer.bytes_value) / 1024:.1f} KiB"
            )

    log_lines.append("")
    log_lines.append("=== TinyLLaMA summary table (batch_size=1) ===")
    log_lines.append(
        "seq_len,total_kv_bytes_llama,kv_bytes_per_token_llama,kv_bytes_per_layer_llama"
    )
    for stats in results:
        log_lines.append(
            f"{stats.seq_len},{stats.total_bytes},{stats.bytes_per_token:.2f},{stats.bytes_per_layer:.2f}"
        )
    RUN_LOG_PATH.write_text("\n".join(log_lines).strip() + "\n", encoding="utf-8")

    write_llama_csvs(results)
    write_gpt2_csv(gpt2_results)
    comparison_rows = write_comparison_csv(results, gpt2_results)
    make_plots(results, comparison_rows)
    build_report(
        results_llama=results,
        results_gpt2=gpt2_results,
        comparison_rows=comparison_rows,
        arch_notes=arch_notes,
        execution_device=exec_device,
        model_dtype=str(dtype),
    )

    print(f"Wrote TinyLLaMA run log to: {RUN_LOG_PATH}")
    print(f"Wrote summary CSV to: {LLAMA_CSV_PATH}")
    print(f"Wrote per-layer CSV to: {LLAMA_LAYER_CSV_PATH}")
    print(f"Wrote comparison CSV to: {COMPARISON_CSV_PATH}")
    print(f"Wrote report to: {REPORT_MD_PATH}")
    print(f"Wrote figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_phase2()
