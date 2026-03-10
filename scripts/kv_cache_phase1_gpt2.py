import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SEQUENCE_LENGTHS: List[int] = [8, 32, 128, 512, 1024]
MODEL_NAME = "gpt2"


@dataclass
class KVLayerStats:
    layer_index: int
    key_shape: Tuple[int, ...]
    value_shape: Tuple[int, ...]
    dtype: str
    bytes_key: int
    bytes_value: int


@dataclass
class KVRunStats:
    seq_len: int
    batch_size: int
    total_bytes: int
    bytes_per_token: float
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
    # Fallback: assume 4 bytes
    return 4


def summarize_kv(past_key_values, seq_len: int, batch_size: int) -> KVRunStats:
    per_layer_stats: List[KVLayerStats] = []
    total_bytes = 0

    for layer_idx, layer_kv in enumerate(past_key_values):
        # layer_kv is typically (key, value)
        if isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2:
            key, value = layer_kv[0], layer_kv[1]
        else:
            raise ValueError(f"Unexpected past_key_values structure at layer {layer_idx}: {type(layer_kv)}")

        k_shape = tuple(key.shape)
        v_shape = tuple(value.shape)
        if key.dtype != value.dtype:
            raise ValueError(f"Mismatched dtypes in layer {layer_idx}: {key.dtype} vs {value.dtype}")

        bpe = bytes_per_element(key.dtype)
        bytes_k = int(np.prod(k_shape)) * bpe
        bytes_v = int(np.prod(v_shape)) * bpe
        total_bytes += bytes_k + bytes_v

        per_layer_stats.append(
            KVLayerStats(
                layer_index=layer_idx,
                key_shape=k_shape,
                value_shape=v_shape,
                dtype=str(key.dtype),
                bytes_key=bytes_k,
                bytes_value=bytes_v,
            )
        )

    bytes_per_tok = total_bytes / (seq_len * batch_size)
    return KVRunStats(
        seq_len=seq_len,
        batch_size=batch_size,
        total_bytes=total_bytes,
        bytes_per_token=bytes_per_tok,
        per_layer=per_layer_stats,
    )


def make_prompt(target_tokens: int) -> str:
    # Simple heuristic: assume ~1.3 tokens per word and ~5 chars per word.
    approx_words = max(1, int(target_tokens / 1.3))
    base_sentence = "This is a test sentence about KV cache analysis."
    words = base_sentence.split()
    repetitions = math.ceil(approx_words / len(words))
    return (" ".join(words) + " ") * repetitions


def run_experiment() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    batch_size = 1
    results: List[KVRunStats] = []

    with torch.no_grad():
        for seq_len in SEQUENCE_LENGTHS:
            prompt = make_prompt(seq_len)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            actual_len = input_ids.shape[1]
            print(f"\n=== Sequence length target: {seq_len}, actual tokens: {actual_len} ===")

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values
            if past_key_values is None:
                raise RuntimeError("Model did not return past_key_values. Ensure use_cache=True.")

            stats = summarize_kv(past_key_values, seq_len=actual_len, batch_size=batch_size)
            results.append(stats)

            print(f"Total KV size: {stats.total_bytes / (1024 ** 2):.4f} MiB")
            print(f"Bytes per token (all layers, K+V): {stats.bytes_per_token:.2f} bytes")
            # Print a one-line layer summary
            for layer in stats.per_layer:
                print(
                    f"  Layer {layer.layer_index:2d} | "
                    f"K shape {layer.key_shape}, V shape {layer.value_shape}, "
                    f"dtype={layer.dtype}, "
                    f"size={(layer.bytes_key + layer.bytes_value) / 1024:.1f} KiB"
                )

    print("\n=== Summary table (batch_size=1) ===")
    print("seq_len,total_kv_bytes,bytes_per_token")
    for stats in results:
        print(f"{stats.seq_len},{stats.total_bytes},{stats.bytes_per_token:.2f}")


if __name__ == "__main__":
    run_experiment()
