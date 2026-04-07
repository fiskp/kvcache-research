# Milestone 2 Task 3 — KV Cache Serialization Layer Plan

## Goal

Build the serialization layer that converts real TinyLLaMA KV cache tensors
into a compact binary format suitable for P2P storage and retrieval via the
Kademlia DHT.

## Context

The project has completed KV cache profiling (Phase 1 & 2) and splitting
strategy experiments, but these work with synthetic byte counts — no real tensor
data flows through the DHT yet. This task bridges that gap.

## Module Structure

```
kv_serialization/
  __init__.py              # Public API exports
  format.py                # Binary format: header struct, magic, constants
  serialize.py             # chunk_kv_cache, serialize_chunk, deserialize_chunk, reassemble
  compress.py              # lz4/zstd compression adapter (optional deps)
  integration.py           # store_kv_chunk / retrieve_kv_chunk wrappers for Kademlia
scripts/
  benchmark_serialization.py   # Perf measurement: timing, compression ratios
tests/
  test_serialization.py        # Comprehensive round-trip, chunking, integration tests
```

## Binary Format

See `SERIALIZATION_FORMAT.md` for the full specification.

64-byte fixed header + variable-length payload. Supports optional lz4/zstd
compression and MD5 integrity checksums.

## Core API

| Function | Description |
|----------|-------------|
| `chunk_kv_cache(past_key_values, seq_len, token_block=32, layer_group=2)` | Split real KV tensors into chunks |
| `serialize_chunk(chunk, compression=0)` | Convert a KVChunk to binary bytes |
| `deserialize_chunk(data)` | Reconstruct a KVChunk from binary bytes |
| `reassemble_kv_cache(chunks, num_layers, seq_len, ...)` | Reconstruct full past_key_values |
| `store_kv_chunk(node, chunk, compression=0)` | Store serialized chunk in Kademlia DHT |
| `retrieve_kv_chunk(node, chunk_id)` | Retrieve and deserialize chunk from DHT |

## Compatibility

- Chunk IDs match existing convention from `experiment_tinyllama_kv_kademlia.py`
- Uses `DHTNode.hash_key(chunk_id, 16)` for DHT keys
- `KademliaNode.store(key, value)` accepts raw bytes directly
- Test patterns follow `tests/test_protocols.py` conventions

## TinyLLaMA KV Cache Structure

- 22 layers, 4 KV heads per layer, 64-dim head
- Tensor shape: `(batch=1, num_kv_heads=4, seq_len, head_dim=64)`
- 45,056 bytes per token across all layers (float32)

## Verification

```bash
pytest tests/test_serialization.py -v -s
python scripts/benchmark_serialization.py
```
