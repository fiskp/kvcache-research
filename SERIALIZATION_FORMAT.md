# KV Cache Serialization Binary Format Specification

## Overview

This document specifies the binary format used to serialize TinyLLaMA KV cache
tensors for storage and retrieval across the P2P DHT network. The format is
designed for compactness, integrity verification, and optional compression.

## Format Layout

Each serialized chunk consists of a **64-byte fixed header** followed by a
variable-length **payload**.

### Header (64 bytes)

| Offset | Size (bytes) | Field | Type | Description |
|--------|-------------|-------|------|-------------|
| 0 | 4 | magic | bytes | `b"KVCH"` — identifies valid KV cache chunks |
| 4 | 2 | version | uint16 | Format version, currently `1` |
| 6 | 1 | compression | uint8 | 0 = none, 1 = lz4, 2 = zstd |
| 7 | 1 | dtype_code | uint8 | 0 = float32, 1 = float16, 2 = bfloat16 |
| 8 | 2 | num_layers | uint16 | Number of layers in this chunk |
| 10 | 2 | num_kv_heads | uint16 | Number of KV attention heads |
| 12 | 2 | head_dim | uint16 | Dimension per head |
| 14 | 2 | seq_len | uint16 | Sequence length (tokens) in this chunk |
| 16 | 2 | token_start | uint16 | Start token index (inclusive) |
| 18 | 2 | token_end | uint16 | End token index (exclusive) |
| 20 | 2 | layer_start | uint16 | Start layer index (inclusive) |
| 22 | 2 | layer_end | uint16 | End layer index (exclusive) |
| 24 | 4 | uncompressed_size | uint32 | Size of uncompressed payload in bytes |
| 28 | 4 | compressed_size | uint32 | Size of (possibly compressed) payload in bytes |
| 32 | 16 | checksum | bytes | MD5 digest of the uncompressed payload |
| 48 | 16 | reserved | bytes | Reserved for future use, zero-padded |

### Payload

The payload contains raw contiguous bytes of K and V tensors in C-order (row-major).
The batch dimension is squeezed (removed) before serialization.

**Tensor layout within payload:**
For each layer in `[layer_start, layer_end)`:
1. Key tensor: shape `(num_kv_heads, seq_len, head_dim)`, contiguous C-order bytes
2. Value tensor: shape `(num_kv_heads, seq_len, head_dim)`, contiguous C-order bytes

## Chunk ID Convention

Chunk IDs follow the existing pipe-delimited format used in KV cache experiments:

```
m=tinyllama|l={seq_len}|s=token_layer_group|tb={token_block}|lg={layer_group}|ts={token_start}|te={token_end}|ls={layer_start}|le={layer_end}
```

DHT keys are derived via `DHTNode.hash_key(chunk_id, 16)`.

## Compression

- **0 (none):** Payload stored as raw bytes.
- **1 (lz4):** LZ4 frame compression. Fast compression/decompression, moderate ratio.
- **2 (zstd):** Zstandard compression. Better ratio, slightly slower.

When compression is enabled, `compressed_size` reflects the actual stored size,
while `uncompressed_size` records the original size for buffer pre-allocation
during decompression.

## Integrity

The `checksum` field contains an MD5 digest of the **uncompressed** payload.
On deserialization, the checksum is recomputed and compared to detect corruption.

## Versioning

The `version` field allows future format changes while maintaining backward
compatibility. Version `1` is the initial release described in this document.
