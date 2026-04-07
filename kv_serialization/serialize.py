"""KV cache chunking, serialization, and reassembly.

Provides the core API for splitting real KV cache tensors into chunks,
serializing them to the binary format, and reassembling them.
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .format import (
    COMPRESS_NONE,
    CODE_TO_DTYPE,
    DTYPE_TO_CODE,
    HEADER_SIZE,
    ChunkHeader,
    encode_header,
    decode_header,
)


@dataclass
class KVChunk:
    """A chunk of KV cache data with metadata.

    Holds the key and value tensors for a subset of layers and token positions.
    """
    chunk_id: str
    layer_start: int
    layer_end: int
    token_start: int
    token_end: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    keys: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)


def _build_chunk_id(
    seq_len: int,
    token_block: int,
    layer_group: int,
    token_start: int,
    token_end: int,
    layer_start: int,
    layer_end: int,
) -> str:
    """Build a chunk ID matching the existing experiment convention."""
    return (
        f"m=tinyllama|l={seq_len}|s=token_layer_group|tb={token_block}|"
        f"lg={layer_group}|ts={token_start}|te={token_end}|"
        f"ls={layer_start}|le={layer_end}"
    )


def chunk_kv_cache(
    past_key_values: Tuple,
    seq_len: int,
    token_block: int = 32,
    layer_group: int = 2,
) -> List[KVChunk]:
    """Split a full KV cache into chunks along token and layer dimensions.

    Args:
        past_key_values: Tuple of (key, value) tensor pairs per layer.
            Each tensor has shape (batch, num_kv_heads, seq_len, head_dim).
        seq_len: Total sequence length.
        token_block: Number of tokens per chunk along the sequence dimension.
        layer_group: Number of layers per chunk.

    Returns:
        List of KVChunk objects covering the entire KV cache.
    """
    num_layers = len(past_key_values)
    first_key = past_key_values[0][0]
    num_kv_heads = first_key.shape[1]
    head_dim = first_key.shape[3]
    dtype = first_key.dtype

    chunks = []
    for layer_start in range(0, num_layers, layer_group):
        layer_end = min(layer_start + layer_group, num_layers)
        for token_start in range(0, seq_len, token_block):
            token_end = min(token_start + token_block, seq_len)

            chunk_id = _build_chunk_id(
                seq_len, token_block, layer_group,
                token_start, token_end, layer_start, layer_end,
            )

            keys = []
            values = []
            for layer_idx in range(layer_start, layer_end):
                k, v = past_key_values[layer_idx]
                # Squeeze batch dim, slice token range
                keys.append(k.squeeze(0)[:, token_start:token_end, :].contiguous())
                values.append(v.squeeze(0)[:, token_start:token_end, :].contiguous())

            chunks.append(KVChunk(
                chunk_id=chunk_id,
                layer_start=layer_start,
                layer_end=layer_end,
                token_start=token_start,
                token_end=token_end,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                keys=keys,
                values=values,
            ))

    return chunks


def _concat_payload(chunk: KVChunk) -> bytes:
    """Concatenate all K,V tensors in a chunk into a single byte buffer."""
    parts = []
    for k, v in zip(chunk.keys, chunk.values):
        parts.append(k.numpy().tobytes())
        parts.append(v.numpy().tobytes())
    return b"".join(parts)


def _compute_checksum(payload: bytes) -> bytes:
    """Compute MD5 checksum of the payload."""
    return hashlib.md5(payload).digest()


def serialize_chunk(chunk: KVChunk, compression: int = COMPRESS_NONE) -> bytes:
    """Serialize a KVChunk into the binary format.

    Args:
        chunk: The KV cache chunk to serialize.
        compression: Compression method (0=none, 1=lz4, 2=zstd).

    Returns:
        Binary data: 64-byte header + payload.
    """
    payload = _concat_payload(chunk)
    checksum = _compute_checksum(payload)
    uncompressed_size = len(payload)

    if compression != COMPRESS_NONE:
        from .compress import compress
        payload = compress(payload, compression)

    header = ChunkHeader(
        version=1,
        compression=compression,
        dtype_code=DTYPE_TO_CODE[chunk.dtype],
        num_layers=chunk.layer_end - chunk.layer_start,
        num_kv_heads=chunk.num_kv_heads,
        head_dim=chunk.head_dim,
        seq_len=chunk.token_end - chunk.token_start,
        token_start=chunk.token_start,
        token_end=chunk.token_end,
        layer_start=chunk.layer_start,
        layer_end=chunk.layer_end,
        uncompressed_size=uncompressed_size,
        compressed_size=len(payload),
        checksum=checksum,
    )

    return encode_header(header) + payload


def deserialize_chunk(data: bytes, chunk_id: str = "") -> KVChunk:
    """Deserialize binary data back into a KVChunk.

    Args:
        data: Binary data produced by serialize_chunk().
        chunk_id: Optional chunk ID to attach to the result.

    Returns:
        Reconstructed KVChunk with tensors.

    Raises:
        ValueError: If the data is corrupt or the checksum doesn't match.
    """
    header = decode_header(data)
    payload = data[HEADER_SIZE:]

    if header.compression != COMPRESS_NONE:
        from .compress import decompress
        payload = decompress(payload, header.compression, header.uncompressed_size)

    # Verify checksum
    actual_checksum = _compute_checksum(payload)
    if actual_checksum != header.checksum:
        raise ValueError(
            f"Checksum mismatch: expected {header.checksum.hex()}, "
            f"got {actual_checksum.hex()}"
        )

    dtype = CODE_TO_DTYPE[header.dtype_code]
    num_layers = header.layer_end - header.layer_start
    seq_len = header.seq_len
    tensor_shape = (header.num_kv_heads, seq_len, header.head_dim)
    element_size = torch.tensor([], dtype=dtype).element_size()
    tensor_bytes = header.num_kv_heads * seq_len * header.head_dim * element_size

    keys = []
    values = []
    offset = 0
    for _ in range(num_layers):
        k_buf = payload[offset:offset + tensor_bytes]
        offset += tensor_bytes
        v_buf = payload[offset:offset + tensor_bytes]
        offset += tensor_bytes

        k = torch.frombuffer(bytearray(k_buf), dtype=dtype).reshape(tensor_shape).clone()
        v = torch.frombuffer(bytearray(v_buf), dtype=dtype).reshape(tensor_shape).clone()
        keys.append(k)
        values.append(v)

    return KVChunk(
        chunk_id=chunk_id,
        layer_start=header.layer_start,
        layer_end=header.layer_end,
        token_start=header.token_start,
        token_end=header.token_end,
        num_kv_heads=header.num_kv_heads,
        head_dim=header.head_dim,
        dtype=dtype,
        keys=keys,
        values=values,
    )


def reassemble_kv_cache(
    chunks: List[KVChunk],
    num_layers: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple:
    """Reassemble a full past_key_values tuple from KVChunk objects.

    Args:
        chunks: List of KVChunk objects covering the full KV cache.
        num_layers: Total number of layers.
        seq_len: Total sequence length.
        num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension.
        dtype: Tensor dtype.

    Returns:
        Tuple of (key, value) tensor pairs per layer, each with shape
        (1, num_kv_heads, seq_len, head_dim) — batch dim restored.
    """
    # Pre-allocate full tensors
    all_keys = [
        torch.zeros(num_kv_heads, seq_len, head_dim, dtype=dtype)
        for _ in range(num_layers)
    ]
    all_values = [
        torch.zeros(num_kv_heads, seq_len, head_dim, dtype=dtype)
        for _ in range(num_layers)
    ]

    for chunk in chunks:
        for i, layer_idx in enumerate(range(chunk.layer_start, chunk.layer_end)):
            ts, te = chunk.token_start, chunk.token_end
            all_keys[layer_idx][:, ts:te, :] = chunk.keys[i]
            all_values[layer_idx][:, ts:te, :] = chunk.values[i]

    # Restore batch dimension
    return tuple(
        (k.unsqueeze(0), v.unsqueeze(0))
        for k, v in zip(all_keys, all_values)
    )
