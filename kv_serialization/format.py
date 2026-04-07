"""Binary format constants and header encode/decode for KV cache chunks.

See SERIALIZATION_FORMAT.md for the full specification.
"""

import struct
from dataclasses import dataclass
from typing import Tuple

import torch

# -- Magic & version --------------------------------------------------------
MAGIC = b"KVCH"
FORMAT_VERSION = 1

# -- Header layout -----------------------------------------------------------
# Total header size: 64 bytes
HEADER_SIZE = 64
HEADER_STRUCT = struct.Struct(
    "<"       # little-endian
    "4s"      # magic (4 bytes)
    "H"       # version (uint16)
    "B"       # compression (uint8)
    "B"       # dtype_code (uint8)
    "8H"      # dimensions: num_layers, num_kv_heads, head_dim, seq_len,
              #             token_start, token_end, layer_start, layer_end
    "I"       # uncompressed_size (uint32)
    "I"       # compressed_size (uint32)
    "16s"     # checksum (16 bytes, MD5)
    "16s"     # reserved (16 bytes)
)
assert HEADER_STRUCT.size == HEADER_SIZE

# -- Compression codes -------------------------------------------------------
COMPRESS_NONE = 0
COMPRESS_LZ4 = 1
COMPRESS_ZSTD = 2

# -- Dtype codes --------------------------------------------------------------
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_BF16 = 2

DTYPE_TO_CODE = {
    torch.float32: DTYPE_F32,
    torch.float16: DTYPE_F16,
    torch.bfloat16: DTYPE_BF16,
}

CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}


@dataclass
class ChunkHeader:
    """Decoded 64-byte header for a serialized KV cache chunk."""
    version: int
    compression: int
    dtype_code: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    token_start: int
    token_end: int
    layer_start: int
    layer_end: int
    uncompressed_size: int
    compressed_size: int
    checksum: bytes


def encode_header(header: ChunkHeader) -> bytes:
    """Encode a ChunkHeader into 64 bytes."""
    return HEADER_STRUCT.pack(
        MAGIC,
        header.version,
        header.compression,
        header.dtype_code,
        header.num_layers,
        header.num_kv_heads,
        header.head_dim,
        header.seq_len,
        header.token_start,
        header.token_end,
        header.layer_start,
        header.layer_end,
        header.uncompressed_size,
        header.compressed_size,
        header.checksum,
        b"\x00" * 16,  # reserved
    )


def decode_header(data: bytes) -> ChunkHeader:
    """Decode 64 bytes into a ChunkHeader. Validates magic and version."""
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"Header too short: expected {HEADER_SIZE} bytes, got {len(data)}"
        )

    (
        magic, version, compression, dtype_code,
        num_layers, num_kv_heads, head_dim, seq_len,
        token_start, token_end, layer_start, layer_end,
        uncompressed_size, compressed_size,
        checksum, _reserved,
    ) = HEADER_STRUCT.unpack(data[:HEADER_SIZE])

    if magic != MAGIC:
        raise ValueError(f"Invalid magic: expected {MAGIC!r}, got {magic!r}")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported version: expected {FORMAT_VERSION}, got {version}"
        )

    return ChunkHeader(
        version=version,
        compression=compression,
        dtype_code=dtype_code,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        token_start=token_start,
        token_end=token_end,
        layer_start=layer_start,
        layer_end=layer_end,
        uncompressed_size=uncompressed_size,
        compressed_size=compressed_size,
        checksum=checksum,
    )
