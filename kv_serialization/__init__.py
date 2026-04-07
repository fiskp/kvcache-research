"""KV Cache Serialization for P2P-RAGCache (Milestone 2, Task 3).

Converts real TinyLLaMA KV cache tensors into a compact binary format
for storage and retrieval across the Kademlia DHT network.
"""

from .format import (
    MAGIC,
    FORMAT_VERSION,
    HEADER_SIZE,
    COMPRESS_NONE,
    COMPRESS_LZ4,
    COMPRESS_ZSTD,
    DTYPE_F32,
    DTYPE_F16,
    DTYPE_BF16,
    DTYPE_TO_CODE,
    CODE_TO_DTYPE,
    ChunkHeader,
    encode_header,
    decode_header,
)

__all__ = [
    "MAGIC",
    "FORMAT_VERSION",
    "HEADER_SIZE",
    "COMPRESS_NONE",
    "COMPRESS_LZ4",
    "COMPRESS_ZSTD",
    "DTYPE_F32",
    "DTYPE_F16",
    "DTYPE_BF16",
    "DTYPE_TO_CODE",
    "CODE_TO_DTYPE",
    "ChunkHeader",
    "encode_header",
    "decode_header",
]
