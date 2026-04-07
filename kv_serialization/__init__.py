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
from .serialize import (
    KVChunk,
    chunk_kv_cache,
    serialize_chunk,
    deserialize_chunk,
    reassemble_kv_cache,
)
from .integration import (
    store_kv_chunk,
    retrieve_kv_chunk,
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
    "KVChunk",
    "chunk_kv_cache",
    "serialize_chunk",
    "deserialize_chunk",
    "reassemble_kv_cache",
    "store_kv_chunk",
    "retrieve_kv_chunk",
]
