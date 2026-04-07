"""Kademlia DHT integration for serialized KV cache chunks.

Provides store/retrieve wrappers that serialize KVChunk objects and
store them in the Kademlia network using the existing DHTNode.hash_key
convention.
"""

from typing import Tuple

from dht_comparison.base import DHTNode
from dht_comparison.kademlia import KademliaNode

from .format import COMPRESS_NONE
from .serialize import KVChunk, serialize_chunk, deserialize_chunk

ID_BITS = 16


def store_kv_chunk(
    node: KademliaNode,
    chunk: KVChunk,
    compression: int = COMPRESS_NONE,
    id_bits: int = ID_BITS,
) -> Tuple[int, int]:
    """Serialize and store a KV cache chunk in the Kademlia DHT.

    Args:
        node: A KademliaNode to issue the store through.
        chunk: The KVChunk to store.
        compression: Compression method (0=none, 1=lz4, 2=zstd).
        id_bits: Bit width for DHT key hashing.

    Returns:
        Tuple of (dht_key, bytes_stored).

    Raises:
        RuntimeError: If the DHT store operation fails.
    """
    data = serialize_chunk(chunk, compression=compression)
    dht_key = DHTNode.hash_key(chunk.chunk_id, id_bits)

    success = node.store(dht_key, data)
    if not success:
        raise RuntimeError(
            f"Failed to store chunk {chunk.chunk_id!r} (key={dht_key})"
        )

    return dht_key, len(data)


def retrieve_kv_chunk(
    node: KademliaNode,
    chunk_id: str,
    id_bits: int = ID_BITS,
) -> KVChunk:
    """Retrieve and deserialize a KV cache chunk from the Kademlia DHT.

    Args:
        node: A KademliaNode to issue the retrieve through.
        chunk_id: The chunk ID string used when storing.
        id_bits: Bit width for DHT key hashing.

    Returns:
        Deserialized KVChunk.

    Raises:
        KeyError: If the chunk is not found in the DHT.
        ValueError: If the retrieved data is corrupt.
    """
    dht_key = DHTNode.hash_key(chunk_id, id_bits)
    data = node.retrieve(dht_key)

    if data is None:
        raise KeyError(f"Chunk {chunk_id!r} not found (key={dht_key})")

    return deserialize_chunk(data, chunk_id=chunk_id)
