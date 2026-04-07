"""Tests for the KV cache serialization module.

Covers header encode/decode, chunking, serialization round-trips,
checksum verification, and Kademlia integration.

Run:  pytest tests/test_serialization.py -v -s
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kv_serialization.format import (
    HEADER_SIZE,
    MAGIC,
    COMPRESS_NONE,
    DTYPE_F32,
    DTYPE_F16,
    ChunkHeader,
    encode_header,
    decode_header,
    DTYPE_TO_CODE,
)
from kv_serialization.serialize import (
    KVChunk,
    chunk_kv_cache,
    serialize_chunk,
    deserialize_chunk,
    reassemble_kv_cache,
    _build_chunk_id,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def make_fake_kv_cache(
    num_layers: int = 22,
    num_kv_heads: int = 4,
    seq_len: int = 64,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float32,
):
    """Create a fake past_key_values matching TinyLLaMA structure."""
    past_kv = []
    for layer_idx in range(num_layers):
        # Use deterministic values so we can verify round-trips
        k = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype)
        past_kv.append((k, v))
    return tuple(past_kv)


# ── Header tests ─────────────────────────────────────────────────────────

class TestHeader:
    def test_encode_decode_roundtrip(self):
        header = ChunkHeader(
            version=1,
            compression=0,
            dtype_code=DTYPE_F32,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            seq_len=32,
            token_start=0,
            token_end=32,
            layer_start=0,
            layer_end=2,
            uncompressed_size=65536,
            compressed_size=65536,
            checksum=b"\x00" * 16,
        )
        data = encode_header(header)
        assert len(data) == HEADER_SIZE

        decoded = decode_header(data)
        assert decoded.version == header.version
        assert decoded.compression == header.compression
        assert decoded.dtype_code == header.dtype_code
        assert decoded.num_layers == header.num_layers
        assert decoded.num_kv_heads == header.num_kv_heads
        assert decoded.head_dim == header.head_dim
        assert decoded.seq_len == header.seq_len
        assert decoded.token_start == header.token_start
        assert decoded.token_end == header.token_end
        assert decoded.layer_start == header.layer_start
        assert decoded.layer_end == header.layer_end
        assert decoded.uncompressed_size == header.uncompressed_size
        assert decoded.compressed_size == header.compressed_size
        assert decoded.checksum == header.checksum

    def test_invalid_magic_raises(self):
        header = ChunkHeader(
            version=1, compression=0, dtype_code=0,
            num_layers=1, num_kv_heads=4, head_dim=64, seq_len=32,
            token_start=0, token_end=32, layer_start=0, layer_end=1,
            uncompressed_size=0, compressed_size=0, checksum=b"\x00" * 16,
        )
        data = bytearray(encode_header(header))
        data[0:4] = b"XXXX"
        with pytest.raises(ValueError, match="Invalid magic"):
            decode_header(bytes(data))

    def test_short_data_raises(self):
        with pytest.raises(ValueError, match="Header too short"):
            decode_header(b"\x00" * 10)


# ── Chunk ID tests ───────────────────────────────────────────────────────

class TestChunkId:
    def test_format_matches_convention(self):
        cid = _build_chunk_id(
            seq_len=512, token_block=32, layer_group=2,
            token_start=0, token_end=32, layer_start=0, layer_end=2,
        )
        assert cid == (
            "m=tinyllama|l=512|s=token_layer_group|tb=32|"
            "lg=2|ts=0|te=32|ls=0|le=2"
        )


# ── Chunking tests ───────────────────────────────────────────────────────

class TestChunking:
    def test_chunk_count(self):
        """22 layers / 2 = 11 groups, 64 tokens / 32 = 2 blocks → 22 chunks."""
        past_kv = make_fake_kv_cache(num_layers=22, seq_len=64)
        chunks = chunk_kv_cache(past_kv, seq_len=64, token_block=32, layer_group=2)
        assert len(chunks) == 22  # 11 * 2

    def test_chunk_covers_all_layers_and_tokens(self):
        past_kv = make_fake_kv_cache(num_layers=4, seq_len=64)
        chunks = chunk_kv_cache(past_kv, seq_len=64, token_block=32, layer_group=2)

        layer_ranges = set()
        token_ranges = set()
        for c in chunks:
            layer_ranges.add((c.layer_start, c.layer_end))
            token_ranges.add((c.token_start, c.token_end))

        assert layer_ranges == {(0, 2), (2, 4)}
        assert token_ranges == {(0, 32), (32, 64)}

    def test_chunk_tensor_shapes(self):
        past_kv = make_fake_kv_cache(num_layers=4, seq_len=64)
        chunks = chunk_kv_cache(past_kv, seq_len=64, token_block=32, layer_group=2)

        for chunk in chunks:
            num_layers_in_chunk = chunk.layer_end - chunk.layer_start
            assert len(chunk.keys) == num_layers_in_chunk
            assert len(chunk.values) == num_layers_in_chunk
            for k, v in zip(chunk.keys, chunk.values):
                assert k.shape == (4, 32, 64)  # (heads, tokens, dim)
                assert v.shape == (4, 32, 64)

    def test_uneven_chunk_boundaries(self):
        """When layers/tokens don't divide evenly, last chunk is smaller."""
        past_kv = make_fake_kv_cache(num_layers=5, seq_len=50)
        chunks = chunk_kv_cache(past_kv, seq_len=50, token_block=32, layer_group=2)

        # 3 layer groups (0-2, 2-4, 4-5), 2 token blocks (0-32, 32-50)
        assert len(chunks) == 6

        # Check last token block has 18 tokens
        last_chunk = [c for c in chunks if c.token_end == 50][0]
        assert last_chunk.keys[0].shape[1] == 18

        # Check last layer group has 1 layer
        single_layer = [c for c in chunks if c.layer_end == 5][0]
        assert len(single_layer.keys) == 1


# ── Serialization round-trip tests ───────────────────────────────────────

class TestSerializeDeserialize:
    def test_roundtrip_no_compression(self):
        torch.manual_seed(42)
        past_kv = make_fake_kv_cache(num_layers=4, seq_len=32)
        chunks = chunk_kv_cache(past_kv, seq_len=32, token_block=16, layer_group=2)

        for chunk in chunks:
            data = serialize_chunk(chunk, compression=COMPRESS_NONE)
            restored = deserialize_chunk(data, chunk_id=chunk.chunk_id)

            assert restored.chunk_id == chunk.chunk_id
            assert restored.layer_start == chunk.layer_start
            assert restored.layer_end == chunk.layer_end
            assert restored.token_start == chunk.token_start
            assert restored.token_end == chunk.token_end
            assert restored.num_kv_heads == chunk.num_kv_heads
            assert restored.head_dim == chunk.head_dim
            assert restored.dtype == chunk.dtype

            for k_orig, k_rest in zip(chunk.keys, restored.keys):
                assert torch.equal(k_orig, k_rest)
            for v_orig, v_rest in zip(chunk.values, restored.values):
                assert torch.equal(v_orig, v_rest)

    def test_header_size_in_output(self):
        past_kv = make_fake_kv_cache(num_layers=2, seq_len=16)
        chunks = chunk_kv_cache(past_kv, seq_len=16, token_block=16, layer_group=2)
        data = serialize_chunk(chunks[0])
        assert len(data) > HEADER_SIZE
        assert data[:4] == MAGIC

    def test_checksum_detects_corruption(self):
        past_kv = make_fake_kv_cache(num_layers=2, seq_len=16)
        chunks = chunk_kv_cache(past_kv, seq_len=16, token_block=16, layer_group=2)
        data = bytearray(serialize_chunk(chunks[0]))

        # Corrupt a byte in the payload
        data[HEADER_SIZE + 10] ^= 0xFF

        with pytest.raises(ValueError, match="Checksum mismatch"):
            deserialize_chunk(bytes(data))

    def test_float16_roundtrip(self):
        past_kv = make_fake_kv_cache(num_layers=2, seq_len=16, dtype=torch.float16)
        chunks = chunk_kv_cache(past_kv, seq_len=16, token_block=16, layer_group=2)

        data = serialize_chunk(chunks[0])
        restored = deserialize_chunk(data)

        assert restored.dtype == torch.float16
        for k_orig, k_rest in zip(chunks[0].keys, restored.keys):
            assert torch.equal(k_orig, k_rest)


# ── Reassembly tests ─────────────────────────────────────────────────────

class TestReassemble:
    def test_full_roundtrip(self):
        """chunk → serialize → deserialize → reassemble matches original."""
        torch.manual_seed(123)
        num_layers, seq_len = 6, 48
        past_kv = make_fake_kv_cache(num_layers=num_layers, seq_len=seq_len)

        chunks = chunk_kv_cache(past_kv, seq_len=seq_len, token_block=16, layer_group=2)

        restored_chunks = []
        for chunk in chunks:
            data = serialize_chunk(chunk)
            restored_chunks.append(deserialize_chunk(data, chunk_id=chunk.chunk_id))

        rebuilt = reassemble_kv_cache(
            restored_chunks,
            num_layers=num_layers,
            seq_len=seq_len,
            num_kv_heads=4,
            head_dim=64,
        )

        assert len(rebuilt) == num_layers
        for layer_idx in range(num_layers):
            k_orig, v_orig = past_kv[layer_idx]
            k_rebuilt, v_rebuilt = rebuilt[layer_idx]
            assert k_rebuilt.shape == k_orig.shape
            assert v_rebuilt.shape == v_orig.shape
            assert torch.equal(k_orig, k_rebuilt)
            assert torch.equal(v_orig, v_rebuilt)

    def test_reassemble_restores_batch_dim(self):
        past_kv = make_fake_kv_cache(num_layers=2, seq_len=16)
        chunks = chunk_kv_cache(past_kv, seq_len=16, token_block=16, layer_group=2)

        rebuilt = reassemble_kv_cache(
            chunks, num_layers=2, seq_len=16, num_kv_heads=4, head_dim=64,
        )

        for k, v in rebuilt:
            assert k.shape[0] == 1  # batch dim
            assert v.shape[0] == 1
