"""Compression adapter for KV cache chunks.

Supports lz4 and zstd compression with graceful handling of missing
optional dependencies.
"""

from .format import COMPRESS_LZ4, COMPRESS_ZSTD, COMPRESS_NONE


def _require_lz4():
    try:
        import lz4.frame
        return lz4.frame
    except ImportError:
        raise ImportError(
            "lz4 compression requested but lz4 is not installed. "
            "Install it with: pip install lz4"
        )


def _require_zstd():
    try:
        import zstandard
        return zstandard
    except ImportError:
        raise ImportError(
            "zstd compression requested but zstandard is not installed. "
            "Install it with: pip install zstandard"
        )


def compress(data: bytes, method: int) -> bytes:
    """Compress data using the specified method.

    Args:
        data: Raw bytes to compress.
        method: COMPRESS_LZ4 (1) or COMPRESS_ZSTD (2).

    Returns:
        Compressed bytes.
    """
    if method == COMPRESS_LZ4:
        lz4_frame = _require_lz4()
        return lz4_frame.compress(data)
    elif method == COMPRESS_ZSTD:
        zstd = _require_zstd()
        cctx = zstd.ZstdCompressor()
        return cctx.compress(data)
    else:
        raise ValueError(f"Unknown compression method: {method}")


def decompress(data: bytes, method: int, uncompressed_size: int) -> bytes:
    """Decompress data using the specified method.

    Args:
        data: Compressed bytes.
        method: COMPRESS_LZ4 (1) or COMPRESS_ZSTD (2).
        uncompressed_size: Expected size of decompressed data (for buffer pre-allocation).

    Returns:
        Decompressed bytes.
    """
    if method == COMPRESS_LZ4:
        lz4_frame = _require_lz4()
        return lz4_frame.decompress(data)
    elif method == COMPRESS_ZSTD:
        zstd = _require_zstd()
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data, max_output_size=uncompressed_size)
    else:
        raise ValueError(f"Unknown compression method: {method}")
