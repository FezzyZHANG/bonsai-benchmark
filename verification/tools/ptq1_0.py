"""Reference PTQ1_0 block decoder (mirrors dequantize_row_ptq1_0 in ggml-quants.c).

Block layout (128 weights, 28 bytes):
    qs[24]  5 trits/byte -> elements 0..119
    qh[2]   4 trits/byte -> elements 120..127
    d        fp16 group scale
"""
import numpy as np

POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
BLOCK_SIZE = 128
BLOCK_BYTES = 28


def dequantize_blocks(raw: np.ndarray) -> np.ndarray:
    """raw: (n_blocks, 28) uint8 -> (n_blocks, 128) float16."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if raw.ndim != 2 or raw.shape[1] != BLOCK_BYTES:
        raise ValueError(f"expected (n, {BLOCK_BYTES}) blocks, got {raw.shape}")
    nb = raw.shape[0]

    d = raw[:, 26:28].copy().view("<f2").reshape(nb)  # already float16
    qs = raw[:, :24].astype(np.uint16)
    qh = raw[:, 24:26].astype(np.uint16)

    out = np.empty((nb, BLOCK_SIZE), dtype=np.float16)

    # stage c=16: qs[0..15], 5 trits each -> elements 0..79
    for n in range(5):
        q = (qs[:, 0:16] * POW3[n]) & 0xFF
        xi = ((q * 3) >> 8).astype(np.int16)
        out[:, n * 16:(n + 1) * 16] = (xi - 1).astype(np.float16) * d[:, None]

    # stage c=8: qs[16..23], 5 trits each -> elements 80..119
    for n in range(5):
        q = (qs[:, 16:24] * POW3[n]) & 0xFF
        xi = ((q * 3) >> 8).astype(np.int16)
        out[:, 80 + n * 8:80 + (n + 1) * 8] = (xi - 1).astype(np.float16) * d[:, None]

    # qh: 4 trits per byte, 2 bytes -> elements 120..127
    for n in range(4):
        for h in range(2):
            q = (qh[:, h] * POW3[n]) & 0xFF
            xi = ((q * 3) >> 8).astype(np.int16)
            out[:, 120 + n * 2 + h] = (xi - 1).astype(np.float16) * d

    return out


def dequantize_tensor(raw_flat: np.ndarray, ne0: int, ne1: int) -> np.ndarray:
    """Dequantize a row-major GGUF tensor; returns shape (ne1, ne0) float16."""
    raw = np.asarray(raw_flat, dtype=np.uint8).reshape(-1)
    if ne0 % BLOCK_SIZE != 0:
        raise ValueError(f"ne0={ne0} not divisible by {BLOCK_SIZE}")
    if raw.size != (ne0 // BLOCK_SIZE) * BLOCK_BYTES * ne1:
        raise ValueError(f"raw size {raw.size} does not match {ne0}x{ne1} PTQ1_0")
    blocks = raw.reshape(-1, BLOCK_BYTES)
    vals = dequantize_blocks(blocks)
    return vals.reshape(ne1, ne0)
