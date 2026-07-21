"""Payload packing (uint16 quantize, delta, gzip)."""

from __future__ import annotations

import base64
import gzip as _gzip

import numpy as np

def delta_u16(a: np.ndarray) -> np.ndarray:
    """Column-wise delta mod 2^16 (lossless; tiny values on ordered data)."""
    d = a.astype(np.int32)
    d[1:] = (d[1:] - d[:-1]) % 65536
    return d.astype("<u2")


def pack_u16(q: np.ndarray, compress: bool) -> str:
    """uint16 array -> base64; compressed = delta + byte-plane shuffle + gzip."""
    if compress:
        d = delta_u16(q).view(np.uint8).reshape(-1, 2)
        raw = _gzip.compress(
            np.ascontiguousarray(d[:, 0]).tobytes()
            + np.ascontiguousarray(d[:, 1]).tobytes(),
            6,
        )
    else:
        raw = np.ascontiguousarray(q, dtype="<u2").tobytes()
    return base64.b64encode(raw).decode("ascii")


def pack_f32(v: np.ndarray, compress: bool) -> str:
    raw = np.ascontiguousarray(v, dtype="<f4").tobytes()
    if compress:
        raw = _gzip.compress(raw, 6)
    return base64.b64encode(raw).decode("ascii")


def encode_norm(v: np.ndarray, lo: float, hi: float, *, quantize: bool,
                 compress: bool) -> dict:
    """Encode values normalized to [0,1] over [lo,hi] (u16 or f32)."""
    span = (hi - lo) or 1.0
    t = (np.asarray(v, dtype=np.float64) - lo) / span
    if quantize:
        q = np.round(np.clip(t, 0.0, 1.0) * 65535.0).astype("<u2")
        return {"dtype": "u16", "b64": pack_u16(q, compress)}
    return {"dtype": "f32", "b64": pack_f32(t.astype(np.float32), compress)}


def encode_codes(codes: np.ndarray, compress: bool) -> dict:
    q = np.ascontiguousarray(codes, dtype="<u2")
    return {"dtype": "u16", "b64": pack_u16(q, compress), "raw": True}


def pack_u32(values: np.ndarray, compress: bool) -> str:
    """uint32 array -> base64 (optional gzip; no delta)."""
    raw = np.ascontiguousarray(values, dtype="<u4").tobytes()
    if compress:
        raw = _gzip.compress(raw, 6)
    return base64.b64encode(raw).decode("ascii")

