"""
TurboQuant
==========
Python implementation of TurboQuant: near-optimal online vector quantization.

Based on:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  Zandieh, Daliri, Hadian, Mirrokni — Google Research, ICLR 2026
  arXiv: https://arxiv.org/abs/2504.19874

Core algorithm:
  1. Random orthogonal rotation (induces Beta distribution on coordinates)
  2. Lloyd-Max scalar quantization tuned to Beta((d-1)/2, (d-1)/2)

Expected quality at dim=256:
  4-bit: cosine sim 0.995, 7.9x compression vs fp32
  3-bit: cosine sim 0.983, 10.4x compression vs fp32
  2-bit: cosine sim 0.940, 15.5x compression vs fp32
"""

from .core import TurboQuant, CompressedVector
from .kv_cache import TurboQuantKVCache
from .codebook import lloyd_max_codebook

__version__ = "0.4.1"
__paper__ = "https://arxiv.org/abs/2504.19874"

__all__ = [
    "TurboQuant",
    "CompressedVector",
    "TurboQuantKVCache",
    "lloyd_max_codebook",
]
