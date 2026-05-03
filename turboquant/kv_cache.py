"""
TurboQuantKVCache
==================
Production-quality compressed KV cache using correct Lloyd-Max codebooks.

Design decisions (learned from real implementation experience):
  - Uses TurboQuant_mse only (all bits to Lloyd-Max, no QJL in cache)
  - Stores keys compressed (uint8 indices + norm) → big memory savings
  - Values are stored compressed too (decompress on attend)
  - inner_product_batch() uses vectorized centroid gather for fast attention
  - One TurboQuant instance per head (each head gets its own rotation seed)

Expected quality (matches dejan.ai reference implementation):
  4-bit keys: cosine sim 0.995+, identical model outputs
  3-bit keys: cosine sim 0.983+, negligible quality loss
  2-bit keys: cosine sim 0.940,  minor rephrase
"""

import numpy as np
from typing import Optional, List, Tuple

from .core import TurboQuant, CompressedVector


class TurboQuantKVCache:
    """
    Compressed KV cache for transformer inference.

    Parameters
    ----------
    num_heads : int
        Number of KV heads (use num_key_value_heads for GQA models).
    head_dim : int
        Dimension per head.
    bits : int
        Bits per dimension. 4 = lossless, 3 = aggressive. Default: 4.
    seed : int
        Random seed.
    verbose : bool
        Print codebook build progress on first use.

    Usage
    -----
    >>> cache = TurboQuantKVCache(num_heads=8, head_dim=128, bits=4)
    >>> # During each forward pass token:
    >>> cache.append(keys, values)  # keys: (num_heads, head_dim)
    >>> # During attention:
    >>> output = cache.attend(query)  # query: (num_heads, head_dim)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        bits: int = 4,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bits = bits

        # One quantizer per head — different rotation per head (seed + h)
        self.quantizers = [
            TurboQuant(dim=head_dim, bits=bits, mode="mse", seed=seed + h, verbose=(verbose and h == 0))
            for h in range(num_heads)
        ]

        # Storage: list of CompressedVector per head, for keys and values
        self._keys:   List[List[CompressedVector]] = [[] for _ in range(num_heads)]
        self._values: List[List[CompressedVector]] = [[] for _ in range(num_heads)]
        self._seq_len = 0

    # ── Append ────────────────────────────────────────────────────────────────

    def append(self, keys: np.ndarray, values: np.ndarray):
        """
        Add one token's KV pair to the cache.

        Parameters
        ----------
        keys   : np.ndarray, shape (num_heads, head_dim), float32
        values : np.ndarray, shape (num_heads, head_dim), float32
        """
        keys   = np.asarray(keys,   dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        assert keys.shape   == (self.num_heads, self.head_dim)
        assert values.shape == (self.num_heads, self.head_dim)

        for h in range(self.num_heads):
            self._keys[h].append(self.quantizers[h].compress(keys[h]))
            self._values[h].append(self.quantizers[h].compress(values[h]))
        self._seq_len += 1

    # ── Attend ────────────────────────────────────────────────────────────────

    def attend(
        self,
        query: np.ndarray,
        scale: Optional[float] = None,
        causal_mask: bool = True,
        return_weights: bool = False,
    ) -> np.ndarray:
        """
        Compute multi-head attention output against the compressed KV cache.

        Uses vectorized inner product computation — queries all compressed
        keys in batch without materializing full fp16 key tensors.

        Parameters
        ----------
        query : np.ndarray, shape (num_heads, head_dim), float32
        scale : float, optional. Default: 1/sqrt(head_dim).
        causal_mask : bool. Apply causal mask (all past positions attend). Default True.
        return_weights : bool. If True, also return attention weights.

        Returns
        -------
        output : np.ndarray, shape (num_heads, head_dim)
        weights : np.ndarray, shape (num_heads, seq_len), only if return_weights=True
        """
        if self._seq_len == 0:
            raise ValueError("Cache is empty. Call append() first.")

        query = np.asarray(query, dtype=np.float32)
        if scale is None:
            scale = 1.0 / np.sqrt(self.head_dim)

        outputs  = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
        all_attn = np.zeros((self.num_heads, self._seq_len), dtype=np.float32)

        for h in range(self.num_heads):
            q_h = query[h]

            # Vectorized: compute all attention logits at once
            logits = self.quantizers[h].inner_product_batch(q_h, self._keys[h])
            logits = logits * scale

            # Numerically stable softmax
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum() + 1e-9
            all_attn[h] = weights

            # Weighted sum of decompressed values
            # Decompress all values at once for efficiency
            V_hat = self.quantizers[h].decompress_batch(self._values[h])  # (seq_len, head_dim)
            outputs[h] = weights @ V_hat  # (head_dim,)

        if return_weights:
            return outputs, all_attn
        return outputs

    # ── Utilities ─────────────────────────────────────────────────────────────

    def clear(self):
        """Reset the cache."""
        self._keys   = [[] for _ in range(self.num_heads)]
        self._values = [[] for _ in range(self.num_heads)]
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def memory_bytes(self) -> int:
        """Approximate bytes used by all compressed KV vectors."""
        bytes_per_vec = self.quantizers[0].bytes_per_vector()
        return self._seq_len * self.num_heads * 2 * bytes_per_vec  # *2 for K and V

    def memory_bytes_fp16(self) -> int:
        """Memory if stored in fp16."""
        return self._seq_len * self.num_heads * self.head_dim * 2 * 2  # *2 bytes *2 for KV

    def compression_ratio(self) -> float:
        comp = self.memory_bytes()
        return self.memory_bytes_fp16() / comp if comp > 0 else float("inf")

    def stats(self) -> dict:
        return {
            "seq_len": self._seq_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "bits": self.bits,
            "memory_compressed_kb": self.memory_bytes() / 1024,
            "memory_fp16_kb": self.memory_bytes_fp16() / 1024,
            "compression_ratio": self.compression_ratio(),
            "effective_bits": self.quantizers[0].effective_bits_per_element(),
        }

    def __repr__(self) -> str:
        return (
            f"TurboQuantKVCache(heads={self.num_heads}, head_dim={self.head_dim}, "
            f"bits={self.bits}, seq_len={self._seq_len}, "
            f"compression={self.compression_ratio():.2f}x vs fp16)"
        )
