"""
TurboQuant PyTorch Backend
===========================
GPU-accelerated version of the core TurboQuant operations.
Works with torch.Tensor instead of np.ndarray.

This is the backend used by the HuggingFace cache integration.
It keeps everything in PyTorch so there's no CPU<->GPU transfer overhead.

Key operations:
  - compress(x)            : rotate + quantize, returns (packed_bytes, norms)
  - decompress(packed, norms): unpack + centroid gather + undo rotation
  - attention_scores(q, k_packed, k_norms): fused or fallback logits

Bit-packing (v2):
  For bits <= 4, two indices are stored per byte (nibble packing).
  This halves storage vs naïve uint8-per-index, matching paper claims.
"""

import math
import torch
import numpy as np
from typing import Tuple, Optional

from .codebook import lloyd_max_codebook, build_boundaries
from .triton_kernel import turboquant_attention_scores, is_triton_available


class TurboQuantTorch:
    """
    PyTorch-native TurboQuant quantizer.

    All operations run on the same device as the input tensors.
    Rotation matrix and codebook are registered as buffers so they
    move with .to(device) calls.

    Parameters
    ----------
    dim : int
        Vector dimension.
    bits : int
        Bits per dimension (2, 3, or 4).
    seed : int
        Random seed for rotation matrix.
    verbose : bool
        Print codebook build progress.
    """

    def __init__(self, dim: int, bits: int = 4, seed: int = 42, verbose: bool = False):
        self.dim = dim
        self.bits = bits
        self.num_levels = 2 ** bits

        # ── Rotation matrix (orthogonal, from QR decomposition) ──────────────
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((dim, dim)).astype(np.float32)
        Q, _ = np.linalg.qr(G)
        self._R   = torch.from_numpy(Q)         # (dim, dim)
        self._R_T = torch.from_numpy(Q.T)       # (dim, dim)

        # ── Lloyd-Max codebook ────────────────────────────────────────────────
        centroids_np  = lloyd_max_codebook(dim=dim, bits=bits, verbose=verbose)
        boundaries_np = build_boundaries(centroids_np)
        self._centroids  = torch.from_numpy(centroids_np)           # (num_levels,)
        self._boundaries = torch.from_numpy(boundaries_np)          # (num_levels-1,)

        self._device = torch.device("cpu")

    def to(self, device) -> "TurboQuantTorch":
        """Move all tensors to device (mirrors nn.Module.to())."""
        self._device     = torch.device(device)
        self._R          = self._R.to(device)
        self._R_T        = self._R_T.to(device)
        self._centroids  = self._centroids.to(device)
        self._boundaries = self._boundaries.to(device)
        return self

    @property
    def device(self):
        return self._device

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _vals_per_byte(self) -> int:
        """How many quantized indices fit in one byte for this bit-width."""
        if self.bits >= 8:
            return 1
        return 8 // self.bits  # e.g. 4 for 2-bit, 2 for 4-bit

    def _pack_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Bit-pack quantized indices into a compact byte tensor.

        Packs `vpb = 8 // bits` indices per byte using left-shifted slots.
        If dim is not a multiple of vpb the tensor is zero-padded.

        Parameters
        ----------
        indices : Tensor, shape (..., dim), uint8, values in [0, num_levels-1]

        Returns
        -------
        packed : Tensor, shape (..., packed_dim), uint8
                 packed_dim = ceil(dim / vpb)
        """
        if self.bits >= 8:
            return indices

        vpb = self._vals_per_byte()   # values per byte
        mask = (1 << self.bits) - 1   # e.g. 0x0F for 4-bit, 0x03 for 2-bit

        *batch, dim = indices.shape
        # Pad so dim is a multiple of vpb
        pad_len = (-dim) % vpb
        if pad_len:
            pad = torch.zeros(*batch, pad_len, dtype=torch.uint8, device=indices.device)
            indices = torch.cat([indices, pad], dim=-1)

        # Reshape to (..., packed_dim, vpb) and fold into bytes
        padded_dim = dim + pad_len
        grouped = indices.reshape(*batch, padded_dim // vpb, vpb)  # (..., pd, vpb)

        packed = torch.zeros(*batch, padded_dim // vpb, dtype=torch.uint8, device=indices.device)
        for slot in range(vpb):
            shift = (vpb - 1 - slot) * self.bits   # MSB first
            packed = packed | ((grouped[..., slot] & mask) << shift).to(torch.uint8)

        return packed

    def _unpack_indices(self, packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
        """
        Unpack a bit-packed byte tensor back to per-element indices.

        Parameters
        ----------
        packed   : Tensor, shape (..., packed_dim), uint8
        orig_dim : int — original unpadded dimension

        Returns
        -------
        indices : Tensor, shape (..., orig_dim), uint8
        """
        if self.bits >= 8:
            return packed

        vpb  = self._vals_per_byte()
        mask = (1 << self.bits) - 1

        *batch, pd = packed.shape
        # Extract each slot
        slots = []
        for slot in range(vpb):
            shift = (vpb - 1 - slot) * self.bits
            slots.append(((packed >> shift) & mask).to(torch.uint8))  # (..., pd)

        # Interleave slots → (..., pd * vpb) then trim
        interleaved = torch.stack(slots, dim=-1).reshape(*batch, pd * vpb)
        return interleaved[..., :orig_dim]

    # ── Compress ─────────────────────────────────────────────────────────────

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a batch of vectors with bit-packing.

        Parameters
        ----------
        x : Tensor, shape (..., dim), any dtype

        Returns
        -------
        packed : Tensor, shape (..., packed_dim), dtype uint8
                 packed_dim = ceil(dim * bits / 8)
        norms  : Tensor, shape (...,), dtype float32
        """
        x = x.to(dtype=torch.float32, device=self._device)

        # Extract norms
        norms = x.norm(dim=-1, keepdim=True)          # (..., 1)
        safe_norms = norms.clamp(min=1e-9)
        x_unit = x / safe_norms                        # (..., dim)

        # Random rotation
        x_rot = x_unit @ self._R.T                     # (..., dim)

        # Lloyd-Max quantize: searchsorted on boundaries
        flat = x_rot.reshape(-1, self.dim)             # (N, dim)
        indices_flat = torch.bucketize(flat, self._boundaries)  # (N, dim)
        indices = indices_flat.reshape(x_rot.shape).to(torch.uint8)

        # Bit-pack indices to save storage
        packed = self._pack_indices(indices)

        return packed, norms.squeeze(-1).to(torch.float32)

    # ── Decompress ────────────────────────────────────────────────────────────

    def decompress(
        self, packed: torch.Tensor, norms: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct vectors from bit-packed compressed representation.

        Parameters
        ----------
        packed : Tensor, shape (..., packed_dim), uint8
        norms  : Tensor, shape (...,), float32

        Returns
        -------
        Tensor, shape (..., dim), float32
        """
        # Unpack nibble-packed bytes back to per-dimension indices
        indices = self._unpack_indices(packed, orig_dim=self.dim)  # (..., dim)

        # Gather centroid values
        x_hat_rot = self._centroids[indices.long()]    # (..., dim)

        # Undo rotation
        x_hat_unit = x_hat_rot @ self._R               # (..., dim)  (R^T)^T = R

        # Restore norm
        return x_hat_unit * norms.unsqueeze(-1)

    # ── Attention scores ─────────────────────────────────────────────────────

    def attention_scores(
        self,
        query: torch.Tensor,          # (..., dim)
        k_packed: torch.Tensor,       # (seq_len, packed_dim) uint8
        k_norms: torch.Tensor,        # (seq_len,) float32
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute attention logits between query and compressed keys.
        Accepts bit-packed k_packed from compress().
        Uses Triton kernel on CUDA, PyTorch on CPU.

        Parameters
        ----------
        query    : Tensor (..., dim) — raw (unrotated) query
        k_packed : Tensor (seq_len, packed_dim) uint8 — nibble-packed key indices
        k_norms  : Tensor (seq_len,) float32
        scale    : float, default 1/sqrt(dim)

        Returns
        -------
        logits : Tensor (..., seq_len)
        """
        if scale is None:
            scale = self.dim ** -0.5

        query = query.to(dtype=torch.float32, device=self._device)

        # Unpack key indices
        k_indices = self._unpack_indices(k_packed, orig_dim=self.dim)  # (seq_len, dim)

        # Pre-rotate query: (dim,) or (batch, dim)
        q_rot = query @ self._R.T

        if q_rot.dim() == 1:
            # Single query: use fused kernel path
            return turboquant_attention_scores(
                q_rot, k_indices, k_norms, self._centroids, scale,
                use_triton=True,
            )
        else:
            # Batched queries: loop (or batch matmul)
            k_hat = self._centroids[k_indices.long()]   # (seq_len, dim)
            # q_rot: (batch, dim), k_hat: (seq_len, dim)
            logits = (q_rot @ k_hat.T) * scale          # (batch, seq_len)
            logits = logits * k_norms.unsqueeze(0)      # broadcast norms
            return logits

    # ── Memory stats ──────────────────────────────────────────────────────────

    def bytes_per_vector(self) -> int:
        """
        Bytes used per compressed vector after bit-packing.

        For bits <= 4: ceil(dim * bits / 8) packed bytes + 4 bytes for float32 norm.
        For bits > 4 : dim bytes (one uint8 per index) + 4 bytes for norm.
        """
        if self.bits <= 4:
            packed_bytes = math.ceil(self.dim * self.bits / 8)
        else:
            packed_bytes = self.dim  # one byte per index
        return packed_bytes + 4

    def compression_ratio(self, original_dtype_bytes: int = 2) -> float:
        """Ratio of original float bytes to compressed bytes per vector."""
        return (self.dim * original_dtype_bytes) / self.bytes_per_vector()

    def __repr__(self) -> str:
        return (
            f"TurboQuantTorch(dim={self.dim}, bits={self.bits}, "
            f"device={self._device}, "
            f"triton={'✓' if is_triton_available() else '✗ (PyTorch fallback)'})"
        )
