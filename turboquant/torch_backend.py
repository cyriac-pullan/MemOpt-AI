"""
TurboQuant PyTorch Backend
===========================
GPU-accelerated version of the core TurboQuant operations.
Works with torch.Tensor instead of np.ndarray.

This is the backend used by the HuggingFace cache integration.
It keeps everything in PyTorch so there's no CPU<->GPU transfer overhead.

Key operations:
  - compress_tensor(x): rotate + quantize, returns (indices, norms) tensors
  - decompress_tensor(indices, norms): centroid gather + undo rotation
  - attention_scores(q, k_indices, k_norms): fused or fallback logits
"""

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

    # ── Compress ─────────────────────────────────────────────────────────────

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a batch of vectors.

        Parameters
        ----------
        x : Tensor, shape (..., dim), any dtype

        Returns
        -------
        indices : Tensor, shape (..., dim), dtype uint8
        norms   : Tensor, shape (...,), dtype float32
        """
        x = x.to(dtype=torch.float32, device=self._device)

        # Extract norms
        norms = x.norm(dim=-1, keepdim=True)          # (..., 1)
        safe_norms = norms.clamp(min=1e-9)
        x_unit = x / safe_norms                        # (..., dim)

        # Random rotation
        x_rot = x_unit @ self._R.T                     # (..., dim)

        # Lloyd-Max quantize: searchsorted on boundaries
        # x_rot: (..., dim) → find which bin each value falls in
        flat = x_rot.reshape(-1, self.dim)             # (N, dim)
        # boundaries: (num_levels-1,) → insert each value
        # Result: index into [0, num_levels-1]
        indices_flat = torch.bucketize(flat, self._boundaries)  # (N, dim)
        indices = indices_flat.reshape(x_rot.shape).to(torch.uint8)

        return indices, norms.squeeze(-1).to(torch.float32)

    # ── Decompress ────────────────────────────────────────────────────────────

    def decompress(
        self, indices: torch.Tensor, norms: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct vectors from compressed representation.

        Parameters
        ----------
        indices : Tensor, shape (..., dim), uint8
        norms   : Tensor, shape (...,), float32

        Returns
        -------
        Tensor, shape (..., dim), float32
        """
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
        k_indices: torch.Tensor,      # (seq_len, dim) uint8
        k_norms: torch.Tensor,        # (seq_len,) float32
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute attention logits between query and compressed keys.
        Uses Triton kernel on CUDA, PyTorch on CPU.

        Parameters
        ----------
        query     : Tensor (..., dim) — raw (unrotated) query
        k_indices : Tensor (seq_len, dim) uint8
        k_norms   : Tensor (seq_len,) float32
        scale     : float, default 1/sqrt(dim)

        Returns
        -------
        logits : Tensor (..., seq_len)
        """
        if scale is None:
            scale = self.dim ** -0.5

        query = query.to(dtype=torch.float32, device=self._device)

        # Pre-rotate query: (dim,) or (batch, dim)
        q_rot = query @ self._R.T   # same as R @ q for each row

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
        """Bytes used per compressed vector: uint8 indices + float32 norm."""
        return self.dim + 4  # 1 byte per index + 4 bytes for norm

    def compression_ratio(self, original_dtype_bytes: int = 2) -> float:
        return (self.dim * original_dtype_bytes) / self.bytes_per_vector()

    def __repr__(self) -> str:
        return (
            f"TurboQuantTorch(dim={self.dim}, bits={self.bits}, "
            f"device={self._device}, "
            f"triton={'✓' if is_triton_available() else '✗ (PyTorch fallback)'})"
        )
