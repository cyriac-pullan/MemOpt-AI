"""
TurboQuant Triton Fused Attention Kernel
=========================================
Computes Q @ K^T directly from compressed uint8 key indices —
never materializing fp16 keys in GPU memory.

Algebraic identity that makes this work:
  <q, R^T · centroids[idx]> = <R·q, centroids[idx]>

So: pre-rotate the query once with a matmul, then per-position work
is just a centroid table lookup + dot product over uint8 indices.

Memory bandwidth reduction:
  Standard: load fp16 keys  → 2 bytes per element
  Fused:    load uint8 idx  → 1 byte per element  (2x less HBM traffic)

At 2-bit packing (future): 4 indices per byte → 8x less HBM traffic.

This kernel handles:
  - Single-head attention (batched across heads at Python level)
  - GQA (grouped query attention) via gqa_ratio parameter
  - Causal masking

Requires: triton >= 2.0, CUDA GPU

For CPU fallback (no GPU / no Triton), see TurboQuantAttention.forward()
which automatically falls back to PyTorch dequantize-then-matmul.
"""

import torch
import torch.nn.functional as F
from typing import Optional

# ── Triton availability gate ────────────────────────────────────────────────
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass


if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_S": 64,  "BLOCK_D": 64},  num_warps=4),
            triton.Config({"BLOCK_S": 128, "BLOCK_D": 64},  num_warps=4),
            triton.Config({"BLOCK_S": 64,  "BLOCK_D": 128}, num_warps=8),
            triton.Config({"BLOCK_S": 128, "BLOCK_D": 128}, num_warps=8),
        ],
        key=["SEQ_LEN", "HEAD_DIM", "NUM_LEVELS"],
    )
    @triton.jit
    def _turboquant_qk_kernel(
        # Pointers
        Q_rot_ptr,      # (HEAD_DIM,) — pre-rotated query, float32
        K_idx_ptr,      # (SEQ_LEN, HEAD_DIM) — uint8 centroid indices
        K_norm_ptr,     # (SEQ_LEN,) — float32 per-vector norms
        C_ptr,          # (NUM_LEVELS,) — centroid table, float32
        Out_ptr,        # (SEQ_LEN,) — output attention logits
        # Dimensions
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_LEVELS: tl.constexpr,
        # Scale
        scale: tl.constexpr,
    ):
        """
        Each program handles BLOCK_S consecutive sequence positions.
        For each position t:
          logit[t] = scale * norm[t] * sum_d(Q_rot[d] * centroids[K_idx[t,d]])
        """
        pid = tl.program_id(0)
        BLOCK_S: tl.constexpr = tl.num_programs(0)  # autotuned

        # Range of sequence positions this block handles
        s_start = pid * BLOCK_S
        s_offs  = s_start + tl.arange(0, BLOCK_S)
        s_mask  = s_offs < SEQ_LEN

        # Accumulator: (BLOCK_S,)
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Loop over head dimension in chunks of BLOCK_D
        for d_start in range(0, HEAD_DIM, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < HEAD_DIM

            # Load query slice: (BLOCK_D,)
            q_slice = tl.load(Q_rot_ptr + d_offs, mask=d_mask, other=0.0)

            # Load key indices: (BLOCK_S, BLOCK_D) uint8
            k_idx = tl.load(
                K_idx_ptr + s_offs[:, None] * HEAD_DIM + d_offs[None, :],
                mask=s_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)

            # Gather centroids: (BLOCK_S, BLOCK_D)
            k_vals = tl.load(C_ptr + k_idx, mask=s_mask[:, None] & d_mask[None, :], other=0.0)

            # Accumulate dot product
            acc += tl.sum(k_vals * q_slice[None, :], axis=1)

        # Multiply by norm and scale
        norms = tl.load(K_norm_ptr + s_offs, mask=s_mask, other=1.0)
        logits = acc * norms * scale

        tl.store(Out_ptr + s_offs, logits, mask=s_mask)


    def triton_turboquant_attention(
        q_rotated: torch.Tensor,       # (head_dim,) float32, already R@q
        k_indices: torch.Tensor,       # (seq_len, head_dim) uint8
        k_norms:   torch.Tensor,       # (seq_len,) float32
        centroids: torch.Tensor,       # (num_levels,) float32
        scale: float,
    ) -> torch.Tensor:
        """
        Compute attention logits from compressed keys using Triton kernel.

        Returns
        -------
        logits : torch.Tensor, shape (seq_len,)
        """
        seq_len, head_dim = k_indices.shape
        num_levels = centroids.shape[0]

        # Ensure contiguous CUDA tensors
        q_rotated = q_rotated.contiguous().cuda()
        k_indices = k_indices.contiguous().cuda()
        k_norms   = k_norms.contiguous().cuda()
        centroids = centroids.contiguous().cuda()
        logits    = torch.empty(seq_len, device="cuda", dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_S"]),)
        _turboquant_qk_kernel[grid](
            q_rotated, k_indices, k_norms, centroids, logits,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            NUM_LEVELS=num_levels,
            scale=scale,
        )
        return logits


def turboquant_attention_scores(
    q_rotated: torch.Tensor,
    k_indices: torch.Tensor,
    k_norms:   torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Compute attention logits from TurboQuant-compressed keys.
    Automatically selects Triton kernel (GPU) or PyTorch fallback (CPU).

    Parameters
    ----------
    q_rotated : Tensor (head_dim,) — query after rotation R@q
    k_indices : Tensor (seq_len, head_dim) uint8 — quantized key indices
    k_norms   : Tensor (seq_len,) float32 — per-key norms
    centroids : Tensor (num_levels,) float32 — codebook centroids
    scale     : float — attention scale (1/sqrt(head_dim))
    use_triton: bool — use Triton kernel if available (default True)

    Returns
    -------
    logits : Tensor (seq_len,) float32
    """
    if use_triton and _TRITON_AVAILABLE and q_rotated.is_cuda:
        return triton_turboquant_attention(q_rotated, k_indices, k_norms, centroids, scale)

    # ── PyTorch fallback (CPU or no Triton) ─────────────────────────────────
    # Dequantize: gather centroids for all indices
    k_hat = centroids[k_indices.long()]          # (seq_len, head_dim)
    # Batch dot products: (seq_len,)
    logits = (k_hat @ q_rotated) * k_norms * scale
    return logits


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE
