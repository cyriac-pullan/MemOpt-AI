"""
Lloyd-Max Codebook for TurboQuant
===================================
This is the critical piece that makes TurboQuant actually work at paper quality.

The key mathematical insight from the paper:
  After applying a random orthogonal rotation R to a vector x on the unit sphere
  S^{d-1}, each coordinate follows a Beta((d-1)/2, (d-1)/2) distribution on [-1, 1].

  For large d:
    - Mean = 0
    - Std ≈ 1/sqrt(d)
    - Distribution concentrates tightly around zero

  Because this distribution is KNOWN analytically (doesn't depend on the data,
  only on the dimension d), you can precompute the OPTIMAL scalar quantizer
  (Lloyd-Max quantizer) for it ONCE and reuse it for every vector — with zero
  per-vector normalization overhead.

  This is exactly why TurboQuant beats KIVI, SqueezeLLM, and other methods:
  they need stored scale/zero-point constants per block (adds ~1-2 bits overhead).
  TurboQuant needs none.

Lloyd-Max algorithm:
  Given a probability distribution p(x) and k = 2^bits quantization levels:
  1. Initialize k centroids
  2. Assign each grid point to its nearest centroid (Voronoi partition)
  3. Update each centroid = weighted mean of its partition under p(x)
  4. Repeat until convergence (MSE-optimal quantizer)

Result: codebook[i] = optimal reconstruction value for the i-th quantization bin.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from functools import lru_cache
from typing import Tuple
import os, hashlib, pickle


# ── Codebook Cache ──────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_codebook_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(dim: int, bits: int) -> str:
    key = f"d{dim}_b{bits}"
    return os.path.join(_CACHE_DIR, f"{key}.pkl")


def _load_cached(dim: int, bits: int):
    path = _cache_path(dim, bits)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cached(dim: int, bits: int, data):
    path = _cache_path(dim, bits)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ── Beta Distribution Setup ─────────────────────────────────────────────────

def beta_params(dim: int) -> Tuple[float, float]:
    """
    After random rotation of a unit-sphere vector in R^d, each coordinate
    follows Beta((d-1)/2, (d-1)/2) on [-1, 1].
    
    For practical use, we work on [-1, 1] using the two-parameter Beta
    distribution. The scipy convention uses a, b parameters where:
        a = b = (d-1)/2
    """
    alpha = (dim - 1) / 2.0
    return alpha, alpha


def beta_pdf_on_grid(grid: np.ndarray, dim: int) -> np.ndarray:
    """
    Evaluate Beta((d-1)/2, (d-1)/2) PDF on the grid [-1, 1].
    scipy's beta distribution is on [0,1]; we shift to [-1,1].
    """
    alpha, _ = beta_params(dim)
    # Transform grid from [-1,1] to [0,1]
    x_unit = (grid + 1.0) / 2.0
    # Evaluate PDF (factor of 0.5 for Jacobian of the transformation)
    pdf = beta_dist.pdf(x_unit, alpha, alpha) / 2.0
    # Clip and normalize
    pdf = np.maximum(pdf, 0.0)
    total = np.trapezoid(pdf, grid) if hasattr(np, 'trapezoid') else np.trapz(pdf, grid)
    if total > 0:
        pdf /= total
    return pdf.astype(np.float32)


# ── Lloyd-Max Algorithm ─────────────────────────────────────────────────────

def lloyd_max_codebook(
    dim: int,
    bits: int,
    n_grid: int = 50_000,
    n_iters: int = 300,
    sigma_range: float = 6.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute the Lloyd-Max optimal scalar quantizer for Beta((d-1)/2, (d-1)/2).

    Parameters
    ----------
    dim : int
        Vector dimension (controls the Beta distribution shape).
    bits : int
        Quantization bit-width. Returns 2^bits centroids.
    n_grid : int
        Number of grid points for numerical integration. 50k is accurate.
    n_iters : int
        Lloyd-Max iterations. 300 is well past convergence.
    sigma_range : float
        Grid range in units of std dev: ±sigma_range * (1/sqrt(dim)).
    verbose : bool
        Print convergence info.

    Returns
    -------
    centroids : np.ndarray, shape (2^bits,)
        Sorted reconstruction values (codebook).
        These are the optimal dequantization values.
    """
    # Check cache first
    cached = _load_cached(dim, bits)
    if cached is not None:
        return cached

    if verbose:
        print(f"  Building Lloyd-Max codebook (d={dim}, bits={bits})...", end=" ", flush=True)

    k = 2 ** bits  # number of quantization levels
    std = 1.0 / np.sqrt(dim)

    # Build fine numerical grid focused on ±sigma_range standard deviations
    # (where >99.9% of the Beta distribution mass lives for large d)
    lo = max(-1.0, -sigma_range * std)
    hi = min(1.0, sigma_range * std)
    grid = np.linspace(lo, hi, n_grid, dtype=np.float64)
    pdf  = beta_pdf_on_grid(grid, dim).astype(np.float64)

    # Normalize pdf to sum to 1 over grid
    pdf_sum = np.sum(pdf)
    if pdf_sum > 0:
        pdf /= pdf_sum

    # Initialize centroids: uniform quantile spacing
    # Use quantiles of the distribution for better initialization
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    init_quantiles = np.linspace(1.0 / (2 * k), 1.0 - 1.0 / (2 * k), k)
    centroids = np.interp(init_quantiles, cdf, grid)

    # Lloyd-Max iterations
    prev_mse = float("inf")
    for iteration in range(n_iters):
        # Step 1: Voronoi partition — assign each grid point to nearest centroid
        # boundaries[i] = midpoint between centroids[i] and centroids[i+1]
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # For each grid point, find its centroid assignment
        assignments = np.searchsorted(boundaries, grid)  # 0..k-1

        # Step 2: Update centroids as weighted mean of their partition
        new_centroids = np.zeros(k, dtype=np.float64)
        for c in range(k):
            mask = assignments == c
            w = pdf[mask]
            if w.sum() > 1e-15:
                new_centroids[c] = np.sum(grid[mask] * w) / w.sum()
            else:
                new_centroids[c] = centroids[c]  # keep if empty

        # Convergence check
        mse = np.sum(pdf * (grid - centroids[assignments]) ** 2)
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids

        if shift < 1e-10:
            break

    centroids = np.sort(centroids).astype(np.float32)

    if verbose:
        print(f"done. MSE={mse:.6f}, centroids range=[{centroids[0]:.4f}, {centroids[-1]:.4f}]")

    _save_cached(dim, bits, centroids)
    return centroids


def build_boundaries(centroids: np.ndarray) -> np.ndarray:
    """
    Compute decision boundaries for nearest-centroid assignment.
    boundaries[i] = threshold between centroid i and i+1.
    """
    return ((centroids[:-1] + centroids[1:]) / 2.0).astype(np.float32)


def quantize_with_codebook(x: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """
    Quantize values to centroid indices using precomputed boundaries.
    Uses binary search (searchsorted) — O(log k) per element.

    Parameters
    ----------
    x : np.ndarray, any shape, float32
    boundaries : np.ndarray, shape (k-1,)

    Returns
    -------
    indices : np.ndarray, same shape as x, dtype uint8
    """
    return np.searchsorted(boundaries, x).astype(np.uint8)


def dequantize_with_codebook(indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Reconstruct float values from quantization indices.

    Parameters
    ----------
    indices : np.ndarray, dtype uint8
    centroids : np.ndarray, shape (k,)

    Returns
    -------
    np.ndarray, float32
    """
    return centroids[indices.astype(np.int32)].astype(np.float32)
