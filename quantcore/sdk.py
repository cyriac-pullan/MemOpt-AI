"""
QuantCore SDK — Core API
=========================
The single entry point for compressing LLM KV caches.

Usage
-----
    from quantcore import optimize_model

    model = optimize_model(model)                          # balanced (3-bit)
    model = optimize_model(model, mode="fast")            # 4-bit, best quality
    model = optimize_model(model, mode="max_memory_save") # 2-bit, max savings
"""

from __future__ import annotations

import sys
from typing import Optional, TYPE_CHECKING

from .exceptions import QuantCoreModeError, QuantCoreDependencyError, QuantCoreCompatError

if TYPE_CHECKING:
    pass


# ── Mode → Bits mapping ───────────────────────────────────────────────────────

_MODE_BITS: dict[str, int] = {
    "fast":            4,   # cosine sim 0.995,  ~1.9x vs fp16
    "balanced":        3,   # cosine sim 0.983,  ~2.8x vs fp16
    "max_memory_save": 2,   # cosine sim 0.940,  ~4.0x vs fp16
    "adaptive":        None, # dynamic — changes at runtime
}

_MODE_DESCRIPTION: dict[str, dict] = {
    "fast": {
        "bits": 4, "cosine_sim": 0.995, "compression_vs_fp16": 1.94,
        "description": "Best quality. Recommended for production chatbots.",
    },
    "balanced": {
        "bits": 3, "cosine_sim": 0.983, "compression_vs_fp16": 2.76,
        "description": "Great tradeoff. Recommended for most use cases.",
    },
    "max_memory_save": {
        "bits": 2, "cosine_sim": 0.940, "compression_vs_fp16": 3.88,
        "description": "Maximum compression. Good for RAG and edge deployment.",
    },
    "adaptive": {
        "bits": "dynamic", "cosine_sim": "varies", "compression_vs_fp16": "varies",
        "description": "Adaptive runtime. Escalates compression as context grows.",
    },
}


def _validate_mode(mode: str) -> Optional[int]:
    """Returns bits for mode. Returns None for adaptive (resolved at runtime)."""
    if mode not in _MODE_BITS:
        raise QuantCoreModeError(mode)
    return _MODE_BITS[mode]


# ── HuggingFace model detection ───────────────────────────────────────────────

def _is_hf_model(model) -> bool:
    """True if model looks like a HuggingFace PreTrainedModel."""
    return hasattr(model, "config") and hasattr(model, "generate")


def _optimize_hf_model(
    model,
    bits: int,
    mode: str,
    compress_values: bool,
    verbose: bool,
    adaptive_policy=None,
    max_cache_len: int = None,
):
    """Patch a HuggingFace model to use TurboQuant KV cache."""
    try:
        from transformers import __version__ as tf_version
    except ImportError:
        raise QuantCoreDependencyError(
            "transformers", "HuggingFace model optimization"
        )

    # Import our HF integration (already exists in the turboquant package)
    try:
        import sys, os
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from turboquant.hf_integration import patch_model_cache
    except ImportError as e:
        raise QuantCoreDependencyError(
            "turboquant", "QuantCore core engine",
        ) from e

    patch_model_cache(
        model,
        bits=bits,
        compress_values=compress_values,
        verbose=verbose,
        adaptive_policy=adaptive_policy,
        max_cache_len=max_cache_len,
    )

    # Inject stats helper onto the model
    _inject_stats(model, mode, bits if bits else 3, adaptive_policy)
    return model


def _inject_stats(model, mode: str, bits: int, adaptive_policy=None):
    """Inject model.quantcore_stats() and model.quantcore_info attributes."""
    from .compat import extract_model_info

    info = None
    try:
        info = extract_model_info(model.config)
    except Exception:
        pass

    model.quantcore_info = {
        "mode": mode,
        "bits": bits if mode != "adaptive" else "dynamic",
        "model_info": info,
        **_MODE_DESCRIPTION[mode],
    }

    # Expose adaptive policy for monitoring
    if adaptive_policy:
        model.quantcore_policy = adaptive_policy

    def quantcore_stats(seq_len: int = 4096) -> dict:
        """
        Show memory savings for this model at a given sequence length.

        Parameters
        ----------
        seq_len : int
            Context length to estimate memory for. Default: 4096.

        Returns
        -------
        dict with keys: fp16_mb, compressed_mb, ratio, mode, bits
        """
        if info is None:
            return {"error": "Model info could not be extracted."}
        kv = info.kv_cache_mb(seq_len=seq_len, bits=bits)
        return {
            "mode": mode,
            "bits": bits,
            "seq_len": seq_len,
            "fp16_mb": kv["fp16_mb"],
            "compressed_mb": kv["compressed_mb"],
            "compression_ratio": kv["ratio"],
            "memory_saved_mb": round(kv["fp16_mb"] - kv["compressed_mb"], 2),
        }

    model.quantcore_stats = quantcore_stats


# ── Pure PyTorch model support ────────────────────────────────────────────────

def _optimize_torch_model(model, bits: int, mode: str, verbose: bool):
    """
    For raw PyTorch models (not HuggingFace), return a TurboQuantKVCache
    helper alongside the model.

    Since raw PyTorch models have no standard generate() interface, we can't
    monkey-patch them. Instead we return (model, cache_factory).
    """
    try:
        import torch
    except ImportError:
        raise QuantCoreDependencyError("torch", "PyTorch model optimization", "torch")

    import sys, os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from turboquant import TurboQuantKVCache

    if verbose:
        print(
            f"[QuantCore] Non-HuggingFace model detected. Returning cache factory.\n"
            f"  Use: cache = model.make_kv_cache(num_heads, head_dim)"
        )

    def make_kv_cache(num_heads: int, head_dim: int) -> "TurboQuantKVCache":
        """Create a TurboQuant KV cache for this model."""
        return TurboQuantKVCache(
            num_heads=num_heads,
            head_dim=head_dim,
            bits=bits,
            verbose=verbose,
        )

    model.make_kv_cache = make_kv_cache
    model.quantcore_info = {"mode": mode, "bits": bits, **_MODE_DESCRIPTION[mode]}
    return model


# ── Public API ────────────────────────────────────────────────────────────────

def optimize_model(
    model,
    mode: str = "balanced",
    max_memory=None,
    max_cache_len: int = None,
    compress_values: bool = False,
    verbose: bool = True,
):
    """
    Compress a model's KV cache using TurboQuant.

    This is the main QuantCore entry point. It patches the model in-place
    and returns it — no wrappers, no new objects.

    Parameters
    ----------
    model : PreTrainedModel or nn.Module
        Any HuggingFace model or raw PyTorch model.
    mode : str
        Compression mode:
          "fast"            -> 4-bit (cosine sim 0.995, ~1.9x vs fp16)
          "balanced"        -> 3-bit (cosine sim 0.983, ~2.8x vs fp16) [default]
          "max_memory_save" -> 2-bit (cosine sim 0.940, ~4.0x vs fp16)
          "adaptive"        -> dynamic compression (escalates as context grows)
    max_memory : str, int, float, optional
        Memory budget. Activates adaptive policy when combined with mode="adaptive".
        Supports: "6GB", "512MB", 6144 (MB), 0.8 (fraction of GPU).
        If set to 0, automatically detects available GPU memory.
    max_cache_len : int, optional
        Maximum KV cache length before eviction (sliding window).
        If None, cache grows without bound.
    compress_values : bool
        Also compress value vectors (not just keys). Default False.
    verbose : bool
        Print compression info on first call. Default True.

    Returns
    -------
    model : same object, patched in-place

    Examples
    --------
    >>> model = optimize_model(model)                        # balanced 3-bit
    >>> model = optimize_model(model, mode="adaptive")       # dynamic compression
    >>> model = optimize_model(model, max_memory="6GB")      # budget mode
    >>> model = optimize_model(model, mode="adaptive",
    ...                        max_memory="8GB",
    ...                        max_cache_len=8192)           # full adaptive
    """
    adaptive_policy = None

    # Parse max_memory if provided
    budget_mb = None
    if max_memory is not None:
        if max_memory == 0:
            # Auto-detect
            from .policy import detect_gpu_memory_mb
            budget_mb = detect_gpu_memory_mb()
        else:
            from .policy import parse_memory
            budget_mb = parse_memory(max_memory)

    # Handle adaptive mode
    if mode == "adaptive":
        from .adaptive import AdaptivePolicy
        adaptive_policy = AdaptivePolicy(
            memory_budget_mb=budget_mb,
            stability_steps=32,
        )
        bits = 4  # start with light compression, policy adjusts at runtime
        if verbose:
            budget_str = f", budget={budget_mb:.0f}MB" if budget_mb else ""
            print(f"[QuantCore] Adaptive mode enabled{budget_str}")
            print(f"  Compression will escalate: 4-bit -> 3-bit -> 2-bit as needed")
            if max_cache_len:
                print(f"  Sliding window eviction at {max_cache_len} tokens")
    elif budget_mb and mode != "adaptive":
        # max_memory without adaptive: just auto-select static mode
        from .policy import auto_select_mode
        mode = auto_select_mode(budget_mb)
        bits = _validate_mode(mode)
        if verbose:
            print(f"[QuantCore] Policy Engine auto-selected: {mode} ({bits}-bit)")
    else:
        bits = _validate_mode(mode)

    if _is_hf_model(model):
        return _optimize_hf_model(
            model, bits=bits, mode=mode,
            compress_values=compress_values, verbose=verbose,
            adaptive_policy=adaptive_policy,
            max_cache_len=max_cache_len,
        )
    else:
        return _optimize_torch_model(model, bits=bits, mode=mode, verbose=verbose)


def mode_info(mode: str = None) -> dict:
    """
    Return information about compression modes.

    Parameters
    ----------
    mode : str, optional
        If given, return info for that mode only. Otherwise return all.
    """
    if mode is not None:
        _validate_mode(mode)
        return {mode: _MODE_DESCRIPTION[mode]}
    return dict(_MODE_DESCRIPTION)
