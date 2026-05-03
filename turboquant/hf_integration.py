"""
TurboQuant HuggingFace Cache Integration
==========================================
Drop-in KV cache compression for ANY HuggingFace model using TurboQuant.

Compatible with transformers >= 4.47 (new layer-based DynamicCache architecture).

Three ways to use:

  1. One-liner patch (simplest):
       from turboquant.hf_integration import patch_model_cache
       patch_model_cache(model, bits=4)
       output = model.generate(...)  # automatically compressed

  2. Explicit cache (most control):
       from turboquant.hf_integration import make_turboquant_cache
       cache = make_turboquant_cache(model.config, bits=4)
       output = model.generate(..., past_key_values=cache)

  3. Layer-level (advanced / custom models):
       from turboquant.hf_integration import TurboQuantLayer
       layer = TurboQuantLayer(head_dim=128, bits=4)

How it works
------------
HuggingFace's DynamicCache is layer-based. Each attention layer calls
cache.update(key, value, layer_idx), which delegates to a DynamicLayer
instance. We subclass DynamicLayer to override update() so keys are
compressed to uint8 on write and reconstructed to fp16 on read.

Memory saved (at head_dim=128, 4-bit):
  FP16 keys:       128 * 2 = 256 bytes / vector
  TurboQuant keys: 128 * 1 + 4 = 132 bytes / vector  →  ~1.94x
"""

import torch
from typing import Optional, Dict, Tuple

try:
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    DynamicLayer = object

from .torch_backend import TurboQuantTorch
from .triton_kernel import is_triton_available


# ── Per-layer compressed cache ────────────────────────────────────────────────

class TurboQuantLayer(DynamicLayer if _HF_AVAILABLE else object):
    """
    TurboQuant-compressed attention cache layer.
    Replaces DynamicLayer: keys compressed to uint8 on write, fp16 on read.

    Parameters
    ----------
    head_dim : int
    bits : int  (4=lossless, 3=aggressive, 2=maximum compression)
    layer_idx : int  (used to seed the rotation matrix)
    compress_values : bool  (also compress values, default False)
    verbose : bool
    adaptive_policy : AdaptivePolicy, optional
        If set, bits may change at runtime based on seq_len and memory.
    max_cache_len : int, optional
        If set, applies sliding window eviction beyond this length.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 4,
        layer_idx: int = 0,
        compress_values: bool = False,
        verbose: bool = False,
        adaptive_policy=None,
        max_cache_len: int = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.compress_values = compress_values
        self._layer_idx = layer_idx
        self._adaptive_policy = adaptive_policy
        self._max_cache_len = max_cache_len

        self._tq_keys = TurboQuantTorch(
            dim=head_dim, bits=bits, seed=layer_idx * 137, verbose=verbose
        )
        self._tq_vals = TurboQuantTorch(
            dim=head_dim, bits=bits, seed=layer_idx * 137 + 1000, verbose=False
        ) if compress_values else None

        # Quantizer cache: pre-built quantizers for each bit depth (adaptive)
        self._tq_cache = {bits: self._tq_keys}
        if compress_values:
            self._tq_val_cache = {bits: self._tq_vals}

        # Compressed key storage tensors
        self._k_indices: Optional[torch.Tensor] = None   # (B, H, S, D) uint8
        self._k_norms:   Optional[torch.Tensor] = None   # (B, H, S) float32
        self._v_indices: Optional[torch.Tensor] = None
        self._v_norms:   Optional[torch.Tensor] = None

        self.cumulative_length = 0

    def lazy_initialization(self, key_states, value_states):
        self.dtype  = key_states.dtype
        self.device = key_states.device
        self._tq_keys.to(self.device)
        if self._tq_vals:
            self._tq_vals.to(self.device)
        # Required by DynamicLayer
        self.keys   = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress incoming keys, append to store, return full decompressed KV.
        Checks adaptive policy and applies sliding window eviction if configured.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, H, T, D = key_states.shape
        self.cumulative_length += T

        # -- Adaptive policy check (only layer 0 makes the decision) --
        if self._adaptive_policy and self._layer_idx == 0:
            try:
                gpu_used = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            except Exception:
                gpu_used, gpu_total = 0, 0

            new_bits = self._adaptive_policy.select_bits(
                seq_len=self.cumulative_length,
                gpu_mem_used_mb=gpu_used,
                gpu_mem_total_mb=gpu_total,
            )
            if new_bits != self.bits and new_bits in (2, 3, 4):
                self._switch_bits(new_bits)

        # -- Sliding window eviction --
        if self._max_cache_len and self._k_indices is not None:
            current_len = self._k_indices.shape[2]
            if current_len >= self._max_cache_len:
                keep = self._max_cache_len - T
                self._k_indices = self._k_indices[:, :, -keep:, :]
                self._k_norms = self._k_norms[:, :, -keep:]
                if self.compress_values and self._v_indices is not None:
                    self._v_indices = self._v_indices[:, :, -keep:, :]
                    self._v_norms = self._v_norms[:, :, -keep:]
                elif not self.compress_values:
                    self.values = self.values[:, :, -keep:, :]

        # -- Keys: compress & append --
        k_flat   = key_states.reshape(-1, D).float()
        idx, nms = self._tq_keys.compress(k_flat)
        idx = idx.reshape(B, H, T, D)
        nms = nms.reshape(B, H, T)

        self._k_indices = idx if self._k_indices is None else \
            torch.cat([self._k_indices, idx], dim=2)
        self._k_norms   = nms if self._k_norms is None else \
            torch.cat([self._k_norms, nms], dim=2)

        # -- Values: compress or accumulate fp16 --
        if self.compress_values:
            v_flat   = value_states.reshape(-1, D).float()
            vidx, vnms = self._tq_vals.compress(v_flat)
            vidx = vidx.reshape(B, H, T, D)
            vnms = vnms.reshape(B, H, T)
            self._v_indices = vidx if self._v_indices is None else \
                torch.cat([self._v_indices, vidx], dim=2)
            self._v_norms   = vnms if self._v_norms is None else \
                torch.cat([self._v_norms, vnms], dim=2)
        else:
            self.values = value_states if self.values.numel() == 0 else \
                torch.cat([self.values, value_states], dim=-2)

        # -- Reconstruct full keys --
        S  = self._k_indices.shape[2]
        fk = self._tq_keys.decompress(
            self._k_indices.reshape(-1, D),
            self._k_norms.reshape(-1),
        ).reshape(B, H, S, D).to(key_states.dtype)

        # -- Reconstruct full values --
        if self.compress_values:
            Sv = self._v_indices.shape[2]
            fv = self._tq_vals.decompress(
                self._v_indices.reshape(-1, D),
                self._v_norms.reshape(-1),
            ).reshape(B, H, Sv, D).to(value_states.dtype)
        else:
            fv = self.values

        return fk, fv

    def _switch_bits(self, new_bits: int):
        """Switch quantizer to a new bit depth (for adaptive policy)."""
        if new_bits not in self._tq_cache:
            self._tq_cache[new_bits] = TurboQuantTorch(
                dim=self.head_dim, bits=new_bits,
                seed=self._layer_idx * 137, verbose=False,
            )
            self._tq_cache[new_bits].to(self.device)
            if self.compress_values:
                if not hasattr(self, '_tq_val_cache'):
                    self._tq_val_cache = {}
                self._tq_val_cache[new_bits] = TurboQuantTorch(
                    dim=self.head_dim, bits=new_bits,
                    seed=self._layer_idx * 137 + 1000, verbose=False,
                )
                self._tq_val_cache[new_bits].to(self.device)

        self._tq_keys = self._tq_cache[new_bits]
        if self.compress_values and hasattr(self, '_tq_val_cache'):
            self._tq_vals = self._tq_val_cache.get(new_bits, self._tq_vals)
        self.bits = new_bits

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def memory_bytes(self) -> Dict[str, int]:
        if self._k_indices is None:
            return {"compressed": 0, "fp16_equiv": 0}
        k_compressed = self._k_indices.numel() + self._k_norms.numel() * 4
        k_fp16       = self._k_indices.numel() * 2
        v_compressed = self.values.numel() * 2 if not self.compress_values else \
            (self._v_indices.numel() + self._v_norms.numel() * 4)
        v_fp16       = self.values.numel() * 2 if not self.compress_values else \
            self._v_indices.numel() * 2
        return {
            "compressed": k_compressed + v_compressed,
            "fp16_equiv": k_fp16 + v_fp16,
        }


# ── Factory + helpers ─────────────────────────────────────────────────────────

def make_turboquant_cache(
    config,
    bits: int = 4,
    compress_values: bool = False,
    verbose: bool = True,
    adaptive_policy=None,
    max_cache_len: int = None,
) -> "DynamicCache":
    """
    Build a DynamicCache pre-populated with TurboQuantLayer instances.

    Parameters
    ----------
    config : PretrainedConfig
    bits : int. Default 4.
    compress_values : bool. Default False.
    verbose : bool. Default True.
    adaptive_policy : AdaptivePolicy, optional.
    max_cache_len : int, optional. Sliding window eviction length.
    """
    if not _HF_AVAILABLE:
        raise ImportError("pip install transformers")

    head_dim   = _get_head_dim(config)
    num_layers = _get_num_layers(config)

    mode_str = "adaptive" if adaptive_policy else f"{bits}-bit"
    if verbose:
        print(f"[TurboQuant] {mode_str} cache | {num_layers} layers | head_dim={head_dim}")
        if max_cache_len:
            print(f"             sliding window: {max_cache_len} tokens")
        print(f"             triton: {'yes' if is_triton_available() else 'no'}")

    cache = DynamicCache()
    cache.layers = []
    for i in range(num_layers):
        cache.layers.append(TurboQuantLayer(
            head_dim=head_dim,
            bits=bits,
            layer_idx=i,
            compress_values=compress_values,
            verbose=(verbose and i == 0),
            adaptive_policy=adaptive_policy,
            max_cache_len=max_cache_len,
        ))

    return cache


def patch_model_cache(
    model,
    bits: int = 4,
    compress_values: bool = False,
    verbose: bool = True,
    mode: str = None,
    adaptive_policy=None,
    max_cache_len: int = None,
):
    """
    Patch model.generate() to use TurboQuant cache automatically.

    Parameters
    ----------
    model : PreTrainedModel
    bits : int
    mode : str, optional
    adaptive_policy : AdaptivePolicy, optional
    max_cache_len : int, optional
    compress_values : bool
    verbose : bool
    """
    if not _HF_AVAILABLE:
        raise ImportError("pip install transformers")

    # Resolve mode -> bits
    if mode is not None and mode != "adaptive":
        _mode_bits = {"fast": 4, "balanced": 3, "max_memory_save": 2}
        if mode not in _mode_bits:
            raise ValueError(f"mode must be one of {list(_mode_bits)}")
        bits = _mode_bits[mode]

    original_generate = model.generate

    def patched_generate(*args, **kwargs):
        if kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = make_turboquant_cache(
                model.config, bits=bits,
                compress_values=compress_values, verbose=verbose,
                adaptive_policy=adaptive_policy,
                max_cache_len=max_cache_len,
            )
        return original_generate(*args, **kwargs)

    model.generate = patched_generate
    model._turboquant_bits = bits
    model._turboquant_mode = mode or f"{bits}-bit"
    model._turboquant_adaptive = adaptive_policy is not None
    if verbose:
        mode_label = "adaptive" if adaptive_policy else f"{bits}-bit"
        print(f"[TurboQuant] {mode_label} patched onto {type(model).__name__}")


def unpatch_model_cache(model):
    if hasattr(model, "_turboquant_bits"):
        del model.generate
        del model._turboquant_bits


# ── Config helpers ────────────────────────────────────────────────────────────
# Delegate to quantcore.compat for robust multi-architecture support.
# Falls back to inline logic if quantcore is not installed.

def _get_head_dim(config) -> int:
    try:
        from quantcore.compat import extract_model_info
        return extract_model_info(config).head_dim
    except Exception:
        if hasattr(config, "head_dim"):
            return config.head_dim
        if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            return config.hidden_size // config.num_attention_heads
        raise ValueError("Cannot infer head_dim from config.")


def _get_num_layers(config) -> int:
    try:
        from quantcore.compat import extract_model_info
        return extract_model_info(config).num_hidden_layers
    except Exception:
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers",
                     "num_decoder_layers"):
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError("Cannot infer num_layers from config.")
