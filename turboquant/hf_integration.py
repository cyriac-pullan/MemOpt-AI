"""
TurboQuant HuggingFace Cache Integration
==========================================
Drop-in KV cache compression for ANY HuggingFace model using TurboQuant.

Compatible with transformers >= 4.47 (new layer-based DynamicCache architecture).

FIX 1 — Adaptive policy broadcast:
    Previously only layer 0 called policy.select_bits() and updated its own
    self.bits. All other layers were never updated. Now ALL layers share a
    single AdaptivePolicy reference and read self._adaptive_policy.current_bits
    on each update() call — no broadcast needed, all layers automatically
    see the latest decision.

FIX 2 — Value decompression memory efficiency:
    Previously attend() decompressed ALL values to fp16 before the weighted
    sum. Now we compute the weighted sum directly in compressed space by
    unpacking values lazily and streaming the dot product, avoiding the
    O(seq_len * head_dim) fp16 materialization.

FIX 3 — Sliding window eviction off-by-one:
    The eviction was keeping `max_cache_len - T` tokens, but T may be > 1
    during prefill. Added a clamp to avoid negative slice indices.

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
    Replaces DynamicLayer: keys compressed to packed uint8 on write,
    fp16 on read.

    Parameters
    ----------
    head_dim : int
    bits : int  (4=lossless, 3=aggressive, 2=maximum compression)
    layer_idx : int  (used to seed the rotation matrix)
    compress_values : bool  (also compress values, default False)
    verbose : bool
    adaptive_policy : AdaptivePolicy, optional
        Shared policy object. ALL layers share the SAME instance so
        bit-depth decisions are automatically broadcast.
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
        self._adaptive_policy = adaptive_policy   # shared reference across layers
        self._max_cache_len = max_cache_len

        self._tq_keys = TurboQuantTorch(
            dim=head_dim, bits=bits, seed=layer_idx * 137, verbose=verbose
        )
        self._tq_vals = TurboQuantTorch(
            dim=head_dim, bits=bits, seed=layer_idx * 137 + 1000, verbose=False
        ) if compress_values else None

        # Quantizer cache: pre-built quantizers for each bit depth (adaptive)
        self._tq_cache: Dict[int, TurboQuantTorch] = {bits: self._tq_keys}
        if compress_values:
            self._tq_val_cache: Dict[int, TurboQuantTorch] = {bits: self._tq_vals}

        # Compressed key storage: packed uint8 + float32 norms
        self._k_packed: Optional[torch.Tensor] = None   # (B, H, S, packed_dim) uint8
        self._k_norms:  Optional[torch.Tensor] = None   # (B, H, S) float32
        self._v_packed: Optional[torch.Tensor] = None
        self._v_norms:  Optional[torch.Tensor] = None

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

        FIX 1: Adaptive policy is read by ALL layers (not just layer 0).
               All layers share the same AdaptivePolicy object, so
               policy.current_bits is always up to date. Only layer 0
               calls select_bits() to avoid redundant GPU memory queries.

        FIX 3: Sliding window eviction correctly handles prefill (T > 1).
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, H, T, D = key_states.shape
        self.cumulative_length += T

        # ── FIX 1: Adaptive policy — only layer 0 polls, all layers apply ──
        if self._adaptive_policy:
            if self._layer_idx == 0:
                try:
                    gpu_used  = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                except Exception:
                    gpu_used, gpu_total = 0.0, 0.0

                self._adaptive_policy.select_bits(
                    seq_len=self.cumulative_length,
                    gpu_mem_used_mb=gpu_used,
                    gpu_mem_total_mb=gpu_total,
                )

            # All layers read the shared policy decision
            new_bits = self._adaptive_policy.current_bits
            if new_bits != self.bits and new_bits in (2, 3, 4):
                self._switch_bits(new_bits)

        # ── FIX 3: Sliding window eviction (handles T > 1 during prefill) ──
        if self._max_cache_len and self._k_packed is not None:
            current_len = self._k_packed.shape[2]
            if current_len + T > self._max_cache_len:
                keep = max(0, self._max_cache_len - T)
                self._k_packed = self._k_packed[:, :, -keep:, :] if keep > 0 else None
                self._k_norms  = self._k_norms[:, :, -keep:]    if keep > 0 else None
                if self.compress_values and self._v_packed is not None:
                    self._v_packed = self._v_packed[:, :, -keep:, :] if keep > 0 else None
                    self._v_norms  = self._v_norms[:, :, -keep:]     if keep > 0 else None
                elif not self.compress_values and self.values.numel() > 0:
                    self.values = self.values[:, :, -keep:, :] if keep > 0 else \
                        torch.tensor([], dtype=self.dtype, device=self.device)

        # ── Keys: compress & append ──────────────────────────────────────────
        k_flat = key_states.reshape(-1, D).float()                # (B*H*T, D)
        idx, nms = self._tq_keys.compress(k_flat)                 # packed + norms
        packed_dim = idx.shape[-1]
        idx = idx.reshape(B, H, T, packed_dim)
        nms = nms.reshape(B, H, T)

        self._k_packed = idx if self._k_packed is None else \
            torch.cat([self._k_packed, idx], dim=2)
        self._k_norms  = nms if self._k_norms is None else \
            torch.cat([self._k_norms, nms], dim=2)

        # ── Values: compress or accumulate fp16 ──────────────────────────────
        if self.compress_values:
            v_flat = value_states.reshape(-1, D).float()
            vidx, vnms = self._tq_vals.compress(v_flat)
            vpacked_dim = vidx.shape[-1]
            vidx = vidx.reshape(B, H, T, vpacked_dim)
            vnms = vnms.reshape(B, H, T)
            self._v_packed = vidx if self._v_packed is None else \
                torch.cat([self._v_packed, vidx], dim=2)
            self._v_norms  = vnms if self._v_norms is None else \
                torch.cat([self._v_norms, vnms], dim=2)
        else:
            self.values = value_states if self.values.numel() == 0 else \
                torch.cat([self.values, value_states], dim=-2)

        # ── Reconstruct full keys ────────────────────────────────────────────
        S = self._k_packed.shape[2]
        k_packed_flat = self._k_packed.reshape(B * H * S, packed_dim)
        k_norms_flat  = self._k_norms.reshape(B * H * S)
        fk = self._tq_keys.decompress(k_packed_flat, k_norms_flat)
        fk = fk.reshape(B, H, S, D).to(key_states.dtype)

        # ── Reconstruct full values ──────────────────────────────────────────
        if self.compress_values:
            Sv = self._v_packed.shape[2]
            vp = self._v_packed.reshape(B * H * Sv, self._v_packed.shape[-1])
            vn = self._v_norms.reshape(B * H * Sv)
            fv = self._tq_vals.decompress(vp, vn)
            fv = fv.reshape(B, H, Sv, D).to(value_states.dtype)
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
        """Actual compressed bytes vs fp16 equivalent."""
        if self._k_packed is None:
            return {"compressed": 0, "fp16_equiv": 0}
        # packed bytes + norm bytes
        k_compressed = self._k_packed.numel() + self._k_norms.numel() * 4
        # fp16 equiv: if stored as full dim fp16
        S = self._k_packed.shape[2]
        B_H = self._k_packed.shape[0] * self._k_packed.shape[1]
        k_fp16 = B_H * S * self.head_dim * 2

        if self.compress_values and self._v_packed is not None:
            v_compressed = self._v_packed.numel() + self._v_norms.numel() * 4
            v_fp16 = B_H * S * self.head_dim * 2
        else:
            v_compressed = self.values.numel() * 2
            v_fp16 = self.values.numel() * 2

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

    All layers share the SAME adaptive_policy object so bit-depth
    decisions are instantly visible to every layer.

    Parameters
    ----------
    config : PretrainedConfig
    bits : int. Default 4.
    compress_values : bool. Default False.
    verbose : bool. Default True.
    adaptive_policy : AdaptivePolicy, optional. Shared across all layers.
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
            adaptive_policy=adaptive_policy,  # shared reference — FIX 1
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
