"""
QuantCore Model Compatibility Layer
=====================================
Reliably extracts head_dim, num_kv_heads, and num_layers from any
HuggingFace PretrainedConfig — across Llama, Mistral, Phi, Gemma,
Qwen, Falcon, GPT-NeoX, GPT-2, and more.

This is the single source of truth for model metadata in QuantCore.
"""

from dataclasses import dataclass
from typing import Optional

from .exceptions import QuantCoreCompatError


@dataclass
class ModelInfo:
    """Extracted metadata for a HuggingFace model."""
    architecture: str        # e.g. "LlamaForCausalLM"
    family: str              # e.g. "llama"
    num_hidden_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    num_attention_heads: int

    def expected_compression(self, bits: int) -> float:
        """Approx compression ratio vs fp16 for KV keys at given bits."""
        bytes_fp16 = self.head_dim * 2
        bytes_compressed = self.head_dim + 4  # uint8 indices + float32 norm
        return bytes_fp16 / bytes_compressed

    def kv_cache_mb(self, seq_len: int, bits: int) -> dict:
        """
        Estimate KV cache memory (MB) for keys at a given sequence length.
        """
        # FP16 baseline: seq * layers * kv_heads * head_dim * 2 bytes * 2 (K+V)
        fp16 = (
            seq_len * self.num_hidden_layers * self.num_kv_heads
            * self.head_dim * 2 * 2 / 1024 / 1024
        )
        # Compressed: uint8 indices (1 byte/elem) + float32 norm (4 bytes) per vector
        bytes_per_vec = self.head_dim + 4
        compressed = (
            seq_len * self.num_hidden_layers * self.num_kv_heads
            * bytes_per_vec * 2 / 1024 / 1024
        )
        return {
            "fp16_mb": round(fp16, 2),
            "compressed_mb": round(compressed, 2),
            "ratio": round(fp16 / compressed, 2),
        }

    def summary(self) -> str:
        lines = [
            f"  Architecture : {self.architecture}",
            f"  Family       : {self.family}",
            f"  Layers       : {self.num_hidden_layers}",
            f"  KV Heads     : {self.num_kv_heads}",
            f"  Head Dim     : {self.head_dim}",
        ]
        for bits, label in [(4, "fast"), (3, "balanced"), (2, "max_memory_save")]:
            kv = self.kv_cache_mb(seq_len=4096, bits=bits)
            lines.append(
                f"  [{label:>16}]  {kv['fp16_mb']:.0f} MB fp16  →  "
                f"{kv['compressed_mb']:.0f} MB ({kv['ratio']:.1f}x)  at seq=4096"
            )
        return "\n".join(lines)


# ── Extraction logic ──────────────────────────────────────────────────────────

def _get_num_layers(config) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers",
                 "num_decoder_layers"):
        if hasattr(config, attr):
            v = getattr(config, attr)
            if isinstance(v, int) and v > 0:
                return v
    raise QuantCoreCompatError(
        f"Cannot determine number of layers from config of type "
        f"{type(config).__name__}. Tried: num_hidden_layers, n_layer, "
        f"num_layers, n_layers, num_decoder_layers."
    )


def _get_num_attention_heads(config) -> int:
    for attr in ("num_attention_heads", "n_head", "num_heads"):
        if hasattr(config, attr):
            v = getattr(config, attr)
            if isinstance(v, int) and v > 0:
                return v
    raise QuantCoreCompatError(
        f"Cannot determine num_attention_heads from config of type "
        f"{type(config).__name__}."
    )


def _get_num_kv_heads(config, num_attention_heads: int) -> int:
    """
    Returns num_key_value_heads (GQA) or falls back to num_attention_heads
    for MHA models. Handles Falcon's non-standard attribute name.
    """
    for attr in ("num_key_value_heads", "num_kv_heads", "multi_query_group_num"):
        if hasattr(config, attr):
            v = getattr(config, attr)
            if isinstance(v, int) and v > 0:
                return v

    # Falcon uses new_decoder_architecture flag + num_kv_heads
    if hasattr(config, "new_decoder_architecture") and hasattr(config, "num_kv_heads"):
        return config.num_kv_heads

    # MHA: kv_heads == attention_heads
    return num_attention_heads


def _get_hidden_size(config) -> int:
    for attr in ("hidden_size", "d_model", "n_embd"):
        if hasattr(config, attr):
            v = getattr(config, attr)
            if isinstance(v, int) and v > 0:
                return v
    raise QuantCoreCompatError(
        f"Cannot determine hidden_size from config of type "
        f"{type(config).__name__}."
    )


def _get_head_dim(config, hidden_size: int, num_attention_heads: int) -> int:
    # Some models (Phi-3, Gemma-2) store head_dim explicitly
    if hasattr(config, "head_dim"):
        v = config.head_dim
        if isinstance(v, int) and v > 0:
            return v
    # Standard: hidden_size / num_attention_heads
    if hidden_size % num_attention_heads == 0:
        return hidden_size // num_attention_heads
    raise QuantCoreCompatError(
        f"Cannot compute head_dim: hidden_size={hidden_size} is not evenly "
        f"divisible by num_attention_heads={num_attention_heads}."
    )


def _detect_family(config) -> str:
    """Best-effort model family detection from config class name."""
    name = type(config).__name__.lower()
    for family in ("llama", "mistral", "mixtral", "phi", "gemma", "qwen",
                   "falcon", "gpt_neox", "gpt2", "bloom", "opt", "t5",
                   "stablelm"):
        if family.replace("_", "") in name.replace("_", ""):
            return family.replace("_", "-")
    # Fall back to the model_type attribute if present
    if hasattr(config, "model_type"):
        return str(config.model_type)
    return "unknown"


def _get_architecture(config) -> str:
    if hasattr(config, "architectures") and config.architectures:
        return config.architectures[0]
    return type(config).__name__.replace("Config", "ForCausalLM")


# ── Public API ────────────────────────────────────────────────────────────────

def extract_model_info(config) -> ModelInfo:
    """
    Extract QuantCore-relevant metadata from any HuggingFace PretrainedConfig.

    Parameters
    ----------
    config : PretrainedConfig
        e.g. model.config for any HuggingFace model.

    Returns
    -------
    ModelInfo

    Raises
    ------
    QuantCoreCompatError
        If metadata cannot be reliably extracted.
    """
    hidden_size = _get_hidden_size(config)
    num_heads   = _get_num_attention_heads(config)
    num_kv      = _get_num_kv_heads(config, num_heads)
    head_dim    = _get_head_dim(config, hidden_size, num_heads)
    num_layers  = _get_num_layers(config)
    family      = _detect_family(config)
    arch        = _get_architecture(config)

    return ModelInfo(
        architecture=arch,
        family=family,
        num_hidden_layers=num_layers,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
    )


def check_compatibility(config) -> tuple[bool, str]:
    """
    Check if a model config is compatible with QuantCore.

    Returns
    -------
    (compatible: bool, message: str)
    """
    try:
        info = extract_model_info(config)
        if info.head_dim < 16:
            return False, f"head_dim={info.head_dim} is too small (min 16)."
        if info.head_dim > 1024:
            return False, f"head_dim={info.head_dim} is very large — may be slow."
        return True, f"Compatible ✓  ({info.family}, head_dim={info.head_dim})"
    except QuantCoreCompatError as e:
        return False, str(e)
