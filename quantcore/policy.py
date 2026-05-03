"""
QuantCore Policy Engine
=======================
Memory budget management and automatic mode selection.

Supports three max_memory formats:
    optimize_model(model, max_memory="6GB")    # string
    optimize_model(model, max_memory=6144)     # MB as int
    optimize_model(model, max_memory=0.8)      # 80% of GPU
"""

from __future__ import annotations
from typing import Union, Optional


def detect_gpu_memory_mb() -> float:
    """Returns total GPU memory in MB. Returns 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def detect_gpu_used_mb() -> float:
    """Returns currently allocated GPU memory in MB. Returns 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def parse_memory(value: Union[str, int, float]) -> float:
    """
    Parse a memory specification into MB.

    Supports:
        "6GB"   -> 6144.0 MB
        "512MB" -> 512.0 MB
        6144    -> 6144.0 MB (int = MB)
        0.8     -> 80% of GPU total memory

    Parameters
    ----------
    value : str, int, or float
        Memory specification.

    Returns
    -------
    float : memory in MB

    Raises
    ------
    ValueError : if string format is invalid
    TypeError : if type is unsupported
    """
    if isinstance(value, str):
        v = value.strip().upper()
        if v.endswith("GB"):
            return float(v[:-2]) * 1024
        elif v.endswith("MB"):
            return float(v[:-2])
        else:
            try:
                return float(v)
            except ValueError:
                raise ValueError(
                    f"Invalid memory string: '{value}'. "
                    f"Use '6GB', '512MB', or a number."
                )

    elif isinstance(value, float) and 0 < value < 1:
        # Fraction of GPU total
        total = detect_gpu_memory_mb()
        if total == 0:
            raise RuntimeError(
                "Cannot use fractional max_memory without GPU. "
                "Use '6GB' or an integer (MB) instead."
            )
        return value * total

    elif isinstance(value, (int, float)):
        return float(value)

    else:
        raise TypeError(
            f"max_memory must be str, int, or float. Got {type(value).__name__}."
        )


def auto_select_bits(max_memory_mb: float = None) -> int:
    """
    Selects the best bit-depth based on maximum available memory.
    If max_memory_mb is not provided, it detects GPU memory automatically.
    """
    if max_memory_mb is None:
        max_memory_mb = detect_gpu_memory_mb()

    if max_memory_mb == 0:
        return 3

    if max_memory_mb < 8192:
        return 2
    elif max_memory_mb < 16384:
        return 3
    else:
        return 4


def auto_select_mode(max_memory_mb: float = None) -> str:
    """Returns the mode string based on memory."""
    bits = auto_select_bits(max_memory_mb)
    if bits == 2:
        return "max_memory_save"
    elif bits == 3:
        return "balanced"
    else:
        return "fast"


class MemoryBudget:
    """
    Tracks GPU memory against a hard budget.

    Usage:
        budget = MemoryBudget(max_memory_mb=6144)
        status = budget.check()
        if status["action"] == "compress_more":
            # switch to more aggressive compression
    """

    def __init__(self, max_memory_mb: float):
        self.max_memory_mb = max_memory_mb

    def check(self) -> dict:
        """
        Returns current memory state and recommended action.

        Returns
        -------
        dict with keys: used_mb, remaining_mb, pressure, action
            action is one of: "ok", "compress_more", "critical"
        """
        used = detect_gpu_used_mb()
        remaining = self.max_memory_mb - used
        pressure = used / self.max_memory_mb if self.max_memory_mb > 0 else 0

        if pressure > 0.95:
            action = "critical"
        elif pressure > 0.85:
            action = "compress_more"
        else:
            action = "ok"

        return {
            "used_mb": round(used, 1),
            "budget_mb": round(self.max_memory_mb, 1),
            "remaining_mb": round(remaining, 1),
            "pressure": round(pressure, 3),
            "action": action,
        }

    def __repr__(self) -> str:
        status = self.check()
        return (
            f"MemoryBudget({status['used_mb']:.0f}/{status['budget_mb']:.0f} MB, "
            f"pressure={status['pressure']:.1%}, action={status['action']})"
        )
