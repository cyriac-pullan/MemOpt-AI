"""
QuantCore Adaptive Compression Engine
======================================
Dynamically adjusts KV cache compression based on runtime conditions.

This is the core differentiator — no major inference engine does this.

Architecture:
    AdaptivePolicy monitors seq_len + GPU memory pressure and returns
    the optimal bit depth. It uses hysteresis to prevent rapid switching.

    Segments track mixed-precision regions in the KV cache:
    [4-bit tokens 0-1023] [3-bit tokens 1024-4095] [2-bit tokens 4096+]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CacheSegment:
    """One contiguous region of the KV cache at a single bit depth."""
    bits: int
    start_idx: int  # inclusive
    end_idx: int    # exclusive
    token_count: int = 0

    @property
    def range_str(self) -> str:
        return f"[{self.start_idx}:{self.end_idx}] @ {self.bits}-bit"


class AdaptivePolicy:
    """
    Decides the optimal bit depth based on runtime state.

    Uses hysteresis to prevent rapid oscillation:
    - Must stay at current level for `stability_steps` before switching
    - Only switches when the *difference* in recommended bits is >= 1

    Parameters
    ----------
    memory_budget_mb : float, optional
        Hard memory limit. If set, policy aggressively compresses to stay under.
    stability_steps : int
        Minimum steps at current bits before allowing a switch. Default 32.
    thresholds : dict, optional
        Override default seq_len/pressure thresholds.
    """

    # Default thresholds: (seq_len, memory_pressure) -> bits
    DEFAULT_THRESHOLDS = [
        # (max_seq_len, max_mem_pressure, bits)
        (1024,  0.70, 16),   # no compression for short context, low pressure
        (2048,  0.80,  4),   # light compression
        (4096,  0.88,  4),   # still 4-bit
        (8192,  0.92,  3),   # balanced
        (99999, 1.00,  2),   # emergency: max compression
    ]

    def __init__(
        self,
        memory_budget_mb: float = None,
        stability_steps: int = 32,
        thresholds: list = None,
    ):
        self.memory_budget_mb = memory_budget_mb
        self.stability_steps = stability_steps
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

        # State tracking
        self._current_bits: int = 16  # start uncompressed
        self._steps_at_current: int = 0
        self._history: List[dict] = []

    def select_bits(
        self,
        seq_len: int,
        gpu_mem_used_mb: float = 0,
        gpu_mem_total_mb: float = 0,
    ) -> int:
        """
        Returns the recommended bit depth for the current state.

        Parameters
        ----------
        seq_len : int
            Current total sequence length (input + generated so far).
        gpu_mem_used_mb : float
            Current GPU memory used in MB.
        gpu_mem_total_mb : float
            Total GPU memory in MB.

        Returns
        -------
        int : recommended bits (2, 3, 4, or 16 for no compression)
        """
        # Calculate memory pressure
        if gpu_mem_total_mb > 0:
            pressure = gpu_mem_used_mb / gpu_mem_total_mb
        else:
            pressure = 0.0

        # Memory budget override: if we're over budget, force max compression
        if self.memory_budget_mb and gpu_mem_used_mb > 0:
            budget_pressure = gpu_mem_used_mb / self.memory_budget_mb
            if budget_pressure > 0.95:
                recommended = 2  # emergency
            elif budget_pressure > 0.85:
                recommended = 3
            else:
                recommended = self._threshold_lookup(seq_len, pressure)
        else:
            recommended = self._threshold_lookup(seq_len, pressure)

        # Apply hysteresis: don't switch unless stable
        self._steps_at_current += 1

        if recommended != self._current_bits:
            if self._steps_at_current >= self.stability_steps:
                # Only allow downward transitions (more compression)
                # or upward if pressure has dropped significantly
                if recommended < self._current_bits:
                    # More compression needed — allow immediately after stability
                    self._current_bits = recommended
                    self._steps_at_current = 0
                elif pressure < 0.5 and recommended > self._current_bits:
                    # Pressure dropped a lot — can relax compression
                    self._current_bits = recommended
                    self._steps_at_current = 0

        # Log for observability
        self._history.append({
            "seq_len": seq_len,
            "pressure": round(pressure, 3),
            "recommended": recommended,
            "applied": self._current_bits,
            "stable_for": self._steps_at_current,
        })

        return self._current_bits

    def _threshold_lookup(self, seq_len: int, pressure: float) -> int:
        """Walk the threshold table and return first matching bits."""
        for max_seq, max_pressure, bits in self.thresholds:
            if seq_len <= max_seq and pressure <= max_pressure:
                return bits
        # Fallback: max compression
        return 2

    @property
    def current_bits(self) -> int:
        return self._current_bits

    @property
    def history(self) -> List[dict]:
        return self._history

    def stats(self) -> dict:
        """Return current policy state for dashboard/logging."""
        return {
            "current_bits": self._current_bits,
            "steps_at_current": self._steps_at_current,
            "stability_threshold": self.stability_steps,
            "memory_budget_mb": self.memory_budget_mb,
            "total_decisions": len(self._history),
        }

    def __repr__(self) -> str:
        return (
            f"AdaptivePolicy(bits={self._current_bits}, "
            f"stable_for={self._steps_at_current}/{self.stability_steps}, "
            f"budget={self.memory_budget_mb}MB)"
        )
