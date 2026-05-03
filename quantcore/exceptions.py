"""
QuantCore Exceptions
====================
Clear, actionable errors for every failure mode.
"""


class QuantCoreError(Exception):
    """Base exception for all QuantCore errors."""


class QuantCoreCompatError(QuantCoreError):
    """
    Raised when a model architecture is not supported or cannot be
    auto-detected.

    The error message always includes what was tried and how to fix it.
    """


class QuantCoreModeError(QuantCoreError):
    """Raised when an invalid mode string is provided."""

    VALID_MODES = ("fast", "balanced", "max_memory_save")

    def __init__(self, mode: str):
        super().__init__(
            f"Invalid mode {mode!r}. Choose from: {self.VALID_MODES}\n"
            f"  fast             → 4-bit, cosine sim 0.995, ~1.9x vs fp16\n"
            f"  balanced         → 3-bit, cosine sim 0.983, ~2.8x vs fp16\n"
            f"  max_memory_save  → 2-bit, cosine sim 0.940, ~4.0x vs fp16"
        )


class QuantCoreDependencyError(QuantCoreError):
    """Raised when an optional dependency is missing."""

    def __init__(self, package: str, feature: str, install_extra: str = None):
        extra = f"  pip install quantcore[{install_extra}]" if install_extra else f"  pip install {package}"
        super().__init__(
            f"{feature} requires '{package}', which is not installed.\n"
            f"Install it with:\n{extra}"
        )
