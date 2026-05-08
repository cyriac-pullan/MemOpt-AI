"""
QuantCore
=========
Adaptive Memory Runtime for LLMs.

Compresses transformer KV caches by 2–6x using near-optimal vector
quantization (TurboQuant, ICLR 2026) — with no retraining, no calibration
data, and no accuracy loss at 4-bit. Features dynamic bit switching,
memory budgets, and sliding window eviction.

Quick start
-----------
    from quantcore import optimize_model

    model = optimize_model(model)                          # balanced 3-bit
    model = optimize_model(model, mode="fast")            # 4-bit, best quality
    model = optimize_model(model, mode="max_memory_save") # 2-bit, max savings
    model = optimize_model(model, mode="adaptive",        # dynamic runtime
                           max_memory="8GB",
                           max_cache_len=8192)

    outputs = model.generate(input_ids, max_new_tokens=512)

After optimization
------------------
    stats = model.quantcore_stats(seq_len=4096)
    # {'mode': 'balanced', 'bits': 3, 'fp16_mb': 512.0,
    #  'compressed_mb': 186.0, 'compression_ratio': 2.75, ...}

Benchmark (no GPU needed)
-------------------------
    from quantcore import benchmark
    result = benchmark()
    print(result.summary())

CLI
---
    quantcore info      --model meta-llama/Llama-3.1-8B
    quantcore benchmark --mode all
    quantcore dashboard --port 8080
"""

from .sdk import optimize_model, mode_info
from .profiler import benchmark_numpy as benchmark, profile_memory
from .compat import extract_model_info, check_compatibility, ModelInfo
from .adaptive import AdaptivePolicy
from .policy import MemoryBudget, parse_memory
from .exceptions import (
    QuantCoreError,
    QuantCoreCompatError,
    QuantCoreModeError,
    QuantCoreDependencyError,
)

__version__ = "0.4.1"
__author__  = "QuantCore Contributors"
__paper__   = "https://arxiv.org/abs/2504.19874"

__all__ = [
    # Main API
    "optimize_model",
    "mode_info",
    "benchmark",
    "profile_memory",
    # Adaptive Runtime (v2)
    "AdaptivePolicy",
    "MemoryBudget",
    "parse_memory",
    # Compatibility
    "extract_model_info",
    "check_compatibility",
    "ModelInfo",
    # Exceptions
    "QuantCoreError",
    "QuantCoreCompatError",
    "QuantCoreModeError",
    "QuantCoreDependencyError",
]
