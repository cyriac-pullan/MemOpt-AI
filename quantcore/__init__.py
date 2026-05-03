"""
QuantCore
=========
AI Memory Optimization Layer for LLMs.

Compresses transformer KV caches by 2–6x using near-optimal vector
quantization (TurboQuant, ICLR 2026) — with no retraining, no calibration
data, and no accuracy loss at 4-bit.

Quick start
-----------
    from quantcore import optimize_model

    model = optimize_model(model)                          # balanced 3-bit
    model = optimize_model(model, mode="fast")            # 4-bit, best quality
    model = optimize_model(model, mode="max_memory_save") # 2-bit, max savings

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
from .exceptions import (
    QuantCoreError,
    QuantCoreCompatError,
    QuantCoreModeError,
    QuantCoreDependencyError,
)

__version__ = "0.1.0"
__author__  = "QuantCore Contributors"
__paper__   = "https://arxiv.org/abs/2504.19874"

__all__ = [
    # Main API
    "optimize_model",
    "mode_info",
    "benchmark",
    "profile_memory",
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
