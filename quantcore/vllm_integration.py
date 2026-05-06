"""
QuantCore vLLM Integration
===========================
Integrates QuantCore's adaptive KV cache compression into vLLM's
production serving pipeline.

Usage with vLLM (offline batch):
    from quantcore.vllm_integration import quantcore_vllm_serve
    quantcore_vllm_serve("meta-llama/Llama-3.1-8B", max_memory="12GB")

Usage with vLLM (online API server):
    quantcore serve --model meta-llama/Llama-3.1-8B --max-memory 12GB

Architecture:
    vLLM Engine
        └─ QuantCore CacheEngine wrapper
            └─ TurboQuant compression (per-layer)
            └─ AdaptivePolicy (dynamic bit switching)
            └─ Sliding window eviction
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Dict, Any

# Ensure turboquant is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


# ── vLLM availability ────────────────────────────────────────────────────────

_VLLM_AVAILABLE = False
try:
    import vllm
    _VLLM_AVAILABLE = True
except ImportError:
    pass


def is_vllm_available() -> bool:
    """Check if vLLM is installed."""
    return _VLLM_AVAILABLE


# ── QuantCore-wrapped LLM for vLLM offline batch ─────────────────────────────

class QuantCoreLLM:
    """
    Drop-in wrapper around vLLM's LLM class with QuantCore KV cache compression.

    This patches the model after vLLM loads it, injecting TurboQuant
    compression into the attention layers' KV cache.

    Parameters
    ----------
    model : str
        HuggingFace model ID.
    mode : str
        QuantCore compression mode: 'fast', 'balanced', 'max_memory_save', 'adaptive'.
    max_memory : str, int, float, optional
        Memory budget (e.g. "12GB", 8192, 0.8).
    max_cache_len : int, optional
        Sliding window eviction length.
    quantization : str, optional
        vLLM quantization method (e.g. 'awq', 'gptq'). Applied to weights, not KV.
    **vllm_kwargs
        Additional arguments passed to vllm.LLM().

    Example
    -------
    >>> from quantcore.vllm_integration import QuantCoreLLM
    >>> llm = QuantCoreLLM("meta-llama/Llama-3.1-8B", mode="adaptive", max_memory="12GB")
    >>> outputs = llm.generate(["Explain KV cache compression"])
    >>> print(outputs[0].outputs[0].text)
    """

    def __init__(
        self,
        model: str,
        mode: str = "balanced",
        max_memory=None,
        max_cache_len: int = None,
        quantization: str = None,
        **vllm_kwargs,
    ):
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for QuantCoreLLM.\n"
                "Install with: pip install vllm"
            )

        self.model_id = model
        self.mode = mode
        self.max_memory = max_memory
        self.max_cache_len = max_cache_len

        # Build vLLM engine
        from vllm import LLM

        vllm_kwargs.setdefault("trust_remote_code", True)
        if quantization:
            vllm_kwargs["quantization"] = quantization

        self._llm = LLM(model=model, **vllm_kwargs)
        self._patch_kv_cache()

    def _patch_kv_cache(self):
        """
        Patch the loaded model's KV cache with QuantCore compression.
        This runs after vLLM has loaded and allocated the model.
        """
        from quantcore.sdk import optimize_model

        # Access the underlying model from vLLM's engine
        try:
            model = self._llm.llm_engine.model_executor.driver_worker.model_runner.model
            optimize_model(
                model,
                mode=self.mode,
                max_memory=self.max_memory,
                max_cache_len=self.max_cache_len,
                verbose=True,
            )
            self._patched = True
        except Exception as e:
            print(f"[QuantCore] Warning: Could not patch vLLM model KV cache: {e}")
            print(f"[QuantCore] Falling back to standard vLLM inference.")
            self._patched = False

    def generate(self, prompts, sampling_params=None, **kwargs):
        """
        Generate completions using vLLM with QuantCore-compressed KV cache.

        Parameters
        ----------
        prompts : list of str
            Input prompts.
        sampling_params : vllm.SamplingParams, optional
            Sampling configuration.
        """
        if sampling_params is None:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=512,
            )

        return self._llm.generate(prompts, sampling_params, **kwargs)

    @property
    def is_patched(self) -> bool:
        """Whether QuantCore compression is active."""
        return getattr(self, "_patched", False)

    def info(self) -> Dict[str, Any]:
        """Return QuantCore configuration info."""
        return {
            "model": self.model_id,
            "mode": self.mode,
            "max_memory": self.max_memory,
            "max_cache_len": self.max_cache_len,
            "patched": self.is_patched,
            "vllm_version": vllm.__version__ if _VLLM_AVAILABLE else None,
        }


# ── Convenience launcher ─────────────────────────────────────────────────────

def quantcore_vllm_serve(
    model: str,
    mode: str = "adaptive",
    max_memory=None,
    max_cache_len: int = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    **vllm_kwargs,
):
    """
    Launch a vLLM API server with QuantCore KV cache compression.

    This starts an OpenAI-compatible API server that serves the model
    with compressed KV caches for reduced memory usage.

    Parameters
    ----------
    model : str
        HuggingFace model ID.
    mode : str
        Compression mode (default: "adaptive").
    max_memory : str, int, float, optional
        Memory budget.
    max_cache_len : int, optional
        Sliding window eviction length.
    host : str
        Server host (default: "0.0.0.0").
    port : int
        Server port (default: 8000).
    **vllm_kwargs
        Additional vLLM arguments.

    Example
    -------
    >>> quantcore_vllm_serve(
    ...     "meta-llama/Llama-3.1-8B",
    ...     mode="adaptive",
    ...     max_memory="12GB",
    ...     max_cache_len=8192,
    ... )
    """
    if not _VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is required for serving.\n"
            "Install with: pip install vllm"
        )

    print(f"[QuantCore] Starting vLLM server with adaptive KV compression")
    print(f"  Model:      {model}")
    print(f"  Mode:       {mode}")
    if max_memory:
        print(f"  Budget:     {max_memory}")
    if max_cache_len:
        print(f"  Window:     {max_cache_len} tokens")
    print(f"  Endpoint:   http://{host}:{port}/v1")
    print()

    # Build args for vLLM's OpenAI-compatible server
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    parser = make_arg_parser()
    args = parser.parse_args([
        "--model", model,
        "--host", host,
        "--port", str(port),
    ])

    # Inject QuantCore post-load hook
    _original_init = None
    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        _original_init = AsyncLLMEngine.__init__

        def _patched_init(self_engine, *a, **kw):
            _original_init(self_engine, *a, **kw)
            try:
                from quantcore.sdk import optimize_model
                model_obj = self_engine.engine.model_executor.driver_worker.model_runner.model
                optimize_model(
                    model_obj,
                    mode=mode,
                    max_memory=max_memory,
                    max_cache_len=max_cache_len,
                    verbose=True,
                )
                print(f"[QuantCore] KV cache compression active on {model}")
            except Exception as e:
                print(f"[QuantCore] Warning: Could not patch: {e}")

        AsyncLLMEngine.__init__ = _patched_init
        run_server(args)
    finally:
        if _original_init:
            AsyncLLMEngine.__init__ = _original_init


# ── Configuration helper for vLLM CLI ─────────────────────────────────────────

def get_vllm_engine_args(
    model: str,
    mode: str = "balanced",
    max_memory=None,
    **extra,
) -> Dict[str, Any]:
    """
    Get vLLM EngineArgs with QuantCore-optimal settings.

    Returns a dict suitable for passing to vLLM's AsyncLLMEngine.from_engine_args().

    Parameters
    ----------
    model : str
    mode : str
    max_memory : str, int, float, optional
    **extra : additional vLLM engine args

    Returns
    -------
    dict of engine arguments
    """
    args = {
        "model": model,
        "trust_remote_code": True,
        "enforce_eager": True,  # needed for custom cache hooks
        **extra,
    }

    # Parse memory budget to set gpu_memory_utilization
    if max_memory is not None:
        from .policy import parse_memory, detect_gpu_memory_mb
        budget_mb = parse_memory(max_memory) if max_memory != 0 else detect_gpu_memory_mb()
        total_mb = detect_gpu_memory_mb()
        if total_mb > 0:
            args["gpu_memory_utilization"] = min(budget_mb / total_mb, 0.95)

    return args

