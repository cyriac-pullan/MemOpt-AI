"""
QuantCore Test Suite
====================
Tests the quantcore SDK layer on top of the turboquant engine.
All tests run without a GPU — NumPy only.
"""

import sys
import os
import pytest
import numpy as np

# Ensure both turboquant and quantcore are importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


# ── SDK tests ─────────────────────────────────────────────────────────────────

class TestModeValidation:
    def test_invalid_mode_raises(self):
        from quantcore.exceptions import QuantCoreModeError
        from quantcore.sdk import _validate_mode
        with pytest.raises(QuantCoreModeError):
            _validate_mode("ultra")

    def test_valid_modes(self):
        from quantcore.sdk import _validate_mode
        assert _validate_mode("fast") == 4
        assert _validate_mode("balanced") == 3
        assert _validate_mode("max_memory_save") == 2

    def test_mode_info(self):
        from quantcore import mode_info
        info = mode_info()
        assert set(info.keys()) == {"fast", "balanced", "max_memory_save"}
        assert info["fast"]["bits"] == 4
        assert info["balanced"]["cosine_sim"] > 0.98

    def test_mode_info_single(self):
        from quantcore import mode_info
        info = mode_info("balanced")
        assert "balanced" in info


# ── Compat tests ──────────────────────────────────────────────────────────────

class _FakeConfig:
    """Minimal fake HF config for testing compat.py without transformers."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.architectures = ["FakeModelForCausalLM"]
        self.model_type = "fake"


class TestCompat:
    def test_basic_extraction(self):
        from quantcore.compat import extract_model_info
        cfg = _FakeConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=32,
        )
        info = extract_model_info(cfg)
        assert info.head_dim == 128          # 4096 / 32
        assert info.num_kv_heads == 8
        assert info.num_hidden_layers == 32

    def test_explicit_head_dim(self):
        from quantcore.compat import extract_model_info
        cfg = _FakeConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=32,
            head_dim=128,
        )
        info = extract_model_info(cfg)
        assert info.head_dim == 128

    def test_mha_fallback(self):
        """MHA model: num_kv_heads should equal num_attention_heads."""
        from quantcore.compat import extract_model_info
        cfg = _FakeConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
        )
        info = extract_model_info(cfg)
        assert info.num_kv_heads == 12

    def test_alternative_layer_attr(self):
        from quantcore.compat import extract_model_info
        cfg = _FakeConfig(
            hidden_size=768,
            num_attention_heads=12,
            n_layers=6,
        )
        info = extract_model_info(cfg)
        assert info.num_hidden_layers == 6

    def test_missing_config_raises(self):
        from quantcore.compat import extract_model_info
        from quantcore.exceptions import QuantCoreCompatError
        cfg = _FakeConfig(something_irrelevant=42)
        with pytest.raises(QuantCoreCompatError):
            extract_model_info(cfg)

    def test_kv_cache_mb(self):
        from quantcore.compat import extract_model_info
        cfg = _FakeConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=32,
        )
        info = extract_model_info(cfg)
        kv = info.kv_cache_mb(seq_len=1024, bits=4)
        assert kv["fp16_mb"] > kv["compressed_mb"]
        assert kv["ratio"] > 1.0

    def test_check_compatibility(self):
        from quantcore.compat import check_compatibility
        cfg = _FakeConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=32,
        )
        ok, msg = check_compatibility(cfg)
        assert ok is True
        assert "✓" in msg


# ── Profiler / benchmark tests ────────────────────────────────────────────────

class TestBenchmark:
    def test_benchmark_runs(self):
        from quantcore import benchmark
        result = benchmark(dim=64, num_heads=4, num_layers=4,
                           seq_lens=(128, 256), bits=4, mode="fast",
                           n_vectors=16)
        assert result.cosine_similarity > 0.98
        assert result.compression_ratio > 1.0
        assert len(result.seq_results) == 2

    def test_benchmark_modes(self):
        from quantcore import benchmark
        for mode, bits, min_cos in [
            ("fast",            4, 0.99),
            ("balanced",        3, 0.97),
            ("max_memory_save", 2, 0.90),
        ]:
            r = benchmark(dim=128, num_heads=2, num_layers=2,
                          seq_lens=(256,), bits=bits, mode=mode, n_vectors=32)
            assert r.cosine_similarity >= min_cos, \
                f"{mode}: cosine sim {r.cosine_similarity:.4f} < {min_cos}"

    def test_benchmark_to_json(self, tmp_path):
        from quantcore import benchmark
        result = benchmark(dim=64, num_heads=2, num_layers=2,
                           seq_lens=(128,), bits=4, mode="fast", n_vectors=8)
        path = str(tmp_path / "result.json")
        result.to_json(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["cosine_similarity"] > 0
        assert "seq_results" in data

    def test_profile_result_summary(self):
        from quantcore import benchmark
        result = benchmark(dim=64, num_heads=2, num_layers=2,
                           seq_lens=(128, 256), bits=4, mode="fast", n_vectors=8)
        summary = result.summary()
        assert "QuantCore" in summary
        assert "cosine sim" in summary
        assert "Memory" in summary


# ── Core TurboQuant engine tests (sanity — existing engine still works) ───────

class TestEngine:
    """Smoke tests confirming the turboquant engine is intact."""

    def test_compress_decompress(self):
        from turboquant import TurboQuant
        tq = TurboQuant(dim=64, bits=4)
        x = np.random.randn(64).astype(np.float32)
        c = tq.compress(x)
        x_hat = tq.decompress(c)
        cos = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat) + 1e-9)
        assert cos > 0.98, f"Cosine sim too low: {cos:.4f}"

    def test_compress_batch(self):
        from turboquant import TurboQuant
        tq = TurboQuant(dim=128, bits=3)
        X = np.random.randn(32, 128).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        compressed = tq.compress_batch(X)
        X_hat = tq.decompress_batch(compressed)
        cos = np.mean(
            np.einsum("ij,ij->i", X, X_hat) /
            (np.linalg.norm(X, axis=1) * np.linalg.norm(X_hat, axis=1) + 1e-9)
        )
        assert cos > 0.97, f"Batch cosine sim too low: {cos:.4f}"

    def test_kv_cache(self):
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(num_heads=4, head_dim=64, bits=4, verbose=False)
        for _ in range(10):
            k = np.random.randn(4, 64).astype(np.float32)
            v = np.random.randn(4, 64).astype(np.float32)
            cache.append(k, v)
        q = np.random.randn(4, 64).astype(np.float32)
        out = cache.attend(q)
        assert out.shape == (4, 64)
