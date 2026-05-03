"""
QuantCore Dashboard Server
===========================
Lightweight Flask server for the QuantCore monitoring dashboard.

Start with:
    quantcore dashboard --port 8080

Or programmatically:
    from quantcore.dashboard.server import create_app
    app = create_app()
    app.run(port=8080)

Endpoints
---------
  GET  /                  Serve dashboard HTML
  GET  /api/status        Server status + live metric snapshot
  POST /api/benchmark     Run benchmark, return JSON results
  POST /api/push          SDK pushes live metrics here
  GET  /api/stream        SSE stream of live metrics
"""

from __future__ import annotations

import json
import time
import threading
import os
import sys
from typing import Optional
from collections import deque

# ── Ensure turboquant is importable ──────────────────────────────────────────
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)


# ── In-memory metric store ───────────────────────────────────────────────────

class MetricStore:
    """Thread-safe store for live inference metrics pushed by the SDK."""

    def __init__(self, maxlen: int = 500):
        self._lock = threading.Lock()
        self._metrics: deque = deque(maxlen=maxlen)
        self._latest: dict = {}
        self._session_start: float = time.time()

    def push(self, data: dict):
        with self._lock:
            data["ts"] = time.time()
            self._metrics.append(data)
            self._latest = data

    def latest(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def all(self) -> list:
        with self._lock:
            return list(self._metrics)

    def uptime(self) -> float:
        return time.time() - self._session_start


_store = MetricStore()


# ── Flask app factory ─────────────────────────────────────────────────────────

def create_app() -> "Flask":
    try:
        from flask import Flask, jsonify, request, Response, send_file
        from flask_cors import CORS
    except ImportError:
        raise ImportError(
            "Dashboard requires Flask and flask-cors.\n"
            "Install with: pip install quantcore[dashboard]"
        )

    app = Flask(__name__, static_folder=None)
    CORS(app)

    _dashboard_html = os.path.join(os.path.dirname(__file__), "index.html")

    # ── Routes ────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        if os.path.exists(_dashboard_html):
            return send_file(_dashboard_html)
        return "<h1>QuantCore Dashboard</h1><p>index.html not found.</p>", 404

    @app.route("/api/status")
    def status():
        return jsonify({
            "status": "running",
            "uptime_s": round(_store.uptime(), 1),
            "latest_metric": _store.latest(),
            "metric_count": len(_store.all()),
            "version": _get_version(),
        })

    @app.route("/api/push", methods=["POST"])
    def push_metrics():
        """SDK pushes live memory stats here during inference."""
        data = request.get_json(force=True, silent=True) or {}
        _store.push(data)
        return jsonify({"ok": True})

    @app.route("/api/metrics")
    def get_metrics():
        return jsonify(_store.all())

    @app.route("/api/stream")
    def stream():
        """Server-Sent Events stream of live metrics."""
        def generate():
            last_count = 0
            while True:
                metrics = _store.all()
                if len(metrics) > last_count:
                    for m in metrics[last_count:]:
                        yield f"data: {json.dumps(m)}\n\n"
                    last_count = len(metrics)
                time.sleep(0.5)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    @app.route("/api/benchmark", methods=["POST"])
    def run_benchmark():
        """
        Run a QuantCore NumPy benchmark.
        Body (JSON, all optional):
          { "dim": 128, "heads": 8, "layers": 32,
            "mode": "balanced", "seq_lens": [128, 512, 1024, 2048, 4096] }
        """
        body = request.get_json(force=True, silent=True) or {}
        dim     = int(body.get("dim",    128))
        heads   = int(body.get("heads",  8))
        layers  = int(body.get("layers", 32))
        mode    = str(body.get("mode",   "balanced"))
        seq_lens = tuple(body.get("seq_lens", [128, 512, 1024, 2048, 4096]))

        from quantcore.profiler import benchmark_numpy
        from quantcore.sdk import _MODE_BITS

        valid_modes = list(_MODE_BITS.keys())
        if mode not in valid_modes:
            return jsonify({"error": f"Invalid mode. Choose from {valid_modes}"}), 400

        result = benchmark_numpy(
            dim=dim, num_heads=heads, num_layers=layers,
            seq_lens=seq_lens, mode=mode,
        )
        return jsonify(result.to_dict())

    @app.route("/api/info", methods=["POST"])
    def model_info():
        """
        Check HuggingFace model compatibility.
        Body: { "model_id": "meta-llama/Llama-3.1-8B" }
        """
        body = request.get_json(force=True, silent=True) or {}
        model_id = body.get("model_id", "").strip()
        if not model_id:
            return jsonify({"error": "model_id is required"}), 400

        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            from quantcore.compat import extract_model_info, check_compatibility
            ok, msg = check_compatibility(config)
            info = None
            if ok:
                m = extract_model_info(config)
                info = {
                    "architecture": m.architecture,
                    "family": m.family,
                    "num_hidden_layers": m.num_hidden_layers,
                    "num_kv_heads": m.num_kv_heads,
                    "head_dim": m.head_dim,
                    "kv_cache": {
                        str(sl): m.kv_cache_mb(sl, 4)
                        for sl in [1024, 2048, 4096, 8192]
                    },
                }
            return jsonify({"compatible": ok, "message": msg, "info": info})
        except ImportError:
            return jsonify({"error": "transformers not installed"}), 500
        except Exception as e:
            return jsonify({"compatible": False, "error": str(e)}), 200

    return app


def _get_version() -> str:
    try:
        import quantcore
        return quantcore.__version__
    except Exception:
        return "unknown"
