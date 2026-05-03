"""
QuantCore CLI
=============
Command-line tools for QuantCore.

Commands
--------
  quantcore info      --model <hf_model_id>    Show model compatibility & compression estimates
  quantcore benchmark [--dim D] [--bits B]      Run NumPy benchmark (no GPU needed)
  quantcore dashboard [--port P] [--host H]     Start the monitoring web dashboard
  quantcore version                             Show version
"""

from __future__ import annotations

import argparse
import sys
import os
import json


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_turboquant():
    """Ensure turboquant root is on sys.path."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def _color(text: str, code: str) -> str:
    """ANSI color if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t):    return _color(t, "1")
def _green(t):   return _color(t, "32")
def _yellow(t):  return _color(t, "33")
def _red(t):     return _color(t, "31")
def _cyan(t):    return _color(t, "36")


# ── info command ──────────────────────────────────────────────────────────────

def cmd_info(args):
    """
    Show compatibility and expected compression for a HuggingFace model.
    Requires: pip install transformers
    """
    model_id = args.model

    print(f"\n{_bold('QuantCore')} — Model Info")
    print(f"  Checking: {_cyan(model_id)}\n")

    try:
        from transformers import AutoConfig
    except ImportError:
        print(_red("  ✗ 'transformers' not installed. Run: pip install transformers"))
        sys.exit(1)

    print("  Loading config (no weights downloaded)...", end=" ", flush=True)
    try:
        config = AutoConfig.from_pretrained(model_id)
        print(_green("done"))
    except Exception as e:
        print(_red(f"failed\n  Error: {e}"))
        sys.exit(1)

    _ensure_turboquant()
    from quantcore.compat import extract_model_info, check_compatibility

    compatible, msg = check_compatibility(config)
    status = _green("✓ Compatible") if compatible else _red(f"✗ {msg}")
    print(f"\n  Status : {status}")

    if compatible:
        info = extract_model_info(config)
        print(f"\n  Model details:")
        print(info.summary())

        print(f"\n  {_bold('Quick start:')}")
        print(f"    from transformers import AutoModelForCausalLM")
        print(f"    from quantcore import optimize_model")
        print(f"    model = AutoModelForCausalLM.from_pretrained({model_id!r})")
        print(f"    model = optimize_model(model)  # balanced 3-bit")
        print()


# ── benchmark command ─────────────────────────────────────────────────────────

def cmd_benchmark(args):
    """
    Run a full QuantCore benchmark on synthetic data (no GPU, no model download).
    Accurate quality and compression numbers from the real TurboQuant algorithm.
    """
    _ensure_turboquant()

    from quantcore.sdk import _MODE_BITS
    from quantcore.profiler import benchmark_numpy

    seq_lens = tuple(int(x) for x in args.seq_lens.split(","))
    bits = _MODE_BITS.get(args.mode, args.bits)

    print(f"\n{_bold('QuantCore')} — Benchmark")
    print(f"  dim={args.dim}, mode={args.mode} ({bits}-bit), heads={args.heads}x{args.layers}L")
    print(f"  seq_lens={seq_lens}\n")

    for mode, b in [("fast", 4), ("balanced", 3), ("max_memory_save", 2)]:
        if args.mode != "all" and args.mode != mode:
            continue
        print(f"  Running {_bold(mode)} ({b}-bit)...", end=" ", flush=True)
        result = benchmark_numpy(
            dim=args.dim,
            num_heads=args.heads,
            num_layers=args.layers,
            seq_lens=seq_lens,
            bits=b,
            mode=mode,
        )
        print(_green("done"))
        print(result.summary())

        if args.output:
            out_path = f"{args.output}_{mode}.json"
            result.to_json(out_path)
            print(f"\n  Saved: {out_path}")
        print()


# ── dashboard command ─────────────────────────────────────────────────────────

def cmd_dashboard(args):
    """Start the QuantCore monitoring dashboard."""
    try:
        from flask import Flask
    except ImportError:
        print(_red(
            "\n  ✗ Flask not installed. Run:\n"
            "    pip install quantcore[dashboard]\n"
            "  or:\n"
            "    pip install flask flask-cors\n"
        ))
        sys.exit(1)

    _ensure_turboquant()
    from quantcore.dashboard.server import create_app

    app = create_app()
    url = f"http://{args.host}:{args.port}"
    print(f"\n{_bold('QuantCore Dashboard')}")
    print(f"  Running at: {_cyan(url)}")
    print(f"  Open {url} in your browser.")
    print(f"  Press Ctrl+C to stop.\n")

    app.run(host=args.host, port=args.port, debug=False)


# ── version command ───────────────────────────────────────────────────────────

def cmd_version(args):
    _ensure_turboquant()
    import quantcore
    print(f"quantcore {quantcore.__version__}")
    try:
        import turboquant
        print(f"turboquant {turboquant.__version__} (engine)")
    except Exception:
        pass


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantcore",
        description="QuantCore — AI Memory Optimization Layer for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  quantcore info --model meta-llama/Llama-3.1-8B\n"
            "  quantcore benchmark --dim 128 --heads 8 --layers 32\n"
            "  quantcore benchmark --mode all --output results\n"
            "  quantcore dashboard --port 8080\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # info
    p_info = sub.add_parser("info", help="Check model compatibility and compression estimates")
    p_info.add_argument("--model", "-m", required=True, metavar="HF_MODEL_ID",
                        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run benchmark (no GPU or model download needed)")
    p_bench.add_argument("--dim",     type=int, default=128, help="Head dimension (default: 128)")
    p_bench.add_argument("--heads",   type=int, default=8,   help="Number of KV heads (default: 8)")
    p_bench.add_argument("--layers",  type=int, default=32,  help="Number of layers (default: 32)")
    p_bench.add_argument("--bits",    type=int, default=4,   help="Bits (used if mode is not set)")
    p_bench.add_argument("--mode",    type=str, default="all",
                         choices=["fast", "balanced", "max_memory_save", "all"],
                         help="Mode to benchmark (default: all)")
    p_bench.add_argument("--seq-lens", type=str, default="128,512,1024,2048,4096",
                         help="Comma-separated sequence lengths (default: 128,512,1024,2048,4096)")
    p_bench.add_argument("--output", "-o", type=str, default=None,
                         help="Save results to JSON file (prefix, one per mode)")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Start the monitoring web dashboard")
    p_dash.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    p_dash.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")

    # version
    sub.add_parser("version", help="Show version info")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "info":      cmd_info,
        "benchmark": cmd_benchmark,
        "dashboard": cmd_dashboard,
        "version":   cmd_version,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(_red(f"\n  Error: {e}"))
        raise


if __name__ == "__main__":
    main()
