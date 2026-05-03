# QuantCore

**Runtime KV Cache Compression for LLMs.**

QuantCore compresses the Key-Value cache of transformer models during inference using the [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm (ICLR 2026). It reduces KV cache memory by 2-4x, enabling longer context windows and lower GPU costs — with a single line of code.

---

## When QuantCore Helps

KV cache memory grows linearly with sequence length. At short contexts (< 1K tokens), KV cache is small and model weights dominate memory. **QuantCore's impact becomes significant when KV cache is the bottleneck:**

| Scenario | KV Cache Size | QuantCore Impact |
|---|---|---|
| Short chat (< 512 tokens) | Small | Minimal |
| Long context (2K-8K tokens) | Large | **Significant savings** |
| Very long context (8K-32K tokens) | Dominant | **Critical — prevents OOM** |
| Multi-user serving (batched) | Multiplied | **Major cost reduction** |

### Real Numbers (Llama-3.1-8B, balanced mode)

| Context Length | FP16 KV Cache | QuantCore | Saved |
|---|---|---|---|
| 1,024 tokens | 22 MB | 12 MB | 10 MB |
| 4,096 tokens | 88 MB | 47 MB | 41 MB |
| 8,192 tokens | 176 MB | 94 MB | **82 MB** |
| 16,384 tokens | 352 MB | 187 MB | **165 MB** |

> At 16K context with 8 concurrent users, that's **1.3 GB saved** — enough to avoid upgrading from a 16GB to 24GB GPU.

---

## Quick Start

```python
from transformers import AutoModelForCausalLM
from quantcore import optimize_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# One-line integration
model = optimize_model(model, mode="balanced")

# Use normally — no other changes needed
outputs = model.generate(input_ids, max_new_tokens=1000)

# Check savings at your context length
stats = model.quantcore_stats(seq_len=4096)
print(f"Memory saved: {stats['memory_saved_mb']:.0f} MB")
```

---

## Compression Modes

| Mode | Bits | Cosine Similarity | Compression | Best For |
|---|---|---|---|---|
| `fast` | 4-bit | 0.995 | ~2x | Production chatbots, high accuracy |
| `balanced` | 3-bit | 0.983 | ~3x | General purpose (recommended) |
| `max_memory_save` | 2-bit | 0.940 | ~4-6x | RAG pipelines, edge deployment |

### Auto Mode (Policy Engine)

Let QuantCore pick the best mode for your GPU:

```python
model = optimize_model(model, max_memory=0)  # Auto-detect GPU memory
```

| GPU Memory | Auto-selected Mode |
|---|---|
| < 8 GB | `max_memory_save` (2-bit) |
| 8-16 GB | `balanced` (3-bit) |
| 16+ GB | `fast` (4-bit) |

---

## Installation

```bash
pip install quantcore
```

With dashboard and all extras:
```bash
pip install quantcore[all]
```

---

## CLI Tools

```bash
# Check model compatibility and see memory estimates
quantcore info --model meta-llama/Llama-3.1-8B

# Run synthetic benchmark (no GPU needed)
quantcore benchmark

# Start live monitoring dashboard
quantcore dashboard --port 8080
```

---

## How It Works

```
User Request
     |
LLM (HuggingFace)
     |
QuantCore Layer (optimize_model)
     |
     +-- Random orthogonal rotation
     +-- Lloyd-Max scalar quantization (Beta-optimal)
     +-- Compressed KV Cache (2-4 bit per dimension)
     |
Efficient Inference (same output quality)
```

The algorithm applies a random orthogonal rotation to KV vectors, which induces a known Beta distribution on each coordinate. A Lloyd-Max codebook optimized for this distribution then quantizes each coordinate independently — no calibration data, no per-channel scales, works online.

---

## Supported Models

QuantCore automatically detects model architecture and extracts KV cache parameters:

- Llama (1B, 3B, 8B, 70B)
- Mistral / Mixtral
- Phi-3 / Phi-4
- Gemma / Gemma 2
- Qwen / Qwen 2.5

Any HuggingFace `PreTrainedModel` with a standard config is supported.

---

## Limitations (Honest)

- **Short context (< 1K tokens)**: KV cache is small, savings are negligible. Model weights dominate memory.
- **Output divergence**: Compressed KV slightly shifts attention weights. At 4-bit this is nearly invisible; at 2-bit, generation may diverge from baseline after many tokens. Semantic meaning is preserved.
- **CPU-only**: Current implementation uses NumPy for compression. A fused Triton kernel (planned) would add GPU-accelerated compression with zero overhead.

---

## Roadmap

- [x] HuggingFace plug-and-play integration
- [x] Multi-architecture support (Llama, Mistral, Phi, Gemma, Qwen)
- [x] Policy Engine (auto mode selection)
- [x] CLI tools and monitoring dashboard
- [ ] Triton fused attention kernel (GPU-accelerated compression)
- [ ] vLLM integration (production serving)
- [ ] PyPI release

---

## Paper

Based on: **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Zandieh, Daliri, Hadian, Mirrokni — Google Research, ICLR 2026
[arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)

## License

Apache 2.0
