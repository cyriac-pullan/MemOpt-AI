"""
QuantCore - Basic Usage Example
-------------------------------
Shows how to apply the QuantCore optimization layer to a HuggingFace model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantcore import optimize_model, profile_memory

model_id = "meta-llama/Llama-3.1-8B"

print(f"Loading {model_id}...")
# Note: In a real environment, you might load in 8-bit or 4-bit weights 
# to save parameter memory, while QuantCore saves KV cache memory.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

# --- 1. Apply QuantCore ---
# Using max_memory=8192 (8GB limit) invokes the Policy Engine
model = optimize_model(model, max_memory=8192)

# Check the stats that were injected into the model
stats = model.quantcore_stats(seq_len=4096)
print(f"Memory Saved at 4k context: {stats['memory_saved_mb']} MB")

# --- 2. Run Inference ---
inputs = tokenizer("QuantCore is a memory optimization layer that", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print("\nGenerated Text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# --- 3. Run Profiler ---
print("\nRunning hardware profiler...")
result = profile_memory(model, inputs["input_ids"], max_new_tokens=64)
print(result.summary())
