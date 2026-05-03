"""
QuantCore Policy Engine
=======================
Automatically selects the optimal compression mode based on available hardware.
"""

from .exceptions import QuantCoreCompatError

def detect_gpu_memory_mb() -> float:
    """Returns total GPU memory in MB. Returns 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory of device 0
            return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    except ImportError:
        pass
    return 0.0

def auto_select_bits(max_memory_mb: float = None) -> int:
    """
    Selects the best bit-depth (mode) based on maximum available memory.
    If max_memory_mb is not provided, it detects GPU memory automatically.
    """
    if max_memory_mb is None:
        max_memory_mb = detect_gpu_memory_mb()
    
    # If we still don't know the memory (e.g. running on CPU without limit), default to 3-bit
    if max_memory_mb == 0:
        return 3

    # Logic: 
    # Under 8GB (e.g., consumer GPU) -> Max memory save (2-bit)
    # Under 16GB (e.g., T4, standard server) -> Balanced (3-bit)
    # 16GB+ (e.g., A10G, A100) -> Fast (4-bit)
    if max_memory_mb < 8192:
        return 2
    elif max_memory_mb < 16384:
        return 3
    else:
        return 4

def auto_select_mode(max_memory_mb: float = None) -> str:
    """Returns the mode string ('fast', 'balanced', 'max_memory_save') based on memory."""
    bits = auto_select_bits(max_memory_mb)
    if bits == 2:
        return "max_memory_save"
    elif bits == 3:
        return "balanced"
    else:
        return "fast"
