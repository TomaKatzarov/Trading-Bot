import logging
import os
import torch
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def setup_gpu(memory_fraction: float = 0.95, 
              enable_optimization: bool = True) -> Dict[str, Any]:
    """
    Setup and optimize GPU for transformer models.
    
    Args:
        memory_fraction: Fraction of GPU memory to use (0.0 to 1.0)
        enable_optimization: Enable additional performance optimizations
        
    Returns:
        Dict with GPU status information
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Running on CPU.")
        return {"available": False, "device": "cpu", "name": "CPU", "memory": None}
    
    # Get GPU information
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    logger.info(f"Found {device_count} GPU(s). Using: {device_name}")
    
    # Set environment variables for transformers to use GPU
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # Set optimal memory allocation for CUDA 12.x (use new API name)
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Enable optimization flags
    if enable_optimization:
        # Enable TF32 for NVIDIA Ampere+ GPUs (A100, A6000, RTX 30xx, RTX 40xx, etc.)
        capability = torch.cuda.get_device_capability(current_device)
        major_version = capability[0]
        
        # Support for newer architectures (Ampere, Ada Lovelace, Blackwell)
        # RTX 5070 Ti has reported capability 12.0
        if major_version >= 8:
            logger.info(f"Enabling TF32 precision for GPU with compute capability {major_version}.{capability[1]}")
            # Use new PyTorch 2.9+ API to avoid deprecation warnings
            try:
                if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
                    torch.backends.cuda.matmul.fp32_precision = "tf32"
                else:
                    torch.backends.cuda.matmul.allow_tf32 = True
            except (AttributeError, RuntimeError):
                pass
            try:
                if hasattr(torch.backends.cudnn.conv, "fp32_precision"):
                    torch.backends.cudnn.conv.fp32_precision = "tf32"
                else:
                    torch.backends.cudnn.allow_tf32 = True
            except (AttributeError, RuntimeError):
                pass
        
        # Enable CUDA graph capture for repeated operations
        if hasattr(torch, '__version__'):
            logger.info(f"Using PyTorch {torch.__version__} with CUDA optimization")
            
        # Use cudnn benchmark for optimized kernels when input sizes don't change
        torch.backends.cudnn.benchmark = True
        
        # Enable flash attention if available on Ada Lovelace/Blackwell (RTX 40xx/50xx)
        if major_version >= 9:
            try:
                # Only attempt to enable if transformers is installed
                import transformers
                os.environ["TRANSFORMERS_ENABLE_FLASH_ATTN"] = "1"
                logger.info("Flash Attention 2.0 enabled for Ada/Blackwell architecture")
            except ImportError:
                pass
    
    # Return device information with additional RTX details
    return {
        "available": True,
        "device": f"cuda:{current_device}",
        "name": device_name,
        "memory": {
            "total": torch.cuda.get_device_properties(current_device).total_memory,
            "fraction": memory_fraction
        },
        "compute_capability": f"{capability[0]}.{capability[1]}"
    }

def get_optimal_batch_size(model_name: str, gpu_info: Dict[str, Any]) -> int:
    """
    Determine optimal batch size based on model and GPU.
    
    Args:
        model_name: Name of the model
        gpu_info: GPU information from setup_gpu()
        
    Returns:
        Recommended batch size
    """
    # Default conservative batch sizes
    if not gpu_info["available"]:
        return 2  # CPU default
    
    # Extract GPU memory in GB (conservative estimate)
    try:
        gpu_memory_gb = gpu_info["memory"]["total"] / (1024 ** 3)
    except (KeyError, TypeError):
        gpu_memory_gb = 4  # Conservative default
    
    # RTX 5070 Ti specific optimizations (16GB VRAM)
    if "RTX 5070" in gpu_info.get("name", ""):
        if "bert" in model_name.lower() or "finbert" in model_name.lower():
            return 32  # Optimized for RTX 5070 Ti with FinBERT
        else:
            return 24  # Other models on RTX 5070 Ti
    
    # Adjust based on available memory and model size
    if "bert" in model_name.lower() or "finbert" in model_name.lower():
        if gpu_memory_gb >= 24:  # A100, A6000, etc.
            return 64
        elif gpu_memory_gb >= 16:  # RTX 3090, 4090, etc.
            return 32
        elif gpu_memory_gb >= 10:  # RTX 3080, 2080 Ti, etc.
            return 24
        elif gpu_memory_gb >= 8:   # RTX 3070, 2080, etc.
            return 16
        else:                      # Smaller GPUs
            return 8
    else:  # Default for other models
        if gpu_memory_gb >= 16:
            return 24
        elif gpu_memory_gb >= 8:
            return 16
        else:
            return 8

def enable_mixed_precision() -> bool:
    """
    Enable mixed precision training if supported by the GPU.
    This can significantly speed up transformer models.
    
    Returns:
        bool: True if mixed precision was enabled, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    # Check if GPU supports mixed precision
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    
    # Volta+ GPUs (V100, RTX 20XX+) support mixed precision
    if capability[0] >= 7:
        try:
            # Try to import and use torch.amp
            from torch.cuda.amp import autocast, GradScaler
            logger.info(f"Mixed precision training enabled for compute capability {capability[0]}.{capability[1]}")
            return True
        except ImportError:
            logger.warning("torch.cuda.amp not available. Mixed precision disabled.")
            return False
    else:
        logger.info(f"GPU compute capability {capability[0]}.{capability[1]} doesn't support mixed precision.")
        return False

def check_xformers_compatibility() -> Dict[str, Any]:
    """
    Check if xFormers is properly installed and compatible with the current environment.
    
    Returns:
        Dict with status information about xFormers
    """
    status = {
        "installed": False,
        "compatible": False,
        "version": None,
        "cuda_version": None,
        "pytorch_version": torch.__version__,
        "action_required": None
    }
    
    try:
        import warnings
        # Suppress warnings during xformers import check
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import xformers
            
        status["installed"] = True
        status["version"] = getattr(xformers, "__version__", "unknown")
        
        # Check if xFormers CUDA extensions are working
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from xformers.ops import memory_efficient_attention
                dummy_input = torch.rand(1, 8, 16, 16).to("cuda")
                memory_efficient_attention(dummy_input, dummy_input, dummy_input)
                
            status["compatible"] = True
            logger.info(f"✅ xFormers {status['version']} is working correctly with PyTorch {status['pytorch_version']}")
        except (ImportError, RuntimeError, AttributeError) as e:
            status["compatible"] = False
            status["error"] = str(e)
            logger.warning(f"xFormers is installed but CUDA extensions aren't working: {e}")
            
            # Extract CUDA version for which xFormers was built
            if "built for" in str(e) and "CUDA" in str(e):
                import re
                cuda_match = re.search(r'CUDA (\d+)', str(e))
                if cuda_match:
                    status["cuda_version"] = cuda_match.group(1)
            
            # Provide specific advice based on environment
            if torch.__version__.startswith("2.8"):
                status["action_required"] = "install_nightly"
            else:
                status["action_required"] = "reinstall"
                
    except ImportError:
        logger.info("xFormers is not installed.")
        status["action_required"] = "install"
        
    return status

def install_xformers_instructions(status: Dict[str, Any]) -> str:
    """
    Generate installation instructions for xFormers based on environment.
    
    Args:
        status: Dict from check_xformers_compatibility()
    
    Returns:
        String with installation instructions
    """
    pytorch_version = status["pytorch_version"]
    action = status["action_required"]
    
    if action == "install_nightly":
        return """
To install xFormers for your development version of PyTorch 2.8.0 with CUDA 12.8:

1. Install from GitHub:
   pip install --no-deps git+https://github.com/facebookresearch/xformers.git@main

2. Or build from source:
   git clone https://github.com/facebookresearch/xformers.git
   cd xformers
   pip install -e .
"""
    elif action == "reinstall":
        return f"""
To reinstall xFormers for PyTorch {pytorch_version}:

1. Uninstall current version:
   pip uninstall -y xformers
   
2. Install via pip:
   pip install xformers
"""
    else:
        return f"""
To install xFormers for PyTorch {pytorch_version}:

1. Install via pip:
   pip install xformers
"""

def setup_efficient_attention() -> bool:
    """
    Configure efficient attention mechanisms for transformers models.
    Falls back gracefully if efficient attention options are not available.
    
    Returns:
        bool: True if any efficient attention method was successfully enabled
    """
    # Check xFormers compatibility
    xformers_status = check_xformers_compatibility()
    
    # If xFormers is compatible, enable memory efficient attention
    if xformers_status["compatible"]:
        # Enable xFormers memory efficient attention in transformers
        os.environ["TRANSFORMERS_USE_XFORMERS"] = "1"
        logger.info("Enabled xFormers memory-efficient attention for transformers")
        return True
    
    # If xFormers is not available/compatible, check for other options
    
    # Check for CUDA 11.6+ for new efficient attention mechanism
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        
        # For RTX 30xx/40xx/50xx, enable native attention optimization in transformers
        if capability[0] >= 8:
            try:
                import transformers
                os.environ["TRANSFORMERS_USE_NATIVE_ATTN"] = "1"
                logger.info("Using transformers native efficient attention (xFormers not available)")
                
                # For non-compatible xFormers, log installation instructions once
                if xformers_status["installed"] and not xformers_status["compatible"]:
                    instructions = install_xformers_instructions(xformers_status)
                    logger.info(f"To fix xFormers compatibility issues:\n{instructions}")
                    
                return True
            except ImportError:
                logger.warning("Transformers not available for native attention optimization")
                
    # No efficient attention methods available
    logger.warning("No efficient attention mechanisms available")
    return False
