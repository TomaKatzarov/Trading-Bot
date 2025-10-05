#!/usr/bin/env python3
"""
GPU Check Utility - Verifies GPU availability and compatibility
"""
import torch
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.gpu_utils import setup_gpu

def check_gpu_compatibility():
    """Check GPU compatibility and print detailed information"""
    print("\n===== GPU COMPATIBILITY CHECK =====\n")
    
    # Basic PyTorch CUDA check
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA is not available! Using CPU mode. ⚠️")
        print("To use GPU, make sure CUDA is properly installed and compatible with PyTorch.")
        return False
    
    # Get detailed GPU information
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Compute capability: {props.major}.{props.minor}")
        print(f"  - Total memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  - CUDA Cores: {props.multi_processor_count}")
        
    # Check transformers integration if available
    try:
        from transformers import pipeline
        print("\nTesting transformers pipeline with GPU...")
        classifier = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
        result = classifier("Testing GPU acceleration")
        print(f"Transformers pipeline test result: {result}")
        print("✅ Transformers can use GPU successfully!")
    except ImportError:
        print("\n⚠️ Transformers library not installed. Cannot test pipeline integration.")
    except Exception as e:
        print(f"\n⚠️ Error testing transformers pipeline: {e}")
    
    # Use the GPU utils
    print("\nTesting GPU utilities...")
    gpu_info = setup_gpu()
    print(f"GPU Status: {gpu_info}")
    
    print("\n===== GPU CHECK COMPLETE =====")
    return True

if __name__ == "__main__":
    check_gpu_compatibility()
