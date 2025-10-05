"""
Model Performance Testing Script for Large Language Models
Specifically for testing Qwen 2.5 14B on RTX 5070TI (16GB VRAM)
"""

import os
import sys
import time
import logging
import torch
import numpy as np
import psutil
import gc
from pathlib import Path
from contextlib import contextmanager
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

# Add project root to Python path if necessary
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "tests", "model_performance_log.txt"))
    ]
)
logger = logging.getLogger("model_performance_test")

# Import project-specific utilities if available
try:
    from utils.gpu_utils import setup_gpu, setup_efficient_attention
except ImportError as e:
    logger.warning(f"Could not import GPU utilities: {e}. Using default settings.")
    def setup_gpu(**kwargs): return {"available": False, "name": "Unknown", "memory": {"total": 0, "free": 0}}
    def setup_efficient_attention(): return False

try:
    from models.llm_handler import LLMHandler
except ImportError:
    logger.warning("Could not import LLMHandler. Will use transformers directly.")
    LLMHandler = None


# --- Utility Functions ---

@contextmanager
def track_gpu_memory():
    """Context manager to track GPU memory usage before and after an operation."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Record starting memory
    if torch.cuda.is_available():
        start_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        start_reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
    else:
        start_allocated = 0
        start_reserved = 0
    
    yield  # Execute the code block
    
    # Record ending memory
    if torch.cuda.is_available():
        end_allocated = torch.cuda.memory_allocated() / (1024 ** 2)    # MB
        end_reserved = torch.cuda.memory_reserved() / (1024 ** 2)      # MB
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        logger.info(f"GPU Memory Usage:")
        logger.info(f"  Allocated: {start_allocated:.2f} MB → {end_allocated:.2f} MB (Delta: {end_allocated - start_allocated:.2f} MB)")
        logger.info(f"  Reserved:  {start_reserved:.2f} MB → {end_reserved:.2f} MB (Delta: {end_reserved - start_reserved:.2f} MB)")
        logger.info(f"  Peak:      {peak_allocated:.2f} MB")


def get_system_info():
    """Get detailed system information including GPU and RAM."""
    system_info = {}
    
    # RAM information
    vm = psutil.virtual_memory()
    system_info["ram"] = {
        "total_gb": vm.total / (1024**3),
        "available_gb": vm.available / (1024**3),
        "used_percent": vm.percent
    }
    
    # CUDA/GPU information
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        system_info["cuda"] = {
            "available": True,
            "device_count": device_count,
            "current_device": torch.cuda.current_device(),
            "devices": []
        }
        
        # Get information for each GPU
        for i in range(device_count):
            device_info = {
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
            }
            
            # Try to get memory info
            try:
                mem_info = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
                free_mem = torch.cuda.memory_reserved(i) / (1024**3)
                device_info["memory_gb"] = mem_info
                device_info["free_memory_gb"] = free_mem
            except:
                device_info["memory_gb"] = "Unknown"
            
            system_info["cuda"]["devices"].append(device_info)
    else:
        system_info["cuda"] = {"available": False}
    
    return system_info


def print_system_info(info):
    """Pretty print system information."""
    logger.info("=== SYSTEM INFORMATION ===")
    
    # RAM info
    logger.info(f"RAM: {info['ram']['total_gb']:.2f} GB total, {info['ram']['available_gb']:.2f} GB available ({info['ram']['used_percent']}% used)")
    
    # CUDA/GPU info
    if info["cuda"]["available"]:
        logger.info(f"CUDA: Available ({info['cuda']['device_count']} device(s))")
        for i, device in enumerate(info["cuda"]["devices"]):
            logger.info(f"  GPU {i}: {device['name']} - {device['memory_gb']:.2f} GB")
            if "free_memory_gb" in device:
                logger.info(f"       Available: {device['free_memory_gb']:.2f} GB")
            logger.info(f"       Capability: {device['capability']}")
    else:
        logger.info("CUDA: Not available")
    
    logger.info("=" * 25)


def measure_inference_time(model, tokenizer, input_text, num_runs=5, warm_up=2, max_new_tokens=20):
    """Measure inference time for a model."""
    times = []
    
    # First do warm-up runs
    for _ in range(warm_up):
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Then measure actual runs
    for i in range(num_runs):
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Log the result for this run
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Run {i+1}/{num_runs}: {elapsed:.4f}s - Output: {generated_text[:50]}...")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "times": times
    }


# --- Test Functions ---

def test_model_loading():
    """Test if the model loads correctly and fits in GPU memory."""
    logger.info("\n=== TESTING MODEL LOADING ===")
    
    model_path = os.path.join(project_root, "models1")
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Try to load with increasing quantization levels only if needed
        loading_attempts = [
            {"name": "BF16", "kwargs": {"torch_dtype": torch.bfloat16}},
            {"name": "4-bit Quantization", "kwargs": {"load_in_4bit": True}},
            {"name": "8-bit Quantization", "kwargs": {"load_in_8bit": True}}
        ]
        
        model = None
        tokenizer = None
        success = False
        
        for attempt in loading_attempts:
            logger.info(f"Attempting to load with {attempt['name']}...")
            
            try:
                with track_gpu_memory():
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, 
                        device_map="auto",
                        **attempt["kwargs"]
                    )
                
                # If we got here, loading worked
                logger.info(f"Successfully loaded model with {attempt['name']}")
                success = True
                
                # Check what device(s) the model is on
                if hasattr(model, "hf_device_map"):
                    logger.info("Model device map:")
                    for layer, device in model.hf_device_map.items():
                        logger.info(f"  {layer}: {device}")
                else:
                    logger.info(f"Model device: {next(model.parameters()).device}")
                
                break  # Exit the loop if successful
            
            except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
                logger.error(f"Failed to load with {attempt['name']}: {str(e)}")
                # Clean up to try next approach
                if model:
                    del model
                torch.cuda.empty_cache()
                gc.collect()
        
        if not success:
            logger.error("Could not load model with any tested configuration")
            return False, None, None
        
        # Test if model can generate text
        logger.info("Testing if model can generate text...")
        prompt = "Testing the Qwen 2.5 model. What is the capital of France?"
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=20)
            
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Model output: {output_text}")
        
        return True, model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None, None


def test_memory_profile(model, tokenizer):
    """Test memory usage during forward and generate passes."""
    logger.info("\n=== MEMORY PROFILING ===")
    
    if not model or not tokenizer:
        logger.error("Model or tokenizer not provided. Skipping memory profiling.")
        return False
    
    prompt = "The capital of France is"
    
    # Memory for tokenization
    logger.info("Memory for tokenization:")
    with track_gpu_memory():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Memory for forward pass
    logger.info("Memory for forward pass:")
    with track_gpu_memory():
        with torch.no_grad():
            outputs = model(**inputs)
    
    # Memory for generation with short output
    logger.info("Memory for generation (short):")
    with track_gpu_memory():
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
    
    # Memory for generation with longer output
    logger.info("Memory for generation (longer):")
    with track_gpu_memory():
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
    
    return True


def test_latency(model, tokenizer):
    """Test inference latency under various conditions."""
    logger.info("\n=== LATENCY TESTING ===")
    
    if not model or not tokenizer:
        logger.error("Model or tokenizer not provided. Skipping latency testing.")
        return False
    
    # Test single token generation
    prompt = "The capital of France is"
    logger.info("Testing single token generation latency:")
    single_results = measure_inference_time(model, tokenizer, prompt, num_runs=5, max_new_tokens=1)
    
    logger.info(f"Average time: {single_results['avg_time']:.4f}s per token")
    logger.info(f"Min time: {single_results['min_time']:.4f}s, Max time: {single_results['max_time']:.4f}s")
    
    # Test short paragraph generation
    logger.info("Testing short paragraph generation latency:")
    para_results = measure_inference_time(model, tokenizer, prompt, num_runs=3, max_new_tokens=50)
    
    avg_per_token = para_results['avg_time'] / 50
    logger.info(f"Average time: {para_results['avg_time']:.4f}s total, {avg_per_token:.4f}s per token")
    
    return True


def test_model_functionality(model, tokenizer):
    """Test basic model functionality and quality."""
    logger.info("\n=== FUNCTIONALITY TESTING ===")
    
    if not model or not tokenizer:
        logger.error("Model or tokenizer not provided. Skipping functionality testing.")
        return False
    
    test_cases = [
        {
            "name": "Basic completion",
            "prompt": "The capital of France is",
            "expected_contains": ["Paris"]
        },
        {
            "name": "Math test",
            "prompt": "What is 2+2?",
            "expected_contains": ["4", "four"]
        },
        {
            "name": "Context understanding",
            "prompt": "I have 5 apples. I eat 2 apples and give 1 to my friend. How many apples do I have left?",
            "expected_contains": ["2", "two"]
        },
        {
            "name": "Trading-specific prompt",
            "prompt": "Input: Context=[0.215, 0.512, 0.823, -0.431, 0.122]... -> Output: Signal=",
            "expected_contains": ["0", "1", "2", "3", "4"] # One of the signal values should be present
        }
    ]
    
    results = []
    
    for test in test_cases:
        logger.info(f"Running test: {test['name']}")
        inputs = tokenizer(test["prompt"], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if output contains any of the expected substrings
        contains_expected = any(exp in output_text for exp in test["expected_contains"])
        
        logger.info(f"Prompt: {test['prompt']}")
        logger.info(f"Output: {output_text}")
        # Use ASCII alternatives instead of Unicode characters
        logger.info(f"Contains expected text: {'[PASS]' if contains_expected else '[FAIL]'}")
        logger.info("---")
        
        results.append({
            "name": test["name"],
            "prompt": test["prompt"],
            "output": output_text,
            "passed": contains_expected
        })
    
    # Calculate pass rate
    passed = sum(1 for r in results if r["passed"])
    logger.info(f"Tests passed: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    return results


def test_lora_compatibility():
    """Test if the model is compatible with LoRA adapters."""
    logger.info("\n=== LORA COMPATIBILITY TESTING ===")
    
    if not LLMHandler:
        logger.warning("LLMHandler not available. Skipping LoRA compatibility test.")
        return False
    
    try:
        # First try loading base model
        handler_base = LLMHandler(use_lora=False)
        
        if not handler_base.base_model_loaded:
            logger.error("Failed to load base model in LLMHandler")
            return False
            
        logger.info("Successfully loaded base model in LLMHandler")
        
        # Try to load with LoRA
        handler_lora = LLMHandler(use_lora=True)
        
        if handler_lora.adapter_loaded:
            logger.info(f"Successfully loaded LoRA adapter: {handler_lora.adapter_path_used}")
        else:
            logger.warning("LoRA adapter not loaded, but base model loaded successfully")
        
        # Test a simple prompt to verify functionality
        test_prompt = {"context_vector": list(np.random.rand(64))}
        result = handler_lora.analyze_market(test_prompt)
        
        logger.info(f"LLMHandler analysis result: {result}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing LoRA compatibility: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_all_tests():
    """Run all performance tests on the model."""
    logger.info("=" * 50)
    logger.info("STARTING QWEN 2.5 14B MODEL PERFORMANCE TESTS")
    logger.info("=" * 50)
    
    # Get system info
    system_info = get_system_info()
    print_system_info(system_info)
    
    # Run tests
    success, model, tokenizer = test_model_loading()
    
    if not success:
        logger.error("Model loading failed. Cannot continue with tests.")
        return
    
    test_memory_profile(model, tokenizer)
    test_latency(model, tokenizer)
    test_model_functionality(model, tokenizer)
    test_lora_compatibility()
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("=" * 50)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_all_tests()
