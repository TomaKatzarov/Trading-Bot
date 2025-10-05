"""
Model Index Generator

This script analyzes model weights files in a directory and generates the appropriate
index files (model.safetensors.index.json or pytorch_model.bin.index.json) needed for
loading sharded models.

Usage:
  python tools/generate_model_index.py [--model_dir=PATH]
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
import re
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate model index file for sharded models")
    parser.add_argument("--model_dir", type=str, default=str(project_root / "models1"),
                        help="Path to the model directory")
    return parser.parse_args()

def get_file_size(file_path):
    """Get file size in bytes."""
    try:
        return Path(file_path).stat().st_size
    except Exception as e:
        print(f"Error getting size of {file_path}: {e}")
        return 0

def analyze_model_config(model_dir):
    """Analyze the model's config.json to determine model type and architecture."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        print(f"Warning: No config.json found in {model_dir}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract key model architecture info
        model_info = {
            "model_type": config.get("model_type", "unknown"),
            "architectures": config.get("architectures", ["unknown"]),
            "hidden_size": config.get("hidden_size", 0),
            "num_hidden_layers": config.get("num_hidden_layers", 0),
            "num_attention_heads": config.get("num_attention_heads", 0),
            "vocab_size": config.get("vocab_size", 0),
        }
        
        return model_info
    except Exception as e:
        print(f"Error analyzing config.json: {e}")
        return {}

def find_weight_files(model_dir):
    """Find all model weight files in directory."""
    model_dir = Path(model_dir)
    
    # Look for various weight file formats
    safetensors_files = list(model_dir.glob("*.safetensors"))
    pytorch_files = list(model_dir.glob("*.bin"))
    shard_files = list(model_dir.glob("*-*.safetensors")) + list(model_dir.glob("*-*.bin"))
    
    # Determine file type priority
    if safetensors_files:
        file_type = "safetensors"
        weight_files = sorted(safetensors_files)
    elif pytorch_files:
        file_type = "bin"
        weight_files = sorted(pytorch_files)
    else:
        file_type = None
        weight_files = []
    
    # Check if there are shard numbers in filenames (model.xx.safetensors, etc)
    shard_pattern = re.compile(r".*?[.-](\d+)(?:of|-)(\d+)\.(safetensors|bin|pt)$")
    
    # Extract shard info if available
    shards_info = []
    for file_path in weight_files:
        match = shard_pattern.match(file_path.name)
        if match:
            shard_num, total_shards, _ = match.groups()
            shards_info.append((int(shard_num), int(total_shards), file_path))
    
    # Check if we found organized shards
    if shards_info and len(set(info[1] for info in shards_info)) == 1:  # All have same total_shards
        print(f"Found {shards_info[0][1]} sharded weight files")
        # Sort by shard number
        weight_files = [info[2] for info in sorted(shards_info, key=lambda x: x[0])]
    
    return weight_files, file_type

def generate_weight_map(model_info, weight_files):
    """Generate a weight map dictionary mapping model parameters to files."""
    # For a generic approach, we'll map by balanced distribution or by prefix pattern
    weight_map = {}
    
    # Get list of parameter names based on typical transformer architecture
    param_prefixes = []
    
    # Base model components
    param_prefixes.extend([
        "model.embed_tokens", 
        "model.norm",
        "lm_head",
    ])
    
    # Attention layers
    num_layers = model_info.get("num_hidden_layers", 0) or 32  # Default if unknown
    for i in range(num_layers):
        param_prefixes.append(f"model.layers.{i}")
    
    # Now map parameters to files evenly
    if len(weight_files) == 1:
        # Single file case - map all params to this file
        for prefix in param_prefixes:
            weight_map[prefix] = weight_files[0].name
    else:
        # Multiple files - distribute parameters across files
        files_per_category = max(1, len(weight_files) // len(param_prefixes))
        
        for i, prefix in enumerate(param_prefixes):
            file_idx = min(i // files_per_category, len(weight_files) - 1)
            weight_map[prefix] = weight_files[file_idx].name
    
    return weight_map

def create_index_file(model_dir, file_type="safetensors"):
    """Create index file for sharded model."""
    model_dir = Path(model_dir)
    model_info = analyze_model_config(model_dir)
    weight_files, detected_file_type = find_weight_files(model_dir)
    
    if not weight_files:
        print(f"Error: No weight files found in {model_dir}")
        return False
    
    # Use detected file type if none specified
    if not file_type:
        file_type = detected_file_type
    
    # Use appropriate index filename
    if file_type == "safetensors":
        index_filename = "model.safetensors.index.json"
    else:
        index_filename = "pytorch_model.bin.index.json"
    
    # Calculate total size
    total_size = sum(get_file_size(f) for f in weight_files)
    
    # Generate weight map
    weight_map = generate_weight_map(model_info, weight_files)
    
    # Create index content
    index_data = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }
    
    # Write index file
    index_path = model_dir / index_filename
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        print(f"Successfully created {index_path}")
        return True
    except Exception as e:
        print(f"Error creating index file: {e}")
        return False

def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists() or not model_dir.is_dir():
        print(f"Error: Model directory {model_dir} not found or is not a directory")
        return 1
    
    print(f"Analyzing model files in {model_dir}...")
    
    # Check if index files already exist
    safetensors_index = model_dir / "model.safetensors.index.json"
    pytorch_index = model_dir / "pytorch_model.bin.index.json"
    
    if safetensors_index.exists():
        print(f"Warning: {safetensors_index} already exists. Will back it up.")
        backup_path = safetensors_index.with_suffix(".json.bak")
        safetensors_index.rename(backup_path)
    
    if pytorch_index.exists():
        print(f"Warning: {pytorch_index} already exists. Will back it up.")
        backup_path = pytorch_index.with_suffix(".json.bak")
        pytorch_index.rename(backup_path)
    
    # Try to create safetensors index first, then pytorch index if needed
    if not create_index_file(model_dir, "safetensors"):
        if not create_index_file(model_dir, "bin"):
            print("Failed to create any index files. Check if model files exist.")
            return 1
    
    print("Index file generation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
