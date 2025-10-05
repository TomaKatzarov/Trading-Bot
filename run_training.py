# ==============================================================================
# LoRA Training Script using Unsloth
# ==============================================================================

# --- Environment Setup ---
# Disable PyTorch compile features early to prevent conflicts
import sys
import os
from pathlib import Path  # ensure Path is defined before using it
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["PYTORCH_DISABLE_TORCH_COMPILE"] = "1"
os.environ["UNSLOTH_OFFLOAD"] = "0"  # Disable offloading to CPU
# After setting TORCH environment variables, ensure Triton caches are persisted
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).parent / "unsloth_compiled_cache"))

# --- Unsloth Import (MUST be before other transformers/peft imports if patching) ---
try:
    from unsloth import FastLanguageModel
    
    # Monkey-patch tokenizer base to stub Unsloth’s push-to-hub method in worker processes
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    def _stub_unsloth_push_to_hub(self, *args, **kwargs):
        pass
    setattr(PreTrainedTokenizerBase, 'unsloth_push_to_hub', _stub_unsloth_push_to_hub)
except ImportError as e:
    print("="*80)
    print("ERROR: Unsloth import failed!")
    print("Please ensure Unsloth is installed correctly for your environment.")
    print("Command: pip install \"unsloth[cu128] @ git+https://github.com/unslothai/unsloth.git\"")
    print(f"Original Error: {e}")
    print("="*80)
    sys.exit(1)
# --- Project-Specific Imports ---
# Add project root to Python path if necessary
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Standard Library Imports ---
import json
import argparse
import logging
from datetime import datetime

# --- Third-Party Imports ---
import numpy as np
import torch
import evaluate
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig


try:
    from utils.gpu_utils import setup_gpu, setup_efficient_attention, get_optimal_batch_size, enable_mixed_precision
except ImportError as e:
    # Provide fallback functions if gpu_utils is missing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(f"Could not import GPU utilities: {e}. Using default settings.")
    def setup_gpu(**kwargs): return {"available": False, "name": "CPU", "memory": {"total": 0}}
    def setup_efficient_attention(): return False
    def get_optimal_batch_size(model_name, gpu_info): return 4
    def enable_mixed_precision(): return False

# --- Setup Logger ---
# Ensure logger is configured even if gpu_utils fails
if 'logger' not in locals():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# ==============================================================================
# Argument Parsing
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA using Unsloth.")
    parser.add_argument("--dataset_fraction", type=float, default=1, help="Fraction of the dataset to use for training/validation (default: 0.3)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=16, help="Override automatic batch size detection (default: None)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank (r) (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (default: 16)")
    parser.add_argument("--output_suffix", type=str, default=None, help="Optional suffix for the output directory name (default: None)")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization (default: 512)")
    parser.add_argument("--data_file", type=str, default="training_data_pnl_v1.jsonl", help="Name of the training data file within data/training data/ (default: training_data_pnl_v1.jsonl)")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup ratio for learning rate scheduler (default: 0.03)")
    parser.add_argument("--gradient_accumulation", type=int, default=None, help="Override gradient accumulation steps (default: auto based on batch size)")

    args = parser.parse_args()
    logger.info(f"Parsed Arguments: {args}")
    return args

# ==============================================================================
# Helper Functions
# ==============================================================================
def prepare_datasets(dataset_file: Path, dataset_fraction: float, validation_split_fraction: float = 0.10):
    """Loads, subsets, and splits the dataset."""
    logger.info(f"--- Loading and Preparing Dataset ---")
    logger.info(f"Loading dataset from: {dataset_file}")
    if not dataset_file.exists():
        logger.error(f"Training data file not found at {dataset_file}. Run data_preparation.py first.")
        sys.exit(1)

    dataset = load_dataset("json", data_files=str(dataset_file), split='train')
    logger.info(f"Original dataset size: {len(dataset)}")
    if len(dataset) == 0:
        logger.error("Dataset is empty. Cannot proceed.")
        sys.exit(1)

    # --- Subsetting ---
    if dataset_fraction < 1.0:
        subset_size = max(1, int(len(dataset) * dataset_fraction))
        if subset_size < len(dataset): # Only subset if fraction is actually smaller
            try:
                # Use train_test_split to get a reproducible subset
                dataset_subset = dataset.train_test_split(train_size=subset_size, shuffle=True, seed=42)['train']
                logger.info(f"Using subset for training/validation: {len(dataset_subset)} samples ({dataset_fraction*100:.1f}% of original).")
            except ValueError as e:
                logger.warning(f"Could not create subset with fraction {dataset_fraction}: {e}. Using full dataset.")
                dataset_subset = dataset
        else:
            dataset_subset = dataset # Fraction is 1.0 or dataset too small
            logger.info(f"Using full dataset for training/validation: {len(dataset_subset)} samples.")
    else:
        dataset_subset = dataset
        logger.info(f"Using full dataset for training/validation: {len(dataset_subset)} samples.")

    # --- Splitting ---
    if len(dataset_subset) < 2:
        logger.warning("Subset too small to create validation split. Using all for training.")
        train_dataset = dataset_subset
        eval_dataset = None
    else:
        try:
            train_val_split = dataset_subset.train_test_split(test_size=validation_split_fraction, shuffle=True, seed=42)
            train_dataset = train_val_split['train']
            eval_dataset = train_val_split['test']
            if len(train_dataset) == 0 or len(eval_dataset) == 0:
                 logger.warning("Splitting resulted in empty train or eval set. Using all data for training.")
                 train_dataset = dataset_subset
                 eval_dataset = None
        except ValueError as e:
             logger.warning(f"Could not split dataset: {e}. Using all data for training.")
             train_dataset = dataset_subset
             eval_dataset = None

    logger.info(f"Final dataset sizes -> Train: {len(train_dataset)}, Validation: {len(eval_dataset) if eval_dataset else 0}")
    logger.info("-----------------------------------")
    return train_dataset, eval_dataset

def formatting_prompts_func(examples):
    """Formats the prompt string for the model."""
    texts = []
    for i in range(len(examples['context_vector'])):
        context_vec = examples['context_vector'][i]
        # Truncate context vector string representation for brevity in prompt
        if isinstance(context_vec, (list, np.ndarray)) and len(context_vec) >= 10:
             context_str = str(np.round(context_vec[:10], 3)) + "..."
        else:
             context_str = str(context_vec)
        label = str(examples['target_signal'][i]) # Target class label (0-4)
        # Standard Input/Output format
        text = f"Input: Context={context_str} -> Output: Signal={label}"
        texts.append(text)
    return { "text" : texts }

def tokenize_function(examples, tokenizer, max_len):
    """Tokenizes the formatted text."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_len,
        padding="max_length", # Pad to max_length
        return_tensors=None, # Trainer prefers lists
    )
    # Labels are the same as input_ids for Causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def compute_metrics(eval_pred, tokenizer):
    """Computes accuracy and F1 score for evaluation."""
    logits, labels = eval_pred
    # Logits might be tuple if past_key_values are returned, take first element
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    true_classes = []
    predicted_classes = []

    # Iterate through each example in the batch
    for i in range(len(labels)):
        # Find the indices of the actual labels (where labels != -100)
        label_indices = np.where(labels[i] != -100)[0]
        if len(label_indices) > 0:
            # We only care about the *first* token of the generated label sequence
            label_start_idx = label_indices[0]

            true_label_token_id = labels[i][label_start_idx]
            predicted_token_id = predictions[i][label_start_idx]

            # Decode tokens to compare strings
            true_label_str = tokenizer.decode([true_label_token_id], skip_special_tokens=True).strip()
            predicted_label_str = tokenizer.decode([predicted_token_id], skip_special_tokens=True).strip()

            # Convert to integers if they are valid digits (0-4)
            try:
                if true_label_str.isdigit() and predicted_label_str.isdigit() and len(true_label_str) == 1 and len(predicted_label_str) == 1:
                    true_cls = int(true_label_str)
                    pred_cls = int(predicted_label_str)
                    if 0 <= true_cls <= 4 and 0 <= pred_cls <= 4:
                        true_classes.append(true_cls)
                        predicted_classes.append(pred_cls)
                    # else: logger.debug(f"Decoded label out of range (0-4): True='{true_label_str}', Pred='{predicted_label_str}'")
                # else: logger.debug(f"Decoded label not a single digit: True='{true_label_str}', Pred='{predicted_label_str}'")
            except ValueError:
                # logger.debug(f"Could not convert decoded tokens to int: True='{true_label_str}', Pred='{predicted_label_str}'")
                pass # Skip if conversion fails

    # Calculate metrics if we collected valid pairs
    if true_classes and predicted_classes:
        try:
            accuracy_metric = evaluate.load("accuracy")
            f1_metric = evaluate.load("f1")
            accuracy = accuracy_metric.compute(predictions=predicted_classes, references=true_classes)["accuracy"]
            f1 = f1_metric.compute(predictions=predicted_classes, references=true_classes, average="weighted")["f1"]
            logger.info(f"Eval Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f} (based on {len(true_classes)} valid samples)")
            return {"accuracy": accuracy, "f1": f1}
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {"accuracy": 0.0, "f1": 0.0}
    else:
        logger.warning("No valid label pairs found for metric calculation during this evaluation step.")
        return {"accuracy": 0.0, "f1": 0.0}

# ==============================================================================
# Main Training Logic
# ==============================================================================
def main():
    args = parse_arguments()
    # --- Multiprocessing Setup ---
    num_workers = os.cpu_count() or 1
    logger.info(f"Using {num_workers} CPU cores for data preprocessing & DataLoader")
    MAX_LEN = args.max_seq_length

    # --- GPU Setup ---
    logger.info("--- Setting up GPU Environment ---")
    gpu_info = setup_gpu()
    setup_efficient_attention()
    logger.info(f"GPU Info: {gpu_info}")
    optimal_batch_size = args.batch_size if args.batch_size else get_optimal_batch_size("llama", gpu_info)
    use_bf16 = enable_mixed_precision() and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and enable_mixed_precision()
    logger.info(f"Batch Size: {optimal_batch_size}, BF16: {use_bf16}, FP16: {use_fp16}")
    logger.info("---------------------------------")

    # --- Paths ---
    project_root_dir = Path(".")
    local_model_path = project_root_dir / "models1"
    data_dir = project_root_dir / "data" / "training data"
    dataset_file = data_dir / args.data_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    base_output_dir = project_root_dir / "training" / "adapter_runs"
    output_dir_name = f"lora_pnl_r{args.lora_r}_a{args.lora_alpha}_ep{args.epochs}_bs{optimal_batch_size}{output_suffix}_{timestamp}"
    output_dir = base_output_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- Load Model & Tokenizer ---
    logger.info(f"--- Loading Base Model & Tokenizer from {local_model_path} ---")
    if not local_model_path.exists():
        logger.error(f"Base model directory not found at {local_model_path}")
        sys.exit(1)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(local_model_path), max_seq_length=MAX_LEN,
        dtype=torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None),
        load_in_4bit=True,
        
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    logger.info("---------------------------------------------")

    # --- Apply LoRA ---
    logger.info("--- Applying LoRA Adapter ---")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
        lora_dropout=0.00, bias="none",
        use_gradient_checkpointing=False, # *** DISABLE GRADIENT CHECKPOINTING ***
        random_state=42, max_seq_length=MAX_LEN,
    )
    logger.info("-----------------------------")
    # Warm-up: dummy forward+backward to compile FlashAttention kernels
    model.train()
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (2, MAX_LEN), device=model.device)
    dummy_attention_mask = torch.ones((2, MAX_LEN), device=model.device)
    # provide labels so that loss is computed
    outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, labels=dummy_input_ids)
    loss = outputs.loss
    loss.backward()
    model.zero_grad()

    # --- Prepare Datasets ---
    train_dataset, eval_dataset = prepare_datasets(dataset_file, args.dataset_fraction)

    # --- Format & Tokenize ---
    # 1. Format
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    if eval_dataset:
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    logger.info("Dataset formatting complete.")

    # 2. Tokenize (and remove original/text columns)
    original_cols = list(train_dataset.column_names) # Get columns before tokenization
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, MAX_LEN),
        batched=True, remove_columns=original_cols
    )
    if eval_dataset:
        original_eval_cols = list(eval_dataset.column_names)
        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, MAX_LEN),
            batched=True, remove_columns=original_eval_cols
        )
    else:
        tokenized_eval_dataset = None
    logger.info("Tokenization complete.")
    logger.info(f"Columns in tokenized train dataset: {tokenized_train_dataset.column_names}")
    logger.info("-----------------------------------")
    # Set torch format for datasets to optimize DataLoader and avoid pickling tokenizer
    tokenized_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    if tokenized_eval_dataset:
        tokenized_eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # --- Training Arguments ---
    logger.info("--- Configuring Training Arguments ---")
    eval_strategy = "steps" if tokenized_eval_dataset else "no"
    # Calculate gradient accumulation steps
    if args.gradient_accumulation:
        grad_accum_steps = args.gradient_accumulation
    else:
        # Default to 2 for potentially better throughput with larger models
        grad_accum_steps = 2 # Changed from max(1, 32 // optimal_batch_size)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=optimal_batch_size,
        gradient_accumulation_steps=grad_accum_steps, # Use the calculated/overridden value
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_strategy="steps", # Log metrics at each logging step
        logging_steps=10,
        save_strategy="no", # Explicitly set save strategy to 'steps' to align with save_steps
        save_steps=500, # Increase save frequency (from 100)
        eval_strategy="no", # Keep evaluation disabled
        save_total_limit=3,
        bf16=use_bf16,
        fp16=use_fp16,
        load_best_model_at_end=True if tokenized_eval_dataset else False,
        metric_for_best_model="f1" if tokenized_eval_dataset else None,
        greater_is_better=True if tokenized_eval_dataset else None,
        report_to="none", # Keep this as "none" to avoid TensorBoard errors
        optim="adamw_8bit", # Changed from adamw_torch
        seed=42,
        remove_unused_columns=False, # Keep False
        dataloader_num_workers=4, # Use CPU cores for parallel data loading
        dataloader_pin_memory=True, # Pin memory enabled for performance
    )
    logger.info("------------------------------------")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset, # Keep eval_dataset
        compute_metrics=lambda p: compute_metrics(p, tokenizer) if tokenized_eval_dataset else None,
    )

    # --- Train ---
    logger.info("--- Starting Training ---")
    trainer.train()
    logger.info("-------------------------")

    # --- Save Final Model ---
    logger.info(f"--- Saving Final LoRA Adapter to: {output_dir} ---")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    train_args_path = output_dir / "training_args.json"
    with open(train_args_path, 'w') as f:
         json.dump(args.__dict__, f, indent=4)
    logger.info(f"Training arguments saved to {train_args_path}")
    logger.info("-------------------------------------------------")
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    main()