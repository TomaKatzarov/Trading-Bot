"""Diagnostic script to understand why options aren't being selected.

Run this to trace through the wrapper state and understand the issue.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

# Key Questions to Answer:
# 1. Is the wrapper being attached correctly?
# 2. Is select_actions being called?
# 3. Is the condition `if state.current_option_idx is None` ever True?
# 4. Is _batch_select_options being called?
# 5. Are options being created properly?

print("="*80)
print("DIAGNOSTIC ANALYSIS - Options Selection Issue")
print("="*80)

print("\nKEY OBSERVATIONS FROM OUTPUT:")
print("  • total_selections: 0  <- OPTIONS NEVER SELECTED!")
print("  • all option metrics are 0")
print("  • Training completes but options framework not engaged")

print("\nPOSSIBLE ROOT CAUSES:")
print("  1. state.current_option_idx might be set initially (not None)")
print("  2. _predict_with_options might not be called properly")
print("  3. Wrapper might not be attached to model.predict")
print("  4. select_actions might be bypassed somehow")
print("  5. Options controller might not have options registered")

print("\nNEXT DEBUGGING STEPS:")
print("  1. Add debug logging to select_actions entry point")
print("  2. Log state.current_option_idx value on first call")
print("  3. Verify _predict_with_options is being used (not original SAC)")
print("  4. Check if options_controller.select_option is working")
print("  5. Verify option_usage_counts is being updated")

print("\nRECOMMENDED FIX:")
print("  Add debug logging at key points:")
print("    - HierarchicalSACWrapper.select_actions() entry")
print("    - Before 'if state.current_option_idx is None' check")
print("    - Inside _batch_select_options()")
print("    - In options_controller.select_option()")
print("\n" + "="*80)
