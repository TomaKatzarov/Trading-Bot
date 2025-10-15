#!/usr/bin/env python3
"""
Script to fix TradingConfig instantiations in test files.
Removes action_mode parameter and properly sets continuous_settings attribute.
"""
import re
from pathlib import Path

def fix_test_file(filepath: Path):
    """Fix TradingConfig instantiations in a test file."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match TradingConfig with action_mode
    # This regex finds TradingConfig( ... action_mode="continuous", ... )
    pattern = r'''
        (config\s*=\s*TradingConfig\s*\(\s*)  # config = TradingConfig(
        ([^)]*?)                               # capture everything until )
        (action_mode="continuous",\s*)         # action_mode="continuous",
        ([^)]*?)                               # capture rest
        (\))                                   # closing )
    '''
    
    def replacer(match):
        prefix = match.group(1)
        before_action = match.group(2)
        after_action = match.group(4)
        closing = match.group(5)
        
        # Combine before and after, removing action_mode
        combined = before_action + after_action
        
        # Extract episode_length and continuous_settings from combined
        ep_match = re.search(r'episode_length\s*=\s*(\d+)', combined)
        cs_match = re.search(r'continuous_settings\s*=\s*(\{[^}]+\})', combined, re.DOTALL)
        
        episode_length = ep_match.group(1) if ep_match else '50'
        continuous_settings = cs_match.group(1) if cs_match else '{}'
        
        # Remove episode_length and continuous_settings from combined
        combined = re.sub(r',?\s*episode_length\s*=\s*\d+\s*,?', '', combined)
        combined = re.sub(r',?\s*continuous_settings\s*=\s*\{[^}]+\}\s*,?', '', combined, flags=re.DOTALL)
        
        # Clean up extra commas and whitespace
        combined = re.sub(r',\s*,', ',', combined)
        combined = re.sub(r',\s*$', '', combined.strip())
        
        # Build proper config
        result = f'''{prefix}{combined.strip()},
        episode_length={episode_length},
    {closing}
    # Add continuous settings as attribute (ContinuousTradingEnvironment reads via getattr)
    config.continuous_settings = {continuous_settings}'''
        
        return result
    
    # Apply fixes
    fixed_content = re.sub(pattern, replacer, content, flags=re.VERBOSE | re.DOTALL)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"✓ Fixed {filepath}")

if __name__ == '__main__':
    test_file = Path('tests/test_action_space_behavior.py')
    if test_file.exists():
        fix_test_file(test_file)
    else:
        print(f"File not found: {test_file}")
