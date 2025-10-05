#!/usr/bin/env python3
"""
Script to create a dummy PyTorch model that produces NO BUY signals for edge case testing.
This model always outputs a probability below the signal_threshold (0.7).
"""

import torch
import torch.nn as nn
import os

class DummyNoSignalsModel(nn.Module):
    """
    A dummy neural network model that NEVER produces BUY signals.
    
    The model always outputs a low probability (0.1) which is below the 
    default signal_threshold of 0.7, ensuring no BUY trades are executed.
    """
    
    def __init__(self, num_features=17, lookback_window=24):
        super(DummyNoSignalsModel, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        
        # Simple linear layer (not actually used in forward pass for predictable behavior)
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, asset_id_tensor=None):
        """
        Forward pass that always returns low probability.
        
        Args:
            x: Input tensor of shape (batch_size, lookback_window, num_features)
            asset_id_tensor: Optional asset ID tensor (not used in dummy model)
            
        Returns:
            Probability tensor of shape (batch_size, 1) - always 0.1
        """
        batch_size = x.shape[0]
        
        # Always return low probability (0.1) regardless of input
        # This ensures no BUY signals are generated (below 0.7 threshold)
        probabilities = torch.full((batch_size, 1), 0.1, dtype=torch.float32)
        
        return probabilities

def create_dummy_no_signals_model():
    """Create and save a dummy PyTorch model that produces no BUY signals."""
    
    # Create the model
    model = DummyNoSignalsModel(num_features=17, lookback_window=24)
    
    # Set to evaluation mode
    model.eval()
    
    # Test the model with dummy input to verify it works
    batch_size = 3
    lookback_window = 24
    num_features = 17
    
    # Create test input with various values
    test_input = torch.randn(batch_size, lookback_window, num_features)
    
    # Test forward pass
    with torch.no_grad():
        output = model(test_input)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test output shape: {output.shape}")
        print(f"Test output values: {output.flatten()}")
        
        # Verify all outputs are 0.1 (below signal_threshold of 0.7)
        expected_value = 0.1
        all_correct = torch.allclose(output, torch.tensor(expected_value))
        print(f"All outputs are {expected_value}: {all_correct}")
        print(f"All outputs below signal_threshold (0.7): {torch.all(output < 0.7).item()}")
    
    # Save the complete model
    output_path = "models/dummy_test_artifacts/dummy_model_no_signals.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model, output_path)
    
    print(f"\nDummy 'no signals' model created and saved to: {output_path}")
    print(f"Model architecture: {model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total parameters")
    print(f"Expected behavior: Always outputs 0.1 (below 0.7 threshold) -> No BUY trades")
    
    return model

if __name__ == "__main__":
    create_dummy_no_signals_model()