#!/usr/bin/env python3
"""
Script to create a dummy PyTorch model for backtesting integration testing.
"""

import torch
import torch.nn as nn
import os

class DummyNNModel(nn.Module):
    """
    A simple dummy neural network model that produces predictable outputs.
    
    The model implements simple logic:
    - If the sum of the last feature vector in the input sequence is > 10, output high probability (0.8)
    - Otherwise, output low probability (0.2)
    
    This makes it easy to verify that the backtesting strategy behaves correctly.
    """
    
    def __init__(self, num_features=17, lookback_window=24):
        super(DummyNNModel, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        
        # Simple linear layer (not actually used in forward pass for predictable behavior)
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, asset_id_tensor=None):
        """
        Forward pass with predictable logic.
        
        Args:
            x: Input tensor of shape (batch_size, lookback_window, num_features)
            asset_id_tensor: Optional asset ID tensor (not used in dummy model)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Get the last timestep's features for each sample in the batch
        last_features = x[:, -1, :]  # Shape: (batch_size, num_features)
        
        # Sum the features for each sample
        feature_sums = torch.sum(last_features, dim=1)  # Shape: (batch_size,)
        
        # Apply simple logic: if sum > 10, high probability (0.8), else low (0.2)
        probabilities = torch.where(feature_sums > 10.0, 
                                   torch.tensor(0.8, dtype=torch.float32), 
                                   torch.tensor(0.2, dtype=torch.float32))
        
        # Reshape to (batch_size, 1)
        probabilities = probabilities.unsqueeze(1)
        
        return probabilities

def create_dummy_model():
    """Create and save a dummy PyTorch model."""
    
    # Create the model
    model = DummyNNModel(num_features=17, lookback_window=24)
    
    # Set to evaluation mode
    model.eval()
    
    # Test the model with dummy input to verify it works
    batch_size = 2
    lookback_window = 24
    num_features = 17
    
    # Create test input
    test_input = torch.randn(batch_size, lookback_window, num_features)
    
    # Test forward pass
    with torch.no_grad():
        output = model(test_input)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test output shape: {output.shape}")
        print(f"Test output values: {output.flatten()}")
        
        # Test the logic with known inputs
        # Create input where last timestep sum > 10 (should give 0.8)
        test_input_high = torch.ones(1, lookback_window, num_features) * 1.0  # Sum = 18 > 10
        output_high = model(test_input_high)
        print(f"High sum input (sum=18): {output_high.item()}")
        
        # Create input where last timestep sum < 10 (should give 0.2)
        test_input_low = torch.ones(1, lookback_window, num_features) * 0.1  # Sum = 1.8 < 10
        output_low = model(test_input_low)
        print(f"Low sum input (sum=1.8): {output_low.item()}")
    
    # Save the complete model (not just state_dict)
    output_path = "models/dummy_test_artifacts/dummy_model.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model, output_path)
    
    print(f"\nDummy model created and saved to: {output_path}")
    print(f"Model architecture: {model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total parameters")
    
    # Also save the model class definition for loading
    model_class_path = "models/dummy_test_artifacts/dummy_model_class.py"
    with open(model_class_path, 'w') as f:
        f.write('''
import torch
import torch.nn as nn

class DummyNNModel(nn.Module):
    """
    A simple dummy neural network model that produces predictable outputs.
    """
    
    def __init__(self, num_features=18, lookback_window=24):
        super(DummyNNModel, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        
        # Simple linear layer (not actually used in forward pass for predictable behavior)
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, asset_id_tensor=None):
        """
        Forward pass with predictable logic.
        
        Args:
            x: Input tensor of shape (batch_size, lookback_window, num_features)
            asset_id_tensor: Optional asset ID tensor (not used in dummy model)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Get the last timestep's features for each sample in the batch
        last_features = x[:, -1, :]  # Shape: (batch_size, num_features)
        
        # Sum the features for each sample
        feature_sums = torch.sum(last_features, dim=1)  # Shape: (batch_size,)
        
        # Apply simple logic: if sum > 10, high probability (0.8), else low (0.2)
        probabilities = torch.where(feature_sums > 10.0,
                                   torch.tensor(0.8, dtype=torch.float32),
                                   torch.tensor(0.2, dtype=torch.float32))
        
        # Reshape to (batch_size, 1)
        probabilities = probabilities.unsqueeze(1)
        
        return probabilities
''')
    print(f"Model class definition saved to: {model_class_path}")
    
    return model

if __name__ == "__main__":
    create_dummy_model()