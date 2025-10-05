
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
