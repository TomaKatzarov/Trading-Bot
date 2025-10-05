"""
Neural Network Architecture Implementations for Trading Signal Prediction

This module implements custom PyTorch neural network architectures for predicting
trading signals based on technical indicators, sentiment data, and asset embeddings.

Architectures implemented:
1. MLPModel - Multi-Layer Perceptron baseline
2. LSTMAttentionModel - LSTM with self-attention mechanism
3. GRUAttentionModel - GRU with self-attention mechanism  
4. CNNLSTMModel - CNN-LSTM hybrid with improvements

All models are designed for binary classification (BUY_SIGNAL vs NO_BUY_SIGNAL)
with +5%/-2% profit/stop-loss targets over 8-hour prediction horizon.

Author: Flow-Code
Date: 2025-05-28
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for sequence models.
    
    Computes attention weights over sequence positions to focus on
    the most relevant time steps for prediction.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        """
        Initialize self-attention layer.
        
        Args:
            hidden_dim: Dimension of input hidden states
            attention_dim: Dimension of attention computation
        """
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Linear layers for attention computation
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor for attention scores
        self.scale = math.sqrt(attention_dim)
        
    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention to hidden states.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Compute queries, keys, and values
        Q = self.query(hidden_states)  # (batch_size, seq_len, attention_dim)
        K = self.key(hidden_states)    # (batch_size, seq_len, attention_dim)
        V = self.value(hidden_states)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_output = torch.matmul(attention_weights, V)
        
        return attended_output, attention_weights


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron baseline model.
    
    Flattens sequence input and processes through fully connected layers.
    Serves as a simple baseline for comparison with sequence models.
    """
    
    def __init__(
        self,
        n_features: int,
        lookback_window: int,
        num_assets: int,
        asset_embedding_dim: int = 8,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.3,
        **kwargs
    ):
        """
        Initialize MLP model.
        
        Args:
            n_features: Number of input features per time step
            lookback_window: Length of input sequence
            num_assets: Number of unique assets for embedding
            asset_embedding_dim: Dimension of asset embeddings
            hidden_dims: Tuple of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(MLPModel, self).__init__()
        
        self.n_features = n_features
        self.lookback_window = lookback_window
        self.num_assets = num_assets
        self.asset_embedding_dim = asset_embedding_dim
        
        # Asset embedding layer
        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
        
        # Calculate input dimension: flattened sequence + asset embedding
        input_dim = lookback_window * n_features + asset_embedding_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        # Removed nn.Sigmoid() - model will output logits
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            features: Input features of shape (batch_size, lookback_window, n_features)
            asset_ids: Asset IDs of shape (batch_size,)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size = features.size(0)
        
        # Flatten sequence features
        flattened_features = features.view(batch_size, -1)
        
        # Get asset embeddings
        asset_embeds = self.asset_embedding(asset_ids)
        
        # Concatenate flattened features with asset embeddings
        combined_input = torch.cat([flattened_features, asset_embeds], dim=1)
        
        # Pass through MLP
        output = self.mlp(combined_input)
        
        return output


class LSTMAttentionModel(nn.Module):
    """
    LSTM model with self-attention mechanism.
    
    Uses LSTM layers to capture temporal dependencies, followed by
    self-attention to focus on the most relevant time steps.
    """
    
    def __init__(
        self,
        n_features: int,
        num_assets: int,
        asset_embedding_dim: int = 8,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        attention_dim: int = 64,
        dropout_rate: float = 0.3,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """
        Initialize LSTM with attention model.
        
        Args:
            n_features: Number of input features per time step
            num_assets: Number of unique assets for embedding
            asset_embedding_dim: Dimension of asset embeddings
            lstm_hidden_dim: Hidden dimension of LSTM layers
            lstm_num_layers: Number of LSTM layers
            attention_dim: Dimension for attention computation
            dropout_rate: Dropout rate for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super(LSTMAttentionModel, self).__init__()
        
        self.n_features = n_features
        self.num_assets = num_assets
        self.asset_embedding_dim = asset_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        
        # Asset embedding layer
        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
        
        # Input projection to combine features with asset embedding
        self.input_projection = nn.Linear(n_features + asset_embedding_dim, lstm_hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_hidden_dim)
        
        # Self-attention mechanism
        self.attention = SelfAttention(lstm_hidden_dim, attention_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim // 2, 1)
            # Removed nn.Sigmoid() - model will output logits
        )
        
    def forward(self, features: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM with attention.
        
        Args:
            features: Input features of shape (batch_size, lookback_window, n_features)
            asset_ids: Asset IDs of shape (batch_size,)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = features.size()
        
        # Get asset embeddings and expand to sequence length
        asset_embeds = self.asset_embedding(asset_ids)  # (batch_size, asset_embedding_dim)
        asset_embeds = asset_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, asset_embedding_dim)
        
        # Concatenate features with asset embeddings
        combined_input = torch.cat([features, asset_embeds], dim=2)  # (batch_size, seq_len, n_features + asset_embedding_dim)
        
        # Project to LSTM input dimension
        projected_input = self.input_projection(combined_input)  # (batch_size, seq_len, lstm_hidden_dim)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(projected_input)  # (batch_size, seq_len, lstm_hidden_dim)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            lstm_output = self.layer_norm(lstm_output)
        
        # Apply self-attention
        attended_output, attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, lstm_hidden_dim)
        
        # Global average pooling over sequence dimension
        pooled_output = torch.mean(attended_output, dim=1)  # (batch_size, lstm_hidden_dim)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Final classification
        output = self.classifier(pooled_output)  # (batch_size, 1)
        
        return output


class GRUAttentionModel(nn.Module):
    """
    GRU model with self-attention mechanism.
    
    Similar to LSTM but uses GRU layers which are simpler and often
    train faster while maintaining comparable performance.
    """
    
    def __init__(
        self,
        n_features: int,
        num_assets: int,
        asset_embedding_dim: int = 8,
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 2,
        attention_dim: int = 64,
        dropout_rate: float = 0.3,
        use_layer_norm: bool = True,
        **kwargs
    ):
        """
        Initialize GRU with attention model.
        
        Args:
            n_features: Number of input features per time step
            num_assets: Number of unique assets for embedding
            asset_embedding_dim: Dimension of asset embeddings
            gru_hidden_dim: Hidden dimension of GRU layers
            gru_num_layers: Number of GRU layers
            attention_dim: Dimension for attention computation
            dropout_rate: Dropout rate for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super(GRUAttentionModel, self).__init__()
        
        self.n_features = n_features
        self.num_assets = num_assets
        self.asset_embedding_dim = asset_embedding_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        
        # Asset embedding layer
        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
        
        # Input projection to combine features with asset embedding
        self.input_projection = nn.Linear(n_features + asset_embedding_dim, gru_hidden_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=gru_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=dropout_rate if gru_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(gru_hidden_dim)
        
        # Self-attention mechanism
        self.attention = SelfAttention(gru_hidden_dim, attention_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gru_hidden_dim // 2, 1)
            # Removed nn.Sigmoid() - model will output logits
        )
        
    def forward(self, features: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRU with attention.
        
        Args:
            features: Input features of shape (batch_size, lookback_window, n_features)
            asset_ids: Asset IDs of shape (batch_size,)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = features.size()
        
        # Get asset embeddings and expand to sequence length
        asset_embeds = self.asset_embedding(asset_ids)  # (batch_size, asset_embedding_dim)
        asset_embeds = asset_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, asset_embedding_dim)
        
        # Concatenate features with asset embeddings
        combined_input = torch.cat([features, asset_embeds], dim=2)  # (batch_size, seq_len, n_features + asset_embedding_dim)
        
        # Project to GRU input dimension
        projected_input = self.input_projection(combined_input)  # (batch_size, seq_len, gru_hidden_dim)
        
        # Pass through GRU
        gru_output, _ = self.gru(projected_input)  # (batch_size, seq_len, gru_hidden_dim)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            gru_output = self.layer_norm(gru_output)
        
        # Apply self-attention
        attended_output, attention_weights = self.attention(gru_output)  # (batch_size, seq_len, gru_hidden_dim)
        
        # Global average pooling over sequence dimension
        pooled_output = torch.mean(attended_output, dim=1)  # (batch_size, gru_hidden_dim)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Final classification
        output = self.classifier(pooled_output)  # (batch_size, 1)
        
        return output


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model with improvements from diagnostic report.
    
    Uses 1D CNN layers for local feature extraction followed by
    LSTM layers for temporal modeling. Includes batch normalization
    and optional attention mechanism.
    """
    
    def __init__(
        self,
        n_features: int,
        num_assets: int,
        asset_embedding_dim: int = 8,
        cnn_filters: Tuple[int, ...] = (32, 64),
        cnn_kernel_sizes: Tuple[int, ...] = (3, 5),
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        attention_dim: int = 64,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        use_pooling: bool = False,
        **kwargs
    ):
        """
        Initialize CNN-LSTM hybrid model.
        
        Args:
            n_features: Number of input features per time step
            num_assets: Number of unique assets for embedding
            asset_embedding_dim: Dimension of asset embeddings
            cnn_filters: Tuple of filter counts for CNN layers
            cnn_kernel_sizes: Tuple of kernel sizes for CNN layers
            lstm_hidden_dim: Hidden dimension of LSTM layers
            lstm_num_layers: Number of LSTM layers
            attention_dim: Dimension for attention computation
            dropout_rate: Dropout rate for regularization
            use_attention: Whether to use attention mechanism
            use_pooling: Whether to use max pooling (use cautiously)
        """
        super(CNNLSTMModel, self).__init__()
        
        self.n_features = n_features
        self.num_assets = num_assets
        self.asset_embedding_dim = asset_embedding_dim
        self.use_attention = use_attention
        self.use_pooling = use_pooling
        
        # Asset embedding layer
        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        input_channels = n_features + asset_embedding_dim
        
        for i, (filters, kernel_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
            # 1D Convolution
            conv_layer = nn.Conv1d(
                in_channels=input_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Same padding
                stride=1
            )
            
            # Batch normalization
            batch_norm = nn.BatchNorm1d(filters)
            
            # Activation and dropout
            activation = nn.ReLU()
            dropout = nn.Dropout(dropout_rate)
            
            # Optional max pooling
            if use_pooling:
                # Corrected MaxPool1d for typical downsampling (halves sequence length)
                pooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                layer_block = nn.Sequential(conv_layer, batch_norm, activation, dropout, pooling)
            else:
                layer_block = nn.Sequential(conv_layer, batch_norm, activation, dropout)
            
            self.cnn_layers.append(layer_block)
            input_channels = filters
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],  # Output from last CNN layer
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Optional attention mechanism
        if use_attention:
            self.attention = SelfAttention(lstm_hidden_dim, attention_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim // 2, 1)
            # Removed nn.Sigmoid() - model will output logits
        )
        
    def forward(self, features: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-LSTM hybrid.
        
        Args:
            features: Input features of shape (batch_size, lookback_window, n_features)
            asset_ids: Asset IDs of shape (batch_size,)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = features.size()
        
        # Get asset embeddings and expand to sequence length
        asset_embeds = self.asset_embedding(asset_ids)  # (batch_size, asset_embedding_dim)
        asset_embeds = asset_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, asset_embedding_dim)
        
        # Concatenate features with asset embeddings
        combined_input = torch.cat([features, asset_embeds], dim=2)  # (batch_size, seq_len, n_features + asset_embedding_dim)
        
        # Transpose for CNN: (batch_size, channels, seq_len)
        cnn_input = combined_input.transpose(1, 2)  # (batch_size, n_features + asset_embedding_dim, seq_len)
        
        # Pass through CNN layers
        cnn_output = cnn_input
        for cnn_layer in self.cnn_layers:
            cnn_output = cnn_layer(cnn_output)  # (batch_size, filters, seq_len)
        
        # Transpose back for LSTM: (batch_size, seq_len, filters)
        lstm_input = cnn_output.transpose(1, 2)  # (batch_size, seq_len, filters)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(lstm_input)  # (batch_size, seq_len, lstm_hidden_dim)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_output, attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, lstm_hidden_dim)
            # Global average pooling over sequence dimension
            pooled_output = torch.mean(attended_output, dim=1)  # (batch_size, lstm_hidden_dim)
        else:
            # Use last time step output
            pooled_output = lstm_output[:, -1, :]  # (batch_size, lstm_hidden_dim)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Final classification
        output = self.classifier(pooled_output)  # (batch_size, 1)
        
        return output


def create_model(model_type: str, model_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create neural network models.
    
    Args:
        model_type: Type of model to create ('mlp', 'lstm', 'gru', 'cnn_lstm')
        model_config: Configuration dictionary for model parameters
        
    Returns:
        Initialized PyTorch model
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    
    if model_type == 'mlp':
        return MLPModel(**model_config)
    elif model_type == 'lstm':
        return LSTMAttentionModel(**model_config)
    elif model_type == 'gru':
        return GRUAttentionModel(**model_config)
    elif model_type == 'cnn_lstm':
        return CNNLSTMModel(**model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: 'mlp', 'lstm', 'gru', 'cnn_lstm'")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model including parameter count and architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_type': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture_summary': str(model)
    }


# Example usage and configuration templates
if __name__ == "__main__":
    # Example model configurations
    
    # Common parameters
    n_features = 17  # 14 TIs + 2 DoW + 1 sentiment
    num_assets = 154  # From asset_id_mapping.json
    lookback_window = 24  # 24-hour lookback window
    
    # MLP configuration
    mlp_config = {
        'n_features': n_features,
        'lookback_window': lookback_window,
        'num_assets': num_assets,
        'asset_embedding_dim': 8,
        'hidden_dims': (128, 64, 32),
        'dropout_rate': 0.3
    }
    
    # LSTM configuration
    lstm_config = {
        'n_features': n_features,
        'num_assets': num_assets,
        'asset_embedding_dim': 8,
        'lstm_hidden_dim': 64,
        'lstm_num_layers': 2,
        'attention_dim': 64,
        'dropout_rate': 0.3,
        'use_layer_norm': True
    }
    
    # GRU configuration
    gru_config = {
        'n_features': n_features,
        'num_assets': num_assets,
        'asset_embedding_dim': 8,
        'gru_hidden_dim': 64,
        'gru_num_layers': 2,
        'attention_dim': 64,
        'dropout_rate': 0.3,
        'use_layer_norm': True
    }
    
    # CNN-LSTM configuration
    cnn_lstm_config = {
        'n_features': n_features,
        'num_assets': num_assets,
        'asset_embedding_dim': 8,
        'cnn_filters': (32, 64),
        'cnn_kernel_sizes': (3, 5),
        'lstm_hidden_dim': 64,
        'lstm_num_layers': 1,
        'attention_dim': 64,
        'dropout_rate': 0.3,
        'use_attention': True,
        'use_pooling': False
    }
    
    # Create example models
    print("Creating example models...")
    
    mlp_model = create_model('mlp', mlp_config)
    lstm_model = create_model('lstm', lstm_config)
    gru_model = create_model('gru', gru_config)
    cnn_lstm_model = create_model('cnn_lstm', cnn_lstm_config)
    
    # Print model information
    models = [
        ('MLP', mlp_model),
        ('LSTM+Attention', lstm_model),
        ('GRU+Attention', gru_model),
        ('CNN-LSTM', cnn_lstm_model)
    ]
    
    for name, model in models:
        info = get_model_info(model)
        print(f"\n{name} Model:")
        print(f"  Parameters: {info['trainable_parameters']:,}")
        print(f"  Size: {info['model_size_mb']:.2f} MB")
    
    # Test forward pass with dummy data
    batch_size = 32
    dummy_features = torch.randn(batch_size, lookback_window, n_features)
    dummy_asset_ids = torch.randint(0, num_assets, (batch_size,))
    
    print(f"\nTesting forward pass with batch_size={batch_size}...")
    
    for name, model in models:
        model.eval()
        with torch.no_grad():
            output = model(dummy_features, dummy_asset_ids)
            print(f"{name}: Output shape {output.shape}, Range [{output.min():.3f}, {output.max():.3f}]")
    
    print("\nAll models created successfully!")