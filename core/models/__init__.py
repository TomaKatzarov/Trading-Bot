"""
Core Models Package

This package contains neural network architectures and model utilities
for the trading signal prediction system.

Available Models:
- MLPModel: Multi-Layer Perceptron baseline
- LSTMAttentionModel: LSTM with self-attention mechanism
- GRUAttentionModel: GRU with self-attention mechanism
- CNNLSTMModel: CNN-LSTM hybrid with improvements

Factory Functions:
- create_model: Factory function to create models by type
- get_model_info: Get model information and statistics

Author: Flow-Code
Date: 2025-05-28
Version: 1.0
"""

from .nn_architectures import (
    MLPModel,
    LSTMAttentionModel,
    GRUAttentionModel,
    CNNLSTMModel,
    SelfAttention,
    create_model,
    get_model_info
)

__all__ = [
    'MLPModel',
    'LSTMAttentionModel', 
    'GRUAttentionModel',
    'CNNLSTMModel',
    'SelfAttention',
    'create_model',
    'get_model_info'
]

__version__ = '1.0.0'