"""Policy networks, encoders, and supporting utilities."""

from .feature_encoder import EncoderConfig, FeatureEncoder, PositionalEncoding
from .symbol_agent import ActionMasker, SymbolAgent, SymbolAgentConfig
from .initialization import (
    init_actor,
    init_critic,
    init_encoder,
    orthogonal_init,
    verify_initialization,
    xavier_uniform_init,
)
from .weight_sharing import SharedEncoderManager, calculate_savings_for_agents
from .sl_to_rl_transfer import (
    SLTransferWarning,
    create_sl_transfer_experiment,
    load_sl_checkpoint,
    transfer_sl_features_to_encoder,
)

__all__ = [
    "EncoderConfig",
    "FeatureEncoder",
    "PositionalEncoding",
    "SymbolAgent",
    "SymbolAgentConfig",
    "ActionMasker",
    "init_encoder",
    "init_actor",
    "init_critic",
    "verify_initialization",
    "xavier_uniform_init",
    "orthogonal_init",
    "SharedEncoderManager",
    "calculate_savings_for_agents",
    "load_sl_checkpoint",
    "transfer_sl_features_to_encoder",
    "create_sl_transfer_experiment",
    "SLTransferWarning",
]
