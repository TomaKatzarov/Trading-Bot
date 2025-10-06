"""Tests for the shared FeatureEncoder used across all symbol agents.

The FeatureEncoder is a critical shared component that transforms dictionary
observations from the trading environment into unified embeddings. These tests
focus on robustness, numerical stability, and interface guarantees to prevent
regressions that could cascade across the entire multi-agent system.
"""

from __future__ import annotations

import math
from typing import Dict

import pytest
import torch

from core.rl.policies.feature_encoder import (
    EncoderConfig,
    FeatureEncoder,
    PositionalEncoding,
)


@pytest.fixture(name="encoder_config")
def fixture_encoder_config() -> EncoderConfig:
    """Provide the standard encoder configuration used in production."""

    return EncoderConfig(
        technical_seq_len=24,
        technical_feature_dim=23,
        sl_prob_dim=3,
        position_dim=5,
        portfolio_dim=8,
        regime_dim=10,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        output_dim=256,
        max_seq_len=50,
    )


@pytest.fixture(name="sample_observations")
def fixture_sample_observations(encoder_config: EncoderConfig, batch_size: int = 8) -> Dict[str, torch.Tensor]:
    """Create synthetic observation dictionaries that mimic environment output."""

    torch.manual_seed(42)
    return {
        "technical": torch.randn(batch_size, encoder_config.technical_seq_len, encoder_config.technical_feature_dim),
        "sl_probs": torch.rand(batch_size, encoder_config.sl_prob_dim),
        "position": torch.randn(batch_size, encoder_config.position_dim),
        "portfolio": torch.randn(batch_size, encoder_config.portfolio_dim),
        "regime": torch.randn(batch_size, encoder_config.regime_dim),
    }


class TestFeatureEncoder:
    """Comprehensive tests for the FeatureEncoder."""

    def test_encoder_initialization(self, encoder_config: EncoderConfig) -> None:
        """Encoder initializes with the expected hyper-parameters."""

        encoder = FeatureEncoder(encoder_config)

        assert encoder.config.d_model == 256
        assert encoder.config.num_layers == 4
        assert encoder.config.nhead == 8

    def test_forward_pass_shape(self, encoder_config: EncoderConfig, sample_observations: Dict[str, torch.Tensor]) -> None:
        """Forward pass returns pooled embeddings of shape (batch, 256)."""

        encoder = FeatureEncoder(encoder_config).eval()
        with torch.no_grad():
            output = encoder(sample_observations)

        batch_size = sample_observations["technical"].size(0)
        assert output.shape == (batch_size, encoder_config.output_dim)

    def test_forward_pass_no_nans(self, encoder_config: EncoderConfig, sample_observations: Dict[str, torch.Tensor]) -> None:
        """Forward pass should not introduce NaN or Inf values."""

        encoder = FeatureEncoder(encoder_config).eval()
        with torch.no_grad():
            output = encoder(sample_observations)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sequence_output(self, encoder_config: EncoderConfig, sample_observations: Dict[str, torch.Tensor]) -> None:
        """Returning the sequence yields the expected token dimensions."""

        encoder = FeatureEncoder(encoder_config).eval()
        with torch.no_grad():
            pooled, sequence = encoder(sample_observations, return_sequence=True)

        batch_size = sample_observations["technical"].size(0)
        expected_tokens = encoder_config.technical_seq_len + 4  # technical + auxiliary tokens
        assert pooled.shape == (batch_size, encoder_config.output_dim)
        assert sequence.shape == (batch_size, expected_tokens, encoder_config.d_model)

    def test_batch_independence(self, encoder_config: EncoderConfig) -> None:
        """Duplicating a single observation in a batch yields identical embeddings."""

        encoder = FeatureEncoder(encoder_config).eval()

        single_obs = {
            "technical": torch.randn(1, encoder_config.technical_seq_len, encoder_config.technical_feature_dim),
            "sl_probs": torch.rand(1, encoder_config.sl_prob_dim),
            "position": torch.randn(1, encoder_config.position_dim),
            "portfolio": torch.randn(1, encoder_config.portfolio_dim),
            "regime": torch.randn(1, encoder_config.regime_dim),
        }
        batch_obs = {
            key: value.repeat(4, 1, 1) if value.dim() == 3 else value.repeat(4, 1)
            for key, value in single_obs.items()
        }

        with torch.no_grad():
            single_out = encoder(single_obs)
            batch_out = encoder(batch_obs)

        for i in range(batch_out.size(0)):
            torch.testing.assert_close(single_out[0], batch_out[i], rtol=1e-5, atol=1e-5)

    def test_gradient_flow(self, encoder_config: EncoderConfig, sample_observations: Dict[str, torch.Tensor]) -> None:
        """Gradients propagate to all learnable parameters."""

        encoder = FeatureEncoder(encoder_config)
        encoder.train()

        output = encoder(sample_observations)
        loss = output.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Gradient missing for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient detected in: {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient detected in: {name}"

    def test_different_batch_sizes(self, encoder_config: EncoderConfig) -> None:
        """The encoder supports a range of batch sizes without shape issues."""

        encoder = FeatureEncoder(encoder_config).eval()
        batch_sizes = [1, 2, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            observations = {
                "technical": torch.randn(batch_size, encoder_config.technical_seq_len, encoder_config.technical_feature_dim),
                "sl_probs": torch.rand(batch_size, encoder_config.sl_prob_dim),
                "position": torch.randn(batch_size, encoder_config.position_dim),
                "portfolio": torch.randn(batch_size, encoder_config.portfolio_dim),
                "regime": torch.randn(batch_size, encoder_config.regime_dim),
            }
            with torch.no_grad():
                output = encoder(observations)
            assert output.shape == (batch_size, encoder_config.output_dim)

    def test_parameter_count(self, encoder_config: EncoderConfig, capsys: pytest.CaptureFixture[str]) -> None:
        """Parameter budget remains below the 5M target for the shared encoder."""

        encoder = FeatureEncoder(encoder_config)
        total_params = sum(param.numel() for param in encoder.parameters())

        print(f"\nEncoder parameters: {total_params:,}")
        assert total_params < 5_000_000, f"Encoder has {total_params:,} parameters, exceeding 5M"

        captured = capsys.readouterr()
        assert "Encoder parameters" in captured.out

    def test_deterministic_eval_mode(self, encoder_config: EncoderConfig, sample_observations: Dict[str, torch.Tensor]) -> None:
        """Repeated forward passes in eval mode are deterministic."""

        encoder = FeatureEncoder(encoder_config).eval()
        with torch.no_grad():
            output_a = encoder(sample_observations)
            output_b = encoder(sample_observations)
        torch.testing.assert_close(output_a, output_b, rtol=1e-6, atol=1e-6)

    def test_return_sequence_tuple_flag(self, encoder_config: EncoderConfig, sample_observations: Dict[str, torch.Tensor]) -> None:
        """When ``return_sequence`` is True the encoder returns a tuple."""

        encoder = FeatureEncoder(encoder_config).eval()
        with torch.no_grad():
            result = encoder(sample_observations, return_sequence=True)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestPositionalEncoding:
    """Tests for the sinusoidal positional encoding module."""

    def test_positional_encoding_shape_and_determinism(self) -> None:
        """Adding positional encodings preserves shape and remains deterministic."""

        pos_enc = PositionalEncoding(d_model=256, dropout=0.0, max_len=50)
        x = torch.randn(4, 28, 256)

        output_a = pos_enc(x.clone())
        output_b = pos_enc(x.clone())

        assert output_a.shape == x.shape
        torch.testing.assert_close(output_a, output_b)

    def test_positional_encoding_sinusoidal_structure(self) -> None:
        """Sinusoidal encoding follows the expected sin/cos pattern."""

        d_model = 256
        pos_enc = PositionalEncoding(d_model=d_model, dropout=0.0, max_len=60)
        x = torch.zeros(1, 60, d_model)

        with torch.no_grad():
            encoded = pos_enc(x)

        # The first two dimensions should be sin and cos of the same frequency.
        position = torch.arange(60).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        expected_sin = torch.sin(position.unsqueeze(1) * div_term)
        expected_cos = torch.cos(position.unsqueeze(1) * div_term)

        torch.testing.assert_close(encoded[0, :, 0::2], expected_sin, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(encoded[0, :, 1::2], expected_cos, rtol=1e-4, atol=1e-4)


class TestEncoderConfig:
    """Validation tests for EncoderConfig."""

    def test_config_defaults(self) -> None:
        """Default configuration matches specification."""

        config = EncoderConfig()
        assert config.technical_seq_len == 24
        assert config.technical_feature_dim == 23
        assert config.d_model == 256
        assert config.nhead == 8
        assert config.num_layers == 4

    def test_config_invalid_max_seq_len(self) -> None:
        """Configuration enforces sufficient sequence length for auxiliary tokens."""

        with pytest.raises(ValueError):
            EncoderConfig(max_seq_len=10)

    def test_config_invalid_head_dimension(self) -> None:
        """Configuration enforces divisibility of d_model by nhead."""

        with pytest.raises(ValueError):
            EncoderConfig(d_model=250, nhead=7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])