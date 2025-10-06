"""Tests for the symbol agent actor-critic policy network."""

from __future__ import annotations

import pytest
import torch

from core.rl.policies.feature_encoder import EncoderConfig, FeatureEncoder
from core.rl.policies.symbol_agent import (
    ActionMasker,
    SymbolAgent,
    SymbolAgentConfig,
)


class TestActionMasker:
    """Validate action masking rules."""

    def test_no_position_masks_sells(self) -> None:
        """When no position is held, selling-related actions must be masked."""

        masker = ActionMasker()
        observations = {
            "position": torch.zeros(2, 5),
            "portfolio": torch.tensor(
                [[10000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ).repeat(2, 1),
        }

        mask = masker.get_mask(observations)

        assert mask.shape == (2, 7)
        assert bool(mask[0, 0])  # HOLD always available
        assert all(bool(mask[0, idx]) for idx in (1, 2, 3))
        assert all(not bool(mask[0, idx]) for idx in (4, 5, 6))

    def test_has_position_masks_buys(self) -> None:
        """When already holding a position, buying actions should be masked."""

        masker = ActionMasker()
        observations = {
            "position": torch.tensor(
                [[100.0, 10.0, 50.0, 5.0, 1.0]], dtype=torch.float32
            ).repeat(2, 1),
            "portfolio": torch.tensor(
                [[10000.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ).repeat(2, 1),
        }

        mask = masker.get_mask(observations)

        assert all(not bool(mask[0, idx]) for idx in (1, 2, 3))
        assert all(bool(mask[0, idx]) for idx in (4, 5, 6))

    def test_high_exposure_masks_increases(self) -> None:
        """High portfolio exposure should disable position-increasing actions."""

        masker = ActionMasker()
        observations = {
            "position": torch.zeros(1, 5),
            "portfolio": torch.tensor(
                [[10000.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),
        }

        mask = masker.get_mask(observations)

        assert bool(mask[0, 0])  # HOLD
        assert all(not bool(mask[0, idx]) for idx in (1, 2, 3))
        assert not bool(mask[0, 6])

    def test_missing_position_key_raises(self) -> None:
        """Omitting the position entry should raise a KeyError."""

        masker = ActionMasker()
        observations = {
            "portfolio": torch.zeros(1, 8),
        }

        with pytest.raises(KeyError, match="position"):
            masker.get_mask(observations)

    def test_missing_portfolio_key_raises(self) -> None:
        """Omitting the portfolio entry should raise a KeyError."""

        masker = ActionMasker()
        observations = {
            "position": torch.zeros(1, 5),
        }

        with pytest.raises(KeyError, match="portfolio"):
            masker.get_mask(observations)

    def test_invalid_tensor_dimensions_raise(self) -> None:
        """Non-2D tensors should trigger a ValueError."""

        masker = ActionMasker()
        observations = {
            "position": torch.zeros(5),  # 1D tensor
            "portfolio": torch.zeros(1, 8),
        }

        with pytest.raises(ValueError, match="two-dimensional"):
            masker.get_mask(observations)

    def test_mismatched_batch_sizes_raise(self) -> None:
        """Batch size mismatch between tensors should raise a ValueError."""

        masker = ActionMasker()
        observations = {
            "position": torch.zeros(2, 5),
            "portfolio": torch.zeros(1, 8),
        }

        with pytest.raises(ValueError, match="matching sizes"):
            masker.get_mask(observations)


class TestSymbolAgentConfig:
    """Configuration sanity checks."""

    def test_default_config(self) -> None:
        encoder_config = EncoderConfig()
        config = SymbolAgentConfig(encoder_config=encoder_config)

        assert config.action_dim == 7
        assert config.hidden_dim == 128
        assert pytest.approx(config.dropout, rel=0.0) == 0.1
        assert config.use_shared_encoder is True


class TestSymbolAgent:
    """Integration tests for the actor-critic policy."""

    @pytest.fixture()
    def agent(self) -> SymbolAgent:
        encoder_config = EncoderConfig()
        config = SymbolAgentConfig(encoder_config=encoder_config)
        return SymbolAgent(config)

    @pytest.fixture()
    def sample_observations(self) -> dict[str, torch.Tensor]:
        batch = 4
        return {
            "technical": torch.randn(batch, 24, 23),
            "sl_probs": torch.rand(batch, 3),
            "position": torch.zeros(batch, 5),
            "portfolio": torch.tensor(
                [[10000.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ).repeat(batch, 1),
            "regime": torch.randn(batch, 10),
        }

    def test_initialization(self, agent: SymbolAgent) -> None:
        assert isinstance(agent, SymbolAgent)
        assert agent.owns_encoder is True
        assert hasattr(agent, "encoder")
        assert hasattr(agent, "actor")
        assert hasattr(agent, "critic")

    def test_forward_pass_shapes(
        self, agent: SymbolAgent, sample_observations: dict[str, torch.Tensor]
    ) -> None:
        agent.eval()
        actions, log_probs, values = agent(sample_observations)

        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert torch.all((0 <= actions) & (actions < agent.action_dim))

    def test_deterministic_mode(
        self, agent: SymbolAgent, sample_observations: dict[str, torch.Tensor]
    ) -> None:
        agent.eval()
        actions1, _, _ = agent(sample_observations, deterministic=True)
        actions2, _, _ = agent(sample_observations, deterministic=True)

        assert torch.equal(actions1, actions2)

    def test_get_value(
        self, agent: SymbolAgent, sample_observations: dict[str, torch.Tensor]
    ) -> None:
        agent.eval()
        values = agent.get_value(sample_observations)
        assert values.shape == (4,)

    def test_evaluate_actions(
        self, agent: SymbolAgent, sample_observations: dict[str, torch.Tensor]
    ) -> None:
        agent.eval()
        chosen_actions = torch.tensor([0, 1, 2, 0])
        log_probs, values, entropy = agent.evaluate_actions(sample_observations, chosen_actions)

        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert entropy.shape == (4,)

    def test_shared_encoder(self, sample_observations: dict[str, torch.Tensor]) -> None:
        encoder_config = EncoderConfig()
        shared_encoder = FeatureEncoder(encoder_config)
        config = SymbolAgentConfig(encoder_config=encoder_config)

        agent1 = SymbolAgent(config, shared_encoder=shared_encoder)
        agent2 = SymbolAgent(config, shared_encoder=shared_encoder)

        assert agent1.encoder is agent2.encoder
        assert agent1.owns_encoder is False
        assert agent2.owns_encoder is False

        agent1.eval()
        agent2.eval()
        actions1, _, _ = agent1(sample_observations)
        actions2, _, _ = agent2(sample_observations)
        assert actions1.shape == actions2.shape

    def test_parameter_count(self, agent: SymbolAgent) -> None:
        total_params = sum(param.numel() for param in agent.parameters())
        assert total_params < 10_000_000

    def test_gradient_flow(
        self, agent: SymbolAgent, sample_observations: dict[str, torch.Tensor]
    ) -> None:
        agent.train()
        agent.zero_grad()

        actions, log_probs, values = agent(sample_observations)
        loss = (log_probs + values).mean()
        loss.backward()

        gradless = [name for name, p in agent.named_parameters() if p.requires_grad and p.grad is None]
        assert not gradless, f"Missing gradients for parameters: {gradless}"
