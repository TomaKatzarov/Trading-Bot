"""Tests for weight initialization, sharing utilities, and SL transfer hooks."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from core.rl.policies.feature_encoder import EncoderConfig, FeatureEncoder
from core.rl.policies.symbol_agent import SymbolAgent, SymbolAgentConfig
from core.rl.policies.initialization import (
    he_normal_init,
    he_uniform_init,
    init_actor,
    init_critic,
    init_encoder,
    orthogonal_init,
    verify_initialization,
    xavier_normal_init,
    xavier_uniform_init,
)
from core.rl.policies.weight_sharing import (
    ParameterReport,
    SharedEncoderManager,
    calculate_savings_for_agents,
)
from core.rl.policies.sl_to_rl_transfer import (
    SLTransferWarning,
    create_sl_transfer_experiment,
    load_sl_checkpoint,
    transfer_sl_features_to_encoder,
)


def _layer_std(layer: nn.Linear) -> float:
    return float(layer.weight.data.std(unbiased=False).item())


class TestPrimitiveInitializers:
    def test_xavier_uniform_zero_mean(self) -> None:
        layer = nn.Linear(128, 64)
        xavier_uniform_init(layer)
        assert abs(float(layer.weight.data.mean().item())) < 0.1

    def test_xavier_uniform_variance(self) -> None:
        layer = nn.Linear(128, 64)
        xavier_uniform_init(layer)
        expected = math.sqrt(2.0 / (128 + 64))
        assert abs(_layer_std(layer) - expected) < 0.05

    def test_he_uniform_variance(self) -> None:
        layer = nn.Linear(256, 64)
        he_uniform_init(layer)
        expected = math.sqrt(2.0 / 256)
        assert abs(_layer_std(layer) - expected) < 0.05

    def test_orthogonal_is_orthonormal(self) -> None:
        layer = nn.Linear(64, 64)
        orthogonal_init(layer, gain=1.0)
        weight = layer.weight.detach()
        product = weight @ weight.t()
        identity = torch.eye(64, device=weight.device, dtype=weight.dtype)
        error = torch.norm(product - identity).item()
        assert error < 0.1

    def test_xavier_normal_bias_zeroed(self) -> None:
        layer = nn.Linear(64, 32)
        nn.init.constant_(layer.bias, 5.0)
        xavier_normal_init(layer)
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_he_normal_custom_gain_scaling(self) -> None:
        layer = nn.Linear(128, 32)
        he_normal_init(layer, gain=1.0)
        expected = 1.0 / math.sqrt(128)
        assert abs(_layer_std(layer) - expected) < 0.05

    def test_he_uniform_custom_gain_scaling(self) -> None:
        layer = nn.Linear(128, 32)
        he_uniform_init(layer, gain=1.0)
        expected = 1.0 / math.sqrt(128)
        assert abs(_layer_std(layer) - expected) < 0.05


class TestEncoderInitialization:
    @pytest.fixture
    def encoder(self) -> FeatureEncoder:
        torch.manual_seed(0)
        return FeatureEncoder(EncoderConfig())

    def test_init_encoder_xavier(self, encoder: FeatureEncoder) -> None:
        init_encoder(encoder, strategy="xavier_uniform", gain=1.0)
        first_linear = next(module for module in encoder.modules() if isinstance(module, nn.Linear))
        assert abs(float(first_linear.weight.data.mean().item())) < 0.1

    def test_verify_initialization_pass(self, encoder: FeatureEncoder) -> None:
        init_encoder(encoder, strategy="xavier_uniform", gain=1.0)
        result = verify_initialization(encoder, strategy="xavier_uniform", tolerance=0.2, gain=1.0)
        assert result["passed"], result["checks"]

    def test_verify_initialization_invalid_tolerance(self, encoder: FeatureEncoder) -> None:
        with pytest.raises(ValueError):
            verify_initialization(encoder, strategy="xavier_uniform", tolerance=0.0)

    def test_init_encoder_invalid_strategy(self, encoder: FeatureEncoder) -> None:
        with pytest.raises(ValueError):
            init_encoder(encoder, strategy="invalid")

    def test_verify_initialization_detects_nan_inf(self) -> None:
        layer = nn.Linear(16, 16)
        xavier_uniform_init(layer)
        with torch.no_grad():
            layer.weight[0, 0] = float("nan")
            layer.weight[1, 0] = float("inf")
        result = verify_initialization(layer, strategy="xavier_uniform", tolerance=0.2)
        assert not result["passed"]
        assert result["has_nan"] and result["has_inf"]
        assert any("NaN" in check or "Inf" in check for check in result["checks"])

    def test_verify_initialization_detects_mean_violation(self) -> None:
        layer = nn.Linear(8, 8)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        report = verify_initialization(layer, strategy="xavier_uniform", tolerance=0.05)
        assert not report["passed"]
        assert any("mean" in check for check in report["checks"])

    def test_verify_initialization_detects_std_violation(self) -> None:
        layer = nn.Linear(32, 16)
        with torch.no_grad():
            layer.weight.zero_()
        report = verify_initialization(layer, strategy="xavier_uniform", tolerance=0.1, gain=1.0)
        assert not report["passed"]
        assert any("std" in check for check in report["checks"])

    def test_verify_initialization_detects_he_std_violation(self) -> None:
        layer = nn.Linear(64, 32)
        with torch.no_grad():
            layer.weight.fill_(10.0)
        report = verify_initialization(layer, strategy="he_uniform", tolerance=0.1, gain=math.sqrt(2.0))
        assert not report["passed"]
        assert any("std" in check for check in report["checks"])

    def test_verify_initialization_detects_orthogonality_error(self) -> None:
        layer = nn.Linear(16, 16)
        with torch.no_grad():
            layer.weight.fill_(0.0)
        report = verify_initialization(layer, strategy="orthogonal", tolerance=0.05)
        assert not report["passed"]
        assert any("orthogonality" in check for check in report["checks"])


class TestActorCriticInitializers:
    def test_actor_output_small_gain(self) -> None:
        actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 7))
        init_actor(actor, strategy="orthogonal", gain=0.01)
        assert _layer_std(actor[-1]) < 0.1

    def test_critic_hidden_orthogonal(self) -> None:
        critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        init_critic(critic, strategy="orthogonal", gain=1.0)
        first = critic[0].weight.data
        identity = torch.eye(first.size(0))
        expected_gain = math.sqrt(2.0)
        error = torch.norm(first @ first.t() - (expected_gain ** 2) * identity).item()
        assert error < 1.0

    def test_actor_invalid_strategy_raises(self) -> None:
        actor = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        with pytest.raises(ValueError):
            init_actor(actor, strategy="invalid")

    def test_critic_non_default_strategy(self) -> None:
        critic = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        init_critic(critic, strategy="he_normal", gain=1.0, hidden_gain=math.sqrt(2.0))
        assert _layer_std(critic[-1]) < 1.0

    def test_critic_invalid_strategy_raises(self) -> None:
        critic = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
        with pytest.raises(ValueError):
            init_critic(critic, strategy="invalid")


class TestWeightSharingUtilities:
    @pytest.fixture
    def manager(self) -> SharedEncoderManager:
        config = EncoderConfig()
        encoder = FeatureEncoder(config)
        manager = SharedEncoderManager(encoder)
        agent_config = SymbolAgentConfig(encoder_config=config)
        for _ in range(3):
            agent = SymbolAgent(agent_config, shared_encoder=encoder)
            manager.register_agent(agent)
        return manager

    def test_verify_sharing(self, manager: SharedEncoderManager) -> None:
        results = manager.verify_sharing()
        assert results["all_share_encoder"]
        assert results["num_agents"] == 3

    def test_parameter_report(self, manager: SharedEncoderManager) -> None:
        report = manager.get_parameter_report()
        assert isinstance(report, ParameterReport)
        assert report.savings_percent > 60

    def test_calculate_savings_formula(self) -> None:
        total_with, total_without, savings = calculate_savings_for_agents(3_239_168, 66_824, 143)
        expected_with = 3_239_168 + 143 * 66_824
        expected_without = 143 * (3_239_168 + 66_824)
        assert total_with == expected_with
        assert total_without == expected_without
        expected_savings = 100.0 * (1 - (expected_with / expected_without))
        assert math.isclose(savings, expected_savings, rel_tol=1e-6)

    def test_register_agent_missing_encoder(self, manager: SharedEncoderManager) -> None:
        class BadAgent(nn.Module):
            pass

        with pytest.raises(AttributeError):
            manager.register_agent(BadAgent())

    def test_register_agent_wrong_encoder(self, manager: SharedEncoderManager) -> None:
        other_encoder = FeatureEncoder(EncoderConfig())
        config = SymbolAgentConfig(encoder_config=manager.encoder.config)
        agent = SymbolAgent(config, shared_encoder=other_encoder)
        with pytest.raises(ValueError):
            manager.register_agent(agent)

    def test_get_parameter_report_requires_agents(self) -> None:
        encoder = FeatureEncoder(EncoderConfig())
        manager = SharedEncoderManager(encoder)
        with pytest.raises(ValueError):
            manager.get_parameter_report()

    def test_print_report_outputs(self, manager: SharedEncoderManager, capsys: pytest.CaptureFixture[str]) -> None:
        manager.print_report()
        captured = capsys.readouterr()
        assert "WEIGHT SHARING PARAMETER REPORT" in captured.out

    def test_calculate_savings_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            calculate_savings_for_agents(10, 10, 0)
        with pytest.raises(ValueError):
            calculate_savings_for_agents(-1, 10, 5)
        with pytest.raises(ValueError):
            calculate_savings_for_agents(10, -1, 5)


class TestSLTransferInfrastructure:
    def test_load_sl_checkpoint_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_sl_checkpoint(tmp_path / "missing")

    def test_transfer_warns_and_counts(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "sl"
        checkpoint_dir.mkdir()
        torch.save({"encoder.layer.weight": torch.randn(2, 2)}, checkpoint_dir / "model.pt")

        encoder = nn.Sequential(nn.Linear(2, 2))

        with pytest.warns(SLTransferWarning):
            checkpoint = load_sl_checkpoint(checkpoint_dir)

        with pytest.warns(SLTransferWarning):
            transferred = transfer_sl_features_to_encoder(encoder, checkpoint)

        assert transferred >= 0

    def test_create_sl_transfer_experiment(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "sl"
        checkpoint_dir.mkdir()
        encoder = FeatureEncoder(EncoderConfig())
        torch.save(encoder.state_dict(), checkpoint_dir / "model.pt")

        with pytest.warns(SLTransferWarning):
            experiment = create_sl_transfer_experiment(encoder, checkpoint_dir, freeze=False)

        assert "baseline_encoder" in experiment
        assert "transfer_encoder" in experiment
        assert experiment["metadata"]["sl_checkpoint"] == str(checkpoint_dir)
        assert experiment["metadata"]["transferred_params"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])