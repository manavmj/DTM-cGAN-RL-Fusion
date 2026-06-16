"""
tests/conftest.py
------------------
Shared pytest fixtures for the rl_gan_fusion test suite.

Fixtures:
    model_cfg   — minimal model config dict (overrides for fast tests)
    train_cfg   — minimal training config dict
    data_cfg    — minimal data config dict
    device      — "cpu" (tests always run on CPU)
    batch_rgb   — (2, 3, 64, 64) synthetic RGB batch
    batch_th    — (2, 1, 64, 64) synthetic thermal batch
    batch_li    — (2, 1, 64, 64) synthetic lidar batch
    dummy_batch — dict{rgb, thermal, lidar} using above tensors
    dummy_state — (2, 1286) synthetic state vector
    dummy_action — dict{branch (2,), weights (2,3)}
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def model_cfg() -> dict:
    """Minimal config for fast unit tests — small dimensions."""
    return {
        "encoder":               {"backbone": "resnet18", "pretrained": False, "embed_dim": 32},
        "confidence_estimator":  {"hidden_dim": 16, "dropout": 0.0},
        "scene_graph":           {"node_feat_dim": 32, "hidden_dim": 32, "num_gnn_layers": 1,
                                  "num_heads": 2, "graph_embed_dim": 32},
        "knowledge_bank":        {"num_slots": 8, "key_dim": 32, "value_dim": 32,
                                  "retrieval_heads": 2},
        "state_builder":         {"state_dim": 32*3 + 3 + 32 + 32 + 5},  # = 170
        "ppo":                   {"state_dim": 170, "action_discrete_n": 3,
                                  "action_continuous_n": 3,
                                  "actor_hidden": [64, 32], "critic_hidden": [64, 32],
                                  "dropout": 0.0, "log_std_init": -0.5},
        "generator": {
            "stem_channels": 16, "base_channels": 16, "out_channels": 3,
            "deep_fusion_block":  {"channels": [16, 32], "use_se": False},
            "light_fusion_block": {"channels": [8, 16], "depthwise": True},
            "cross_attention":    {"embed_dim": 32, "num_heads": 2, "ff_dim": 64, "dropout": 0.0},
        },
        "discriminator": {
            "fusion_quality_critic":    {"channels": [16, 32], "use_spectral_norm": False},
            "latency_critic":           {"hidden": [32, 16, 8]},
        },
        "loss_weights":   {"lambda_adv": 1.0, "lambda_fusion": 1.0,
                           "lambda_latency": 0.5},
        "reward_weights": {"alpha": 1.0, "gamma": 0.5},
    }


@pytest.fixture
def train_cfg() -> dict:
    return {
        "batch_size": 2, "rollout_steps": 4, "ppo_epochs": 1,
        "ppo_minibatch_size": 2, "num_epochs": 1, "gan_warmup_epochs": 0,
        "num_workers": 0, "pin_memory": False, "use_amp": False,
        "ppo": {"gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
                "vf_coef": 0.5, "ent_coef": 0.01, "max_grad_norm": 0.5},
    }


@pytest.fixture
def batch_rgb() -> torch.Tensor:
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def batch_th() -> torch.Tensor:
    return torch.randn(2, 1, 64, 64)


@pytest.fixture
def batch_li() -> torch.Tensor:
    return torch.randn(2, 1, 64, 64)


@pytest.fixture
def dummy_batch(batch_rgb, batch_th, batch_li) -> dict:
    return {"rgb": batch_rgb, "thermal": batch_th, "lidar": batch_li}


@pytest.fixture
def dummy_action() -> dict:
    return {
        "branch":  torch.tensor([0, 1], dtype=torch.long),
        "weights": torch.ones(2, 3) / 3.0,
    }
