from __future__ import annotations

import numpy as np
import pytest

from hil_pusht_env import PushTEnv, PushTEnvConfig


def make_env(**kwargs) -> PushTEnv:
    cfg = PushTEnvConfig(enable_render=False, enable_teleop=False, **kwargs)
    return PushTEnv(cfg)


def test_reset_observation_contract() -> None:
    env = make_env(render_size=64, max_steps=10)
    try:
        obs, info = env.reset(seed=123)

        assert set(obs.keys()) == {"state", "applied_action", "action_source", "image"}
        assert obs["state"].shape == (5,)
        assert obs["state"].dtype == np.float32
        assert obs["applied_action"].shape == (2,)
        assert obs["applied_action"].dtype == np.float32
        assert obs["action_source"] in {"policy", "teleop"}
        assert obs["image"].shape == (3, 64, 64)
        assert obs["image"].dtype == np.float32
        assert 0.0 <= float(obs["image"].min()) <= 1.0
        assert 0.0 <= float(obs["image"].max()) <= 1.0

        assert "coverage" in info
    finally:
        env.close()


def test_step_observation_and_reward_contract() -> None:
    env = make_env(render_size=64, max_steps=10)
    try:
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.array([0.1, -0.2], dtype=np.float32))

        assert obs["applied_action"].shape == (2,)
        assert obs["action_source"] in {"policy", "teleop"}
        assert 0.0 <= reward <= 1.0
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 0.0 <= info["coverage"] <= 1.0
    finally:
        env.close()


def test_invalid_action_shape_raises() -> None:
    env = make_env()
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="Expected action shape"):
            env.step(np.array([[0.0, 0.0]], dtype=np.float32))
    finally:
        env.close()


def test_consecutive_success_termination() -> None:
    env = make_env(success_threshold=0.0, success_consecutive_steps=2, max_steps=20)
    try:
        env.reset(seed=0)
        _, _, terminated1, truncated1, _ = env.step(np.zeros(2, dtype=np.float32))
        _, _, terminated2, truncated2, _ = env.step(np.zeros(2, dtype=np.float32))

        assert not terminated1
        assert not truncated1
        assert terminated2
        assert not truncated2
    finally:
        env.close()


def test_timeout_truncation() -> None:
    env = make_env(success_threshold=1.1, success_consecutive_steps=1, max_steps=3)
    try:
        env.reset(seed=0)
        term = trunc = False
        for _ in range(3):
            _, _, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
        assert not term
        assert trunc
    finally:
        env.close()
