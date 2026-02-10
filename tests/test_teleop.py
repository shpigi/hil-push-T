from __future__ import annotations

import numpy as np

from hil_pusht_env import PushTEnv, PushTEnvConfig


class FakeTeleop:
    def __init__(self, action: np.ndarray, active: bool = True) -> None:
        self._action = np.asarray(action, dtype=np.float32)
        self._active = active

    def get_action(self, observation: dict | None = None) -> tuple[np.ndarray | None, bool]:
        _ = observation
        return self._action.copy(), self._active

    def close(self) -> None:
        return None


def test_teleop_override_sets_action_source() -> None:
    cfg = PushTEnvConfig(enable_render=False, enable_teleop=True, teleop_override=True)
    env = PushTEnv(config=cfg, teleop_policy=FakeTeleop(np.array([0.5, -0.5], dtype=np.float32)))
    try:
        env.reset(seed=0)
        obs, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

        assert obs["action_source"] == "teleop"
        assert np.allclose(obs["applied_action"], [0.5, -0.5])
        assert info["action_source"] == "teleop"
        assert info["teleop_active"] is True
    finally:
        env.close()


def test_teleop_inactive_keeps_policy_action() -> None:
    cfg = PushTEnvConfig(enable_render=False, enable_teleop=True, teleop_override=True)
    env = PushTEnv(config=cfg, teleop_policy=FakeTeleop(np.array([0.8, 0.8], dtype=np.float32), active=False))
    try:
        env.reset(seed=0)
        obs, _, _, _, info = env.step(np.array([0.1, -0.2], dtype=np.float32))

        assert obs["action_source"] == "policy"
        assert np.allclose(obs["applied_action"], [0.1, -0.2])
        assert info["teleop_active"] is False
    finally:
        env.close()
