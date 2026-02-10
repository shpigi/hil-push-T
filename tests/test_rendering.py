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


def test_rgb_array_render_and_observation_image() -> None:
    env = PushTEnv(PushTEnvConfig(render_size=64, enable_render=False, enable_overlays=True))
    try:
        obs, _ = env.reset(seed=0)
        assert obs["image"].shape == (3, 64, 64)

        frame = env.render(mode="rgb_array")
        assert frame is not None
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_teleop_overlay_border_present() -> None:
    cfg = PushTEnvConfig(
        render_size=64,
        enable_render=False,
        enable_teleop=True,
        teleop_override=True,
        enable_overlays=True,
    )
    env = PushTEnv(config=cfg, teleop_policy=FakeTeleop(np.array([0.2, 0.0], dtype=np.float32)))
    try:
        env.reset(seed=0)
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        frame = env.render(mode="rgb_array")
        assert frame is not None

        top_row = frame[0]
        is_magenta = (top_row[:, 0] > 200) & (top_row[:, 1] < 80) & (top_row[:, 2] > 200)
        assert np.any(is_magenta)
    finally:
        env.close()
