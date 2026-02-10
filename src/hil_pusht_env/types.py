"""Public types for hil_pusht_env."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

ActionSource = Literal["policy", "teleop"]


class TeleopPolicy(Protocol):
    """Protocol for teleoperation policies used by PushTEnv."""

    def get_action(self, observation: dict | None = None) -> tuple[np.ndarray | None, bool]:
        """Return `(action, is_active)` where action is a delta in `[-1, 1]^2`."""

    def close(self) -> None:
        """Release resources used by the teleoperation policy."""


@dataclass(slots=True)
class PushTEnvConfig:
    """Configuration for :class:`hil_pusht_env.env.PushTEnv`."""

    render_size: int = 96
    max_steps: int = 500
    success_threshold: float = 0.95
    success_consecutive_steps: int = 1

    sim_hz: int = 100
    control_hz: int = 10
    delta_scale_pixels: float = 51.2

    enable_render: bool = True
    enable_overlays: bool = True

    enable_teleop: bool = False
    teleop_override: bool = True

    agent_init_range: tuple[tuple[float, float], tuple[float, float]] = ((50.0, 450.0), (50.0, 450.0))
    block_init_range: tuple[tuple[float, float], tuple[float, float]] = (
        (100.0, 400.0),
        (100.0, 400.0),
    )
    block_angle_range: tuple[float, float] = (-np.pi, np.pi)
    goal_pose: tuple[float, float, float] = field(default_factory=lambda: (256.0, 256.0, np.pi / 4.0))
