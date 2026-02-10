"""Xbox teleoperation policies for hil_pusht_env."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class XboxDeltaTeleop:
    """Xbox policy returning relative delta actions in ``[-1, 1]^2``.

    Mapping:
    - LB (button 4): hold to enable teleoperation
    - Right stick (axes 3, 4): direction
    - RT (axis 5): variable speed gain
    """

    def __init__(
        self,
        *,
        stick_side: str = "right",
        deadzone: float = 0.1,
        enable_button: int = 4,
        speed_boost_trigger: int = 5,
        speed_gain: float = 2.5,
        base_gain: float = 0.35,
        enable_pygame: bool = True,
    ) -> None:
        self.deadzone = float(deadzone)
        self.enable_button = int(enable_button)
        self.speed_boost_trigger = int(speed_boost_trigger)
        self.speed_gain = float(speed_gain)
        self.base_gain = float(base_gain)

        if stick_side == "left":
            self.axis_x = 0
            self.axis_y = 1
        elif stick_side == "right":
            self.axis_x = 3
            self.axis_y = 4
        else:
            raise ValueError(f"Invalid stick_side: {stick_side!r}")

        self.joystick: Any | None = None
        self._pump_events: Any | None = None

        if enable_pygame:
            self._init_controller()

    def _init_controller(self) -> None:
        try:
            import pygame

            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= 0:
                logger.warning("No Xbox controller detected")
                return
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self._pump_events = pygame.event.pump
            logger.info("Xbox controller initialized: %s", self.joystick.get_name())
        except Exception as exc:
            logger.warning("Failed to initialize Xbox controller: %s", exc)
            self.joystick = None
            self._pump_events = None

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) <= self.deadzone:
            return 0.0
        scaled = (abs(value) - self.deadzone) / max(1e-6, (1.0 - self.deadzone))
        return float(np.sign(value) * scaled)

    def get_action(self, observation: dict | None = None) -> tuple[np.ndarray | None, bool]:
        """Return `(action, active)` where action is a relative delta in `[-1, 1]^2`."""
        _ = observation
        if self.joystick is None:
            return None, False

        try:
            if self._pump_events is not None:
                self._pump_events()

            if self.enable_button >= self.joystick.get_numbuttons():
                return None, False
            if not bool(self.joystick.get_button(self.enable_button)):
                return None, False

            raw_x = float(self.joystick.get_axis(self.axis_x))
            raw_y = float(self.joystick.get_axis(self.axis_y))
            x = self._apply_deadzone(raw_x)
            y = self._apply_deadzone(raw_y)

            speed_multiplier = 1.0
            if self.speed_boost_trigger < self.joystick.get_numaxes():
                trigger = float(self.joystick.get_axis(self.speed_boost_trigger))
                trigger01 = (trigger + 1.0) / 2.0
                speed_multiplier = 1.0 + trigger01 * (self.speed_gain - 1.0)

            delta = np.array([x, y], dtype=np.float32) * self.base_gain * speed_multiplier
            delta = np.clip(delta, -1.0, 1.0)
            return delta, True
        except Exception as exc:
            logger.warning("Xbox read failed: %s", exc)
            return None, False

    def close(self) -> None:
        if self.joystick is None:
            return
        try:
            self.joystick.quit()
        except Exception:
            pass
