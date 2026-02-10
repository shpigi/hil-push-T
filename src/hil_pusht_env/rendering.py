"""Rendering overlays for PushT visualization."""

from __future__ import annotations

import cv2
import numpy as np


def _to_render_coords(position_xy: np.ndarray, render_size: int, window_size: float) -> tuple[int, int]:
    coord = (np.asarray(position_xy, dtype=np.float32) / float(window_size)) * float(render_size)
    return int(np.clip(coord[0], 0, render_size - 1)), int(np.clip(coord[1], 0, render_size - 1))


def draw_action_arrow(
    frame_bgr: np.ndarray,
    action_start_xy: np.ndarray | None,
    applied_action: np.ndarray | None,
    delta_scale_pixels: float,
    window_size: float,
) -> np.ndarray:
    """Draw an arrow for the currently applied action."""
    if action_start_xy is None or applied_action is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    render_size = min(h, w)
    start = np.asarray(action_start_xy, dtype=np.float32)
    delta = np.asarray(applied_action, dtype=np.float32) * float(delta_scale_pixels)
    end = start + delta

    start_px = _to_render_coords(start, render_size, window_size)
    end_px = _to_render_coords(end, render_size, window_size)

    cv2.arrowedLine(
        frame_bgr,
        start_px,
        end_px,
        color=(0, 255, 0),
        thickness=max(1, render_size // 64),
        tipLength=0.25,
    )
    return frame_bgr


def draw_time_to_termination_bar(frame_bgr: np.ndarray, current_step: int, max_steps: int) -> np.ndarray:
    """Draw a top bar showing remaining episode budget."""
    h, w = frame_bgr.shape[:2]
    padding = max(2, h // 96)
    bar_height = max(4, h // 24)

    if max_steps <= 0:
        ratio = 0.0
    else:
        ratio = np.clip(1.0 - (current_step / max_steps), 0.0, 1.0)

    x0, y0 = padding, padding
    x1, y1 = w - padding, padding + bar_height

    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color=(32, 32, 32), thickness=-1)
    fill_w = int((x1 - x0) * ratio)
    if fill_w > 0:
        cv2.rectangle(frame_bgr, (x0, y0), (x0 + fill_w, y1), color=(0, 200, 255), thickness=-1)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=1)
    return frame_bgr


def draw_coverage_bar(frame_bgr: np.ndarray, coverage: float) -> np.ndarray:
    """Draw a bottom bar showing current target coverage."""
    h, w = frame_bgr.shape[:2]
    padding = max(2, h // 96)
    bar_height = max(4, h // 24)

    coverage = float(np.clip(coverage, 0.0, 1.0))

    x0, y0 = padding, h - padding - bar_height
    x1, y1 = w - padding, h - padding

    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color=(32, 32, 32), thickness=-1)
    fill_w = int((x1 - x0) * coverage)
    if fill_w > 0:
        cv2.rectangle(frame_bgr, (x0, y0), (x0 + fill_w, y1), color=(255, 120, 0), thickness=-1)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=1)
    return frame_bgr


def draw_teleop_indicator(frame_bgr: np.ndarray, teleop_active: bool) -> np.ndarray:
    """Draw a magenta border around the frame when teleop is active."""
    if not teleop_active:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    thickness = max(3, h // 24)
    cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), color=(255, 0, 255), thickness=thickness)
    return frame_bgr


def apply_render_overlays(
    frame_bgr: np.ndarray,
    *,
    current_step: int,
    max_steps: int,
    coverage: float,
    teleop_active: bool,
    action_start_xy: np.ndarray | None,
    applied_action: np.ndarray | None,
    delta_scale_pixels: float,
    window_size: float,
) -> np.ndarray:
    """Apply all standard PushT overlays to a frame."""
    frame = frame_bgr.copy()
    frame = draw_time_to_termination_bar(frame, current_step=current_step, max_steps=max_steps)
    frame = draw_coverage_bar(frame, coverage=coverage)
    frame = draw_action_arrow(
        frame,
        action_start_xy=action_start_xy,
        applied_action=applied_action,
        delta_scale_pixels=delta_scale_pixels,
        window_size=window_size,
    )
    frame = draw_teleop_indicator(frame, teleop_active=teleop_active)
    return frame
