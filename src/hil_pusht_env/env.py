"""PushT environment with relative-action control and built-in teleoperation."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
from gymnasium import spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

from .rendering import apply_render_overlays
from .teleop import XboxDeltaTeleop
from .types import ActionSource, PushTEnvConfig, TeleopPolicy

positive_y_is_up: bool = False


def to_pygame(p: tuple[float, float], surface: pygame.Surface) -> tuple[int, int]:
    """Convert pymunk coordinates to pygame coordinates."""
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    return round(p[0]), round(p[1])


def _light_color(color: SpaceDebugColor) -> SpaceDebugColor:
    arr = np.minimum(
        1.2 * np.float32([color.r, color.g, color.b, color.a]),
        np.float32([255]),
    )
    return SpaceDebugColor(r=arr[0], g=arr[1], b=arr[2], a=arr[3])


class DrawOptions(pymunk.SpaceDebugDrawOptions):
    """Pymunk debug drawing options targeting a pygame surface."""

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        super().__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(
            self.surface,
            _light_color(fill_color).as_int(),
            p,
            max(1, round(radius - 4)),
            0,
        )

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        _ = outline_color
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r <= 2:
            return

        orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
        if orthog[0] == 0 and orthog[1] == 0:
            return
        scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
        orthog[0] = round(orthog[0] * scale)
        orthog[1] = round(orthog[1] * scale)
        points = [
            (p1[0] - orthog[0], p1[1] - orthog[1]),
            (p1[0] + orthog[0], p1[1] + orthog[1]),
            (p2[0] + orthog[0], p2[1] + orthog[1]),
            (p2[0] - orthog[0], p2[1] - orthog[1]),
        ]
        pygame.draw.polygon(self.surface, fill_color.as_int(), points)

    def draw_polygon(
        self,
        verts: Sequence[tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        _ = outline_color
        _ = radius
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]
        pygame.draw.polygon(self.surface, _light_color(fill_color).as_int(), ps)

        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            self.draw_fat_segment(a, b, 2, fill_color, fill_color)

    def draw_dot(self, size: float, pos: tuple[float, float], color: SpaceDebugColor) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def _pymunk_to_shapely(body: pymunk.Body, shapes: Sequence[pymunk.Shape]) -> sg.MultiPolygon:
    geoms: list[sg.Polygon] = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    return sg.MultiPolygon(geoms)


class PushTEnv(gym.Env[dict[str, Any], np.ndarray]):
    """PushT environment with relative delta action control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        config: PushTEnvConfig | None = None,
        teleop_policy: TeleopPolicy | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PushTEnvConfig()

        self.window_size = 512
        self.render_size = int(self.config.render_size)
        self.sim_hz = int(self.config.sim_hz)
        self.control_hz = int(self.config.control_hz)

        self.k_p = 100.0
        self.k_v = 20.0

        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.screen: pygame.Surface | None = None

        self.space: pymunk.Space | None = None
        self.agent: pymunk.Body | None = None
        self.block: pymunk.Body | None = None

        self.goal_pose = np.asarray(self.config.goal_pose, dtype=np.float64)
        self.goal_color = pygame.Color("LightGreen")

        self.current_step = 0
        self.success_streak = 0
        self.last_coverage = 0.0

        self.last_action_source: ActionSource = "policy"
        self.last_applied_action = np.zeros(2, dtype=np.float32)
        self.last_action_start_pos: np.ndarray | None = None

        self.n_contact_points: int = 0
        self._seed: int | None = None

        if teleop_policy is not None:
            self.teleop_policy = teleop_policy
        elif self.config.enable_teleop:
            self.teleop_policy = XboxDeltaTeleop()
        else:
            self.teleop_policy = None

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=np.zeros(5, dtype=np.float32),
                    high=np.full(5, 512.0, dtype=np.float32),
                    shape=(5,),
                    dtype=np.float32,
                ),
                "applied_action": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "action_source": spaces.Text(max_length=6),
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3, self.render_size, self.render_size),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.seed()

    def seed(self, seed: int | None = None) -> None:
        """Set the random seed used by reset sampling."""
        if seed is None:
            seed = int(np.random.randint(0, 65536))
        self._seed = int(seed)
        self.np_random = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment state."""
        if seed is not None:
            self.seed(seed)

        self._setup()
        self.current_step = 0
        self.success_streak = 0

        reset_state = None
        if options is not None:
            reset_state = options.get("reset_to_state")

        if reset_state is None:
            rng = self.np_random
            ax = rng.uniform(*self.config.agent_init_range[0])
            ay = rng.uniform(*self.config.agent_init_range[1])
            bx = rng.uniform(*self.config.block_init_range[0])
            by = rng.uniform(*self.config.block_init_range[1])
            ba = rng.uniform(*self.config.block_angle_range)
            reset_state = np.array([ax, ay, bx, by, ba], dtype=np.float64)

        self._set_state(np.asarray(reset_state, dtype=np.float64))

        self.last_action_source = "policy"
        self.last_applied_action = np.zeros(2, dtype=np.float32)
        self.last_action_start_pos = None
        self.last_coverage = float(self._compute_coverage())

        obs = self._get_obs()
        info = self._get_info()
        info["coverage"] = self.last_coverage
        info["success_streak"] = int(self.success_streak)
        info["action_source"] = self.last_action_source
        info["teleop_active"] = False
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Run one simulation step with a relative delta action."""
        if action is None:
            raise ValueError("Action must be a 2D relative delta; got None")

        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action_arr.shape}")

        action_arr = np.clip(action_arr, -1.0, 1.0)
        applied_action = action_arr.copy()
        action_source: ActionSource = "policy"

        teleop_active = False
        if self.teleop_policy is not None:
            teleop_action, teleop_active = self.teleop_policy.get_action(None)
            if (
                teleop_active
                and teleop_action is not None
                and self.config.teleop_override
            ):
                teleop_arr = np.asarray(teleop_action, dtype=np.float32)
                if teleop_arr.shape != (2,):
                    raise ValueError(
                        f"Teleop policy must produce shape (2,), got {teleop_arr.shape}"
                    )
                applied_action = np.clip(teleop_arr, -1.0, 1.0)
                action_source = "teleop"

        assert self.agent is not None
        assert self.space is not None

        self.last_action_start_pos = np.array(self.agent.position, dtype=np.float32)
        self.last_applied_action = applied_action.astype(np.float32)
        self.last_action_source = action_source

        absolute_target = self.last_action_start_pos + (
            applied_action.astype(np.float64) * float(self.config.delta_scale_pixels)
        )
        absolute_target = np.clip(absolute_target, 0.0, float(self.window_size))

        dt = 1.0 / float(self.sim_hz)
        n_steps = self.sim_hz // self.control_hz
        self.n_contact_points = 0

        for _ in range(n_steps):
            acceleration = self.k_p * (absolute_target - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * dt
            self.space.step(dt)

        coverage = float(self._compute_coverage())
        self.last_coverage = coverage
        self.current_step += 1

        if coverage >= float(self.config.success_threshold):
            self.success_streak += 1
        else:
            self.success_streak = 0

        terminated = self.success_streak >= int(self.config.success_consecutive_steps)
        truncated = (self.current_step >= int(self.config.max_steps)) and not terminated

        reward = float(np.clip(coverage, 0.0, 1.0))

        obs = self._get_obs()
        info = self._get_info()
        info["coverage"] = coverage
        info["success_streak"] = int(self.success_streak)
        info["action_source"] = action_source
        info["teleop_active"] = bool(teleop_active)

        return obs, reward, bool(terminated), bool(truncated), info

    def render(self, mode: str | None = None) -> np.ndarray | None:
        """Render environment frame with overlays."""
        if mode is None:
            mode = "rgb_array"
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {mode!r}")

        if mode == "human" and not self.config.enable_render:
            raise RuntimeError("Human rendering is disabled by config.enable_render=False")

        frame_rgb = self._render_frame(include_overlays=True)

        if mode == "human":
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.render_size, self.render_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, axes=(1, 0, 2)))
            assert self.window is not None
            self.window.blit(surface, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            assert self.clock is not None
            self.clock.tick(self.metadata["render_fps"])
            return None

        return frame_rgb

    def close(self) -> None:
        """Close pygame and teleop resources."""
        if self.teleop_policy is not None:
            self.teleop_policy.close()

        if self.window is not None:
            pygame.display.quit()
        if pygame.get_init():
            pygame.quit()

    def _get_obs(self) -> dict[str, Any]:
        assert self.agent is not None
        assert self.block is not None

        state = np.array(
            [
                float(self.agent.position[0]),
                float(self.agent.position[1]),
                float(self.block.position[0]),
                float(self.block.position[1]),
                float(self.block.angle % (2.0 * np.pi)),
            ],
            dtype=np.float32,
        )

        frame = self._render_frame(include_overlays=False)
        image = np.moveaxis(frame.astype(np.float32) / 255.0, -1, 0)

        return {
            "state": state,
            "applied_action": self.last_applied_action.astype(np.float32),
            "action_source": self.last_action_source,
            "image": image,
        }

    def _render_frame(self, include_overlays: bool) -> np.ndarray:
        if not pygame.get_init():
            pygame.init()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        assert self.block is not None
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface)
                for v in shape.get_vertices()
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        assert self.space is not None
        self.space.debug_draw(draw_options)

        frame_rgb = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        frame_rgb = cv2.resize(
            frame_rgb,
            (self.render_size, self.render_size),
            interpolation=cv2.INTER_AREA,
        )

        if include_overlays and self.config.enable_overlays:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_bgr = apply_render_overlays(
                frame_bgr,
                current_step=self.current_step,
                max_steps=int(self.config.max_steps),
                coverage=self.last_coverage,
                teleop_active=(self.last_action_source == "teleop"),
                action_start_xy=self.last_action_start_pos,
                applied_action=self.last_applied_action,
                delta_scale_pixels=float(self.config.delta_scale_pixels),
                window_size=float(self.window_size),
            )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        return frame_rgb

    def _compute_coverage(self) -> float:
        assert self.block is not None

        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = _pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = _pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        if goal_area <= 0:
            return 0.0
        return float(intersection_area / goal_area)

    def _get_goal_pose_body(self, pose: np.ndarray) -> pymunk.Body:
        mass = 1.0
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = float(pose[2])
        return body

    def _get_info(self) -> dict[str, Any]:
        assert self.agent is not None
        assert self.block is not None

        n_steps = max(1, self.sim_hz // self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))

        return {
            "pos_agent": np.array(self.agent.position, dtype=np.float32),
            "vel_agent": np.array(self.agent.velocity, dtype=np.float32),
            "block_pose": np.array(
                [float(self.block.position[0]), float(self.block.position[1]), float(self.block.angle)],
                dtype=np.float32,
            ),
            "goal_pose": self.goal_pose.astype(np.float32).copy(),
            "n_contacts": n_contact_points_per_step,
        }

    def _handle_collision(
        self,
        arbiter: pymunk.Arbiter,
        space: pymunk.Space,
        data: dict | None,
    ) -> None:
        _ = space
        _ = data
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state: np.ndarray) -> None:
        assert self.space is not None
        assert self.agent is not None
        assert self.block is not None

        state_arr = np.asarray(state, dtype=np.float64)
        if state_arr.shape != (5,):
            raise ValueError(f"State must have shape (5,), got {state_arr.shape}")

        self.agent.position = tuple(state_arr[:2])
        self.block.angle = float(state_arr[4])
        self.block.position = tuple(state_arr[2:4])
        self.space.step(1.0 / float(self.sim_hz))

    def _setup(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.0

        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        self.agent = self._add_circle((256, 400), 15)
        self.block = self._add_tee((256, 300), 0)

        collision_handler = self.space.add_collision_handler(0, 0)
        collision_handler.post_solve = self._handle_collision
        self.n_contact_points = 0

    def _add_segment(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        radius: float,
    ) -> pymunk.Segment:
        assert self.space is not None
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        return shape

    def _add_circle(self, position: tuple[float, float], radius: float) -> pymunk.Body:
        assert self.space is not None
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1.0
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def _add_tee(
        self,
        position: tuple[float, float],
        angle: float,
        scale: float = 30.0,
        color: str = "LightSlateGray",
    ) -> pymunk.Body:
        assert self.space is not None

        mass = 1.0
        length = 4.0
        vertices1 = [
            (-length * scale / 2.0, scale),
            (length * scale / 2.0, scale),
            (length * scale / 2.0, 0.0),
            (-length * scale / 2.0, 0.0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)

        vertices2 = [
            (-scale / 2.0, scale),
            (-scale / 2.0, length * scale),
            (scale / 2.0, length * scale),
            (scale / 2.0, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)

        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)

        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)

        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2.0
        body.position = position
        body.angle = angle
        body.friction = 1.0

        self.space.add(body, shape1, shape2)
        return body
