"""Run hil_pusht_env with Xbox teleoperation enabled."""

from __future__ import annotations

import argparse

import numpy as np

from hil_pusht_env import PushTEnv, PushTEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PushT with Xbox teleoperation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for first reset.")
    parser.add_argument(
        "--render-size",
        type=int,
        default=1024,
        help="Render resolution in pixels (square).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode before truncation.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.9,
        help="Coverage threshold required for success.",
    )
    parser.add_argument(
        "--success-consecutive-steps",
        type=int,
        default=1,
        help="Consecutive successful steps required to terminate.",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=51.2,
        help="Pixels corresponding to a relative action magnitude of 1.0.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 means run forever).",
    )
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Use random policy actions when teleop is inactive (default is zero action).",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=20,
        help="Print status every N steps (0 disables).",
    )
    parser.add_argument(
        "--disable-overlays",
        action="store_true",
        help="Disable visual overlays.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = PushTEnvConfig(
        render_size=args.render_size,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        success_consecutive_steps=args.success_consecutive_steps,
        delta_scale_pixels=args.delta_scale,
        enable_teleop=True,
        teleop_override=True,
        enable_render=True,
        enable_overlays=not args.disable_overlays,
    )
    env = PushTEnv(config=config)

    print("Starting Xbox teleop demo.")
    print("Controls: hold LB to teleoperate, right stick to move, RT to boost speed.")
    print("Press Ctrl+C in terminal to quit.")

    episode_idx = 0
    step_idx = 0

    try:
        _, info = env.reset(seed=args.seed)
        print(f"Episode {episode_idx + 1} reset.")
        _ = info

        while True:
            if args.random_policy:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = np.zeros(2, dtype=np.float32)

            _, reward, terminated, truncated, info = env.step(action)
            env.render(mode="human")
            step_idx += 1

            if args.status_every > 0 and (step_idx % args.status_every == 0):
                coverage = float(info.get("coverage", 0.0))
                source = info.get("action_source", "policy")
                streak = int(info.get("success_streak", 0))
                print(
                    f"step={step_idx} reward={reward:.3f} coverage={coverage:.3f} "
                    f"source={source} success_streak={streak}"
                )

            if terminated or truncated:
                episode_idx += 1
                coverage = float(info.get("coverage", 0.0))
                term_type = "terminated" if terminated else "truncated"
                print(
                    f"Episode {episode_idx} {term_type} at step={step_idx}, "
                    f"coverage={coverage:.3f}"
                )
                if args.episodes > 0 and episode_idx >= args.episodes:
                    break

                _, info = env.reset()
                step_idx = 0
                print(f"Episode {episode_idx + 1} reset.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()

    return 0

