from __future__ import annotations

from gymnasium.utils.env_checker import check_env

from hil_pusht_env import PushTEnv, PushTEnvConfig


def test_gymnasium_env_checker() -> None:
    env = PushTEnv(PushTEnvConfig(render_size=32, max_steps=10, enable_render=False, enable_teleop=False))
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
