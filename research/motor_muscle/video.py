from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco

from .config import ExperimentConfig
from .controllers import FixedOscillator, ResidualPolicy, TeacherController
from .env import MotorMuscleEnv
from .runner import controller_action


def record_episode(
    output: str | Path,
    config: ExperimentConfig,
    policy: ResidualPolicy | None,
    fps: int = 30,
) -> Path:
    env = MotorMuscleEnv(config)
    env.reset(config.seed)
    oscillator = FixedOscillator(env.dof_names)
    teacher = TeacherController(env.dof_names)
    renderer = mujoco.Renderer(env.model, height=720, width=960)
    frames = []
    next_frame = 0.0
    done = False
    while not done:
        action = controller_action(env, config.controller, oscillator, teacher, policy)
        _, _, terminated, truncated, _ = env.step(action)
        if env.data.time >= next_frame:
            renderer.update_scene(env.data)
            frames.append(renderer.render())
            next_frame += 1 / fps
        done = terminated or truncated
    renderer.close()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)
    return path

