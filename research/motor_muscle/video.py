from __future__ import annotations

from pathlib import Path
import json

import imageio.v2 as imageio
import mujoco
import numpy as np

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


def render_mjx_trace(
    trace_path: str | Path,
    output: str | Path,
    fps: int = 30,
) -> Path:
    """Render a qpos/qvel trajectory exported by the GPU MJX backend."""
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = trace["qpos"]
        qvel = trace["qvel"]
        metadata = json.loads(str(trace["metadata"]))
    config = ExperimentConfig(**metadata["config"])
    env = MotorMuscleEnv(config)
    renderer = mujoco.Renderer(env.model, height=720, width=960)
    frames = []
    duration_s = len(qpos) / config.control_hz
    frame_times = np.arange(1 / fps, duration_s + 1e-9, 1 / fps)
    frame_indices = np.minimum(
        np.rint(frame_times * config.control_hz).astype(int) - 1,
        len(qpos) - 1,
    )
    for index in frame_indices:
        env.data.qpos[:] = qpos[index]
        env.data.qvel[:] = qvel[index]
        env.data.time = (index + 1) / config.control_hz
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data)
        frames.append(renderer.render())
    renderer.close()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)
    return path
