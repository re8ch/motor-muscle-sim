from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .controllers import FixedOscillator, ResidualPolicy, TeacherController
from .env import MotorMuscleEnv


@dataclass(slots=True)
class EpisodeResult:
    seed: int
    controller: str
    survived: bool
    survival_time_s: float
    max_tilt_rad: float
    max_com_offset_m: float
    energy_j: float
    peak_temperature_c: float
    mean_action_delta: float
    realtime_factor: float


def controller_action(
    env: MotorMuscleEnv,
    controller_name: str,
    oscillator: FixedOscillator,
    teacher: TeacherController,
    policy: ResidualPolicy | None,
) -> np.ndarray:
    obs = env.observe()
    base = oscillator.command(float(obs["time"]))
    if controller_name == "teacher":
        signed = teacher.command(obs)
        allocation = np.zeros(len(signed) * 2, dtype=np.float32)
    elif controller_name == "oscillator":
        signed = base
        allocation = np.zeros(len(signed) * 2, dtype=np.float32)
    else:
        if policy is None:
            raise FileNotFoundError("neural controller requires a trained checkpoint")
        augmented = np.concatenate([env.flatten_observation(obs), oscillator.state(float(obs["time"]))])
        output = policy(augmented, base)
        signed, allocation = output.signed_drive, output.allocation_bias
    return np.concatenate([signed, allocation]).astype(np.float32)


def run_episode(
    config: ExperimentConfig,
    seed: int,
    policy: ResidualPolicy | None = None,
) -> EpisodeResult:
    env = MotorMuscleEnv(config)
    env.reset(seed)
    oscillator = FixedOscillator(env.dof_names)
    teacher = TeacherController(env.dof_names)
    tilts: list[float] = []
    offsets: list[float] = []
    deltas: list[float] = []
    done = False
    info: dict[str, float] = {}
    while not done:
        action = controller_action(env, config.controller, oscillator, teacher, policy)
        _, _, terminated, truncated, info = env.step(action)
        tilts.append(info["torso_tilt_rad"])
        offsets.append(info["com_offset_m"])
        deltas.append(info["action_delta"])
        done = terminated or truncated
    return EpisodeResult(
        seed=seed,
        controller=config.controller,
        survived=not terminated,
        survival_time_s=info["time"],
        max_tilt_rad=max(tilts, default=0),
        max_com_offset_m=max(offsets, default=0),
        energy_j=info["energy_j"],
        peak_temperature_c=info["peak_temperature_c"],
        mean_action_delta=float(np.mean(deltas)) if deltas else 0,
        realtime_factor=info["realtime_factor"],
    )

