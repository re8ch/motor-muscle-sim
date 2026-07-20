from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ControllerOutput:
    signed_drive: np.ndarray
    allocation_bias: np.ndarray


class FixedOscillator:
    def __init__(self, dof_names: list[str], amplitude: float = 0.04) -> None:
        self.names = dof_names
        self.amplitude = amplitude

    def state(self, time_s: float) -> np.ndarray:
        phase = 2 * np.pi * 0.7 * time_s
        return np.array([np.sin(phase), np.cos(phase), np.sin(phase * 0.5), np.cos(phase * 0.5)], dtype=np.float32)

    def command(self, time_s: float) -> np.ndarray:
        osc = self.state(time_s)
        result = np.zeros(len(self.names), dtype=np.float32)
        for i, name in enumerate(self.names):
            if "ankle" in name:
                result[i] = self.amplitude * osc[0] * (-1 if "right" in name else 1)
            elif "hip" in name:
                result[i] = self.amplitude * 0.4 * osc[2] * (-1 if "right" in name else 1)
        return result


class TeacherController:
    def __init__(self, dof_names: list[str]) -> None:
        self.names = dof_names

    def command(self, observation: dict[str, np.ndarray | float]) -> np.ndarray:
        q = np.asarray(observation["joint_position"])
        qd = np.asarray(observation["joint_velocity"])
        roll, pitch = np.asarray(observation["tilt"])
        root_position = np.asarray(observation["root_position"])
        root_v = np.asarray(observation["root_velocity"])
        cmd = -3.0 * q - 0.32 * qd
        for i, name in enumerate(self.names):
            if "ankle_roll" in name:
                cmd[i] += -3.0 * roll + 12.0 * root_position[1] + 3.0 * root_v[1]
            elif "hip_roll" in name:
                cmd[i] += 0.0
            elif "ankle_pitch" in name:
                cmd[i] += -3.0 * pitch + 12.0 * root_position[0] + 3.0 * root_v[0]
            elif "hip_pitch" in name:
                cmd[i] += 0.0
            elif "spine_roll" in name:
                cmd[i] += -2.0 * roll
            elif "spine_pitch" in name:
                cmd[i] += -2.0 * pitch
        return np.clip(cmd, -1.0, 1.0).astype(np.float32)


def signed_to_antagonists(signed: np.ndarray) -> np.ndarray:
    out = np.empty(len(signed) * 2, dtype=np.float32)
    out[0::2] = np.clip(signed, 0, 1)
    out[1::2] = np.clip(-signed, 0, 1)
    return out


class ResidualPolicy:
    def __init__(self, observation_size: int, dof_count: int, checkpoint: str | None = None) -> None:
        import torch

        self.torch = torch
        self.model = torch.nn.Sequential(
            torch.nn.Linear(observation_size, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, dof_count + dof_count * 2),
            torch.nn.Tanh(),
        )
        self.mean = np.zeros(observation_size, dtype=np.float32)
        self.scale = np.ones(observation_size, dtype=np.float32)
        if checkpoint and Path(checkpoint).exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(payload["model"])
            self.mean = payload["mean"].numpy()
            self.scale = payload["scale"].numpy()
        self.model.eval()

    def __call__(self, flat_observation: np.ndarray, base: np.ndarray) -> ControllerOutput:
        x = (flat_observation - self.mean) / np.maximum(self.scale, 1e-5)
        with self.torch.no_grad():
            y = self.model(self.torch.from_numpy(x).float()).numpy()
        n = len(base)
        signed = np.clip(base + 0.35 * y[:n], -1, 1)
        return ControllerOutput(signed, y[n:])
