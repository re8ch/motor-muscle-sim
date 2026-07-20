from __future__ import annotations

from dataclasses import asdict
import time

import mujoco
import numpy as np

from .config import ExperimentConfig
from .controllers import signed_to_antagonists
from .model import joint_dofs, load_model
from .motors import DenseMotorArray, MotorDiagnostics


def _quat_tilt(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat)
    matrix = matrix.reshape(3, 3)
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[2, 1], matrix[2, 2]))
    return np.array([roll, pitch], dtype=np.float32)


class MotorMuscleEnv:
    """Continuous-control MuJoCo environment with a vectorized dense motor layer."""

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()
        self.config.validate()
        self.model = load_model(self.config.physics_dt)
        self.data = mujoco.MjData(self.model)
        dofs = joint_dofs(self.model)
        self.dof_names = [name for name, _ in dofs]
        self.dof_indices = np.array([index for _, index in dofs], dtype=np.int32)
        self.qpos_indices = np.array(
            [
                int(self.model.jnt_qposadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )])
                for name in self.dof_names
            ],
            dtype=np.int32,
        )
        self.motors = DenseMotorArray(
            self.config.motor_count,
            self.dof_indices,
            self.config.seed,
            self.config.parameter_variation,
        )
        self.pelvis_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.left_foot_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom"
        )
        self.right_foot_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom"
        )
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.rng = np.random.default_rng(self.config.seed)
        self.initial_pelvis_height = 1.0
        self.previous_action = np.zeros(len(self.dof_indices), dtype=np.float32)
        self.action_delay: list[np.ndarray] = []
        self.last_motor_diagnostics = MotorDiagnostics(0, 24, 24, 1, 0)
        self.wall_start = 0.0
        self.sim_start = 0.0

    @property
    def action_size(self) -> int:
        return len(self.dof_indices) * 3

    @property
    def observation_size(self) -> int:
        return len(self.flatten_observation(self.observe()))

    def reset(
        self,
        seed: int | None = None,
        config: ExperimentConfig | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if config is not None:
            self.__init__(config)
        seed = self.config.seed if seed is None else seed
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.95
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        # Small randomized initial condition prevents a trivially symmetric benchmark.
        self.data.qpos[self.qpos_indices] = self.rng.normal(0, 0.003, len(self.qpos_indices))
        mujoco.mj_forward(self.model, self.data)
        self.initial_pelvis_height = float(self.data.xpos[self.pelvis_body, 2])
        self.motors.reset(seed + 101, self.config.failure_ratio, self.config.concentrated_failure)
        self.previous_action.fill(0)
        self.action_delay = [
            np.zeros(self.action_size, dtype=np.float32)
            for _ in range(self.config.control_delay_steps)
        ]
        self.wall_start = time.perf_counter()
        self.sim_start = self.data.time
        obs = self.observe()
        return self.flatten_observation(obs), self._info(obs, 0.0)

    def _foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(2, dtype=np.float32)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.floor_geom not in pair:
                continue
            if self.left_foot_geom in pair:
                contacts[0] = 1
            if self.right_foot_geom in pair:
                contacts[1] = 1
        return contacts

    def observe(self) -> dict[str, np.ndarray | float]:
        quat = self.data.xquat[self.pelvis_body]
        group_temp, group_available = self.motors.group_summary()
        obs: dict[str, np.ndarray | float] = {
            "joint_position": self.data.qpos[self.qpos_indices].astype(np.float32).copy(),
            "joint_velocity": self.data.qvel[self.dof_indices].astype(np.float32).copy(),
            "tilt": _quat_tilt(quat),
            "root_position": self.data.qpos[:2].astype(np.float32).copy(),
            "root_velocity": self.data.qvel[:3].astype(np.float32).copy(),
            "root_angular_velocity": self.data.qvel[3:6].astype(np.float32).copy(),
            "foot_contact": self._foot_contacts(),
            "group_temperature": group_temp,
            "group_availability": group_available,
            "pelvis_height": float(self.data.xpos[self.pelvis_body, 2]),
            "time": float(self.data.time),
        }
        if self.config.sensor_noise:
            for key in ("joint_position", "joint_velocity", "tilt", "root_velocity"):
                value = np.asarray(obs[key])
                obs[key] = value + self.rng.normal(0, self.config.sensor_noise, value.shape)
        return obs

    @staticmethod
    def flatten_observation(obs: dict[str, np.ndarray | float]) -> np.ndarray:
        keys = (
            "joint_position", "joint_velocity", "tilt", "root_position", "root_velocity",
            "root_angular_velocity", "foot_contact", "group_temperature",
            "group_availability",
        )
        normalized: list[np.ndarray] = []
        for key in keys:
            value = np.atleast_1d(obs[key]).astype(np.float32)
            if key == "group_temperature":
                value = (value - 24.0) / 60.0
            normalized.append(value)
        return np.concatenate(normalized)

    def step(
        self, continuous_action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action = np.asarray(continuous_action, dtype=np.float32)
        if action.shape != (self.action_size,):
            raise ValueError(f"expected action shape {(self.action_size,)}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must be finite")
        action = np.clip(action, -1, 1)
        if self.action_delay:
            self.action_delay.append(action.copy())
            action = self.action_delay.pop(0)
        n = len(self.dof_indices)
        signed = action[:n]
        allocation = action[n:]
        group_command = signed_to_antagonists(signed)
        action_delta = float(np.mean(np.square(signed - self.previous_action)))
        self.previous_action[:] = signed

        for _ in range(self.config.control_stride):
            torque, self.last_motor_diagnostics = self.motors.step(
                group_command,
                allocation,
                self.data.qvel[self.dof_indices],
                self.config.physics_dt,
            )
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[self.dof_indices] = torque
            if (
                self.config.impulse_time_s is not None
                and self.config.impulse_time_s <= self.data.time
                < self.config.impulse_time_s + self.config.impulse_duration_s
            ):
                self.data.xfrc_applied[self.pelvis_body, 0] = self.config.impulse_force_n
            else:
                self.data.xfrc_applied[self.pelvis_body] = 0
            mujoco.mj_step(self.model, self.data)

        obs = self.observe()
        tilt = np.asarray(obs["tilt"])
        height = float(obs["pelvis_height"])
        fallen = (
            height < self.initial_pelvis_height * 0.75
            or float(np.max(np.abs(tilt))) > np.deg2rad(45)
        )
        truncated = self.data.time >= self.config.duration_s
        reward = 1.0 - 1.4 * float(tilt @ tilt) - 0.02 * action_delta
        if fallen:
            reward -= 10.0
        info = self._info(obs, action_delta)
        return self.flatten_observation(obs), reward, fallen, truncated, info

    def _info(
        self, obs: dict[str, np.ndarray | float], action_delta: float
    ) -> dict[str, float]:
        elapsed_wall = max(1e-9, time.perf_counter() - self.wall_start) if self.wall_start else 1e-9
        tilt = np.asarray(obs["tilt"])
        return {
            "time": float(self.data.time),
            "pelvis_height": float(obs["pelvis_height"]),
            "torso_tilt_rad": float(np.linalg.norm(tilt)),
            "com_offset_m": float(np.linalg.norm(self.data.subtree_com[0, :2])),
            "foot_contacts": float(np.sum(np.asarray(obs["foot_contact"]))),
            "energy_j": self.last_motor_diagnostics.energy_j,
            "peak_temperature_c": self.last_motor_diagnostics.peak_temperature_c,
            "mean_temperature_c": self.last_motor_diagnostics.mean_temperature_c,
            "saturated_ratio": self.last_motor_diagnostics.saturated_ratio,
            "action_delta": action_delta,
            "realtime_factor": float((self.data.time - self.sim_start) / elapsed_wall),
        }

    def metadata(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "dof_names": self.dof_names,
            "action_size": self.action_size,
            "observation_size": self.observation_size,
            "model_bodies": int(self.model.nbody),
        }
