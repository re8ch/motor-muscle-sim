from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MotorDiagnostics:
    energy_j: float
    peak_temperature_c: float
    mean_temperature_c: float
    active_ratio: float
    saturated_ratio: float


class DenseMotorArray:
    """Vectorized electrical/thermal state for independently varying motors."""

    def __init__(
        self,
        motor_count: int,
        dof_indices: np.ndarray,
        seed: int,
        variation: float = 0.08,
    ) -> None:
        self.count = motor_count
        self.dof_indices = np.asarray(dof_indices, dtype=np.int32)
        self.group_count = len(self.dof_indices) * 2
        rng = np.random.default_rng(seed)
        # Motors are stored contiguously by antagonist group. Their independent
        # parameter draws are unchanged, while the layout enables accelerator
        # backends to replace costly scatter reductions with reshape + sum.
        per_group = int(np.ceil(motor_count / self.group_count))
        self.group = np.repeat(
            np.arange(self.group_count, dtype=np.int32), per_group
        )[:motor_count]
        self.sign = np.where(self.group % 2 == 0, 1.0, -1.0).astype(np.float32)
        self.group_dof = self.group // 2
        # Equivalent geared torque constant. At 20k density each antagonist group
        # must retain enough aggregate ankle authority to balance a human-scale body.
        self.kt = (0.90 * (1 + rng.normal(0, variation, motor_count))).astype(np.float32)
        self.resistance0 = (2.1 * (1 + rng.normal(0, variation, motor_count))).astype(np.float32)
        self.moment_arm = (0.012 * (1 + rng.normal(0, variation * 0.5, motor_count))).astype(np.float32)
        self.current = np.zeros(motor_count, dtype=np.float32)
        self.temperature = np.full(motor_count, 24.0, dtype=np.float32)
        self.failed = np.zeros(motor_count, dtype=bool)
        self.energy_j = 0.0
        self.saturated = np.zeros(motor_count, dtype=bool)
        self._counts = np.maximum(
            1, np.bincount(self.group, minlength=self.group_count)
        ).astype(np.float32)

    def reset(
        self,
        seed: int,
        failure_ratio: float,
        concentrated: bool = False,
    ) -> None:
        self.current.fill(0)
        self.temperature.fill(24)
        self.failed.fill(False)
        self.energy_j = 0.0
        rng = np.random.default_rng(seed)
        failed_count = int(round(self.count * failure_ratio))
        if failed_count:
            if concentrated:
                left_groups = np.arange(self.group_count // 2, dtype=np.int32)
                candidates = np.flatnonzero(np.isin(self.group, left_groups))
                failed_count = min(failed_count, len(candidates))
                failed = rng.choice(candidates, failed_count, replace=False)
            else:
                failed = rng.choice(self.count, failed_count, replace=False)
            self.failed[failed] = True

    def group_summary(self) -> tuple[np.ndarray, np.ndarray]:
        temp_sum = np.bincount(
            self.group, weights=self.temperature, minlength=self.group_count
        )
        available = np.bincount(
            self.group, weights=(~self.failed).astype(np.float32), minlength=self.group_count
        )
        return (temp_sum / self._counts).astype(np.float32), (available / self._counts).astype(np.float32)

    def step(
        self,
        group_command: np.ndarray,
        allocation_bias: np.ndarray,
        joint_velocity: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, MotorDiagnostics]:
        group_command = np.clip(np.asarray(group_command, dtype=np.float32), 0, 1)
        allocation_bias = np.clip(np.asarray(allocation_bias, dtype=np.float32), -1, 1)
        temp_derate = np.clip((95.0 - self.temperature) / 20.0, 0.0, 1.0)
        health = (~self.failed).astype(np.float32) * temp_derate
        raw_weight = health * np.exp(0.35 * allocation_bias[self.group])
        weight_sum = np.bincount(self.group, weights=raw_weight, minlength=self.group_count)
        normalized = raw_weight * self._counts[self.group] / np.maximum(1e-6, weight_sum[self.group])
        voltage = 12.0 * group_command[self.group] * normalized * health
        resistance = self.resistance0 * (1 + 0.0039 * (self.temperature - 24.0))
        back_emf = 0.08 * joint_velocity[self.group_dof] * self.sign
        target_current = np.clip((voltage - back_emf) / resistance, 0.0, 4.0)
        alpha = 1.0 - np.exp(-dt / 0.012)
        self.current += (target_current - self.current) * alpha
        self.current[self.failed] = 0
        self.saturated = target_current >= 3.999

        motor_torque = self.kt * self.current * self.moment_arm * self.sign
        generalized = np.bincount(
            self.group_dof, weights=motor_torque, minlength=len(self.dof_indices)
        ).astype(np.float64)

        joule = self.current * self.current * resistance
        cooling = 0.42 * (self.temperature - 24.0)
        self.temperature += ((joule - cooling) * dt / 1.25).astype(np.float32)
        self.energy_j += float(np.sum(np.abs(voltage * self.current)) * dt)
        diagnostics = MotorDiagnostics(
            energy_j=self.energy_j,
            peak_temperature_c=float(np.max(self.temperature)),
            mean_temperature_c=float(np.mean(self.temperature)),
            active_ratio=float(np.mean(~self.failed)),
            saturated_ratio=float(np.mean(self.saturated)),
        )
        return generalized, diagnostics
