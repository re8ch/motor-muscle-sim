from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class ExperimentConfig:
    motor_count: int = 20_000
    physics_dt: float = 0.001
    sensor_hz: int = 200
    control_hz: int = 100
    duration_s: float = 30.0
    seed: int = 7
    failure_ratio: float = 0.0
    concentrated_failure: bool = False
    parameter_variation: float = 0.08
    sensor_noise: float = 0.0
    control_delay_steps: int = 0
    impulse_time_s: float | None = None
    impulse_duration_s: float = 0.10
    impulse_force_n: float = 0.0
    controller: str = "neural"
    checkpoint: str = "artifacts/policy.pt"

    def validate(self) -> None:
        if self.motor_count < 100:
            raise ValueError("motor_count must be >= 100")
        if self.physics_dt <= 0:
            raise ValueError("physics_dt must be positive")
        if not 0 <= self.failure_ratio < 1:
            raise ValueError("failure_ratio must be in [0, 1)")
        if self.controller not in {"teacher", "oscillator", "neural"}:
            raise ValueError("controller must be teacher, oscillator, or neural")
        control_steps = 1.0 / (self.physics_dt * self.control_hz)
        sensor_steps = 1.0 / (self.physics_dt * self.sensor_hz)
        if abs(control_steps - round(control_steps)) > 1e-9:
            raise ValueError("control_hz must divide the physics rate")
        if abs(sensor_steps - round(sensor_steps)) > 1e-9:
            raise ValueError("sensor_hz must divide the physics rate")

    @property
    def control_stride(self) -> int:
        return round(1.0 / (self.physics_dt * self.control_hz))

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

