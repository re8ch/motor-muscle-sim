from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .config import ExperimentConfig
from .motors import DenseMotorArray


def accelerator_doctor(require_rocm: bool = True) -> dict[str, object]:
    import jax
    import mujoco

    devices = jax.devices()
    device_text = [str(device) for device in devices]
    device_details = [
        {
            "text": str(device),
            "platform": device.platform,
            "kind": getattr(device, "device_kind", ""),
        }
        for device in devices
    ]
    platforms = sorted({device.platform for device in devices})
    rocm_version = ""
    version_path = Path("/opt/rocm/.info/version")
    if version_path.exists():
        rocm_version = version_path.read_text(encoding="utf-8").strip()
    try:
        rocminfo_text = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        rocminfo_text = ""
    gfx1101_visible = "gfx1101" in rocminfo_text.lower() or any(
        "gfx1101" in item.lower() or "7700" in item.lower()
        for item in device_text
    )
    report = {
        "ok": True,
        "python": platform.python_version(),
        "system": platform.platform(),
        "mujoco": mujoco.__version__,
        "jax": jax.__version__,
        "jax_platforms": platforms,
        "jax_devices": device_text,
        "jax_device_details": device_details,
        "rocm_version": rocm_version,
        "gfx1101_visible": gfx1101_visible,
        "wsl_dxg": Path("/dev/dxg").exists(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    if require_rocm:
        gpu = any(
            platform_name in {"rocm", "gpu"} for platform_name in platforms
        ) and any("rocm" in item.lower() for item in device_text)
        report["ok"] = bool(gpu and report["gfx1101_visible"])
        if not report["ok"]:
            raise RuntimeError(f"ROCm GPU is not active: {json.dumps(report)}")
    return report


def motor_parity_report(seed: int = 7, motor_count: int = 2_000) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from .mjx_env import JAXMotorParameters, JAXMotorState, jax_motor_step

    dofs = np.arange(8, dtype=np.int32)
    cpu = DenseMotorArray(motor_count, dofs, seed, 0.08)
    cpu.reset(seed + 101, 0.10)
    command = np.linspace(0.05, 0.95, cpu.group_count, dtype=np.float32)
    allocation = np.linspace(-0.3, 0.3, cpu.group_count, dtype=np.float32)
    velocity = np.linspace(-0.5, 0.5, len(dofs), dtype=np.float32)
    cpu_torque, cpu_diag = cpu.step(command, allocation, velocity, 0.001)

    parameters = JAXMotorParameters(
        *[
            jnp.asarray(value)[None, ...]
            for value in (
                cpu.group,
                cpu.group_dof,
                cpu.sign,
                cpu.kt,
                cpu.resistance0,
                cpu.moment_arm,
                cpu._counts,
            )
        ]
    )
    # Recreate the state immediately before the CPU step.
    reference = DenseMotorArray(motor_count, dofs, seed, 0.08)
    reference.reset(seed + 101, 0.10)
    state = JAXMotorState(
        jnp.asarray(reference.current)[None, :],
        jnp.asarray(reference.temperature)[None, :],
        jnp.asarray(reference.failed)[None, :],
        jnp.zeros((1,), dtype=jnp.float32),
    )
    jax_state, jax_torque, jax_diag = jax.jit(jax_motor_step, static_argnames=("dt",))(
        parameters,
        state,
        jnp.asarray(command)[None, :],
        jnp.asarray(allocation)[None, :],
        jnp.asarray(velocity)[None, :],
        dt=0.001,
    )
    current_error = float(np.max(np.abs(cpu.current - np.asarray(jax_state.current[0]))))
    temperature_error = float(
        np.max(np.abs(cpu.temperature - np.asarray(jax_state.temperature[0])))
    )
    torque_error = float(np.max(np.abs(cpu_torque - np.asarray(jax_torque[0]))))
    energy_error = abs(cpu_diag.energy_j - float(jax_state.energy_j[0]))
    maximum = max(current_error, temperature_error, torque_error, energy_error)
    return {
        "ok": maximum <= 1e-5,
        "motor_count": motor_count,
        "current_max_abs_error": current_error,
        "temperature_max_abs_error": temperature_error,
        "torque_max_abs_error": torque_error,
        "energy_abs_error": energy_error,
        "maximum_error": maximum,
    }


def physics_parity_report(seed: int = 7) -> dict[str, object]:
    from .env import MotorMuscleEnv
    from .mjx_env import MJXMotorMuscleEnv

    config = ExperimentConfig(motor_count=2_000, duration_s=0.1, seed=seed)
    cpu = MotorMuscleEnv(config)
    cpu_obs, _ = cpu.reset(seed)
    gpu = MJXMotorMuscleEnv(config, batch_size=1)
    mjx_obs, _ = gpu.reset(seed)
    reset_error = float(np.max(np.abs(cpu_obs - np.asarray(mjx_obs))))
    action = np.zeros(cpu.action_size, dtype=np.float32)
    cpu_next, _, cpu_done, _, _ = cpu.step(action)
    mjx_next, _, mjx_done, _, _ = gpu.step(action)
    step_rmse = float(
        np.sqrt(np.mean(np.square(cpu_next - np.asarray(mjx_next))))
    )
    return {
        "ok": reset_error <= 1e-5 and step_rmse <= 1e-2 and bool(cpu_done) == bool(mjx_done),
        "reset_max_abs_error": reset_error,
        "control_step_observation_rmse": step_rmse,
        "cpu_terminated": bool(cpu_done),
        "mjx_terminated": bool(mjx_done),
    }


def benchmark_mjx(
    output: str | Path,
    batch_sizes: tuple[int, ...] = (64, 128, 256),
    motor_count: int = 20_000,
    control_steps: int = 20,
) -> dict[str, object]:
    import jax

    from .mjx_env import MJXMotorMuscleEnv
    from .env import MotorMuscleEnv

    cpu_config = ExperimentConfig(
        motor_count=motor_count,
        duration_s=max(1.0, control_steps / 100),
    )
    cpu_env = MotorMuscleEnv(cpu_config)
    cpu_env.reset(0)
    cpu_actions = np.zeros(cpu_env.action_size, dtype=np.float32)
    cpu_env.step(cpu_actions)
    cpu_start = time.perf_counter()
    for _ in range(control_steps):
        cpu_env.step(cpu_actions)
    cpu_elapsed = time.perf_counter() - cpu_start
    cpu_realtime = (control_steps / cpu_config.control_hz) / cpu_elapsed
    records: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        config = ExperimentConfig(
            motor_count=motor_count,
            duration_s=max(1.0, control_steps / 100),
        )
        env = MJXMotorMuscleEnv(config, batch_size=batch_size)
        env.reset_batch(np.arange(batch_size))
        actions = np.zeros(
            (control_steps, batch_size, env.action_size), dtype=np.float32
        )
        compile_start = time.perf_counter()
        compiled = env.rollout_batch(actions)
        jax.block_until_ready(compiled[0])
        compile_seconds = time.perf_counter() - compile_start
        env.reset_batch(np.arange(batch_size))
        start = time.perf_counter()
        result = env.rollout_batch(actions)
        jax.block_until_ready(result[0])
        elapsed = time.perf_counter() - start
        physics_steps = control_steps * config.control_stride * batch_size
        simulated_seconds = control_steps / config.control_hz * batch_size
        estimated_bytes = batch_size * motor_count * 4 * 13
        records.append(
            {
                "batch_size": batch_size,
                "motor_count": motor_count,
                "compile_seconds": compile_seconds,
                "elapsed_seconds": elapsed,
                "physics_steps_per_second": physics_steps / elapsed,
                "aggregate_realtime_factor": simulated_seconds / elapsed,
                "speedup_vs_cpu_single": (simulated_seconds / elapsed)
                / cpu_realtime,
                "per_world_realtime_factor": (control_steps / config.control_hz) / elapsed,
                "estimated_motor_state_gib": estimated_bytes / (1024**3),
                "finite": bool(np.all(np.isfinite(np.asarray(result[0])))),
            }
        )
    valid = [
        item
        for item in records
        if item["estimated_motor_state_gib"] < 10 and item["finite"]
    ]
    selected = max(valid, key=lambda item: item["physics_steps_per_second"]) if valid else None
    report = {
        "device": [str(device) for device in jax.devices()],
        "cpu_single": {
            "elapsed_seconds": cpu_elapsed,
            "realtime_factor": cpu_realtime,
        },
        "records": records,
        "selected_batch_size": selected["batch_size"] if selected else None,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_consistency_report(output: str | Path) -> dict[str, object]:
    report = {
        "motor": motor_parity_report(),
        "physics": physics_parity_report(),
    }
    report["ok"] = bool(report["motor"]["ok"] and report["physics"]["ok"])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
