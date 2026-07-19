from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controllers import FixedOscillator, ResidualPolicy, TeacherController
from .env import MotorMuscleEnv


def generate_teacher_data(
    output: str | Path,
    episodes: int = 12,
    duration_s: float = 8.0,
    seed: int = 7,
) -> Path:
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    for episode in range(episodes):
        config = ExperimentConfig(
            motor_count=2_000,
            duration_s=duration_s,
            seed=seed + episode,
            controller="teacher",
            impulse_time_s=2.0,
            impulse_force_n=float(np.random.default_rng(seed + episode).uniform(-80, 80)),
            parameter_variation=0.10,
        )
        env = MotorMuscleEnv(config)
        env.reset(config.seed)
        oscillator = FixedOscillator(env.dof_names)
        teacher = TeacherController(env.dof_names)
        done = False
        while not done:
            obs = env.observe()
            base = oscillator.command(float(obs["time"]))
            target = teacher.command(obs)
            x = np.concatenate([env.flatten_observation(obs), oscillator.state(float(obs["time"]))])
            residual = np.clip((target - base) / 0.35, -1, 1)
            _, availability = env.motors.group_summary()
            allocation = np.clip((availability - np.mean(availability)) * 2.0, -1, 1)
            y = np.concatenate([residual, allocation]).astype(np.float32)
            rows_x.append(x)
            rows_y.append(y)
            action = np.concatenate([target, np.zeros(len(target) * 2, dtype=np.float32)])
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.stack(rows_x), y=np.stack(rows_y))
    return path


def train_policy(
    dataset: str | Path,
    checkpoint: str | Path,
    epochs: int = 40,
    seed: int = 7,
) -> dict[str, list[float]]:
    import torch

    torch.manual_seed(seed)
    payload = np.load(dataset)
    x = payload["x"].astype(np.float32)
    y = payload["y"].astype(np.float32)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-5)
    normalized = (x - mean) / scale
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    split = max(1, int(len(x) * 0.85))
    train_idx, validation_idx = order[:split], order[split:]

    dof_count = y.shape[1] // 3
    policy = ResidualPolicy(x.shape[1], dof_count)
    model = policy.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    loss_fn = torch.nn.SmoothL1Loss()
    history = {"train_loss": [], "validation_loss": []}
    tx = torch.from_numpy(normalized[train_idx])
    ty = torch.from_numpy(y[train_idx])
    vx = torch.from_numpy(normalized[validation_idx])
    vy = torch.from_numpy(y[validation_idx])

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(tx))
        losses: list[float] = []
        for start in range(0, len(tx), 256):
            batch = permutation[start:start + 256]
            prediction = model(tx[batch])
            loss = loss_fn(prediction, ty[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(vx), vy)) if len(vx) else float(np.mean(losses))
        history["train_loss"].append(float(np.mean(losses)))
        history["validation_loss"].append(validation_loss)

    checkpoint_path = Path(checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "seed": seed,
            "observation_size": x.shape[1],
            "dof_count": dof_count,
        },
        checkpoint_path,
    )
    history_path = checkpoint_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history

