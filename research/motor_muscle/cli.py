from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

from .config import ExperimentConfig


def doctor() -> int:
    import mujoco
    import numpy
    import torch

    from .env import MotorMuscleEnv

    config = ExperimentConfig(motor_count=20_000, duration_s=0.05)
    env = MotorMuscleEnv(config)
    observation, info = env.reset(config.seed)
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "observation_shape": list(observation.shape),
        "initial_info": info,
        **env.metadata(),
    }
    print(json.dumps(metadata, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense motor-muscle MuJoCo research backend")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")

    train = sub.add_parser("train")
    train.add_argument("--episodes", type=int, default=12)
    train.add_argument("--duration", type=float, default=8.0)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--dataset", default="artifacts/teacher.npz")
    train.add_argument("--checkpoint", default="artifacts/policy.pt")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--seeds", type=int, default=100)
    evaluate.add_argument("--duration", type=float, default=30.0)
    evaluate.add_argument("--checkpoint", default="artifacts/policy.pt")
    evaluate.add_argument("--output", default="results")

    report = sub.add_parser("report")
    report.add_argument("--output", default="results")

    video = sub.add_parser("video")
    video.add_argument("--checkpoint", default="artifacts/policy.pt")
    video.add_argument("--output", default="results/neural_episode.mp4")
    video.add_argument("--duration", type=float, default=10.0)
    video.add_argument("--controller", choices=("teacher", "oscillator", "neural"), default="neural")

    all_command = sub.add_parser("all")
    all_command.add_argument("--quick", action="store_true")

    accelerator = sub.add_parser("accelerator-doctor")
    accelerator.add_argument("--allow-cpu", action="store_true")
    accelerator.add_argument("--output")

    parity = sub.add_parser("parity")
    parity.add_argument("--output", default="results/consistency.json")

    rollout = sub.add_parser("rollout")
    rollout.add_argument("--backend", choices=("cpu", "mjx"), default="cpu")
    rollout.add_argument("--batch-size", type=int, default=64)
    rollout.add_argument("--steps", type=int, default=10)
    rollout.add_argument("--motors", type=int, default=20_000)

    benchmark = sub.add_parser("mjx-benchmark")
    benchmark.add_argument("--output", default="results/mjx_benchmark.json")
    benchmark.add_argument("--steps", type=int, default=20)
    benchmark.add_argument("--motors", type=int, default=20_000)
    benchmark.add_argument("--batch-sizes", default="64,128,256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        return doctor()

    if args.command == "accelerator-doctor":
        from .accelerator import accelerator_doctor

        report = accelerator_doctor(require_rocm=not args.allow_cpu)
        text = json.dumps(report, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text, encoding="utf-8")
        print(text)
        return 0
    if args.command == "parity":
        from .accelerator import write_consistency_report

        report = write_consistency_report(args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "rollout":
        import numpy as np

        if args.backend == "cpu":
            from .env import MotorMuscleEnv

            env = MotorMuscleEnv(ExperimentConfig(motor_count=args.motors))
            env.reset()
            result = None
            for _ in range(args.steps):
                result = env.step(np.zeros(env.action_size, dtype=np.float32))
            print(json.dumps({"backend": "cpu", "time": result[4]["time"]}, indent=2))
        else:
            import jax
            from .mjx_env import MJXMotorMuscleEnv

            env = MJXMotorMuscleEnv(
                ExperimentConfig(motor_count=args.motors), args.batch_size
            )
            env.reset_batch(np.arange(args.batch_size))
            result = env.rollout_batch(
                np.zeros(
                    (args.steps, args.batch_size, env.action_size),
                    dtype=np.float32,
                )
            )
            jax.block_until_ready(result[0])
            print(
                json.dumps(
                    {
                        "backend": "mjx",
                        "batch_size": args.batch_size,
                        "shape": list(result[0].shape),
                        "finite": bool(np.all(np.isfinite(np.asarray(result[0])))),
                    },
                    indent=2,
                )
            )
        return 0
    if args.command == "mjx-benchmark":
        from .accelerator import benchmark_mjx

        print(
            json.dumps(
                benchmark_mjx(
                    args.output,
                    batch_sizes=tuple(
                        int(value) for value in args.batch_sizes.split(",")
                    ),
                    motor_count=args.motors,
                    control_steps=args.steps,
                ),
                indent=2,
            )
        )
        return 0

    from .controllers import ResidualPolicy
    from .env import MotorMuscleEnv
    from .evaluate import evaluate_suite, render_report
    from .train import generate_teacher_data, train_policy

    if args.command == "train":
        dataset = generate_teacher_data(args.dataset, args.episodes, args.duration)
        history = train_policy(dataset, args.checkpoint, args.epochs)
        print(json.dumps({"dataset": str(dataset), "checkpoint": args.checkpoint, "final": {k: v[-1] for k, v in history.items()}}, indent=2))
    elif args.command == "evaluate":
        records = evaluate_suite(args.output, args.checkpoint, args.seeds, args.duration)
        print(json.dumps({"episodes": len(records), "output": args.output}, indent=2))
    elif args.command == "report":
        print(render_report(args.output))
    elif args.command == "video":
        from .video import record_episode

        config = ExperimentConfig(
            duration_s=args.duration,
            controller=args.controller,
            checkpoint=args.checkpoint,
            impulse_time_s=3.0,
            impulse_force_n=100.0,
        )
        probe = MotorMuscleEnv(config)
        policy = None
        if args.controller == "neural":
            policy = ResidualPolicy(probe.observation_size + 4, len(probe.dof_indices), args.checkpoint)
        print(record_episode(args.output, config, policy))
    else:
        episodes, data_duration, epochs, seeds, eval_duration = (
            (2, 1.0, 2, 1, 1.0) if args.quick else (12, 8.0, 40, 100, 30.0)
        )
        dataset = generate_teacher_data("artifacts/teacher.npz", episodes, data_duration)
        train_policy(dataset, "artifacts/policy.pt", epochs)
        evaluate_suite("results", "artifacts/policy.pt", seeds, eval_duration)
        print(render_report("results"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
