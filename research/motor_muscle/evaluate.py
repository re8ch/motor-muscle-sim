from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controllers import ResidualPolicy
from .env import MotorMuscleEnv
from .runner import EpisodeResult, run_episode


def benchmark_scenarios(base: ExperimentConfig) -> dict[str, ExperimentConfig]:
    return {
        "nominal_2k": replace(base, motor_count=2_000),
        "nominal_10k": replace(base, motor_count=10_000),
        "nominal_20k": replace(base, motor_count=20_000),
        "impulse": replace(base, impulse_time_s=5.0, impulse_force_n=100.0),
        "failure_5": replace(base, failure_ratio=0.05),
        "failure_10": replace(base, failure_ratio=0.10),
        "failure_20": replace(base, failure_ratio=0.20),
        "left_failure": replace(base, failure_ratio=0.10, concentrated_failure=True),
        "variation": replace(base, parameter_variation=0.15),
        "sensor_noise": replace(base, sensor_noise=0.005),
        "delay": replace(base, control_delay_steps=1),
    }


def evaluate_suite(
    output_dir: str | Path,
    checkpoint: str | Path,
    seeds: int = 100,
    duration_s: float = 30.0,
) -> list[dict[str, object]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = ExperimentConfig(duration_s=duration_s, checkpoint=str(checkpoint))
    probe = MotorMuscleEnv(base)
    policy = ResidualPolicy(
        probe.observation_size + 4,
        len(probe.dof_indices),
        str(checkpoint),
    )
    records: list[dict[str, object]] = []
    for scenario_name, scenario in benchmark_scenarios(base).items():
        for controller in ("teacher", "oscillator", "neural"):
            for seed in range(seeds):
                config = replace(scenario, controller=controller, seed=seed)
                result = run_episode(config, seed, policy if controller == "neural" else None)
                row = {"scenario": scenario_name, **asdict(result)}
                records.append(row)
    _write_results(output, records)
    return records


def _write_results(output: Path, records: list[dict[str, object]]) -> None:
    (output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary: list[dict[str, object]] = []
    keys = sorted({(str(row["scenario"]), str(row["controller"])) for row in records})
    for scenario, controller in keys:
        group = [row for row in records if row["scenario"] == scenario and row["controller"] == controller]
        summary.append(
            {
                "scenario": scenario,
                "controller": controller,
                "episodes": len(group),
                "success_rate": float(np.mean([row["survived"] for row in group])),
                "mean_survival_s": float(np.mean([row["survival_time_s"] for row in group])),
                "mean_energy_j": float(np.mean([row["energy_j"] for row in group])),
                "peak_temperature_c": float(np.max([row["peak_temperature_c"] for row in group])),
                "mean_realtime_factor": float(np.mean([row["realtime_factor"] for row in group])),
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def render_report(output_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    nominal = [row for row in summary if row["scenario"] == "nominal_20k"]
    labels = [row["controller"] for row in nominal]
    success = [row["success_rate"] for row in nominal]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, success, color=["#4078a8", "#d08737", "#3f9c70"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("30 s survival rate")
    ax.set_title("20,000-motor standing benchmark")
    fig.tight_layout()
    fig.savefig(output / "success_rate.png", dpi=160)
    plt.close(fig)

    by_key = {(row["scenario"], row["controller"]): row for row in summary}
    neural_nominal = by_key.get(("nominal_20k", "neural"), {})
    neural_impulse = by_key.get(("impulse", "neural"), {})
    neural_failure = by_key.get(("failure_10", "neural"), {})
    oscillator_impulse = by_key.get(("impulse", "oscillator"), {})
    complete = bool(summary) and all(row.get("episodes", 0) >= 100 for row in summary)
    full_duration = neural_nominal.get("mean_survival_s", 0) >= 29.9
    criteria = {
        "full_100_seed_30s_suite_completed": complete and full_duration,
        "nominal_success_ge_95pct": complete and full_duration and neural_nominal.get("success_rate", 0) >= 0.95,
        "impulse_success_ge_90pct": complete and neural_impulse.get("mean_survival_s", 0) >= 29.9 and neural_impulse.get("success_rate", 0) >= 0.90,
        "failure_10_drop_le_15pct": (
            complete
            and neural_failure.get("mean_survival_s", 0) >= 29.9
            and (
                neural_nominal.get("success_rate", 0)
                - neural_failure.get("success_rate", 0)
            ) <= 0.15
        ),
        "neural_beats_oscillator_on_impulse": complete and neural_impulse.get("success_rate", 0) > oscillator_impulse.get("success_rate", 0),
    }
    lines = [
        "# Dense Motor-Muscle Validation Report",
        "",
        "## Conclusion",
        "",
        f"Acceptance criteria passed: **{sum(criteria.values())}/{len(criteria)}**.",
        "" if complete else "**Preliminary run only:** the full 100-seed, 30-second suite has not completed.",
        "",
        "## Acceptance criteria",
        "",
        *[f"- [{'x' if passed else ' '}] {name}" for name, passed in criteria.items()],
        "",
        "## Evidence",
        "",
        "![20k success-rate comparison](success_rate.png)",
        "",
        "Raw episode data: `results.csv`; aggregated metrics: `summary.json`.",
        "",
        "## Limitation",
        "",
        "This is a concept-fidelity rigid-body and lumped electro-thermal model. Absolute hardware accuracy requires measured motor, transmission, material, and cooling parameters.",
    ]
    report = output / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
