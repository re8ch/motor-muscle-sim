# MuJoCo dense motor-muscle research backend

This backend tests continuous hierarchical control of a MuJoCo humanoid through
2,000–20,000 independently stateful micro-motors. MuJoCo owns rigid-body,
contact, and friction dynamics; NumPy owns the vectorized electrical, thermal,
variation, allocation, and failure model.

## Setup

Python 3.11 is required.

```bash
cd research
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/muscle-research doctor
.venv/bin/python -m pytest
```

## Reproducible workflow

Use the short pipeline as an installation check:

```bash
.venv/bin/muscle-research all --quick
```

Run the full experiment only when ready for a long batch:

```bash
.venv/bin/muscle-research train
.venv/bin/muscle-research evaluate --seeds 100 --duration 30
.venv/bin/muscle-research report
.venv/bin/muscle-research video
```

Outputs are written to `artifacts/` and `results/` and are intentionally ignored
by Git. The complete evaluation is 3 controllers × 11 scenarios × 100 seeds.

## ROCm / MJX remote backend

The CPU MuJoCo backend remains the reference implementation. The optional MJX
backend batches 1–256 worlds and keeps physics plus 20,000-motor state arrays on
the accelerator.

```bash
# From the Mac workspace:
research/scripts/sync_remote.sh

# On amdwsl-zt:
cd /root/projects/motor-muscle-sim/research
scripts/setup_rocm_remote.sh
scripts/run_remote_validation.sh

# Back on the Mac:
research/scripts/fetch_remote_results.sh
```

The remote setup is pinned to Ubuntu 24.04, Python 3.12, ROCm 6.4.2, JAX
0.4.35, and `mujoco-mjx` 3.3.5. It refuses to continue unless the `gfx1101`
device is visible and never silently falls back to CPU.

## Model boundary

The backend targets concept-level and algorithm-comparison fidelity. It does not
claim absolute hardware accuracy until motor constants, transmissions, material
properties, and thermal paths are calibrated from measurements or a local
Abaqus/COMSOL study.
