import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None
    or importlib.util.find_spec("mujoco.mjx") is None,
    reason="MJX optional dependencies are not installed",
)


def test_jax_motor_matches_numpy():
    from motor_muscle.accelerator import motor_parity_report

    report = motor_parity_report(motor_count=1_000)
    assert report["ok"], report


def test_mjx_batch_contract_and_determinism():
    import jax

    from motor_muscle.config import ExperimentConfig
    from motor_muscle.mjx_env import MJXMotorMuscleEnv

    config = ExperimentConfig(motor_count=1_000, duration_s=0.02)
    first = MJXMotorMuscleEnv(config, batch_size=2)
    obs_a, _ = first.reset_batch([3, 4])
    initial_motor_a = [
        np.asarray(value).copy()
        for value in jax.tree_util.tree_leaves(
            (first.motor_parameters, first.motor_state)
        )
    ]
    actions = np.zeros((2, first.action_size), dtype=np.float32)
    result_a = first.step_batch(actions)
    jax.block_until_ready(result_a[0])

    first.reset_batch([3, 4])
    initial_motor_b = [
        np.asarray(value)
        for value in jax.tree_util.tree_leaves(
            (first.motor_parameters, first.motor_state)
        )
    ]
    result_b = first.step_batch(actions)
    jax.block_until_ready(result_b[0])

    assert result_a[0].shape[0] == 2
    assert np.all(np.isfinite(np.asarray(result_a[0])))
    assert all(
        np.array_equal(left, right)
        for left, right in zip(initial_motor_a, initial_motor_b, strict=True)
    )
    # ROCm contact reductions are numerically, but not bitwise, deterministic.
    assert np.allclose(
        np.asarray(result_a[0]), np.asarray(result_b[0]), rtol=1e-5, atol=1e-5
    )
    assert not np.array_equal(np.asarray(result_a[0][0]), np.asarray(result_a[0][1]))


def test_mjx_rollout_contract():
    import jax

    from motor_muscle.config import ExperimentConfig
    from motor_muscle.mjx_env import MJXMotorMuscleEnv

    env = MJXMotorMuscleEnv(
        ExperimentConfig(motor_count=1_000, duration_s=0.03), batch_size=2
    )
    env.reset_batch([8, 9])
    actions = np.zeros((3, 2, env.action_size), dtype=np.float32)
    observation, reward, terminated, truncated, info = env.rollout_batch(actions)
    jax.block_until_ready(observation)
    assert observation.shape == (3, 2, 108)
    assert reward.shape == terminated.shape == truncated.shape == (3, 2)
    assert info["energy_j"].shape == (3, 2)
