import numpy as np

from motor_muscle.config import ExperimentConfig
from motor_muscle.controllers import TeacherController
from motor_muscle.env import MotorMuscleEnv


def make_env(dt=0.001, motors=2_000):
    return MotorMuscleEnv(
        ExperimentConfig(
            motor_count=motors,
            physics_dt=dt,
            duration_s=0.05,
            controller="teacher",
        )
    )


def test_reset_and_step_contract():
    env = make_env()
    observation, info = env.reset(12)
    assert observation.shape == (env.observation_size,)
    teacher = TeacherController(env.dof_names)
    signed = teacher.command(env.observe())
    action = np.concatenate([signed, np.zeros(len(signed) * 2)])
    observation, reward, terminated, truncated, info = env.step(action)
    assert np.all(np.isfinite(observation))
    assert np.isfinite(reward)
    assert info["energy_j"] >= 0
    assert not (terminated and truncated)


def test_invalid_action_is_rejected():
    env = make_env()
    env.reset()
    try:
        env.step(np.zeros(env.action_size + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action shape was accepted")


def test_20k_motor_performance_and_finiteness():
    env = make_env(motors=20_000)
    env.reset()
    for _ in range(5):
        _, _, terminated, _, info = env.step(np.zeros(env.action_size))
        assert not terminated
    assert np.all(np.isfinite(env.motors.temperature))
    assert info["realtime_factor"] > 0


def test_smaller_timestep_stays_close_over_short_horizon():
    coarse = make_env(0.001)
    fine = make_env(0.0005)
    coarse.reset(3)
    fine.reset(3)
    for env in (coarse, fine):
        steps = round(0.04 * env.config.control_hz)
        for _ in range(steps):
            env.step(np.zeros(env.action_size))
    assert abs(coarse.data.qpos[2] - fine.data.qpos[2]) < 0.02

