import numpy as np

from motor_muscle.motors import DenseMotorArray


def test_fixed_seed_is_reproducible():
    dofs = np.arange(8)
    a = DenseMotorArray(2_000, dofs, 5)
    b = DenseMotorArray(2_000, dofs, 5)
    assert np.array_equal(a.group, b.group)
    assert np.allclose(a.kt, b.kt)


def test_failure_saturation_and_finite_torque():
    motors = DenseMotorArray(2_000, np.arange(8), 4)
    motors.reset(9, 0.1)
    command = np.ones(16, dtype=np.float32)
    torque, diagnostics = motors.step(command, np.zeros(16), np.zeros(8), 0.001)
    assert np.all(np.isfinite(torque))
    assert np.all(motors.current[motors.failed] == 0)
    assert 0.89 <= diagnostics.active_ratio <= 0.91
    assert np.max(motors.current) <= 4.0


def test_hot_motors_are_derated():
    motors = DenseMotorArray(1_000, np.arange(4), 3)
    motors.temperature[:100] = 100
    motors.step(np.ones(8), np.zeros(8), np.zeros(4), 0.001)
    assert np.all(motors.current[:100] == 0)

