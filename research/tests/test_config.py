import pytest

from motor_muscle.config import ExperimentConfig


def test_rate_validation():
    ExperimentConfig().validate()
    with pytest.raises(ValueError):
        ExperimentConfig(control_hz=333).validate()


def test_failure_ratio_validation():
    with pytest.raises(ValueError):
        ExperimentConfig(failure_ratio=1.0).validate()

