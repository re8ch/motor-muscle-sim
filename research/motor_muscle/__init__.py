"""Research-grade MuJoCo backend for dense motor-muscle experiments."""

from .config import ExperimentConfig
from .env import MotorMuscleEnv

__all__ = ["ExperimentConfig", "MotorMuscleEnv"]

