"""
Training orchestration for the EVO system.

This package contains training utilities, callbacks, and hyperparameter
management for training trading agents.
"""

from .trainer import Trainer
from .callbacks import TrainingCallback, EntropyAnnealingCallback, RewardLoggerCallback
from .hyperparameters import HyperparameterManager

__all__ = [
    "Trainer",
    "TrainingCallback", 
    "EntropyAnnealingCallback",
    "RewardLoggerCallback",
    "HyperparameterManager"
] 