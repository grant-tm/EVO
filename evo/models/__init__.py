"""
Model layer for the EVO trading system.

This package contains all model-related components including:
- Trading agents (PPO, etc.)
- Trading environments
- Training orchestration
- Hyperparameter management
"""

from .agents import BaseAgent, PPOAgent
from .environments import BaseEnv, TradingEnv
from .training import Trainer, TrainingCallback, HyperparameterManager

__all__ = [
    "BaseAgent",
    "PPOAgent", 
    "BaseEnv",
    "TradingEnv",
    "Trainer",
    "TrainingCallback",
    "HyperparameterManager"
] 