"""
Trading agents for the EVO system.

This package contains implementations of various trading agents
using reinforcement learning algorithms.
"""

from .base_agent import BaseAgent
from .ppo_agent import PPOAgent

__all__ = ["BaseAgent", "PPOAgent"] 