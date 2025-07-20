"""
EVO - Neural Network Powered Trading Agent

An experimental reinforcement learning-based trading system using Proximal 
Policy Optimization (PPO)with genetic hyperparameter and reward function 
optimization.
"""

__version__ = "0.1.0"
__author__ = "grant.t.morgan@gmail.com"

from .core.config import Config
from .core.logging import setup_logging
from .optimization.genetic import Genome, GeneticSearch, FitnessEvaluator
from .optimization.backtesting import BacktestEngine, TradingStrategy, PerformanceMetrics

__all__ = [
    "Config", 
    "setup_logging",
    "Genome",
    "GeneticSearch", 
    "FitnessEvaluator",
    "BacktestEngine",
    "TradingStrategy",
    "PerformanceMetrics"
] 