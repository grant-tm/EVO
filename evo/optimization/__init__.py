"""
Optimization module for EVO trading system.

This module provides genetic algorithms, backtesting engines, and performance
metrics for optimizing trading strategies and hyperparameters.
"""

from .genetic.genome import Genome
from .genetic.genetic_search import GeneticSearch
from .genetic.fitness import FitnessEvaluator
from .backtesting.engine import BacktestEngine
from .backtesting.strategies import TradingStrategy
from .backtesting.metrics import PerformanceMetrics

__all__ = [
    "Genome",
    "GeneticSearch", 
    "FitnessEvaluator",
    "BacktestEngine",
    "TradingStrategy",
    "PerformanceMetrics"
] 