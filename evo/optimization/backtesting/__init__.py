"""
Backtesting framework for evaluating trading strategies and models.
"""

from .engine import BacktestEngine, CSVDataProvider, CrossValidationEngine
from .strategies import TradingStrategy
from .metrics import PerformanceMetrics

__all__ = ["BacktestEngine", "CSVDataProvider", "CrossValidationEngine", "TradingStrategy", "PerformanceMetrics"] 