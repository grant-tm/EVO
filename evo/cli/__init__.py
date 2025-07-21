"""
Command Line Interface for the EVO trading system.

This package provides CLI tools for training, optimization, backtesting, and live trading.
"""

from .train import train_command
from .optimize import optimize_command
from .backtest import backtest_command
from .trade import trade_command

__all__ = [
    'train_command',
    'optimize_command', 
    'backtest_command',
    'trade_command'
] 