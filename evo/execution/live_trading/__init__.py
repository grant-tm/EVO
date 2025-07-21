"""
Live trading components for the EVO trading system.

This package provides live trading orchestration, monitoring, and execution
capabilities for automated trading.
"""

from .trader import LiveTrader
from .monitoring import TradingMonitor

__all__ = ['LiveTrader', 'TradingMonitor'] 