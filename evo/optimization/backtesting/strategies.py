"""
Trading strategies for backtesting framework.

This module provides abstract interfaces and concrete implementations for
various trading strategies used in backtesting.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

from evo.core.logging import get_logger

logger = get_logger(__name__)


class Action(Enum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Trade:
    """Represents a single trade."""
    
    entry_time: int
    exit_time: Optional[int]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    action: Action
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    
    def __post_init__(self):
        """Calculate PnL and return if exit information is available."""
        if self.exit_price is not None and self.exit_time is not None:
            self.pnl = (self.exit_price - self.entry_price) * self.position_size
            self.return_pct = (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class Position:
    """Represents a current position."""
    
    entry_time: int
    entry_price: float
    position_size: float
    action: Action
    
    def get_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        return (current_price - self.entry_price) * self.position_size
    
    def get_return_pct(self, current_price: float) -> float:
        """Calculate unrealized return percentage. Profitable shorts yield positive return."""
        if self.position_size >= 0:
            # Long position
            return (current_price - self.entry_price) / self.entry_price
        else:
            # Short position
            return (self.entry_price - current_price) / self.entry_price


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize trading strategy.
        
        Args:
            initial_capital: Initial capital for trading
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        
    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> Action:
        """
        Generate trading signal based on current data.
        
        Args:
            data: Current market data and features
            
        Returns:
            Trading action to take
        """
        pass
    
    def update(self, data: Dict[str, Any]) -> Optional[Trade]:
        """
        Update strategy with new data and potentially execute trades.
        
        Args:
            data: Current market data
            
        Returns:
            Completed trade if any, None otherwise
        """
        current_price = data.get('close', 0.0)
        current_time = data.get('time', 0)
        
        # Update equity curve
        if self.position is not None:
            unrealized_pnl = self.position.get_pnl(current_price)
            self.current_capital = self.initial_capital + unrealized_pnl
        else:
            self.current_capital = self.initial_capital
        
        self.equity_curve.append(self.current_capital)
        
        # Generate signal
        signal = self.generate_signal(data)
        
        # Execute trades based on signal
        completed_trade = None
        
        if signal == Action.BUY and self.position is None:
            # Open long position
            position_size = self._calculate_position_size(current_price)
            self.position = Position(
                entry_time=current_time,
                entry_price=current_price,
                position_size=position_size,
                action=Action.BUY
            )
            
        elif signal == Action.SELL and self.position is None:
            # Open short position
            position_size = self._calculate_position_size(current_price)
            self.position = Position(
                entry_time=current_time,
                entry_price=current_price,
                position_size=-position_size,  # Negative for short
                action=Action.SELL
            )
            
        elif signal == Action.HOLD and self.position is not None:
            # Close position
            completed_trade = self._close_position(current_time, current_price)
        
        return completed_trade
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on current capital."""
        # Simple position sizing: use 95% of capital
        return (self.current_capital * 0.95) / price
    
    def _close_position(self, exit_time: int, exit_price: float) -> Trade:
        """Close current position and return completed trade."""
        if self.position is None:
            return None
        
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=exit_time,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            position_size=abs(self.position.position_size),
            action=self.position.action
        )
        
        self.trades.append(trade)
        self.position = None
        
        # Update capital
        self.current_capital = self.equity_curve[-1] + trade.pnl
        self.initial_capital = self.current_capital
        
        return trade
    
    def get_returns(self) -> np.ndarray:
        """Calculate daily returns from equity curve."""
        if len(self.equity_curve) < 2:
            return np.array([])
        
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        return returns
    
    def get_trade_returns(self) -> List[float]:
        """Get returns from completed trades."""
        return [trade.return_pct for trade in self.trades if trade.return_pct is not None]


class PPOStrategy(TradingStrategy):
    """Trading strategy based on PPO model predictions."""
    
    def __init__(self, model, initial_capital: float = 100000.0):
        """
        Initialize PPO strategy.
        
        Args:
            model: Trained PPO model
            initial_capital: Initial capital for trading
        """
        super().__init__(initial_capital)
        self.model = model
    
    def generate_signal(self, data: Dict[str, Any]) -> Action:
        """
        Generate signal using PPO model prediction.
        
        Args:
            data: Market data with features
            
        Returns:
            Trading action
        """
        try:
            # Extract features for model prediction
            features = data.get('features', [])
            if features is None or (isinstance(features, (list, np.ndarray)) and len(features) == 0):
                return Action.HOLD
            
            # Convert to numpy array and reshape for model
            features_array = np.array(features, dtype=np.float32)
            if len(features_array.shape) == 1:
                features_array = features_array.reshape(1, -1)
            
            # Get model prediction
            action, _ = self.model.predict(features_array, deterministic=True)
            if isinstance(action, (np.ndarray, list)):
                action = np.asarray(action).flatten()[0]
            
            # Map action to trading signal
            if action == 0:
                return Action.HOLD
            elif action == 1:
                return Action.BUY
            elif action == 2:
                return Action.SELL
            else:
                logger.warning(f"Unknown action from model: {action}")
                return Action.HOLD
                
        except Exception as e:
            logger.error(f"Error generating PPO signal: {e}")
            return Action.HOLD


class MovingAverageStrategy(TradingStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(
        self, 
        short_window: int = 10, 
        long_window: int = 30,
        initial_capital: float = 100000.0
    ):
        """
        Initialize moving average strategy.
        
        Args:
            short_window: Short moving average window
            long_window: Long moving average window
            initial_capital: Initial capital for trading
        """
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        super().__init__(initial_capital)
        self.short_window = short_window
        self.long_window = long_window
        self.price_history: List[float] = []
    
    def generate_signal(self, data: Dict[str, Any]) -> Action:
        """
        Generate signal based on moving average crossover.
        
        Args:
            data: Market data with price information
            
        Returns:
            Trading action
        """
        # Prefer short_ma and long_ma from data if present (for testability)
        if 'short_ma' in data and 'long_ma' in data:
            short_ma = data['short_ma']
            long_ma = data['long_ma']
        else:
            current_price = data.get('close', 0.0)
            self.price_history.append(current_price)
            # Need enough data for both moving averages
            if len(self.price_history) < self.long_window:
                return Action.HOLD
            short_ma = np.mean(self.price_history[-self.short_window:])
            long_ma = np.mean(self.price_history[-self.long_window:])
        
        if self.position is None:
            if short_ma > long_ma:
                return Action.BUY
            elif short_ma < long_ma:
                return Action.SELL
            else:
                return Action.HOLD
        else:
            # If in a long position and crossover reverses, close it
            if self.position.action == Action.BUY and short_ma < long_ma:
                return Action.HOLD  # Close long
            # If in a short position and crossover reverses, close it
            elif self.position.action == Action.SELL and short_ma > long_ma:
                return Action.HOLD  # Close short
            else:
                return Action.HOLD


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy based on Bollinger Bands."""
    
    def __init__(
        self, 
        window: int = 20, 
        std_dev: float = 2.0,
        initial_capital: float = 100000.0
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            window: Window for calculating moving average
            std_dev: Standard deviations for Bollinger Bands
            initial_capital: Initial capital for trading
        """
        super().__init__(initial_capital)
        self.window = window
        self.std_dev = std_dev
        self.price_history: List[float] = []
    
    def generate_signal(self, data: Dict[str, Any]) -> Action:
        """
        Generate signal based on Bollinger Bands.
        
        Args:
            data: Market data with price information
            
        Returns:
            Trading action
        """
        current_price = data.get('close', 0.0)
        self.price_history.append(current_price)
        
        # Need enough data for calculation
        if len(self.price_history) < self.window:
            return Action.HOLD
        
        # Calculate Bollinger Bands
        recent_prices = np.array(self.price_history[-self.window:])
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        # Generate signals
        if current_price <= lower_band and self.position is None:
            return Action.BUY  # Oversold, buy
        elif current_price >= upper_band and self.position is None:
            return Action.SELL  # Overbought, open short
        elif current_price >= upper_band and self.position is not None:
            return Action.HOLD  # Overbought, close long
        elif abs(current_price - sma) < std and self.position is not None:
            return Action.HOLD  # Near mean, close position
        else:
            return Action.HOLD


class MomentumStrategy(TradingStrategy):
    """Momentum strategy based on price momentum."""
    
    def __init__(
        self, 
        lookback_period: int = 20,
        momentum_threshold: float = 0.02,
        initial_capital: float = 100000.0
    ):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Period for calculating momentum
            momentum_threshold: Threshold for momentum signal
            initial_capital: Initial capital for trading
        """
        super().__init__(initial_capital)
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.price_history: List[float] = []
    
    def generate_signal(self, data: Dict[str, Any]) -> Action:
        """
        Generate signal based on price momentum.
        
        Args:
            data: Market data with price information
            
        Returns:
            Trading action
        """
        current_price = data.get('close', 0.0)
        self.price_history.append(current_price)
        
        # Need enough data for momentum calculation
        if len(self.price_history) < self.lookback_period + 1:
            return Action.HOLD
        
        # Calculate momentum
        current_price = self.price_history[-1]
        past_price = self.price_history[-self.lookback_period - 1]
        momentum = (current_price - past_price) / past_price
        
        # Generate signals
        if momentum > self.momentum_threshold and self.position is None:
            return Action.BUY  # Strong positive momentum
        elif momentum < -self.momentum_threshold and self.position is None:
            return Action.SELL  # Strong negative momentum, open short
        elif momentum < -self.momentum_threshold and self.position is not None:
            return Action.HOLD  # Strong negative momentum, close position
        else:
            return Action.HOLD


class MultiSignalStrategy(TradingStrategy):
    """Strategy that combines multiple signals."""
    
    def __init__(
        self, 
        strategies: List[TradingStrategy],
        weights: Optional[List[float]] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize multi-signal strategy.
        
        Args:
            strategies: List of trading strategies
            weights: Weights for each strategy (default: equal weights)
            initial_capital: Initial capital for trading
        """
        super().__init__(initial_capital)
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        
        if len(self.weights) != len(strategies):
            raise ValueError("Number of weights must match number of strategies")
    
    def generate_signal(self, data: Dict[str, Any]) -> Action:
        """
        Generate signal by combining multiple strategy signals.
        
        Args:
            data: Market data
            
        Returns:
            Combined trading action
        """
        signals = [strategy.generate_signal(data) for strategy in self.strategies]
        unique_signals = set(signals)
        # Only return BUY or SELL if all strategies agree (and not HOLD)
        if len(unique_signals) == 1 and signals[0] != Action.HOLD:
            return signals[0]
        else:
            return Action.HOLD


class RandomStrategy(TradingStrategy):
    """Trading strategy that makes random trades (for testing/comparison)."""
    def __init__(self, initial_capital: float = 100000.0, seed: Optional[int] = None):
        """
        Initialize random strategy.
        Args:
            initial_capital: Initial capital for trading
            seed: Optional random seed for reproducibility
        """
        super().__init__(initial_capital)
        self.rng = random.Random(seed)

    def generate_signal(self, data: Dict[str, Any]) -> Action:
        """
        Randomly choose an action (BUY, SELL, HOLD).
        """
        return self.rng.choice([Action.BUY, Action.SELL, Action.HOLD])


class StrategyFactory:
    """Factory for creating trading strategies."""
    
    @staticmethod
    def create_strategy(
        strategy_type: str, 
        **kwargs
    ) -> TradingStrategy:
        """
        Create a trading strategy instance.
        
        Args:
            strategy_type: Type of strategy to create
            **kwargs: Strategy-specific parameters
            
        Returns:
            Trading strategy instance
        """
        if strategy_type.lower() == "ppo":
            model = kwargs.pop("model", None)
            if model is None:
                raise ValueError("PPO strategy requires a model parameter")
            return PPOStrategy(model, **kwargs)
        
        elif strategy_type.lower() == "moving_average":
            return MovingAverageStrategy(**kwargs)
        
        elif strategy_type.lower() == "mean_reversion":
            return MeanReversionStrategy(**kwargs)
        
        elif strategy_type.lower() == "momentum":
            return MomentumStrategy(**kwargs)
        
        elif strategy_type.lower() == "multi_signal":
            strategies = kwargs.get("strategies", [])
            if not strategies:
                raise ValueError("Multi-signal strategy requires strategies parameter")
            return MultiSignalStrategy(strategies, **kwargs)
        
        elif strategy_type.lower() == "random":
            return RandomStrategy(**kwargs)
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}") 