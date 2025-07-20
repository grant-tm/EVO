"""
Tests for trading strategies and strategy factory.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from evo.optimization.backtesting.strategies import (
    Action,
    Trade,
    Position,
    TradingStrategy,
    PPOStrategy,
    MovingAverageStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MultiSignalStrategy,
    StrategyFactory
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.optimization,
    pytest.mark.backtesting
]


class TestAction:
    """Test Action enum."""
    
    def test_action_values(self):
        """Test Action enum values."""
        assert Action.HOLD.value == 0
        assert Action.BUY.value == 1
        assert Action.SELL.value == 2
    
    def test_action_names(self):
        """Test Action enum names."""
        assert Action.HOLD.name == "HOLD"
        assert Action.BUY.name == "BUY"
        assert Action.SELL.name == "SELL"


class TestTrade:
    """Test Trade dataclass."""
    
    def test_trade_creation(self):
        """Test creating a Trade instance."""
        trade = Trade(
            entry_time=100,
            exit_time=110,
            entry_price=100.0,
            exit_price=110.0,
            position_size=100,
            action=Action.BUY
        )
        
        assert trade.entry_time == 100
        assert trade.exit_time == 110
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.position_size == 100
        assert trade.action == Action.BUY
        assert trade.pnl == 1000.0  # (110 - 100) * 100
        assert trade.return_pct == 0.1  # (110 - 100) / 100
    
    def test_trade_without_exit(self):
        """Test creating a Trade without exit information."""
        trade = Trade(
            entry_time=100,
            exit_time=None,
            entry_price=100.0,
            exit_price=None,
            position_size=100,
            action=Action.BUY
        )
        
        assert trade.pnl is None
        assert trade.return_pct is None
    
    def test_short_trade(self):
        """Test creating a short trade."""
        trade = Trade(
            entry_time=100,
            exit_time=110,
            entry_price=110.0,
            exit_price=100.0,
            position_size=100,
            action=Action.SELL
        )
        
        assert np.isclose(trade.pnl, -1000.0)  # (100 - 110) * 100
        assert np.isclose(trade.return_pct, -0.0909, rtol=1e-3)  # (100 - 110) / 110


class TestPosition:
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test creating a Position instance."""
        position = Position(
            entry_time=100,
            entry_price=100.0,
            position_size=100,
            action=Action.BUY
        )
        
        assert position.entry_time == 100
        assert position.entry_price == 100.0
        assert position.position_size == 100
        assert position.action == Action.BUY
    
    def test_long_position_pnl(self):
        """Test PnL calculation for long position."""
        position = Position(
            entry_time=100,
            entry_price=100.0,
            position_size=100,
            action=Action.BUY
        )
        
        pnl = position.get_pnl(110.0)
        assert pnl == 1000.0  # (110 - 100) * 100
        
        return_pct = position.get_return_pct(110.0)
        assert np.isclose(return_pct, 0.1)  # (110 - 100) / 100
    
    def test_short_position_pnl(self):
        """Test PnL calculation for short position."""
        position = Position(
            entry_time=100,
            entry_price=110.0,
            position_size=-100,  # Negative for short
            action=Action.SELL
        )
        
        pnl = position.get_pnl(100.0)
        assert pnl == 1000.0  # (110 - 100) * 100 (positive for short)
        
        return_pct = position.get_return_pct(100.0)
        assert np.isclose(return_pct, 0.0909, rtol=1e-3)  # (100 - 110) / 110


class TestTradingStrategy:
    """Test TradingStrategy abstract base class."""
    
    def test_trading_strategy_abstract(self):
        """Test that TradingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TradingStrategy()
    
    def test_concrete_strategy_creation(self):
        """Test creating a concrete strategy implementation."""
        class ConcreteStrategy(TradingStrategy):
            def generate_signal(self, data):
                return Action.HOLD
        
        strategy = ConcreteStrategy(initial_capital=100000.0)
        
        assert strategy.initial_capital == 100000.0
        assert strategy.current_capital == 100000.0
        assert strategy.position is None
        assert strategy.trades == []
        assert strategy.equity_curve == [100000.0]
    
    def test_strategy_update_no_position(self):
        """Test strategy update when no position is held."""
        class ConcreteStrategy(TradingStrategy):
            def generate_signal(self, data):
                return Action.BUY
        
        strategy = ConcreteStrategy(initial_capital=100000.0)
        
        data = {'close': 100.0, 'time': 0}
        trade = strategy.update(data)
        
        assert trade is None
        assert strategy.position is not None
        assert strategy.position.entry_price == 100.0
        assert strategy.position.action == Action.BUY
    
    def test_strategy_update_with_position(self):
        """Test strategy update when position is held."""
        class ConcreteStrategy(TradingStrategy):
            def __init__(self, initial_capital=100000.0):
                super().__init__(initial_capital)
                self.position = Position(
                    entry_time=0,
                    entry_price=100.0,
                    position_size=100,
                    action=Action.BUY
                )
            
            def generate_signal(self, data):
                return Action.HOLD  # Close position
        
        strategy = ConcreteStrategy(initial_capital=100000.0)
        
        data = {'close': 110.0, 'time': 1}
        trade = strategy.update(data)
        
        assert trade is not None
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.pnl == 1000.0
        assert strategy.position is None
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        class ConcreteStrategy(TradingStrategy):
            def generate_signal(self, data):
                return Action.HOLD
        
        strategy = ConcreteStrategy(initial_capital=100000.0)
        
        position_size = strategy._calculate_position_size(100.0)
        expected_size = (100000.0 * 0.95) / 100.0
        assert position_size == expected_size
    
    def test_get_returns(self):
        """Test getting returns from equity curve."""
        class ConcreteStrategy(TradingStrategy):
            def generate_signal(self, data):
                return Action.HOLD
        
        strategy = ConcreteStrategy(initial_capital=100000.0)
        strategy.equity_curve = [100000, 101000, 102000, 103000]
        
        returns = strategy.get_returns()
        
        assert isinstance(returns, np.ndarray)
        assert len(returns) == 3
        assert np.allclose(returns, [0.01, 0.0099, 0.0098], rtol=1e-3)
    
    def test_get_trade_returns(self):
        """Test getting returns from completed trades."""
        class ConcreteStrategy(TradingStrategy):
            def generate_signal(self, data):
                return Action.HOLD
        
        strategy = ConcreteStrategy(initial_capital=100000.0)
        strategy.trades = [
            Trade(0, 1, 100, 110, 100, Action.BUY),  # 10% return
            Trade(2, 3, 100, 90, 100, Action.BUY),   # -10% return
        ]
        
        returns = strategy.get_trade_returns()
        
        assert returns == [0.1, -0.1]


class TestPPOStrategy:
    """Test PPOStrategy class."""
    
    def test_ppo_strategy_creation(self):
        """Test creating a PPOStrategy instance."""
        mock_model = Mock()
        strategy = PPOStrategy(mock_model, initial_capital=100000.0)
        
        assert strategy.model == mock_model
        assert strategy.initial_capital == 100000.0
        assert isinstance(strategy, TradingStrategy)
    
    def test_generate_signal(self):
        """Test signal generation with PPO model."""
        mock_model = Mock()
        mock_model.predict.return_value = ([1], None)  # BUY action
        
        strategy = PPOStrategy(mock_model, initial_capital=100000.0)
        
        data = {
            'features': np.array([[1.0, 2.0, 3.0]]),
            'close': 100.0,
            'time': 0
        }
        
        signal = strategy.generate_signal(data)
        
        assert signal == Action.BUY
        mock_model.predict.assert_called_once()
    
    def test_generate_signal_hold(self):
        """Test signal generation for HOLD action."""
        mock_model = Mock()
        mock_model.predict.return_value = ([0], None)  # HOLD action
        
        strategy = PPOStrategy(mock_model, initial_capital=100000.0)
        
        data = {
            'features': np.array([[1.0, 2.0, 3.0]]),
            'close': 100.0,
            'time': 0
        }
        
        signal = strategy.generate_signal(data)
        
        assert signal == Action.HOLD
    
    def test_generate_signal_sell(self):
        """Test signal generation for SELL action."""
        mock_model = Mock()
        mock_model.predict.return_value = ([2], None)  # SELL action
        
        strategy = PPOStrategy(mock_model, initial_capital=100000.0)
        
        data = {
            'features': np.array([[1.0, 2.0, 3.0]]),
            'close': 100.0,
            'time': 0
        }
        
        signal = strategy.generate_signal(data)
        
        assert signal == Action.SELL


class TestMovingAverageStrategy:
    """Test MovingAverageStrategy class."""
    
    def test_moving_average_strategy_creation(self):
        """Test creating a MovingAverageStrategy instance."""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            initial_capital=100000.0
        )
        
        assert strategy.short_window == 10
        assert strategy.long_window == 30
        assert strategy.initial_capital == 100000.0
        assert isinstance(strategy, TradingStrategy)
    
    def test_generate_signal_buy(self):
        """Test signal generation for buy signal."""
        strategy = MovingAverageStrategy(short_window=2, long_window=3)
        
        # Create data with short MA > long MA (buy signal)
        data = {
            'close': 100.0,
            'short_ma': 105.0,
            'long_ma': 95.0,
            'time': 0
        }
        
        signal = strategy.generate_signal(data)
        
        assert signal == Action.BUY
    
    def test_generate_signal_sell(self):
        """Test signal generation for sell signal."""
        strategy = MovingAverageStrategy(short_window=2, long_window=3)
        
        # Create data with short MA < long MA (sell signal)
        data = {
            'close': 100.0,
            'short_ma': 95.0,
            'long_ma': 105.0,
            'time': 0
        }
        
        signal = strategy.generate_signal(data)
        
        assert signal == Action.SELL
    
    def test_generate_signal_hold(self):
        """Test signal generation for hold signal."""
        strategy = MovingAverageStrategy(short_window=2, long_window=3)
        
        # Create data with short MA = long MA (hold signal)
        data = {
            'close': 100.0,
            'short_ma': 100.0,
            'long_ma': 100.0,
            'time': 0
        }
        
        signal = strategy.generate_signal(data)
        
        assert signal == Action.HOLD


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy class."""
    
    def test_mean_reversion_strategy_creation(self):
        """Test creating a MeanReversionStrategy instance."""
        strategy = MeanReversionStrategy(
            window=20,
            std_dev=2.0,
            initial_capital=100000.0
        )
        
        assert strategy.window == 20
        assert strategy.std_dev == 2.0
        assert strategy.initial_capital == 100000.0
        assert isinstance(strategy, TradingStrategy)
    
    def test_generate_signal_buy_oversold(self):
        """Test signal generation for oversold condition (buy)."""
        strategy = MeanReversionStrategy(window=5, std_dev=2.0)
        
        # Fill price_history with prices at 100
        for _ in range(10):
            strategy.price_history.append(100.0)
        
        # Now, set the current price much lower than the lower band
        data = {
            'close': 80.0,
            'time': 0
        }
        signal = strategy.generate_signal(data)
        assert signal == Action.BUY
    
    def test_generate_signal_sell_overbought(self):
        """Test signal generation for overbought condition (sell/close position)."""
        strategy = MeanReversionStrategy(window=5, std_dev=2.0)

        # Step 1: Fill price_history with values around 100
        for _ in range(10):
            strategy.price_history.append(100.0)

        # Step 2: Trigger a buy (price below lower band, no position)
        data_buy = {'close': 80.0, 'time': 0}
        signal_buy = strategy.generate_signal(data_buy)
        assert signal_buy == Action.BUY
        # Simulate opening a position
        strategy.position = Position(entry_time=0, entry_price=80.0, position_size=1, action=Action.BUY)

        # Step 3: Add more prices to keep history length
        for _ in range(4):
            strategy.price_history.append(100.0)

        # Step 4: Provide a price well above the upper band to trigger a close (sell/hold signal)
        data_sell = {'close': 120.0, 'time': 1}
        signal_sell = strategy.generate_signal(data_sell)
        assert signal_sell == Action.HOLD  # The strategy closes the position by returning HOLD when overbought
    
    def test_generate_signal_hold_normal(self):
        """Test signal generation for normal condition (hold)."""
        strategy = MeanReversionStrategy(window=20, std_dev=2.0)
        
        data = {
            'close': 100.0,
            'mean': 100.0,
            'std': 5.0,
            'z_score': 0.5,  # Normal
            'time': 0
        }
        
        signal = strategy.generate_signal(data)
        
        assert signal == Action.HOLD


class TestMomentumStrategy:
    """Test MomentumStrategy class."""
    
    def test_momentum_strategy_creation(self):
        """Test creating a MomentumStrategy instance."""
        strategy = MomentumStrategy(
            lookback_period=20,
            momentum_threshold=0.02,
            initial_capital=100000.0
        )
        
        assert strategy.lookback_period == 20
        assert strategy.momentum_threshold == 0.02
        assert strategy.initial_capital == 100000.0
        assert isinstance(strategy, TradingStrategy)
    
    def test_generate_signal_buy_positive_momentum(self):
        """Test signal generation for positive momentum (buy)."""
        strategy = MomentumStrategy(lookback_period=20, momentum_threshold=0.02)
        
        # Pre-fill price_history with 21 prices at 100.0
        strategy.price_history = [100.0] * 21
        
        # Add a new price that is sufficiently higher to trigger a buy
        new_price = 110.0 
        data = {
            'close': new_price,
            'time': 0
        }
        signal = strategy.generate_signal(data)
        assert signal == Action.BUY
    
    def test_generate_signal_sell_negative_momentum(self):
        """Test signal generation for negative momentum (sell)."""
        strategy = MomentumStrategy(lookback_period=20, momentum_threshold=0.02)
        
        # Pre-fill price_history with 21 prices at 100.0
        strategy.price_history = [100.0] * 21

        # Add a new price that is sufficiently lower to trigger a sell
        new_price = 80.0
        data = {
            'close': new_price,
            'time': 0
        }
        signal = strategy.generate_signal(data)
        # The strategy should open a short position (SELL) on strong negative momentum
        assert signal == Action.SELL
    
    def test_generate_signal_hold_weak_momentum(self):
        """Test signal generation for weak momentum (hold)."""
        strategy = MomentumStrategy(lookback_period=20, momentum_threshold=0.02)
        
        # Pre-fill price_history with 21 prices at 100.0
        strategy.price_history = [100.0] * 21
        
        # Add a new price that is only slightly higher, so momentum is weak
        new_price = 100.1
        data = {
            'close': new_price,
            'time': 0
        }
        signal = strategy.generate_signal(data)
        assert signal == Action.HOLD


class TestMultiSignalStrategy:
    """Test MultiSignalStrategy class."""
    
    def test_multi_signal_strategy_creation(self):
        """Test creating a MultiSignalStrategy instance."""
        strategies = [
            MovingAverageStrategy(10, 30),
            MeanReversionStrategy(20, 2.0)
        ]
        weights = [0.6, 0.4]
        
        strategy = MultiSignalStrategy(
            strategies=strategies,
            weights=weights,
            initial_capital=100000.0
        )
        
        assert strategy.strategies == strategies
        assert strategy.weights == weights
        assert strategy.initial_capital == 100000.0
        assert isinstance(strategy, TradingStrategy)
    
    def test_multi_signal_strategy_default_weights(self):
        """Test creating MultiSignalStrategy with default weights."""
        strategies = [
            MovingAverageStrategy(10, 30),
            MeanReversionStrategy(20, 2.0)
        ]
        
        strategy = MultiSignalStrategy(strategies=strategies)
        
        assert strategy.weights == [0.5, 0.5]  # Equal weights
    
    def test_generate_signal_consensus_buy(self):
        """Test signal generation with consensus buy."""
        # Create strategies that both return BUY
        strategy1 = Mock(spec=TradingStrategy)
        strategy1.generate_signal.return_value = Action.BUY
        
        strategy2 = Mock(spec=TradingStrategy)
        strategy2.generate_signal.return_value = Action.BUY
        
        multi_strategy = MultiSignalStrategy(
            strategies=[strategy1, strategy2],
            weights=[0.6, 0.4]
        )
        
        data = {'close': 100.0, 'time': 0}
        signal = multi_strategy.generate_signal(data)
        
        assert signal == Action.BUY
        strategy1.generate_signal.assert_called_once_with(data)
        strategy2.generate_signal.assert_called_once_with(data)
    
    def test_generate_signal_mixed_signals(self):
        """Test signal generation with mixed signals."""
        # Create strategies with different signals
        strategy1 = Mock(spec=TradingStrategy)
        strategy1.generate_signal.return_value = Action.BUY
        
        strategy2 = Mock(spec=TradingStrategy)
        strategy2.generate_signal.return_value = Action.SELL
        
        multi_strategy = MultiSignalStrategy(
            strategies=[strategy1, strategy2],
            weights=[0.6, 0.4]
        )
        
        data = {'close': 100.0, 'time': 0}
        signal = multi_strategy.generate_signal(data)
        
        # Should return HOLD for mixed signals
        assert signal == Action.HOLD


class TestStrategyFactory:
    """Test StrategyFactory class."""
    
    def test_create_moving_average_strategy(self):
        """Test creating MovingAverageStrategy via factory."""
        strategy = StrategyFactory.create_strategy(
            "moving_average",
            short_window=10,
            long_window=30,
            initial_capital=100000.0
        )
        
        assert isinstance(strategy, MovingAverageStrategy)
        assert strategy.short_window == 10
        assert strategy.long_window == 30
        assert strategy.initial_capital == 100000.0
    
    def test_create_mean_reversion_strategy(self):
        """Test creating MeanReversionStrategy via factory."""
        strategy = StrategyFactory.create_strategy(
            "mean_reversion",
            window=20,
            std_dev=2.0,
            initial_capital=100000.0
        )
        
        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.window == 20
        assert strategy.std_dev == 2.0
        assert strategy.initial_capital == 100000.0
    
    def test_create_momentum_strategy(self):
        """Test creating MomentumStrategy via factory."""
        strategy = StrategyFactory.create_strategy(
            "momentum",
            lookback_period=20,
            momentum_threshold=0.02,
            initial_capital=100000.0
        )
        
        assert isinstance(strategy, MomentumStrategy)
        assert strategy.lookback_period == 20
        assert strategy.momentum_threshold == 0.02
        assert strategy.initial_capital == 100000.0
    
    def test_create_unknown_strategy(self):
        """Test creating unknown strategy type."""
        with pytest.raises(ValueError) as exc_info:
            StrategyFactory.create_strategy("unknown_strategy")
        
        assert "Unknown strategy type" in str(exc_info.value)
    
    def test_create_strategy_with_model(self):
        """Test creating PPOStrategy with model."""
        mock_model = Mock()
        
        strategy = StrategyFactory.create_strategy(
            "ppo",
            model=mock_model,
            initial_capital=100000.0
        )
        
        assert isinstance(strategy, PPOStrategy)
        assert strategy.model == mock_model
        assert strategy.initial_capital == 100000.0 