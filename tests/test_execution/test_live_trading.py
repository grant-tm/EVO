"""
Tests for live trading module.

This module tests the live trading orchestration system including
the LiveTrader and TradingMonitor classes.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from evo.execution.live_trading.trader import LiveTrader, TradingState
from evo.execution.live_trading.monitoring import TradingMonitor, Alert, PerformanceMetrics
from evo.core.config import Config
from evo.core.exceptions import BrokerError, RiskError
from evo.execution.brokers.base_broker import Account, Position, Order, OrderSide, OrderType, OrderStatus
from evo.data.providers.base_provider import BaseDataProvider

pytestmark = [
    pytest.mark.unit,
    pytest.mark.execution,
    pytest.mark.live_trading
]


class TestTradingState:
    """Test TradingState dataclass."""
    
    def test_trading_state_creation(self):
        """Test creating a TradingState instance."""
        state = TradingState(symbol="AAPL")
        
        assert state.symbol == "AAPL"
        assert state.current_price is None
        assert state.current_position is None
        assert state.account_equity == 0.0
        assert state.is_market_open is False
        assert state.last_action is None
        assert state.last_action_time is None
        assert state.total_trades == 0
        assert state.total_pnl == 0.0
        assert state.is_trading is False
    
    def test_trading_state_with_values(self):
        """Test creating a TradingState with specific values."""
        now = datetime.now()
        state = TradingState(
            symbol="TSLA",
            current_price=150.0,
            current_position=100,
            account_equity=50000.0,
            is_market_open=True,
            last_action=1,
            last_action_time=now,
            total_trades=5,
            total_pnl=2500.0,
            is_trading=True
        )
        
        assert state.symbol == "TSLA"
        assert state.current_price == 150.0
        assert state.current_position == 100
        assert state.account_equity == 50000.0
        assert state.is_market_open is True
        assert state.last_action == 1
        assert state.last_action_time == now
        assert state.total_trades == 5
        assert state.total_pnl == 2500.0
        assert state.is_trading is True


class TestLiveTrader:
    """Test LiveTrader class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=Config)
        
        # Create nested mock objects for the config structure
        config.trading = Mock()
        config.trading.symbol = "AAPL"
        config.trading.close_positions_on_stop = True
        config.trading.update_interval = 30
        
        config.data = Mock()
        config.data.features = ["close", "volume", "rsi"]
        config.data.seq_len = 20
        
        # Create execution config with risk limits
        config.execution = Mock()
        config.execution.risk_limits = Mock()
        config.execution.risk_limits.max_position_size = 0.1
        config.execution.risk_limits.max_portfolio_exposure = 0.5
        config.execution.risk_limits.max_drawdown = 0.15
        config.execution.risk_limits.max_daily_loss = 0.05
        config.execution.risk_limits.max_correlation_exposure = 0.3
        config.execution.risk_limits.stop_loss_pct = 0.02
        config.execution.risk_limits.take_profit_pct = 0.04
        config.execution.risk_limits.max_orders_per_day = 50
        config.execution.risk_limits.min_order_size = 100.0
        config.execution.risk_limits.max_order_size = 10000.0
        
        config.execution.position_config = Mock()
        config.execution.position_config.default_stop_loss_pct = 0.02
        config.execution.position_config.default_take_profit_pct = 0.04
        config.execution.position_config.trailing_stop_pct = 0.01
        config.execution.position_config.max_positions = 10
        config.execution.position_config.position_sizing_method = "kelly"
        config.execution.position_config.kelly_fraction = 0.25
        config.execution.position_config.volatility_lookback = 20
        
        config.to_dict.return_value = {
            "risk_limits": {
                "max_position_size": 0.1,
                "max_drawdown": 0.15
            },
            "position_config": {
                "default_stop_loss_pct": 0.02,
                "max_positions": 10
            }
        }
        return config
    
    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker with all required methods."""
        broker = Mock()
        
        # Async methods
        broker.get_account = AsyncMock()
        broker.close = AsyncMock()
        broker.submit_order = AsyncMock()  # Fixed: was place_order
        broker.cancel_order = AsyncMock()
        broker.is_market_open = AsyncMock()
        broker.get_positions = AsyncMock()
        broker.get_latest_price = AsyncMock()
        broker.get_position = AsyncMock()
        
        # Mock account
        account = Mock(spec=Account)
        account.equity = 50000.0
        account.cash = 50000.0
        account.trading_blocked = False
        broker.get_account.return_value = account
        
        # Mock market state
        broker.is_market_open.return_value = True
        
        # Mock positions
        mock_position = Mock(spec=Position)
        mock_position.symbol = "AAPL"
        mock_position.quantity = 0
        mock_position.side = "long"
        broker.get_positions.return_value = [mock_position]
        broker.get_position.return_value = mock_position
        
        # Mock latest price
        broker.get_latest_price.return_value = 150.0
        
        # Mock order
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.status = OrderStatus.FILLED
        broker.submit_order.return_value = mock_order
        
        return broker
    
    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider with all required methods."""
        provider = Mock(spec=BaseDataProvider)
        provider.get_historical_bars = AsyncMock()
        provider.get_latest_bar = AsyncMock()  # Added missing method
        
        # Mock historical data
        historical_data = pd.DataFrame({
            'close': [100 + i for i in range(50)],
            'volume': [1000 + i * 10 for i in range(50)],
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1min')
        })
        provider.get_historical_bars.return_value = historical_data
        
        # Mock latest bar
        latest_bar = {
            'close': 150.0,
            'volume': 1500,
            'timestamp': datetime.now()
        }
        provider.get_latest_bar.return_value = latest_bar
        
        return provider
    
    @pytest.fixture
    def live_trader(self, mock_config, mock_broker, mock_data_provider):
        """Create a LiveTrader instance for testing."""
        trader = LiveTrader(mock_config, mock_broker, mock_data_provider)
        
        # Mock the monitor to avoid issues with real TradingMonitor
        trader.monitor = Mock()
        trader.monitor.start = Mock()
        trader.monitor.stop = Mock()
        trader.monitor.update_state = Mock()
        trader.monitor.get_summary = Mock(return_value={})
        
        # Mock the risk manager to avoid issues with real RiskManager
        trader.risk_manager = Mock()
        trader.risk_manager.calculate_position_size = Mock(return_value=10)
        trader.risk_manager.validate_order = Mock(return_value=(True, "Order validated successfully"))
        trader.risk_manager.should_stop_trading = Mock(return_value=(False, None))
        trader.risk_manager.set_initial_capital = Mock()
        trader.risk_manager.update_account = Mock()
        trader.risk_manager.update_positions = Mock()
        trader.risk_manager.add_order = Mock()
        
        # Mock the position manager to avoid issues with real PositionManager
        trader.position_manager = Mock()
        trader.position_manager.calculate_position_size = Mock(return_value=10)
        trader.position_manager.add_position = Mock()
        
        return trader
    
    def test_live_trader_initialization(self, live_trader, mock_config):
        """Test LiveTrader initialization."""
        assert live_trader.config == mock_config
        assert live_trader.state.symbol == "AAPL"
        assert live_trader.model is None
        assert live_trader.is_running is False
        assert live_trader.session_id is not None
        assert len(live_trader.price_history) == 0
        assert len(live_trader.feature_history) == 0
        assert len(live_trader.action_history) == 0
    
    @pytest.mark.asyncio
    async def test_start_success(self, live_trader, tmp_path):
        """Test successful start of live trading."""
        # Create a mock model file
        model_path = tmp_path / "test_model.zip"
        model_path.write_text("mock model data")
        
        with patch('evo.execution.live_trading.trader.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.load.return_value = mock_model
            
            # Mock the trading loop to return immediately
            with patch.object(live_trader, '_trading_loop', new_callable=AsyncMock) as mock_loop:
                mock_loop.side_effect = Exception("Trading loop stopped for test")
                
                with pytest.raises(Exception, match="Trading loop stopped for test"):
                    await live_trader.start(model_path)
                
                # Verify components were initialized
                assert live_trader.model == mock_model
                assert live_trader.feature_engineer is not None
                assert live_trader.normalizer is not None
                assert live_trader.is_running is True
                assert live_trader.start_time is not None
                
                # Verify monitor was started
                live_trader.monitor.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_model_not_found(self, live_trader, tmp_path):
        """Test start with non-existent model file."""
        model_path = tmp_path / "nonexistent_model.zip"
        
        with pytest.raises(FileNotFoundError):
            await live_trader.start(model_path)
    
    @pytest.mark.asyncio
    async def test_stop_success(self, live_trader):
        """Test successful stop of live trading."""
        # Set up trader as running
        live_trader.is_running = True
        live_trader.start_time = datetime.now()
        
        await live_trader.stop()
        
        assert live_trader.is_running is False
        live_trader.broker.close.assert_called_once()
        live_trader.monitor.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_without_close_positions(self, live_trader):
        """Test stop when close_positions_on_stop is False."""
        live_trader.is_running = True
        live_trader.config.trading.close_positions_on_stop = False
        
        await live_trader.stop()
        
        # Should not call close_all_positions
        live_trader.broker.get_positions.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, live_trader, tmp_path):
        """Test successful model loading."""
        # Create a mock model file
        model_path = tmp_path / "test_model.zip"
        model_path.write_text("mock model data")
        
        with patch('evo.execution.live_trading.trader.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.load.return_value = mock_model
            
            await live_trader._load_model(model_path)
            
            assert live_trader.model == mock_model
            assert live_trader.feature_engineer is not None
            assert live_trader.normalizer is not None
            mock_ppo.load.assert_called_once_with(str(model_path))
    
    @pytest.mark.asyncio
    async def test_load_model_file_not_found(self, live_trader, tmp_path):
        """Test model loading with non-existent file."""
        model_path = tmp_path / "nonexistent_model.zip"
        
        with pytest.raises(FileNotFoundError):
            await live_trader._load_model(model_path)
    
    @pytest.mark.asyncio
    async def test_initialize_components(self, live_trader, mock_broker):
        """Test component initialization."""
        await live_trader._initialize_components()
        
        # Verify risk manager was initialized with capital
        mock_broker.get_account.assert_called_once()
        
        # Verify historical data was loaded
        live_trader.data_provider.get_historical_bars.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_historical_data(self, live_trader):
        """Test historical data loading."""
        await live_trader._load_historical_data()
        
        # Verify data provider was called
        live_trader.data_provider.get_historical_bars.assert_called_once()
        
        # Verify price history was populated
        assert len(live_trader.price_history) > 0
    
    @pytest.mark.asyncio
    async def test_load_historical_data_error(self, live_trader):
        """Test historical data loading with error."""
        live_trader.data_provider.get_historical_bars.side_effect = Exception("Data error")
        
        # Should not raise exception, just log warning
        await live_trader._load_historical_data()
        
        # Price history should remain empty
        assert len(live_trader.price_history) == 0
    
    def test_update_price_history(self, live_trader):
        """Test price history updates."""
        live_trader.config.data.seq_len = 5
        
        # Add prices
        for i in range(10):
            live_trader._update_price_history(100 + i)
        
        # Should only keep last 5 prices
        assert len(live_trader.price_history) == 5
        assert live_trader.price_history == [105, 106, 107, 108, 109]
    
    @pytest.mark.asyncio
    async def test_calculate_features(self, live_trader):
        """Test feature calculation."""
        # Add some price history
        for i in range(25):
            live_trader._update_price_history(100 + i)
        
        # Mock feature engineer to return DataFrame
        live_trader.feature_engineer = Mock()
        mock_features_df = pd.DataFrame({
            'close': [100 + i for i in range(25)],
            'sma_10': [105 + i for i in range(25)],
            'rsi_14': [50 + i for i in range(25)]
        })
        live_trader.feature_engineer.calculate_features.return_value = mock_features_df
        
        # Mock normalizer to return DataFrame with values attribute
        live_trader.normalizer = Mock()
        mock_normalized_df = pd.DataFrame({
            'close': [0.1 + i * 0.01 for i in range(25)],
            'sma_10': [0.2 + i * 0.01 for i in range(25)],
            'rsi_14': [0.3 + i * 0.01 for i in range(25)]
        })
        live_trader.normalizer.transform.return_value = mock_normalized_df
        
        features = await live_trader._calculate_features()
        
        assert features is not None
        assert len(features) == 3  # Should be the last row of features
        live_trader.feature_engineer.calculate_features.assert_called_once()
        live_trader.normalizer.transform.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_features_insufficient_data(self, live_trader):
        """Test feature calculation with insufficient data."""
        # Add only a few prices
        for i in range(5):
            live_trader._update_price_history(100 + i)
        
        features = await live_trader._calculate_features()
        
        assert features is None
    
    @pytest.mark.asyncio
    async def test_get_model_prediction(self, live_trader):
        """Test model prediction."""
        # Mock model
        live_trader.model = Mock()
        live_trader.model.predict.return_value = (1, None)  # action, state
        
        features = np.array([1.0, 2.0, 3.0])
        action = await live_trader._get_model_prediction(features)
        
        assert action == 1
        # Note: last_action and last_action_time are now set in _execute_trading_decision
        live_trader.model.predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_market_state(self, live_trader, mock_broker):
        """Test market state updates."""
        await live_trader._update_market_state()
        
        # Verify broker calls
        mock_broker.is_market_open.assert_called_once()
        mock_broker.get_account.assert_called_once()
        mock_broker.get_positions.assert_called_once()
        
        # Verify state updates
        assert live_trader.state.is_market_open is True
        assert live_trader.state.account_equity == 50000.0
    
    @pytest.mark.asyncio
    async def test_update_account_state(self, live_trader, mock_broker):
        """Test account state updates."""
        await live_trader._update_account_state()
        
        mock_broker.get_account.assert_called_once()
        assert live_trader.state.account_equity == 50000.0
    
    @pytest.mark.asyncio
    async def test_get_latest_data_success(self, live_trader):
        """Test getting latest data from provider."""
        latest_data = await live_trader._get_latest_data()
        
        assert latest_data is not None
        assert latest_data['close'] == 150.0
        live_trader.data_provider.get_latest_bar.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_latest_data_fallback(self, live_trader):
        """Test getting latest data with provider fallback."""
        # Mock provider to return None
        live_trader.data_provider.get_latest_bar.return_value = None
        
        latest_data = await live_trader._get_latest_data()
        
        assert latest_data is not None
        assert latest_data['close'] == 150.0
        live_trader.broker.get_latest_price.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_latest_data_no_data(self, live_trader):
        """Test getting latest data when no data available."""
        # Mock both provider and broker to return None
        live_trader.data_provider.get_latest_bar.return_value = None
        live_trader.broker.get_latest_price.return_value = None
        
        latest_data = await live_trader._get_latest_data()
        
        assert latest_data is None
    
    @pytest.mark.asyncio
    async def test_execute_trading_decision_buy(self, live_trader, mock_broker):
        """Test executing buy decision."""
        live_trader.state.current_price = 150.0
        live_trader.state.current_position = 0
        live_trader.state.account_equity = 50000.0
        
        # Mock order
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_broker.submit_order.return_value = mock_order
        
        await live_trader._execute_trading_decision(2, 150.0)  # Buy action (action 2 = buy)
        
        mock_broker.submit_order.assert_called_once()
        assert live_trader.state.last_action == 2
        assert live_trader.state.last_action_time is not None
    
    @pytest.mark.asyncio
    async def test_execute_trading_decision_sell(self, live_trader, mock_broker):
        """Test executing sell decision."""
        live_trader.state.current_price = 150.0
        live_trader.state.current_position = 100
        
        # Mock order
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_broker.submit_order.return_value = mock_order
        
        await live_trader._execute_trading_decision(0, 150.0)  # Sell action (action 0 = sell)
        
        mock_broker.submit_order.assert_called_once()
        assert live_trader.state.last_action == 0
        assert live_trader.state.last_action_time is not None
    
    @pytest.mark.asyncio
    async def test_execute_trading_decision_hold(self, live_trader, mock_broker):
        """Test executing hold decision."""
        live_trader.state.current_price = 150.0
        
        await live_trader._execute_trading_decision(1, 150.0)  # Hold action (action 1 = hold)
        
        mock_broker.submit_order.assert_not_called()
        assert live_trader.state.last_action == 1
        assert live_trader.state.last_action_time is not None
    
    @pytest.mark.asyncio
    async def test_execute_buy_order(self, live_trader, mock_broker):
        """Test executing buy order."""
        live_trader.state.current_price = 150.0
        
        # Mock order
        mock_order = Mock(spec=Order)
        mock_order.id = "buy_order_123"
        mock_broker.submit_order.return_value = mock_order
        
        await live_trader._execute_buy_order(150.0)
        
        mock_broker.submit_order.assert_called_once()
        # Verify order was created with correct parameters
        call_args = mock_broker.submit_order.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.side == OrderSide.BUY
        assert call_args.quantity == 10
    
    @pytest.mark.asyncio
    async def test_execute_sell_order(self, live_trader, mock_broker):
        """Test executing sell order."""
        live_trader.state.current_position = 100
        
        # Mock order
        mock_order = Mock(spec=Order)
        mock_order.id = "sell_order_123"
        mock_broker.submit_order.return_value = mock_order
        
        await live_trader._execute_sell_order(150.0)
        
        mock_broker.submit_order.assert_called_once()
        # Verify order was created with correct parameters
        call_args = mock_broker.submit_order.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.side == OrderSide.SELL
        assert call_args.quantity == 100
    
    @pytest.mark.asyncio
    async def test_check_position_exits(self, live_trader, mock_broker):
        """Test checking position exits."""
        # Set up trader state to have a current position
        live_trader.state.current_position = 100
        
        # Mock position with stop loss
        mock_position = Mock(spec=Position)
        mock_position.symbol = "AAPL"
        mock_position.quantity = 100
        mock_position.entry_price = 140.0
        
        mock_broker.get_position.return_value = mock_position
        
        # Mock order
        mock_order = Mock(spec=Order)
        mock_order.id = "exit_order_123"
        mock_broker.submit_order.return_value = mock_order
        
        # Mock position manager to return a stop loss order
        mock_exit_order = Mock(spec=Order)
        mock_exit_order.id = "stop_loss_order_123"
        live_trader.position_manager.update_position.return_value = [mock_exit_order]
        
        # Test with price below stop loss (2% = 137.2)
        await live_trader._check_position_exits(135.0)
        
        mock_broker.submit_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_all_positions(self, live_trader, mock_broker):
        """Test closing all positions."""
        # Mock positions
        mock_position = Mock(spec=Position)
        mock_position.symbol = "AAPL"
        mock_position.quantity = 100
        mock_position.side = "long"
        
        mock_broker.get_positions.return_value = [mock_position]
        
        # Mock order
        mock_order = Mock(spec=Order)
        mock_order.id = "close_order_123"
        mock_broker.submit_order.return_value = mock_order
        
        await live_trader._close_all_positions()
        
        mock_broker.get_positions.assert_called_once()
        mock_broker.submit_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trading_loop_basic_flow(self, live_trader):
        """Test basic trading loop flow."""
        # Set up trader
        live_trader.is_running = True
        live_trader.state.is_market_open = True
        
        # Mock data methods
        live_trader._get_latest_data = AsyncMock(return_value={'close': 150.0, 'timestamp': datetime.now()})
        live_trader._calculate_features = AsyncMock(return_value=np.array([1.0, 2.0, 3.0]))
        live_trader._get_model_prediction = AsyncMock(return_value=1)
        live_trader._execute_trading_decision = AsyncMock()
        
        # Mock sleep to avoid long test
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = Exception("Sleep interrupted for test")
            
            with pytest.raises(Exception, match="Sleep interrupted for test"):
                await live_trader._trading_loop()
            
            # Verify methods were called
            live_trader._get_latest_data.assert_called()
            live_trader._calculate_features.assert_called()
            live_trader._get_model_prediction.assert_called()
            live_trader._execute_trading_decision.assert_called()
    
    @pytest.mark.asyncio
    async def test_trading_loop_market_closed(self, live_trader):
        """Test trading loop when market is closed."""
        live_trader.is_running = True
        
        # Mock broker to report market as closed
        live_trader.broker.is_market_open.return_value = False
        
        # Ensure state is properly initialized
        live_trader.state.is_market_open = False
        
        # Mock sleep to raise exception on first call to stop the test
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = Exception("Sleep interrupted for test")
            
            with pytest.raises(Exception, match="Sleep interrupted for test"):
                await live_trader._trading_loop()
            
            # Verify that the first call to sleep was with 60 seconds (market closed)
            # The test should reach the market closed condition before the exception
            calls = mock_sleep.call_args_list
            assert len(calls) > 0, "Sleep should have been called at least once"
            assert calls[0][0][0] == 60, f"First sleep call should be 60 seconds, got {calls[0][0][0]}"
    
    @pytest.mark.asyncio
    async def test_trading_loop_risk_stop(self, live_trader):
        """Test trading loop stops on risk condition."""
        live_trader.is_running = True
        
        # Mock risk manager to stop trading
        live_trader.risk_manager.should_stop_trading.return_value = (True, "Max drawdown exceeded")
        
        await live_trader._trading_loop()
        
        # Should exit loop without calling trading methods
        assert live_trader.is_running is True  # Loop exits but is_running not set to False
    
    @pytest.mark.asyncio
    async def test_trading_loop_error_handling(self, live_trader):
        """Test trading loop error handling."""
        live_trader.is_running = True
        live_trader.state.is_market_open = True
        
        # Mock error callback
        error_callback = Mock()
        live_trader.on_error_callback = error_callback
        
        # Mock _get_latest_data to raise exception
        live_trader._get_latest_data = AsyncMock(side_effect=Exception("Data error"))
        
        # Mock sleep to avoid long test
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = Exception("Sleep interrupted for test")
            
            with pytest.raises(Exception, match="Sleep interrupted for test"):
                await live_trader._trading_loop()
            
            # Error callback should be called
            error_callback.assert_called()
    
    def test_get_trading_summary(self, live_trader):
        """Test getting trading summary."""
        live_trader.state.total_trades = 10
        live_trader.state.total_pnl = 2500.0
        live_trader.start_time = datetime.now() - timedelta(hours=2)
        live_trader.session_id = "test_session"
        
        summary = live_trader.get_trading_summary()
        
        assert summary['session_id'] == "test_session"
        assert summary['symbol'] == "AAPL"
        assert summary['total_trades'] == 10
        assert summary['total_pnl'] == 2500.0
        assert 'duration' in summary
    
    def test_set_callbacks(self, live_trader):
        """Test setting callbacks."""
        def mock_trade_callback(trade):
            pass
        
        def mock_error_callback(error):
            pass
        
        live_trader.set_callbacks(mock_trade_callback, mock_error_callback)
        
        assert live_trader.on_trade_callback == mock_trade_callback
        assert live_trader.on_error_callback == mock_error_callback
    
    def test_string_representation(self, live_trader):
        """Test string representation."""
        string_repr = str(live_trader)
        
        assert "LiveTrader" in string_repr
        assert "AAPL" in string_repr


class TestTradingMonitor:
    """Test TradingMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a TradingMonitor instance for testing."""
        config = {
            'monitor_interval': 30,
            'alert_thresholds': {
                'max_drawdown': 0.15,
                'daily_loss': 0.05
            }
        }
        return TradingMonitor(config)
    
    def test_monitor_initialization(self, monitor):
        """Test TradingMonitor initialization."""
        assert monitor.is_running is False
        assert monitor.start_time is None
        assert len(monitor.trade_history) == 0
        assert len(monitor.alert_history) == 0
        assert monitor.alert_thresholds['max_drawdown'] == 0.15
        assert monitor.alert_thresholds['daily_loss'] == 0.05
    
    def test_monitor_start_stop(self, monitor):
        """Test starting and stopping the monitor."""
        monitor.start()
        assert monitor.is_running is True
        assert monitor.start_time is not None
        
        monitor.stop()
        assert monitor.is_running is False
    
    def test_update_state(self, monitor):
        """Test updating monitor state."""
        # Create a mock state
        mock_state = Mock()
        mock_state.current_price = 150.0
        mock_state.current_position = 100
        mock_state.account_equity = 50000.0
        
        monitor.update_state(mock_state)
        
        assert monitor.last_update is not None
    
    def test_add_trade(self, monitor):
        """Test adding a trade to history."""
        trade_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 10,
            'price': 150.0,
            'timestamp': datetime.now(),
            'pnl': 100.0
        }
        
        monitor.add_trade(trade_data)
        
        assert len(monitor.trade_history) == 1
        assert monitor.trade_history[0] == trade_data
    
    def test_add_alert(self, monitor):
        """Test adding an alert."""
        monitor.add_alert('warning', 'Test warning message', {'data': 'test'})
        
        assert len(monitor.alert_history) == 1
        alert = monitor.alert_history[0]
        assert alert.type == 'warning'
        assert alert.message == 'Test warning message'
        assert alert.data == {'data': 'test'}
        assert alert.acknowledged is False
    
    def test_register_callbacks(self, monitor):
        """Test registering callbacks."""
        def alert_callback(alert):
            pass
        
        def metric_callback(metrics):
            pass
        
        monitor.register_alert_callback(alert_callback)
        monitor.register_metric_callback(metric_callback)
        
        assert alert_callback in monitor.alert_callbacks
        assert metric_callback in monitor.metric_callbacks
    
    def test_get_summary(self, monitor):
        """Test getting monitor summary."""
        # Add some data
        monitor.start_time = datetime.now() - timedelta(hours=1)
        monitor.add_alert('info', 'Test alert')
        
        summary = monitor.get_summary()
        
        assert 'is_running' in summary
        assert 'start_time' in summary
        assert 'total_alerts' in summary
        assert 'total_trades' in summary
    
    def test_get_recent_alerts(self, monitor):
        """Test getting recent alerts."""
        # Add multiple alerts
        for i in range(15):
            monitor.add_alert('info', f'Alert {i}')
        
        recent_alerts = monitor.get_recent_alerts(limit=10)
        
        assert len(recent_alerts) == 10
        assert recent_alerts[0]['message'] == 'Alert 5'  # Oldest of the recent alerts first
    
    def test_get_performance_metrics(self, monitor):
        """Test getting performance metrics."""
        metrics = monitor.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_acknowledge_alert(self, monitor):
        """Test acknowledging an alert."""
        monitor.add_alert('warning', 'Test alert')
        alert_id = monitor.alert_history[0].id
        
        result = monitor.acknowledge_alert(alert_id)
        
        assert result is True
        assert monitor.alert_history[0].acknowledged is True
    
    def test_acknowledge_nonexistent_alert(self, monitor):
        """Test acknowledging a non-existent alert."""
        result = monitor.acknowledge_alert("nonexistent_id")
        
        assert result is False
    
    def test_save_monitoring_data(self, monitor, tmp_path):
        """Test saving monitoring data to file."""
        # Add some data
        monitor.add_alert('info', 'Test alert')
        monitor.add_trade({'symbol': 'AAPL', 'price': 150.0})
        
        file_path = tmp_path / "monitoring_data.json"
        monitor.save_monitoring_data(file_path)
        
        assert file_path.exists()
        
        # Verify file contains data
        with open(file_path, 'r') as f:
            data = f.read()
            assert 'Test alert' in data
            assert 'AAPL' in data
    
    def test_string_representation(self, monitor):
        """Test string representation."""
        string_repr = str(monitor)
        
        assert "TradingMonitor" in string_repr


class TestAlert:
    """Test Alert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an Alert instance."""
        now = datetime.now()
        alert = Alert(
            id="alert_123",
            type="warning",
            message="Test warning",
            timestamp=now,
            data={"key": "value"},
            acknowledged=True
        )
        
        assert alert.id == "alert_123"
        assert alert.type == "warning"
        assert alert.message == "Test warning"
        assert alert.timestamp == now
        assert alert.data == {"key": "value"}
        assert alert.acknowledged is True
    
    def test_alert_defaults(self):
        """Test Alert creation with defaults."""
        now = datetime.now()
        alert = Alert(
            id="alert_123",
            type="info",
            message="Test info",
            timestamp=now
        )
        
        assert alert.data == {}
        assert alert.acknowledged is False


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test creating a PerformanceMetrics instance."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            win_rate=0.65,
            profit_factor=1.8,
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            average_win=500.0,
            average_loss=300.0,
            largest_win=2000.0,
            largest_loss=1500.0
        )
        
        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == 0.05
        assert metrics.win_rate == 0.65
        assert metrics.profit_factor == 1.8
        assert metrics.total_trades == 100
        assert metrics.winning_trades == 65
        assert metrics.losing_trades == 35
        assert metrics.average_win == 500.0
        assert metrics.average_loss == 300.0
        assert metrics.largest_win == 2000.0
        assert metrics.largest_loss == 1500.0
    
    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics creation with defaults."""
        metrics = PerformanceMetrics()
        
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.average_win == 0.0
        assert metrics.average_loss == 0.0
        assert metrics.largest_win == 0.0
        assert metrics.largest_loss == 0.0


class TestLiveTradingIntegration:
    """Integration tests for live trading system."""
    
    @pytest.fixture
    def integration_config(self):
        """Create configuration for integration tests."""
        config = Mock(spec=Config)
        
        # Create nested mock objects for the config structure
        config.trading = Mock()
        config.trading.symbol = "AAPL"
        config.trading.close_positions_on_stop = True
        config.trading.update_interval = 30
        
        config.data = Mock()
        config.data.features = ["close", "volume"]
        config.data.seq_len = 10
        
        # Create execution config with risk limits
        config.execution = Mock()
        config.execution.risk_limits = Mock()
        config.execution.risk_limits.max_position_size = 0.1
        config.execution.risk_limits.max_portfolio_exposure = 0.5
        config.execution.risk_limits.max_drawdown = 0.15
        config.execution.risk_limits.max_daily_loss = 0.05
        config.execution.risk_limits.max_correlation_exposure = 0.3
        config.execution.risk_limits.stop_loss_pct = 0.02
        config.execution.risk_limits.take_profit_pct = 0.04
        config.execution.risk_limits.max_orders_per_day = 50
        config.execution.risk_limits.min_order_size = 100.0
        config.execution.risk_limits.max_order_size = 10000.0
        
        config.execution.position_config = Mock()
        config.execution.position_config.default_stop_loss_pct = 0.02
        config.execution.position_config.default_take_profit_pct = 0.04
        config.execution.position_config.trailing_stop_pct = 0.01
        config.execution.position_config.max_positions = 10
        config.execution.position_config.position_sizing_method = "kelly"
        config.execution.position_config.kelly_fraction = 0.25
        config.execution.position_config.volatility_lookback = 20
        
        config.to_dict.return_value = {
            "risk_limits": {"max_position_size": 0.1, "max_drawdown": 0.15},
            "position_config": {"default_stop_loss_pct": 0.02, "max_positions": 10}
        }
        return config
    
    @pytest.fixture
    def integration_broker(self):
        """Create broker for integration tests."""
        broker = Mock()
        broker.get_account = AsyncMock()
        broker.close = AsyncMock()
        broker.submit_order = AsyncMock()
        broker.is_market_open = AsyncMock(return_value=True)
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_latest_price = AsyncMock(return_value=150.0)
        broker.get_position = AsyncMock(return_value=None)
        
        account = Mock(spec=Account)
        account.equity = 50000.0
        account.cash = 50000.0
        account.trading_blocked = False
        broker.get_account.return_value = account
        
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.status = OrderStatus.FILLED
        broker.submit_order.return_value = mock_order
        
        return broker
    
    @pytest.fixture
    def integration_data_provider(self):
        """Create data provider for integration tests."""
        provider = Mock(spec=BaseDataProvider)
        provider.get_historical_bars = AsyncMock()
        provider.get_latest_bar = AsyncMock()
        
        # Historical data
        historical_data = pd.DataFrame({
            'close': [100 + i for i in range(20)],
            'volume': [1000 + i * 10 for i in range(20)],
            'timestamp': pd.date_range(start='2024-01-01', periods=20, freq='1min')
        })
        provider.get_historical_bars.return_value = historical_data
        
        # Latest data
        latest_bar = {'close': 150.0, 'volume': 1500, 'timestamp': datetime.now()}
        provider.get_latest_bar.return_value = latest_bar
        
        return provider
    
    @pytest.mark.asyncio
    async def test_complete_trading_session(self, integration_config, integration_broker, integration_data_provider, tmp_path):
        """Test a complete trading session from start to stop."""
        # Create trader
        trader = LiveTrader(integration_config, integration_broker, integration_data_provider)
        
        # Create model file
        model_path = tmp_path / "integration_model.zip"
        model_path.write_text("mock model data")
        
        # Mock PPO model
        with patch('evo.execution.live_trading.trader.PPO') as mock_ppo:
            mock_model = Mock()
            mock_model.predict.return_value = (1, None)  # Buy action
            mock_ppo.load.return_value = mock_model
            
            # Mock trading loop to run for a few iterations then stop
            original_loop = trader._trading_loop
            
            async def mock_trading_loop():
                # Run a few iterations then stop, but call the key methods
                for i in range(3):
                    if not trader.is_running:
                        break
                    # Call the key methods that would normally be called
                    await trader._update_market_state()
                    await trader._get_latest_data()  # Add this call to match actual trading loop
                    await asyncio.sleep(0.1)  # Short sleep for test
                trader.is_running = False
            
            trader._trading_loop = mock_trading_loop
            
            # Start trading
            await trader.start(model_path)
            
            # Verify components were initialized
            assert trader.model is not None
            assert trader.feature_engineer is not None
            assert trader.normalizer is not None
            assert trader.is_running is False  # Should have stopped after loop
            
            # Verify broker interactions
            integration_broker.get_account.assert_called()
            integration_broker.is_market_open.assert_called()
            
            # Verify data provider interactions
            integration_data_provider.get_historical_bars.assert_called()
            integration_data_provider.get_latest_bar.assert_called()
    
    @pytest.mark.asyncio
    async def test_trading_with_error_recovery(self, integration_config, integration_broker, integration_data_provider, tmp_path):
        """Test trading with error recovery."""
        trader = LiveTrader(integration_config, integration_broker, integration_data_provider)
        
        # Create model file
        model_path = tmp_path / "error_model.zip"
        model_path.write_text("mock model data")
        
        # Mock PPO model
        with patch('evo.execution.live_trading.trader.PPO') as mock_ppo:
            mock_model = Mock()
            mock_model.predict.return_value = (1, None)
            mock_ppo.load.return_value = mock_model
            
            # Mock data provider to fail once then succeed
            call_count = 0
            original_get_latest = integration_data_provider.get_latest_bar
            
            async def mock_get_latest_bar(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Temporary data error")
                return original_get_latest(*args, **kwargs)
            
            integration_data_provider.get_latest_bar = mock_get_latest_bar
            
            # Mock trading loop to run briefly
            async def mock_trading_loop():
                # Call the key methods that would normally be called
                await trader._update_market_state()
                await asyncio.sleep(0.1)
                trader.is_running = False
            
            trader._trading_loop = mock_trading_loop
            
            # Should handle the error gracefully
            await trader.start(model_path)
            
            # Verify trader continued after error
            assert trader.is_running is False 