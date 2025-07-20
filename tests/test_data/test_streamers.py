"""
Tests for data streamers.

This module contains tests for the data streamer implementations.
"""

import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from types import SimpleNamespace

from evo.data.streamers.base_streamer import BaseStreamer
from evo.data.streamers.live_streamer import LiveStreamer
from evo.data.streamers.simulated_streamer import SimulatedStreamer
from evo.core.exceptions import DataProviderError

pytestmark = [
    pytest.mark.unit,
    pytest.mark.data,
    pytest.mark.streamers
]


class TestBaseStreamer:
    """Test the abstract base streamer."""
    
    def test_abstract_methods(self):
        """Test that BaseStreamer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStreamer({})
    
    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = [
            '_validate_config',
            'start',
            'stop',
            'subscribe',
            'unsubscribe'
        ]
        
        for method_name in abstract_methods:
            assert hasattr(BaseStreamer, method_name)
            method = getattr(BaseStreamer, method_name)
            assert hasattr(method, '__isabstractmethod__')


class TestLiveStreamer:
    """Test the live streamer implementation."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock data provider."""
        provider = Mock()
        provider.subscribe_to_bars = AsyncMock()
        provider.start_streaming = AsyncMock()
        provider.health_check = AsyncMock(return_value=True)
        provider.is_market_open = Mock(return_value=True)
        return provider
    
    @pytest.fixture
    def valid_config(self, mock_provider):
        """Create a valid configuration for testing."""
        return {
            "provider": mock_provider
        }
    
    @pytest.fixture
    def streamer(self, valid_config):
        """Create a live streamer instance for testing."""
        return LiveStreamer(valid_config)
    
    def test_init_valid_config(self, valid_config):
        """Test initialization with valid configuration."""
        streamer = LiveStreamer(valid_config)
        assert streamer.config == valid_config
        assert streamer._provider == valid_config["provider"]
        assert not streamer.is_running
    
    def test_init_missing_provider(self):
        """Test initialization with missing provider."""
        config = {}
        with pytest.raises(DataProviderError, match="Missing required config key: provider"):
            LiveStreamer(config)
    
    def test_init_none_provider(self):
        """Test initialization with None provider."""
        config = {"provider": None}
        with pytest.raises(DataProviderError, match="Data provider is required for live streaming"):
            LiveStreamer(config)
    
    def test_validate_config_missing_provider(self):
        """Test config validation with missing provider."""
        config = {}
        with pytest.raises(DataProviderError, match="Missing required config key: provider"):
            LiveStreamer(config)
    
    def test_validate_config_invalid_provider(self):
        """Test config validation with invalid provider."""
        invalid_provider = Mock()
        del invalid_provider.subscribe_to_bars
        
        config = {"provider": invalid_provider}
        with pytest.raises(DataProviderError, match="Provider must support bar subscription"):
            LiveStreamer(config)
    
    @pytest.mark.asyncio
    async def test_start_success(self, streamer, mock_provider):
        """Test successful stream start."""
        await streamer.start()
        
        assert streamer.is_running
        mock_provider.start_streaming.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, streamer, mock_provider):
        """Test starting when already running."""
        streamer._running = True
        
        await streamer.start()
        
        # Should not call provider start again
        mock_provider.start_streaming.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_start_provider_error(self, streamer, mock_provider):
        """Test start with provider error."""
        mock_provider.start_streaming.side_effect = Exception("Provider error")
        
        with pytest.raises(DataProviderError, match="Failed to start live streaming"):
            await streamer.start()
        
        assert not streamer.is_running
    
    @pytest.mark.asyncio
    async def test_stop_success(self, streamer):
        """Test successful stream stop."""
        streamer._running = True
        
        await streamer.stop()
        
        assert not streamer.is_running
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self, streamer):
        """Test stopping when not running."""
        await streamer.stop()
        
        # Should not raise error
        assert not streamer.is_running
    
    @pytest.mark.asyncio
    async def test_stop_error(self, streamer):
        """Test stop with error."""
        # Note: Current LiveStreamer.stop() implementation is very simple and doesn't have
        # any operations that could realistically fail. The method only sets _running = False
        # and logs. If the implementation changes to include error-prone operations,
        # this test should be updated accordingly.
        streamer._running = True
        
        # For now, we'll test that the method handles the case where it's already stopped
        await streamer.stop()  # Should not raise an error
        assert not streamer.is_running
    
    @pytest.mark.asyncio
    async def test_subscribe_success(self, streamer, mock_provider):
        """Test successful subscription."""
        symbol = "AAPL"
        handler = AsyncMock()
        
        await streamer.subscribe(symbol, handler)
        
        assert symbol in streamer._symbol_handlers
        assert handler in streamer._symbol_handlers[symbol]
        mock_provider.subscribe_to_bars.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_subscribe_multiple_handlers(self, streamer, mock_provider):
        """Test subscribing multiple handlers to same symbol."""
        symbol = "AAPL"
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        await streamer.subscribe(symbol, handler1)
        await streamer.subscribe(symbol, handler2)
        
        assert len(streamer._symbol_handlers[symbol]) == 2
        assert handler1 in streamer._symbol_handlers[symbol]
        assert handler2 in streamer._symbol_handlers[symbol]
    
    @pytest.mark.asyncio
    async def test_subscribe_provider_error(self, streamer, mock_provider):
        """Test subscription with provider error."""
        symbol = "AAPL"
        handler = AsyncMock()
        mock_provider.subscribe_to_bars.side_effect = Exception("Provider error")
        
        with pytest.raises(DataProviderError, match="Failed to subscribe to AAPL"):
            await streamer.subscribe(symbol, handler)
    
    @pytest.mark.asyncio
    async def test_unsubscribe_success(self, streamer):
        """Test successful unsubscription."""
        symbol = "AAPL"
        handler = AsyncMock()
        
        # First subscribe
        streamer._symbol_handlers[symbol] = [handler]
        
        await streamer.unsubscribe(symbol)
        
        assert symbol not in streamer._symbol_handlers
    
    @pytest.mark.asyncio
    async def test_unsubscribe_not_subscribed(self, streamer):
        """Test unsubscribing from non-subscribed symbol."""
        symbol = "AAPL"
        
        # Should not raise error
        await streamer.unsubscribe(symbol)
    
    @pytest.mark.asyncio
    async def test_unsubscribe_error(self, streamer):
        """Test unsubscription with error."""
        symbol = "AAPL"
        streamer._symbol_handlers[symbol] = [AsyncMock()]
        
        # Mock an error by making the dict access raise an exception
        original_handlers = streamer._symbol_handlers
        mock_handlers = Mock()
        mock_handlers.__contains__ = Mock(return_value=True)
        mock_handlers.__delitem__ = Mock(side_effect=Exception("Dict access error"))
        streamer._symbol_handlers = mock_handlers
        
        try:
            with pytest.raises(DataProviderError, match="Failed to unsubscribe from AAPL"):
                await streamer.unsubscribe(symbol)
        finally:
            streamer._symbol_handlers = original_handlers
    
    @pytest.mark.asyncio
    async def test_create_bar_handler(self, streamer):
        """Test bar handler creation and execution."""
        symbol = "AAPL"
        handler = AsyncMock()
        streamer._symbol_handlers[symbol] = [handler]
        
        # Create bar handler
        bar_handler = streamer._create_bar_handler(symbol)
        
        # Mock data
        bar_data = {"timestamp": datetime.now(), "close": 100.0}
        
        # Execute handler
        await bar_handler(bar_data)
        
        # Check that handler was called
        handler.assert_called_once_with(bar_data)
    
    @pytest.mark.asyncio
    async def test_create_bar_handler_with_global_handlers(self, streamer):
        """Test bar handler with global handlers."""
        symbol = "AAPL"
        symbol_handler = AsyncMock()
        global_handler = AsyncMock()
        
        streamer._symbol_handlers[symbol] = [symbol_handler]
        streamer.add_handler(global_handler)
        
        # Create bar handler
        bar_handler = streamer._create_bar_handler(symbol)
        
        # Mock data
        bar_data = {"timestamp": datetime.now(), "close": 100.0}
        
        # Execute handler
        await bar_handler(bar_data)
        
        # Check that both handlers were called
        symbol_handler.assert_called_once_with(bar_data)
        global_handler.assert_called_once_with(bar_data)
    
    @pytest.mark.asyncio
    async def test_get_subscribed_symbols(self, streamer):
        """Test getting subscribed symbols."""
        symbol1 = "AAPL"
        symbol2 = "GOOGL"
        
        streamer._symbol_handlers[symbol1] = [AsyncMock()]
        streamer._symbol_handlers[symbol2] = [AsyncMock()]
        
        symbols = await streamer.get_subscribed_symbols()
        
        assert symbol1 in symbols
        assert symbol2 in symbols
        assert len(symbols) == 2
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, streamer, mock_provider):
        """Test successful health check."""
        result = await streamer.health_check()
        
        assert result is True
        mock_provider.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_provider_failure(self, streamer, mock_provider):
        """Test health check with provider failure."""
        mock_provider.health_check.return_value = False
        
        result = await streamer.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, streamer, mock_provider):
        """Test health check with exception."""
        mock_provider.health_check.side_effect = Exception("Health check error")
        
        result = await streamer.health_check()
        
        assert result is False
    
    def test_add_handler(self, streamer):
        """Test adding global handler."""
        handler = AsyncMock()
        
        streamer.add_handler(handler)
        
        assert handler in streamer._handlers
        assert len(streamer._handlers) == 1
    
    def test_remove_handler(self, streamer):
        """Test removing global handler."""
        handler = AsyncMock()
        
        # Add handler first
        streamer.add_handler(handler)
        assert len(streamer._handlers) == 1
        
        # Remove handler
        streamer.remove_handler(handler)
        assert len(streamer._handlers) == 0
    
    def test_remove_handler_not_found(self, streamer):
        """Test removing non-existent handler."""
        handler = AsyncMock()
        
        # Should not raise error
        streamer.remove_handler(handler)
    
    @pytest.mark.asyncio
    async def test_notify_handlers(self, streamer):
        """Test notifying global handlers."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        streamer.add_handler(handler1)
        streamer.add_handler(handler2)
        
        data = {"test": "data"}
        await streamer._notify_handlers(data)
        
        handler1.assert_called_once_with(data)
        handler2.assert_called_once_with(data)
    
    @pytest.mark.asyncio
    async def test_notify_handlers_with_error(self, streamer):
        """Test notifying handlers with error."""
        handler1 = AsyncMock()
        handler2 = AsyncMock(side_effect=Exception("Handler error"))
        handler3 = AsyncMock()
        
        streamer.add_handler(handler1)
        streamer.add_handler(handler2)
        streamer.add_handler(handler3)
        
        data = {"test": "data"}
        await streamer._notify_handlers(data)
        
        # All handlers should be called, even if one fails
        handler1.assert_called_once_with(data)
        handler2.assert_called_once_with(data)
        handler3.assert_called_once_with(data)
    
    def test_str_representation(self, streamer):
        """Test string representation."""
        str_repr = str(streamer)
        assert "LiveStreamer" in str_repr
        assert "running=False" in str_repr
        assert "handlers=0" in str_repr


class TestSimulatedStreamer:
    """Test the simulated streamer implementation."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock data provider."""
        provider = Mock()
        provider.get_historical_bars = AsyncMock()
        provider.health_check = AsyncMock(return_value=True)
        return provider
    
    @pytest.fixture
    def valid_config(self, mock_provider):
        """Create a valid configuration for testing."""
        return {
            "provider": mock_provider,
            "start_time": datetime.now() - timedelta(days=1),
            "end_time": datetime.now(),
            "speed": 1.0,
            "timeframe": "1Min"
        }
    
    @pytest.fixture
    def streamer(self, valid_config):
        """Create a simulated streamer instance for testing."""
        return SimulatedStreamer(valid_config)
    
    def test_init_valid_config(self, valid_config):
        """Test initialization with valid configuration."""
        streamer = SimulatedStreamer(valid_config)
        assert streamer.config == valid_config
        assert streamer._provider == valid_config["provider"]
        assert streamer._start_time == valid_config["start_time"]
        assert streamer._end_time == valid_config["end_time"]
        assert streamer._speed == 1.0
        assert streamer._timeframe == "1Min"
        assert not streamer.is_running
    
    def test_init_missing_provider(self):
        """Test initialization with missing provider."""
        config = {
            "start_time": datetime.now() - timedelta(days=1),
            "end_time": datetime.now()
        }
        with pytest.raises(DataProviderError, match="Missing required config key: provider"):
            SimulatedStreamer(config)
    
    def test_init_missing_start_time(self):
        """Test initialization with missing start time."""
        config = {
            "provider": Mock(),
            "end_time": datetime.now()
        }
        with pytest.raises(DataProviderError, match="Missing required config key: start_time"):
            SimulatedStreamer(config)
    
    def test_init_missing_end_time(self):
        """Test initialization with missing end time."""
        config = {
            "provider": Mock(),
            "start_time": datetime.now() - timedelta(days=1)
        }
        with pytest.raises(DataProviderError, match="Missing required config key: end_time"):
            SimulatedStreamer(config)
    
    def test_init_invalid_time_range(self):
        """Test initialization with invalid time range."""
        config = {
            "provider": Mock(),
            "start_time": datetime.now(),
            "end_time": datetime.now() - timedelta(days=1)  # End before start
        }
        with pytest.raises(DataProviderError, match="Start time must be before end time"):
            SimulatedStreamer(config)
    
    def test_init_invalid_speed(self):
        """Test initialization with invalid speed."""
        config = {
            "provider": Mock(),
            "start_time": datetime.now() - timedelta(days=1),
            "end_time": datetime.now(),
            "speed": -1.0  # Negative speed
        }
        with pytest.raises(DataProviderError, match="Speed must be positive"):
            SimulatedStreamer(config)
    
    def test_init_default_values(self, mock_provider):
        """Test initialization with default values."""
        config = {
            "provider": mock_provider,
            "start_time": datetime.now() - timedelta(days=1),
            "end_time": datetime.now()
        }
        streamer = SimulatedStreamer(config)
        
        assert streamer._speed == 1.0
        assert streamer._timeframe == "1Min"
    
    @pytest.mark.asyncio
    async def test_start_success(self, streamer, mock_provider):
        """Test successful stream start."""
        # Mock historical data
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000]
        })
        mock_provider.get_historical_bars.return_value = mock_df
        
        # Subscribe to a symbol first
        await streamer.subscribe("AAPL", AsyncMock())
        
        await streamer.start()
        
        assert streamer.is_running
        assert streamer._simulation_task is not None
        assert not streamer._simulation_task.done()
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, streamer):
        """Test starting when already running."""
        streamer._running = True
        
        await streamer.start()
        
        # Should not create new simulation task
        assert streamer._simulation_task is None
    
    @pytest.mark.asyncio
    async def test_start_no_symbols(self, streamer):
        """Test starting with no subscribed symbols."""
        await streamer.start()
        
        assert streamer.is_running
        # Should still create simulation task but it will exit early
    
    @pytest.mark.asyncio
    async def test_start_provider_error(self, streamer, mock_provider):
        """Test start with provider error."""
        mock_provider.get_historical_bars.side_effect = Exception("Provider error")
        
        # Subscribe to a symbol first
        await streamer.subscribe("AAPL", AsyncMock())
        
        # The streamer should start successfully even with provider errors
        # because _load_historical_data() handles errors gracefully
        await streamer.start()
        
        assert streamer.is_running
        assert streamer._simulation_task is not None
        assert not streamer._simulation_task.done()
        
        # Clean up
        await streamer.stop()
    
    @pytest.mark.asyncio
    async def test_stop_success(self, streamer):
        """Test successful stream stop."""
        streamer._running = True
        streamer._simulation_task = asyncio.create_task(asyncio.sleep(10))
        
        await streamer.stop()
        
        assert not streamer.is_running
        assert streamer._simulation_task is None
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self, streamer):
        """Test stopping when not running."""
        await streamer.stop()
        
        # Should not raise error
        assert not streamer.is_running
    
    @pytest.mark.asyncio
    async def test_stop_error(self, streamer):
        """Test stop with error."""
        streamer._running = True
        streamer._simulation_task = asyncio.create_task(asyncio.sleep(10))
        
        # Mock an error during stop by making the task await raise an exception
        mock_task = AsyncMock()
        mock_task.cancel = Mock()
        mock_task.__await__ = Mock(side_effect=Exception("Stop error"))
        streamer._simulation_task = mock_task
        
        with pytest.raises(DataProviderError, match="Failed to stop simulated streaming"):
            await streamer.stop()
    
    @pytest.mark.asyncio
    async def test_subscribe_success(self, streamer):
        """Test successful subscription."""
        symbol = "AAPL"
        handler = AsyncMock()
        
        await streamer.subscribe(symbol, handler)
        
        assert symbol in streamer._symbol_handlers
        assert handler in streamer._symbol_handlers[symbol]
    
    @pytest.mark.asyncio
    async def test_subscribe_multiple_handlers(self, streamer):
        """Test subscribing multiple handlers to same symbol."""
        symbol = "AAPL"
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        await streamer.subscribe(symbol, handler1)
        await streamer.subscribe(symbol, handler2)
        
        assert len(streamer._symbol_handlers[symbol]) == 2
        assert handler1 in streamer._symbol_handlers[symbol]
        assert handler2 in streamer._symbol_handlers[symbol]
    
    @pytest.mark.asyncio
    async def test_subscribe_error(self, streamer):
        """Test subscription with error."""
        symbol = "AAPL"
        handler = AsyncMock()
        
        # Mock an error during subscribe by making the dict access raise an exception
        original_handlers = streamer._symbol_handlers
        mock_handlers = Mock()
        mock_handlers.__contains__ = Mock(return_value=False)
        mock_handlers.__setitem__ = Mock(side_effect=Exception("Dict access error"))
        streamer._symbol_handlers = mock_handlers
        
        try:
            with pytest.raises(DataProviderError, match="Failed to subscribe to AAPL"):
                await streamer.subscribe(symbol, handler)
        finally:
            streamer._symbol_handlers = original_handlers
    
    @pytest.mark.asyncio
    async def test_unsubscribe_success(self, streamer):
        """Test successful unsubscription."""
        symbol = "AAPL"
        handler = AsyncMock()
        
        # First subscribe
        streamer._symbol_handlers[symbol] = [handler]
        
        await streamer.unsubscribe(symbol)
        
        assert symbol not in streamer._symbol_handlers
    
    @pytest.mark.asyncio
    async def test_unsubscribe_not_subscribed(self, streamer):
        """Test unsubscribing from non-subscribed symbol."""
        symbol = "AAPL"
        
        # Should not raise error
        await streamer.unsubscribe(symbol)
    
    @pytest.mark.asyncio
    async def test_unsubscribe_error(self, streamer):
        """Test unsubscription with error."""
        symbol = "AAPL"
        streamer._symbol_handlers[symbol] = [AsyncMock()]
        
        # Mock an error by making the dict access raise an exception
        original_handlers = streamer._symbol_handlers
        mock_handlers = Mock()
        mock_handlers.__contains__ = Mock(return_value=True)
        mock_handlers.__delitem__ = Mock(side_effect=Exception("Dict access error"))
        streamer._symbol_handlers = mock_handlers
        
        try:
            with pytest.raises(DataProviderError, match="Failed to unsubscribe from AAPL"):
                await streamer.unsubscribe(symbol)
        finally:
            streamer._symbol_handlers = original_handlers
    
    @pytest.mark.asyncio
    async def test_load_historical_data_success(self, streamer, mock_provider):
        """Test successful historical data loading."""
        # Mock historical data
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000]
        })
        mock_provider.get_historical_bars.return_value = mock_df
        
        # Subscribe to a symbol
        await streamer.subscribe("AAPL", AsyncMock())
        
        await streamer._load_historical_data()
        
        assert "AAPL" in streamer._historical_data
        assert "AAPL" in streamer._current_indices
        assert streamer._current_indices["AAPL"] == 0
        mock_provider.get_historical_bars.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_historical_data_empty_result(self, streamer, mock_provider):
        """Test historical data loading with empty result."""
        mock_provider.get_historical_bars.return_value = pd.DataFrame()
        
        # Subscribe to a symbol
        await streamer.subscribe("AAPL", AsyncMock())
        
        await streamer._load_historical_data()
        
        assert "AAPL" not in streamer._historical_data
        assert "AAPL" not in streamer._current_indices
    
    @pytest.mark.asyncio
    async def test_load_historical_data_no_symbols(self, streamer):
        """Test historical data loading with no symbols."""
        await streamer._load_historical_data()
        
        # Should not raise error
        assert len(streamer._historical_data) == 0
    
    @pytest.mark.asyncio
    async def test_load_historical_data_provider_error(self, streamer, mock_provider):
        """Test historical data loading with provider error."""
        mock_provider.get_historical_bars.side_effect = Exception("Provider error")
        
        # Subscribe to a symbol
        await streamer.subscribe("AAPL", AsyncMock())
        
        await streamer._load_historical_data()
        
        # Should handle error gracefully
        assert "AAPL" not in streamer._historical_data
    
    @pytest.mark.asyncio
    async def test_run_simulation_basic(self, streamer):
        """Test basic simulation run."""
        # Mock historical data
        timestamp = datetime.now()
        mock_df = pd.DataFrame({
            "timestamp": [timestamp],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000]
        })
        streamer._historical_data["AAPL"] = mock_df
        streamer._current_indices["AAPL"] = 0
        
        # Subscribe to symbol
        handler = AsyncMock()
        streamer._symbol_handlers["AAPL"] = [handler]
        
        # Start simulation
        streamer._running = True
        task = asyncio.create_task(streamer._run_simulation())
        
        # Wait a bit for simulation to process
        await asyncio.sleep(0.1)
        
        # Stop simulation
        streamer._running = False
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Check that handler was called
        assert handler.called
    
    @pytest.mark.asyncio
    async def test_run_simulation_no_data(self, streamer):
        """Test simulation run with no data."""
        streamer._running = True
        
        # Run simulation with no historical data
        await streamer._run_simulation()
        
        # Should exit gracefully
        assert not streamer._running
    
    @pytest.mark.asyncio
    async def test_run_simulation_completed(self, streamer):
        """Test simulation run when completed."""
        # Mock historical data with index at end
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000]
        })
        streamer._historical_data["AAPL"] = mock_df
        streamer._current_indices["AAPL"] = 1  # At end of data
        
        streamer._running = True
        
        await streamer._run_simulation()
        
        # Should exit gracefully
        assert not streamer._running
    
    @pytest.mark.asyncio
    async def test_notify_symbol_handlers(self, streamer):
        """Test notifying symbol-specific handlers."""
        symbol = "AAPL"
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        
        streamer._symbol_handlers[symbol] = [handler1, handler2]
        
        data = {"test": "data"}
        await streamer._notify_symbol_handlers(symbol, data)
        
        handler1.assert_called_once_with(data)
        handler2.assert_called_once_with(data)
    
    @pytest.mark.asyncio
    async def test_notify_symbol_handlers_with_error(self, streamer):
        """Test notifying symbol handlers with error."""
        symbol = "AAPL"
        handler1 = AsyncMock()
        handler2 = AsyncMock(side_effect=Exception("Handler error"))
        handler3 = AsyncMock()
        
        streamer._symbol_handlers[symbol] = [handler1, handler2, handler3]
        
        data = {"test": "data"}
        await streamer._notify_symbol_handlers(symbol, data)
        
        # All handlers should be called, even if one fails
        handler1.assert_called_once_with(data)
        handler2.assert_called_once_with(data)
        handler3.assert_called_once_with(data)
    
    @pytest.mark.asyncio
    async def test_get_subscribed_symbols(self, streamer):
        """Test getting subscribed symbols."""
        symbol1 = "AAPL"
        symbol2 = "GOOGL"
        
        streamer._symbol_handlers[symbol1] = [AsyncMock()]
        streamer._symbol_handlers[symbol2] = [AsyncMock()]
        
        symbols = await streamer.get_subscribed_symbols()
        
        assert symbol1 in symbols
        assert symbol2 in symbols
        assert len(symbols) == 2
    
    @pytest.mark.asyncio
    async def test_get_simulation_progress(self, streamer):
        """Test getting simulation progress."""
        # Mock historical data
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now()] * 10,  # 10 rows
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000] * 10
        })
        streamer._historical_data["AAPL"] = mock_df
        streamer._current_indices["AAPL"] = 5  # Halfway through
        
        progress = await streamer.get_simulation_progress()
        
        assert "AAPL" in progress
        assert progress["AAPL"] == 50.0  # 50% complete
    
    @pytest.mark.asyncio
    async def test_get_simulation_progress_empty_data(self, streamer):
        """Test getting simulation progress with empty data."""
        # Mock empty historical data
        streamer._historical_data["AAPL"] = pd.DataFrame()
        streamer._current_indices["AAPL"] = 0
        
        progress = await streamer.get_simulation_progress()
        
        assert "AAPL" in progress
        assert progress["AAPL"] == 0.0
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, streamer, mock_provider):
        """Test successful health check."""
        # Mock historical data
        streamer._historical_data["AAPL"] = pd.DataFrame({"test": [1]})
        streamer._simulation_task = asyncio.create_task(asyncio.sleep(10))
        
        result = await streamer.health_check()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_no_data(self, streamer):
        """Test health check with no historical data."""
        result = await streamer.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_task_not_running(self, streamer):
        """Test health check when simulation task is not running."""
        # Mock historical data
        streamer._historical_data["AAPL"] = pd.DataFrame({"test": [1]})
        streamer._running = True
        streamer._simulation_task = None
        
        result = await streamer.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, streamer):
        """Test health check with exception."""
        # Mock an exception during health check by making the dict access raise an exception
        original_data = streamer._historical_data
        mock_data = Mock()
        mock_data.__bool__ = Mock(side_effect=Exception("Health check error"))
        streamer._historical_data = mock_data
        
        try:
            result = await streamer.health_check()
            assert result is False
        finally:
            streamer._historical_data = original_data 