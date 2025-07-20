"""
Tests for data providers.

This module contains tests for the data provider implementations.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from types import SimpleNamespace

from evo.data.providers.base_provider import BaseDataProvider
from evo.data.providers.alpaca_provider import AlpacaDataProvider
from evo.core.exceptions import DataProviderError

pytestmark = [
    pytest.mark.unit,
    pytest.mark.data,
    pytest.mark.providers
]

class TestBaseDataProvider:
    """Test the abstract base data provider."""
    
    def test_abstract_methods(self):
        """Test that BaseDataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataProvider({})


class TestAlpacaDataProvider:
    """Test the Alpaca data provider implementation."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return SimpleNamespace(
            alpaca=SimpleNamespace(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )
        )
    
    @pytest.fixture
    def provider(self, mock_config):
        """Create a provider instance for testing."""
        # Create mock clients
        mock_historical_client = Mock()
        mock_stream_client = Mock()
        mock_trading_client = Mock()
        
        with patch('evo.data.providers.alpaca_provider.StockHistoricalDataClient', return_value=mock_historical_client), \
             patch('evo.data.providers.alpaca_provider.StockDataStream', return_value=mock_stream_client), \
             patch('evo.data.providers.alpaca_provider.TradingClient', return_value=mock_trading_client):
            provider = AlpacaDataProvider(mock_config)
            # Manually set the mock clients
            provider._historical_client = mock_historical_client
            provider._stream_client = mock_stream_client
            provider._trading_client = mock_trading_client
            return provider
    
    def test_init_valid_config(self, mock_config):
        """Test initialization with valid configuration."""
        with patch('evo.data.providers.alpaca_provider.StockHistoricalDataClient'), \
             patch('evo.data.providers.alpaca_provider.StockDataStream'), \
             patch('evo.data.providers.alpaca_provider.TradingClient'):
            provider = AlpacaDataProvider(mock_config)
            assert provider.config == mock_config
    
    def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        config = SimpleNamespace(alpaca=SimpleNamespace(api_secret="test_secret"))
        with pytest.raises(DataProviderError, match="Missing required config key: api_key"):
            AlpacaDataProvider(config)
    
    def test_init_missing_api_secret(self):
        """Test initialization with missing API secret."""
        config = SimpleNamespace(alpaca=SimpleNamespace(api_key="test_key"))
        with pytest.raises(DataProviderError, match="Missing required config key: api_secret"):
            AlpacaDataProvider(config)
    
    def test_init_empty_credentials(self):
        """Test initialization with empty credentials."""
        config = SimpleNamespace(alpaca=SimpleNamespace(api_key="", api_secret=""))
        with pytest.raises(DataProviderError, match="API key and secret cannot be empty"):
            AlpacaDataProvider(config)
    
    @pytest.mark.asyncio
    async def test_get_historical_bars_success(self, provider):
        """Test successful historical bars retrieval."""
        # Mock data
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000]
        })
        
        # Mock the historical client method
        mock_bars = Mock()
        mock_bars.df = mock_df
        provider._historical_client.get_stock_bars.return_value = mock_bars
        
        # Test
        result = await provider.get_historical_bars(
            symbol="AAPL",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            timeframe="1Min"
        )
        
        assert not result.empty
        assert len(result) == 1
        assert "timestamp" in result.columns
    
    @pytest.mark.asyncio
    async def test_get_historical_bars_empty_result(self, provider):
        """Test historical bars retrieval with empty result."""
        # Mock empty data
        mock_bars = Mock()
        mock_bars.df = pd.DataFrame()
        provider._historical_client.get_stock_bars.return_value = mock_bars
        
        # Test
        result = await provider.get_historical_bars(
            symbol="AAPL",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        
        assert result.empty
    
    @pytest.mark.asyncio
    async def test_get_historical_bars_unsupported_timeframe(self, provider):
        """Test historical bars retrieval with unsupported timeframe."""
        with pytest.raises(DataProviderError, match="Unsupported timeframe: invalid"):
            await provider.get_historical_bars(
                symbol="AAPL",
                start_time=datetime.now() - timedelta(days=1),
                end_time=datetime.now(),
                timeframe="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_get_latest_bar_success(self, provider):
        """Test successful latest bar retrieval."""
        # Mock data
        mock_df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000]
        })
        
        # Mock the historical client method
        mock_bars = Mock()
        mock_bars.df = mock_df
        provider._historical_client.get_stock_bars.return_value = mock_bars
        
        # Test
        result = await provider.get_latest_bar("AAPL")
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert "timestamp" in result
        assert "close" in result
    
    @pytest.mark.asyncio
    async def test_get_latest_bar_empty_result(self, provider):
        """Test latest bar retrieval with empty result."""
        # Mock empty data
        mock_bars = Mock()
        mock_bars.df = pd.DataFrame()
        provider._historical_client.get_stock_bars.return_value = mock_bars
        
        # Test
        result = await provider.get_latest_bar("AAPL")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_symbols_success(self, provider):
        """Test successful symbols retrieval."""
        # Mock assets
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.tradable = True
        
        provider.trading_client.get_all_assets.return_value = [mock_asset]
        
        # Test
        result = await provider.get_symbols()
        
        assert "AAPL" in result
        assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_is_market_open_true(self, provider):
        """Test market open check when market is open."""
        # Mock clock
        mock_clock = Mock()
        mock_clock.is_open = True
        provider.trading_client.get_clock.return_value = mock_clock
        
        # Test
        result = await provider.is_market_open()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_is_market_open_false(self, provider):
        """Test market open check when market is closed."""
        # Mock clock
        mock_clock = Mock()
        mock_clock.is_open = False
        provider.trading_client.get_clock.return_value = mock_clock
        
        # Test
        result = await provider.is_market_open()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check."""
        # Mock successful symbols retrieval
        provider.trading_client.get_all_assets.return_value = []
        
        # Test
        result = await provider.health_check()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test health check failure."""
        # Mock failed symbols retrieval
        provider.trading_client.get_all_assets.side_effect = Exception("API Error")
        
        # Test
        with pytest.raises(DataProviderError, match="Health check failed"):
            await provider.health_check()
    
    def test_timeframe_mapping(self, provider):
        """Test timeframe mapping."""
        assert "1Min" in provider._timeframe_map
        assert "1Hour" in provider._timeframe_map
        assert "1Day" in provider._timeframe_map 