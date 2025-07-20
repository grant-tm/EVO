"""
Alpaca data provider implementation.

This module provides an implementation of the BaseDataProvider interface
for fetching market data from Alpaca Markets.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import pandas as pd
import logging

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest

from .base_provider import BaseDataProvider
from evo.core.exceptions import DataProviderError
from evo.core.logging import get_logger

logger = get_logger(__name__)


class AlpacaDataProvider(BaseDataProvider):
    """
    Alpaca Markets data provider implementation.
    
    Provides access to real-time and historical market data through
    Alpaca's API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpaca data provider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key: Alpaca API key
                - api_secret: Alpaca API secret
                - paper: Whether to use paper trading (default: True)
                - base_url: Base URL for API (optional)
        """
        super().__init__(config)
        
        # Initialize Alpaca clients
        self._historical_client = None
        self._stream_client = None
        self._trading_client = None
        
        # Timeframe mapping - Alpaca only supports certain timeframes
        self._timeframe_map = {
            "1Min": TimeFrame.Minute,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day
        }
    
    def _validate_config(self) -> None:
        """Validate Alpaca configuration."""
        required_keys = ["api_key", "api_secret"]
        for key in required_keys:
            if key not in self.config:
                raise DataProviderError(f"Missing required config key: {key}")
        
        if not self.config["api_key"] or not self.config["api_secret"]:
            raise DataProviderError("API key and secret cannot be empty")
    
    @property
    def historical_client(self) -> StockHistoricalDataClient:
        """Get or create historical data client."""
        if self._historical_client is None:
            self._historical_client = StockHistoricalDataClient(
                api_key=self.config["api_key"],
                secret_key=self.config["api_secret"],
                url_override=self.config.get("base_url")
            )
        return self._historical_client
    
    @property
    def stream_client(self) -> StockDataStream:
        """Get or create streaming data client."""
        if self._stream_client is None:
            self._stream_client = StockDataStream(
                api_key=self.config["api_key"],
                secret_key=self.config["api_secret"],
                url_override=self.config.get("base_url")
            )
        return self._stream_client
    
    @property
    def trading_client(self) -> TradingClient:
        """Get or create trading client."""
        if self._trading_client is None:
            self._trading_client = TradingClient(
                api_key=self.config["api_key"],
                secret_key=self.config["api_secret"],
                paper=self.config.get("paper", True),
                url_override=self.config.get("base_url")
            )
        return self._trading_client
    
    async def get_historical_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: str = "1Min"
    ) -> pd.DataFrame:
        """
        Fetch historical bar data from Alpaca.
        
        Args:
            symbol: Trading symbol
            start_time: Start time for data fetch
            end_time: End time for data fetch
            timeframe: Timeframe for bars
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            DataProviderError: If data fetch fails
        """
        try:
            # Convert timeframe string to Alpaca TimeFrame
            alpaca_timeframe = self._timeframe_map.get(timeframe)
            if alpaca_timeframe is None:
                supported_timeframes = list(self._timeframe_map.keys())
                raise DataProviderError(
                    f"Unsupported timeframe: {timeframe}. "
                    f"Supported timeframes: {supported_timeframes}"
                )
            
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start_time,
                end=end_time
            )
            
            # Fetch data
            bars = self.historical_client.get_stock_bars(request)
            
            if bars is None or bars.df.empty:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = bars.df
            
            # Handle multi-symbol response
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level="symbol")
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            raise DataProviderError(f"Failed to fetch historical data for {symbol}: {str(e)}")
    
    async def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest bar data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary containing latest bar data or None
        """
        try:
            # Get the most recent bar (last 1 minute)
            end_time = datetime.now(timezone.utc)
            start_time = end_time.replace(minute=end_time.minute - 1)
            
            df = await self.get_historical_bars(symbol, start_time, end_time, "1Min")
            
            if df.empty:
                return None
            
            # Get the latest bar
            latest = df.iloc[-1]
            
            return {
                "timestamp": latest["timestamp"],
                "symbol": symbol,
                "open": latest["open"],
                "high": latest["high"],
                "low": latest["low"],
                "close": latest["close"],
                "volume": latest["volume"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol}: {str(e)}")
            return None
    
    async def get_symbols(self) -> List[str]:
        """
        Get list of available symbols from Alpaca.
        
        Returns:
            List of available trading symbols
        """
        try:
            # Get all active assets
            request = GetAssetsRequest(status="active")
            assets = self.trading_client.get_all_assets(request)
            
            # Filter for tradeable assets
            symbols = [asset.symbol for asset in assets if asset.tradable]
            
            logger.info(f"Found {len(symbols)} tradeable symbols")
            return symbols
            
        except Exception as e:
            raise DataProviderError(f"Failed to get symbols: {str(e)}")
    
    async def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market status: {str(e)}")
            return False
    
    async def subscribe_to_bars(
        self,
        symbol: str,
        handler: callable
    ) -> None:
        """
        Subscribe to real-time bar data for a symbol.
        
        Args:
            symbol: Trading symbol to subscribe to
            handler: Async function to handle incoming bars
        """
        try:
            self.stream_client.subscribe_bars(handler, symbol)
            logger.info(f"Subscribed to bars for {symbol}")
        except Exception as e:
            raise DataProviderError(f"Failed to subscribe to bars for {symbol}: {str(e)}")
    
    async def start_streaming(self) -> None:
        """Start the streaming data feed."""
        try:
            await self.stream_client._run_forever()
        except Exception as e:
            raise DataProviderError(f"Failed to start streaming: {str(e)}")
    
    def __del__(self):
        """Cleanup resources."""
        if self._historical_client:
            del self._historical_client
        if self._stream_client:
            del self._stream_client
        if self._trading_client:
            del self._trading_client 