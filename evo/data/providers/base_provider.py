"""
Abstract base class for data providers.

This module defines the interface that all data providers must implement
to be compatible with the EVO trading system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

from evo.core.exceptions import DataProviderError


class BaseDataProvider(ABC):
    """
    Abstract base class for data providers.
    
    All data providers must implement these methods to provide
    consistent interfaces for data access across different sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data provider with configuration.
        
        Args:
            config: Configuration dictionary containing provider-specific settings
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the provider configuration.
        
        Raises:
            DataProviderError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def get_historical_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: str = "1Min"
    ) -> pd.DataFrame:
        """
        Fetch historical bar data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            start_time: Start time for data fetch
            end_time: End time for data fetch
            timeframe: Timeframe for bars (e.g., '1Min', '5Min', '1Hour')
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            DataProviderError: If data fetch fails
        """
        pass
    
    @abstractmethod
    async def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest bar data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary containing latest bar data or None if not available
        """
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """
        Get list of available symbols from the provider.
        
        Returns:
            List of available trading symbols
        """
        pass
    
    @abstractmethod
    async def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        pass
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the data provider.
        
        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Basic health check - try to get available symbols
            await self.get_symbols()
            return True
        except Exception as e:
            raise DataProviderError(f"Health check failed: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the data provider."""
        return f"{self.__class__.__name__}(config={self.config})" 