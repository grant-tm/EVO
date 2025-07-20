"""
Live data streamer implementation.

This module provides an implementation of the BaseStreamer interface
for real-time data streaming from live market data sources.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime
import logging

from .base_streamer import BaseStreamer
from evo.core.exceptions import DataProviderError
from evo.core.logging import get_logger

logger = get_logger(__name__)


class LiveStreamer(BaseStreamer):
    """
    Live data streamer for real-time market data.
    
    Provides real-time streaming capabilities for live market data
    from various data providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize live streamer.
        
        Args:
            config: Configuration dictionary containing:
                - provider: Data provider instance
                - symbols: List of symbols to stream
                - handlers: List of handler functions
        """
        super().__init__(config)
        
        # Symbol-specific handlers
        self._symbol_handlers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {}
        
        # Data provider
        self._provider = config.get("provider")
    
    def _validate_config(self) -> None:
        """Validate live streamer configuration."""
        required_keys = ["provider"]
        for key in required_keys:
            if key not in self.config:
                raise DataProviderError(f"Missing required config key: {key}")
        
        if self.config["provider"] is None:
            raise DataProviderError("Data provider is required for live streaming")
        
        if not hasattr(self.config["provider"], "subscribe_to_bars"):
            raise DataProviderError("Provider must support bar subscription")
    
    async def start(self) -> None:
        """
        Start the live data stream.
        
        Raises:
            DataProviderError: If streaming fails to start
        """
        try:
            if self._running:
                logger.warning("Live streamer is already running")
                return
            
            self._running = True
            logger.info("Starting live data stream")
            
            # Start the provider's streaming
            await self._provider.start_streaming()
            
        except Exception as e:
            self._running = False
            raise DataProviderError(f"Failed to start live streaming: {str(e)}")
    
    async def stop(self) -> None:
        """
        Stop the live data stream.
        
        Raises:
            DataProviderError: If streaming fails to stop
        """
        try:
            if not self._running:
                logger.warning("Live streamer is not running")
                return
            
            self._running = False
            logger.info("Stopping live data stream")
            
            # Note: Alpaca streamer doesn't have a stop method, so we just set running to False
            
        except Exception as e:
            raise DataProviderError(f"Failed to stop live streaming: {str(e)}")
    
    async def subscribe(
        self,
        symbol: str,
        handler: Callable[[Any], Awaitable[None]]
    ) -> None:
        """
        Subscribe to live data for a specific symbol.
        
        Args:
            symbol: Trading symbol to subscribe to
            handler: Async function to handle incoming data
            
        Raises:
            DataProviderError: If subscription fails
        """
        try:
            # Add to symbol handlers
            if symbol not in self._symbol_handlers:
                self._symbol_handlers[symbol] = []
            
            self._symbol_handlers[symbol].append(handler)
            
            # Subscribe to provider
            await self._provider.subscribe_to_bars(symbol, self._create_bar_handler(symbol))
            
            logger.info(f"Subscribed to live data for {symbol}")
            
        except Exception as e:
            raise DataProviderError(f"Failed to subscribe to {symbol}: {str(e)}")
    
    async def unsubscribe(self, symbol: str) -> None:
        """
        Unsubscribe from live data for a specific symbol.
        
        Args:
            symbol: Trading symbol to unsubscribe from
            
        Raises:
            DataProviderError: If unsubscription fails
        """
        try:
            if symbol in self._symbol_handlers:
                del self._symbol_handlers[symbol]
                logger.info(f"Unsubscribed from live data for {symbol}")
            
        except Exception as e:
            raise DataProviderError(f"Failed to unsubscribe from {symbol}: {str(e)}")
    
    def _create_bar_handler(self, symbol: str) -> Callable[[Any], Awaitable[None]]:
        """
        Create a bar handler for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Async function to handle incoming bars
        """
        async def bar_handler(bar_data: Any) -> None:
            """Handle incoming bar data for a symbol."""
            try:
                # Notify symbol-specific handlers
                if symbol in self._symbol_handlers:
                    for handler in self._symbol_handlers[symbol]:
                        await handler(bar_data)
                
                # Notify global handlers
                await self._notify_handlers(bar_data)
                
            except Exception as e:
                logger.error(f"Error in bar handler for {symbol}: {str(e)}")
        
        return bar_handler
    
    async def get_subscribed_symbols(self) -> List[str]:
        """
        Get list of currently subscribed symbols.
        
        Returns:
            List of subscribed symbols
        """
        return list(self._symbol_handlers.keys())
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the live streamer.
        
        Returns:
            True if streamer is healthy, False otherwise
        """
        try:
            # Check base health
            if not await super().health_check():
                return False
            
            # Check provider health
            if not await self._provider.health_check():
                return False
            
            # Check if we're running when expected
            if self._running and not self._provider.is_market_open():
                logger.warning("Market is closed but streamer is running")
            
            return True
            
        except Exception as e:
            logger.error(f"Live streamer health check failed: {str(e)}")
            return False 