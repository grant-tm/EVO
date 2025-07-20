"""
Abstract base class for data streamers.

This module defines the interface that all data streamers must implement
to be compatible with the EVO trading system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime
import asyncio

from evo.core.exceptions import DataProviderError
from evo.core.logging import get_logger

logger = get_logger(__name__)


class BaseStreamer(ABC):
    """
    Abstract base class for data streamers.
    
    All data streamers must implement these methods to provide
    consistent interfaces for data streaming across different sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data streamer with configuration.
        
        Args:
            config: Configuration dictionary containing streamer-specific settings
        """
        self.config = config
        self._validate_config()
        self._running = False
        self._handlers: List[Callable[[Any], Awaitable[None]]] = []
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the streamer configuration.
        
        Raises:
            DataProviderError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the data stream.
        
        Raises:
            DataProviderError: If streaming fails to start
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the data stream.
        
        Raises:
            DataProviderError: If streaming fails to stop
        """
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        symbol: str,
        handler: Callable[[Any], Awaitable[None]]
    ) -> None:
        """
        Subscribe to data for a specific symbol.
        
        Args:
            symbol: Trading symbol to subscribe to
            handler: Async function to handle incoming data
            
        Raises:
            DataProviderError: If subscription fails
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbol: str) -> None:
        """
        Unsubscribe from data for a specific symbol.
        
        Args:
            symbol: Trading symbol to unsubscribe from
            
        Raises:
            DataProviderError: If unsubscription fails
        """
        pass
    
    def add_handler(self, handler: Callable[[Any], Awaitable[None]]) -> None:
        """
        Add a global handler for all incoming data.
        
        Args:
            handler: Async function to handle incoming data
        """
        self._handlers.append(handler)
        logger.info(f"Added global handler: {handler.__name__}")
    
    def remove_handler(self, handler: Callable[[Any], Awaitable[None]]) -> None:
        """
        Remove a global handler.
        
        Args:
            handler: Handler function to remove
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
            logger.info(f"Removed global handler: {handler.__name__}")
    
    async def _notify_handlers(self, data: Any) -> None:
        """
        Notify all registered handlers with incoming data.
        
        Args:
            data: Data to send to handlers
        """
        for handler in self._handlers:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Handler {handler.__name__} failed: {str(e)}")
    
    @property
    def is_running(self) -> bool:
        """Check if the streamer is currently running."""
        return self._running
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the streamer.
        
        Returns:
            True if streamer is healthy, False otherwise
        """
        try:
            # Basic health check - verify configuration
            self._validate_config()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """String representation of the streamer."""
        return f"{self.__class__.__name__}(running={self._running}, handlers={len(self._handlers)})" 