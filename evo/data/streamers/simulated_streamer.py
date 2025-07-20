"""
Simulated data streamer implementation.

This module provides an implementation of the BaseStreamer interface
for simulated data streaming, useful for backtesting and development.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
import pandas as pd
from types import SimpleNamespace

from .base_streamer import BaseStreamer
from evo.core.exceptions import DataProviderError
from evo.core.logging import get_logger

logger = get_logger(__name__)


class SimulatedStreamer(BaseStreamer):
    """
    Simulated data streamer for backtesting and development.
    
    Provides simulated streaming capabilities using historical data
    to mimic real-time market data feeds.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulated streamer.
        
        Args:
            config: Configuration dictionary containing:
                - provider: Data provider instance
                - symbols: List of symbols to stream
                - start_time: Start time for simulation
                - end_time: End time for simulation
                - speed: Simulation speed multiplier (default: 1.0)
                - timeframe: Data timeframe (default: "1Min")
        """
        # Set up attributes before calling parent constructor
        self._start_time = config.get("start_time")
        self._end_time = config.get("end_time")
        self._speed = config.get("speed", 1.0)
        self._timeframe = config.get("timeframe", "1Min")
        
        super().__init__(config)
        
        # Symbol-specific handlers
        self._symbol_handlers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {}
        
        # Data provider
        self._provider = config.get("provider")
        if self._provider is None:
            raise DataProviderError("Data provider is required for simulated streaming")
        
        # Historical data cache
        self._historical_data: Dict[str, pd.DataFrame] = {}
        self._current_indices: Dict[str, int] = {}
        
        # Task for running simulation
        self._simulation_task: Optional[asyncio.Task] = None
    
    def _validate_config(self) -> None:
        """Validate simulated streamer configuration."""
        required_keys = ["provider", "start_time", "end_time"]
        for key in required_keys:
            if key not in self.config:
                raise DataProviderError(f"Missing required config key: {key}")
        
        if self._start_time >= self._end_time:
            raise DataProviderError("Start time must be before end time")
        
        if self._speed <= 0:
            raise DataProviderError("Speed must be positive")
    
    async def start(self) -> None:
        """
        Start the simulated data stream.
        
        Raises:
            DataProviderError: If streaming fails to start
        """
        try:
            if self._running:
                logger.warning("Simulated streamer is already running")
                return
            
            # Load historical data for all subscribed symbols
            await self._load_historical_data()
            
            self._running = True
            logger.info(f"Starting simulated data stream from {self._start_time} to {self._end_time}")
            
            # Start simulation task
            self._simulation_task = asyncio.create_task(self._run_simulation())
            
        except Exception as e:
            self._running = False
            raise DataProviderError(f"Failed to start simulated streaming: {str(e)}")
    
    async def stop(self) -> None:
        """
        Stop the simulated data stream.
        
        Raises:
            DataProviderError: If streaming fails to stop
        """
        try:
            if not self._running:
                logger.warning("Simulated streamer is not running")
                return
            
            self._running = False
            logger.info("Stopping simulated data stream")
            
            # Cancel simulation task
            if self._simulation_task:
                self._simulation_task.cancel()
                try:
                    await self._simulation_task
                except asyncio.CancelledError:
                    pass
                self._simulation_task = None
            
        except Exception as e:
            raise DataProviderError(f"Failed to stop simulated streaming: {str(e)}")
    
    async def subscribe(
        self,
        symbol: str,
        handler: Callable[[Any], Awaitable[None]]
    ) -> None:
        """
        Subscribe to simulated data for a specific symbol.
        
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
            
            logger.info(f"Subscribed to simulated data for {symbol}")
            
        except Exception as e:
            raise DataProviderError(f"Failed to subscribe to {symbol}: {str(e)}")
    
    async def unsubscribe(self, symbol: str) -> None:
        """
        Unsubscribe from simulated data for a specific symbol.
        
        Args:
            symbol: Trading symbol to unsubscribe from
            
        Raises:
            DataProviderError: If unsubscription fails
        """
        try:
            if symbol in self._symbol_handlers:
                del self._symbol_handlers[symbol]
                logger.info(f"Unsubscribed from simulated data for {symbol}")
            
        except Exception as e:
            raise DataProviderError(f"Failed to unsubscribe from {symbol}: {str(e)}")
    
    async def _load_historical_data(self) -> None:
        """Load historical data for all subscribed symbols."""
        symbols = list(self._symbol_handlers.keys())
        
        if not symbols:
            logger.warning("No symbols subscribed for simulation")
            return
        
        logger.info(f"Loading historical data for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                # Fetch historical data
                df = await self._provider.get_historical_bars(
                    symbol=symbol,
                    start_time=self._start_time,
                    end_time=self._end_time,
                    timeframe=self._timeframe
                )
                
                if df.empty:
                    logger.warning(f"No historical data found for {symbol}")
                    continue
                
                # Store data and initialize index
                self._historical_data[symbol] = df
                self._current_indices[symbol] = 0
                
                logger.info(f"Loaded {len(df)} bars for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load historical data for {symbol}: {str(e)}")
    
    async def _run_simulation(self) -> None:
        """Run the simulation loop."""
        try:
            while self._running:
                # Check if we have data for any symbols
                if not self._historical_data:
                    logger.warning("No historical data available for simulation")
                    break
                
                # Find the next timestamp across all symbols
                next_timestamp = None
                symbols_with_data = []
                
                for symbol, df in self._historical_data.items():
                    current_idx = self._current_indices[symbol]
                    if current_idx < len(df):
                        timestamp = df.iloc[current_idx]["timestamp"]
                        if next_timestamp is None or timestamp < next_timestamp:
                            next_timestamp = timestamp
                        symbols_with_data.append(symbol)
                
                if not symbols_with_data:
                    logger.info("Simulation completed - no more data")
                    break
                
                # Process all symbols at this timestamp
                for symbol in symbols_with_data:
                    df = self._historical_data[symbol]
                    current_idx = self._current_indices[symbol]
                    
                    if current_idx < len(df):
                        row = df.iloc[current_idx]
                        if row["timestamp"] == next_timestamp:
                            # Create bar object
                            bar = SimpleNamespace(
                                timestamp=row["timestamp"],
                                symbol=symbol,
                                open=row["open"],
                                high=row["high"],
                                low=row["low"],
                                close=row["close"],
                                volume=row["volume"]
                            )
                            
                            # Notify handlers
                            await self._notify_symbol_handlers(symbol, bar)
                            
                            # Advance index
                            self._current_indices[symbol] += 1
                
                # Wait for next iteration based on speed
                await asyncio.sleep(1.0 / self._speed)
            
            # Set running to False when simulation ends normally
            self._running = False
                
        except asyncio.CancelledError:
            logger.info("Simulation cancelled")
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            self._running = False
    
    async def _notify_symbol_handlers(self, symbol: str, data: Any) -> None:
        """
        Notify symbol-specific handlers with incoming data.
        
        Args:
            symbol: Trading symbol
            data: Data to send to handlers
        """
        if symbol in self._symbol_handlers:
            for handler in self._symbol_handlers[symbol]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Handler {handler.__name__} failed for {symbol}: {str(e)}")
    
    async def get_subscribed_symbols(self) -> List[str]:
        """
        Get list of currently subscribed symbols.
        
        Returns:
            List of subscribed symbols
        """
        return list(self._symbol_handlers.keys())
    
    async def get_simulation_progress(self) -> Dict[str, float]:
        """
        Get simulation progress for each symbol.
        
        Returns:
            Dictionary mapping symbols to progress percentage
        """
        progress = {}
        for symbol, df in self._historical_data.items():
            current_idx = self._current_indices[symbol]
            total_bars = len(df)
            if total_bars > 0:
                progress[symbol] = (current_idx / total_bars) * 100
            else:
                progress[symbol] = 0.0
        return progress
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the simulated streamer.
        
        Returns:
            True if streamer is healthy, False otherwise
        """
        try:
            # Check base health
            if not await super().health_check():
                return False
            
            # Check if we have data
            if not self._historical_data:
                logger.warning("No historical data loaded")
                return False
            
            # Check if simulation task is running
            if self._running and (self._simulation_task is None or self._simulation_task.done()):
                logger.warning("Simulation task is not running")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Simulated streamer health check failed: {str(e)}")
            return False 