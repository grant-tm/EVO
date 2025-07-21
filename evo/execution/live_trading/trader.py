"""
Live trading orchestration system.

This module provides the main trading engine that coordinates data streaming,
model inference, order execution, and risk management for live trading.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from evo.core.exceptions import BrokerError, RiskError
from evo.core.logging import get_logger
from evo.core.config import Config
from evo.data.providers.base_provider import BaseDataProvider
from evo.data.processors.feature_engineer import FeatureEngineer
from evo.data.processors.normalizer import DataNormalizer
from evo.models.environments.trading_env import TradingEnv
from ..brokers.base_broker import BaseBroker, Order, OrderSide, OrderType, OrderStatus
from ..risk.risk_manager import RiskManager
from ..risk.position_manager import PositionManager
from .monitoring import TradingMonitor


@dataclass
class TradingState:
    """Current trading state."""
    symbol: str
    current_price: Optional[float] = None
    current_position: Optional[float] = None
    account_equity: float = 0.0
    is_market_open: bool = False
    last_action: Optional[int] = None
    last_action_time: Optional[datetime] = None
    total_trades: int = 0
    total_pnl: float = 0.0
    is_trading: bool = False


class LiveTrader:
    """
    Live trading orchestration system.
    
    This class coordinates all aspects of live trading including:
    - Data streaming and preprocessing
    - Model inference and decision making
    - Order execution and position management
    - Risk management and monitoring
    """
    
    def __init__(self, config: Config, broker: BaseBroker, data_provider: BaseDataProvider):
        """
        Initialize live trader.
        
        Args:
            config: Trading configuration
            broker: Broker for order execution
            data_provider: Data provider for market data
        """
        self.logger = get_logger(__name__)
        self.config = config
        self.broker = broker
        self.data_provider = data_provider
        
        # Initialize components
        self.risk_manager = RiskManager(config)
        self.position_manager = PositionManager(config)
        self.monitor = TradingMonitor()
        
        # Trading state
        self.state = TradingState(symbol=config.trading.symbol)
        self.model: Optional[PPO] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.normalizer: Optional[DataNormalizer] = None
        
        # Data buffers
        self.price_history: List[float] = []
        self.feature_history: List[np.ndarray] = []
        self.action_history: List[int] = []
        
        # Trading session
        self.session_id = str(uuid.uuid4())
        self.start_time = None
        self.is_running = False
        
        # Callbacks
        self.on_trade_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        
        self.logger.info(f"Live trader initialized for {config.trading.symbol}")
    
    async def start(self, model_path: Path) -> None:
        """
        Start live trading.
        
        Args:
            model_path: Path to trained model file
        """
        try:
            self.logger.info("Starting live trading session")
            self.start_time = datetime.now()
            self.is_running = True
            
            # Load model and components
            await self._load_model(model_path)
            await self._initialize_components()
            
            # Initialize account state
            await self._update_account_state()
            
            # Start monitoring
            self.monitor.start()
            
            # Start trading loop
            await self._trading_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start live trading: {str(e)}")
            if self.on_error_callback:
                self.on_error_callback(e)
            raise
    
    async def stop(self) -> None:
        """Stop live trading."""
        self.logger.info("Stopping live trading")
        self.is_running = False
        
        # Close all positions if configured
        if self.config.trading.close_positions_on_stop:
            await self._close_all_positions()
        
        # Stop monitoring
        self.monitor.stop()
        
        # Close broker connection
        await self.broker.close()
        
        self.logger.info("Live trading stopped")
    
    async def _load_model(self, model_path: Path) -> None:
        """Load the trained model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        self.model = PPO.load(str(model_path))
        
        # Initialize feature engineering
        self.feature_engineer = FeatureEngineer(self.config.data.features)
        self.normalizer = DataNormalizer()
        
        self.logger.info("Model loaded successfully")
    
    async def _initialize_components(self) -> None:
        """Initialize trading components."""
        # Initialize risk manager with initial capital
        account = await self.broker.get_account()
        self.risk_manager.set_initial_capital(account.equity)
        
        # Load historical data for feature calculation
        await self._load_historical_data()
        
        self.logger.info("Trading components initialized")
    
    async def _load_historical_data(self) -> None:
        """Load historical data for feature calculation."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.config.data.seq_len * 2)
        
        self.logger.info(f"Loading historical data from {start_time} to {end_time}")
        
        try:
            historical_data = await self.data_provider.get_historical_bars(
                self.config.trading.symbol,
                start_time,
                end_time,
                "1Min"
            )
            
            # Process historical data
            for _, row in historical_data.iterrows():
                self.price_history.append(row['close'])
                
                if len(self.price_history) > self.config.data.seq_len:
                    self.price_history.pop(0)
            
            self.logger.info(f"Loaded {len(historical_data)} historical bars")
            
        except Exception as e:
            self.logger.warning(f"Failed to load historical data: {str(e)}")
    
    async def _trading_loop(self) -> None:
        """Main trading loop."""
        self.logger.info("Starting trading loop")
        
        while self.is_running:
            try:
                # Check if we should stop trading
                should_stop, reason = self.risk_manager.should_stop_trading()
                if should_stop:
                    self.logger.warning(f"Stopping trading: {reason}")
                    break
                
                # Update market state
                await self._update_market_state()
                
                # Check if market is open
                if not self.state.is_market_open:
                    self.logger.debug("Market is closed, waiting...")
                    await asyncio.sleep(60)  # Wait 1 minute
                    continue
                
                # Get latest market data
                latest_data = await self._get_latest_data()
                if latest_data is None:
                    await asyncio.sleep(5)  # Wait 5 seconds
                    continue
                
                # Update price history
                self._update_price_history(latest_data['close'])
                
                # Calculate features
                features = await self._calculate_features()
                if features is None:
                    continue
                
                # Get model prediction
                action = await self._get_model_prediction(features)
                
                # Execute trading decision
                await self._execute_trading_decision(action, latest_data['close'])
                
                # Update monitoring
                self.monitor.update_state(self.state)
                
                # Sleep between iterations
                await asyncio.sleep(self.config.trading.update_interval or 30)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                if self.on_error_callback:
                    self.on_error_callback(e)
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _update_market_state(self) -> None:
        """Update current market state."""
        try:
            # Check if market is open
            self.state.is_market_open = await self.broker.is_market_open()
            
            # Update account state
            await self._update_account_state()
            
            # Update positions
            positions = await self.broker.get_positions()
            self.risk_manager.update_positions(positions)
            
            # Update position manager
            for position in positions:
                if position.symbol == self.config.trading.symbol:
                    self.state.current_position = position.quantity
                    break
            else:
                self.state.current_position = 0.0
            
        except Exception as e:
            self.logger.error(f"Error updating market state: {str(e)}")
    
    async def _update_account_state(self) -> None:
        """Update account state."""
        try:
            account = await self.broker.get_account()
            self.state.account_equity = account.equity
            self.risk_manager.update_account(account)
        except Exception as e:
            self.logger.error(f"Error updating account state: {str(e)}")
    
    async def _get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest market data."""
        try:
            latest_bar = await self.data_provider.get_latest_bar(self.config.trading.symbol)
            if latest_bar:
                return latest_bar
            
            # Fallback to broker price
            price = await self.broker.get_latest_price(self.config.trading.symbol)
            if price:
                return {'close': price, 'timestamp': datetime.now()}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {str(e)}")
            return None
    
    def _update_price_history(self, price: float) -> None:
        """Update price history buffer."""
        self.price_history.append(price)
        
        # Keep only the required number of prices
        if len(self.price_history) > self.config.data.seq_len:
            self.price_history.pop(0)
    
    async def _calculate_features(self) -> Optional[np.ndarray]:
        """Calculate features for model input."""
        try:
            if len(self.price_history) < self.config.data.seq_len:
                return None
            
            # Create DataFrame for feature engineering
            df = pd.DataFrame({
                'close': self.price_history,
                'open': self.price_history,  # Simplified for now
                'high': self.price_history,
                'low': self.price_history,
                'volume': [1000] * len(self.price_history)  # Placeholder
            })
            
            # Calculate features
            features = self.feature_engineer.calculate_features(df)
            
            # Normalize features
            features_normalized = self.normalizer.transform(features)
            
            return features_normalized.values[-1]  # Return latest features
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            return None
    
    async def _get_model_prediction(self, features: np.ndarray) -> int:
        """Get model prediction for current state."""
        try:
            # Reshape features for model input
            features_reshaped = features.reshape(1, -1)
            
            # Get model prediction
            action, _ = self.model.predict(features_reshaped, deterministic=True)
            
            action_value = int(action[0])
            self.logger.debug(f"Model prediction: {action_value}")
            return action_value
            
        except Exception as e:
            self.logger.error(f"Error getting model prediction: {str(e)}")
            return 1  # Default to hold
    
    async def _execute_trading_decision(self, action: int, current_price: float) -> None:
        """Execute trading decision based on model action."""
        try:
            # Update current price
            self.state.current_price = current_price
            
            # Record the action that was executed
            self.state.last_action = action
            self.state.last_action_time = datetime.now()
            
            # Map action to trading decision
            if action == 0:  # Sell
                await self._execute_sell_order(current_price)
            elif action == 2:  # Buy
                await self._execute_buy_order(current_price)
            else:  # Hold (action == 1)
                # Check for stop loss/take profit on existing positions
                await self._check_position_exits(current_price)
            
        except Exception as e:
            self.logger.error(f"Error executing trading decision: {str(e)}")
    
    async def _execute_buy_order(self, current_price: float) -> None:
        """Execute buy order."""
        if self.state.current_position is not None and self.state.current_position > 0:
            self.logger.debug("Already have long position, skipping buy")
            return
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.config.trading.symbol, current_price, self.state.account_equity
        )
        
        if position_size <= 0:
            self.logger.debug("Position size too small, skipping buy")
            return
        
        # Create buy order
        order = Order(
            id="",
            symbol=self.config.trading.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position_size,
            metadata={'order_type': 'model_buy', 'session_id': self.session_id}
        )
        
        # Validate order
        account = await self.broker.get_account()
        is_valid, reason = self.risk_manager.validate_order(order, account)
        
        if not is_valid:
            self.logger.warning(f"Buy order rejected: {reason}")
            return
        
        # Submit order
        submitted_order = await self.broker.submit_order(order)
        self.risk_manager.add_order(submitted_order)
        
        self.logger.info(f"Buy order submitted: {submitted_order.id} for {position_size} shares at ${current_price:.2f}")
        
        # Update trade count
        self.state.total_trades += 1
        
        # Call trade callback
        if self.on_trade_callback:
            self.on_trade_callback(submitted_order)
    
    async def _execute_sell_order(self, current_price: float) -> None:
        """Execute sell order."""
        if self.state.current_position is None or self.state.current_position <= 0:
            self.logger.debug("No position to sell, skipping sell")
            return
        
        # Create sell order
        order = Order(
            id="",
            symbol=self.config.trading.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=abs(self.state.current_position),
            metadata={'order_type': 'model_sell', 'session_id': self.session_id}
        )
        
        # Validate order
        account = await self.broker.get_account()
        is_valid, reason = self.risk_manager.validate_order(order, account)
        
        if not is_valid:
            self.logger.warning(f"Sell order rejected: {reason}")
            return
        
        # Submit order
        submitted_order = await self.broker.submit_order(order)
        self.risk_manager.add_order(submitted_order)
        
        self.logger.info(f"Sell order submitted: {submitted_order.id} for {abs(self.state.current_position)} shares at ${current_price:.2f}")
        
        # Update trade count
        self.state.total_trades += 1
        
        # Call trade callback
        if self.on_trade_callback:
            self.on_trade_callback(submitted_order)
    
    async def _check_position_exits(self, current_price: float) -> None:
        """Check for stop loss/take profit exits on existing positions."""
        if self.state.current_position is None or self.state.current_position == 0:
            return
        
        # Update position manager
        position = await self.broker.get_position(self.config.trading.symbol)
        if position:
            exit_orders = self.position_manager.update_position(self.config.trading.symbol, current_price)
            
            for order in exit_orders:
                # Submit exit order
                submitted_order = await self.broker.submit_order(order)
                self.risk_manager.add_order(submitted_order)
                
                self.logger.info(f"Exit order submitted: {submitted_order.id} for {order.metadata.get('order_type', 'unknown')}")
                
                # Call trade callback
                if self.on_trade_callback:
                    self.on_trade_callback(submitted_order)
    
    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        try:
            positions = await self.broker.get_positions()
            
            for position in positions:
                if position.quantity != 0:
                    side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                    
                    order = Order(
                        id="",
                        symbol=position.symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=abs(position.quantity),
                        metadata={'order_type': 'close_position', 'session_id': self.session_id}
                    )
                    
                    submitted_order = await self.broker.submit_order(order)
                    self.logger.info(f"Closing position: {submitted_order.id} for {position.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {str(e)}")
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading session summary."""
        # Calculate duration if trading has started
        duration = None
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'session_id': self.session_id,
            'symbol': self.state.symbol,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration': duration,
            'is_running': self.is_running,
            'total_trades': self.state.total_trades,
            'total_pnl': self.state.total_pnl,
            'current_position': self.state.current_position,
            'current_price': self.state.current_price,
            'account_equity': self.state.account_equity,
            'last_action': self.state.last_action,
            'last_action_time': self.state.last_action_time.isoformat() if self.state.last_action_time else None,
            'risk_summary': self.risk_manager.get_risk_summary(),
            'position_summary': self.position_manager.get_position_summary(),
            'monitor_summary': self.monitor.get_summary()
        }
    
    def set_callbacks(self, on_trade: Optional[Callable] = None, on_error: Optional[Callable] = None) -> None:
        """Set trading callbacks."""
        self.on_trade_callback = on_trade
        self.on_error_callback = on_error
    
    def __str__(self) -> str:
        return f"LiveTrader({self.state.symbol}, running={self.is_running}, trades={self.state.total_trades})" 