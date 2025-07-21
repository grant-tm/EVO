"""
Position management for live trading.

This module provides position sizing, stop loss, and take profit management
for individual positions.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

from evo.core.exceptions import RiskError
from evo.core.logging import get_logger
from evo.core.config import Config
from ..brokers.base_broker import Position, Order, OrderSide, OrderType, OrderStatus


@dataclass
class ManagedPosition:
    """Position with management metadata."""
    position: Position
    entry_price: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    entry_time: datetime = None
    last_update: datetime = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()
        if self.last_update is None:
            self.last_update = datetime.now()


class PositionManager:
    """
    Position management system for live trading.
    
    This class handles position sizing, stop losses, take profits, and
    position monitoring for individual positions.
    """
    
    def __init__(self, config: Config):
        """
        Initialize position manager.
        
        Args:
            config: Configuration object containing position settings
        """
        self.logger = get_logger(__name__)
        self.config = config.execution.position_config
        self.managed_positions: Dict[str, ManagedPosition] = {}
        
        self.logger.info(f"Position manager initialized with config: {self.config}")
    
    def add_position(self, position: Position, entry_price: float) -> ManagedPosition:
        """
        Add a new position for management.
        
        Args:
            position: Position to manage
            entry_price: Entry price for the position
            
        Returns:
            Managed position object
        """
        if len(self.managed_positions) >= self.config.max_positions:
            raise RiskError(f"Maximum positions limit reached: {self.config.max_positions}")
        
        managed_pos = ManagedPosition(
            position=position,
            entry_price=entry_price
        )
        
        # Calculate stop loss and take profit prices
        if position.side == "long":
            managed_pos.stop_loss_price = entry_price * (1 - self.config.default_stop_loss_pct)
            managed_pos.take_profit_price = entry_price * (1 + self.config.default_take_profit_pct)
            managed_pos.trailing_stop_price = entry_price * (1 - self.config.trailing_stop_pct)
        else:  # short
            managed_pos.stop_loss_price = entry_price * (1 + self.config.default_stop_loss_pct)
            managed_pos.take_profit_price = entry_price * (1 - self.config.default_take_profit_pct)
            managed_pos.trailing_stop_price = entry_price * (1 + self.config.trailing_stop_pct)
        
        self.managed_positions[position.symbol] = managed_pos
        
        self.logger.info(f"Added managed position: {position.symbol} at ${entry_price:.2f}")
        self.logger.debug(f"Stop loss: ${managed_pos.stop_loss_price:.2f}, Take profit: ${managed_pos.take_profit_price:.2f}")
        
        return managed_pos
    
    def update_position(self, symbol: str, current_price: float) -> List[Order]:
        """
        Update position with current price and check for stop/take profit triggers.
        
        Args:
            symbol: Position symbol
            current_price: Current market price
            
        Returns:
            List of orders to execute (stop loss, take profit, etc.)
        """
        if symbol not in self.managed_positions:
            return []
        
        managed_pos = self.managed_positions[symbol]
        position = managed_pos.position
        orders = []
        
        # Update trailing stop for long positions
        if position.side == "long" and current_price > managed_pos.entry_price:
            new_trailing_stop = current_price * (1 - self.config.trailing_stop_pct)
            if new_trailing_stop > managed_pos.trailing_stop_price:
                managed_pos.trailing_stop_price = new_trailing_stop
                self.logger.debug(f"Updated trailing stop for {symbol}: ${new_trailing_stop:.2f}")
        
        # Update trailing stop for short positions
        elif position.side == "short" and current_price < managed_pos.entry_price:
            new_trailing_stop = current_price * (1 + self.config.trailing_stop_pct)
            if new_trailing_stop < managed_pos.trailing_stop_price:
                managed_pos.trailing_stop_price = new_trailing_stop
                self.logger.debug(f"Updated trailing stop for {symbol}: ${new_trailing_stop:.2f}")
        
        # Check stop loss triggers
        if self._should_trigger_stop_loss(managed_pos, current_price):
            order = self._create_stop_loss_order(position, current_price)
            orders.append(order)
            self.logger.info(f"Stop loss triggered for {symbol} at ${current_price:.2f}")
        
        # Check take profit triggers
        elif self._should_trigger_take_profit(managed_pos, current_price):
            order = self._create_take_profit_order(position, current_price)
            orders.append(order)
            self.logger.info(f"Take profit triggered for {symbol} at ${current_price:.2f}")
        
        # Check trailing stop triggers
        elif self._should_trigger_trailing_stop(managed_pos, current_price):
            order = self._create_trailing_stop_order(position, current_price)
            orders.append(order)
            self.logger.info(f"Trailing stop triggered for {symbol} at ${current_price:.2f}")
        
        managed_pos.last_update = datetime.now()
        return orders
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position from management."""
        if symbol in self.managed_positions:
            del self.managed_positions[symbol]
            self.logger.info(f"Removed managed position: {symbol}")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all managed positions."""
        summary = {
            'total_positions': len(self.managed_positions),
            'positions': {}
        }
        
        for symbol, managed_pos in self.managed_positions.items():
            position = managed_pos.position
            current_price = position.current_price or 0
            
            # Calculate P&L
            if position.side == "long":
                unrealized_pnl = (current_price - managed_pos.entry_price) * position.quantity
            else:
                unrealized_pnl = (managed_pos.entry_price - current_price) * position.quantity
            
            summary['positions'][symbol] = {
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': managed_pos.entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'stop_loss_price': managed_pos.stop_loss_price,
                'take_profit_price': managed_pos.take_profit_price,
                'trailing_stop_price': managed_pos.trailing_stop_price,
                'entry_time': managed_pos.entry_time.isoformat(),
                'last_update': managed_pos.last_update.isoformat()
            }
        
        return summary
    
    def calculate_position_size(self, symbol: str, price: float, account_equity: float, 
                              volatility: Optional[float] = None) -> float:
        """
        Calculate position size using the configured method.
        
        Args:
            symbol: Trading symbol
            price: Current price
            account_equity: Current account equity
            volatility: Historical volatility (optional, for volatility-based sizing)
            
        Returns:
            Recommended position size in shares
        """
        if self.config.position_sizing_method == "kelly":
            return self._calculate_kelly_position_size(price, account_equity, volatility)
        elif self.config.position_sizing_method == "fixed":
            return self._calculate_fixed_position_size(price, account_equity)
        elif self.config.position_sizing_method == "volatility":
            return self._calculate_volatility_position_size(price, account_equity, volatility)
        else:
            self.logger.warning(f"Unknown position sizing method: {self.config.position_sizing_method}")
            return self._calculate_fixed_position_size(price, account_equity)
    
    def _should_trigger_stop_loss(self, managed_pos: ManagedPosition, current_price: float) -> bool:
        """Check if stop loss should be triggered."""
        if managed_pos.position.side == "long":
            return current_price <= managed_pos.stop_loss_price
        else:  # short
            return current_price >= managed_pos.stop_loss_price
    
    def _should_trigger_take_profit(self, managed_pos: ManagedPosition, current_price: float) -> bool:
        """Check if take profit should be triggered."""
        if managed_pos.position.side == "long":
            return current_price >= managed_pos.take_profit_price
        else:  # short
            return current_price <= managed_pos.take_profit_price
    
    def _should_trigger_trailing_stop(self, managed_pos: ManagedPosition, current_price: float) -> bool:
        """Check if trailing stop should be triggered."""
        if managed_pos.position.side == "long":
            return current_price <= managed_pos.trailing_stop_price
        else:  # short
            return current_price >= managed_pos.trailing_stop_price
    
    def _create_stop_loss_order(self, position: Position, current_price: float) -> Order:
        """Create a stop loss order."""
        side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
        
        return Order(
            id="",  # Will be assigned by broker
            symbol=position.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            metadata={'order_type': 'stop_loss', 'position_id': position.symbol}
        )
    
    def _create_take_profit_order(self, position: Position, current_price: float) -> Order:
        """Create a take profit order."""
        side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
        
        return Order(
            id="",  # Will be assigned by broker
            symbol=position.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            metadata={'order_type': 'take_profit', 'position_id': position.symbol}
        )
    
    def _create_trailing_stop_order(self, position: Position, current_price: float) -> Order:
        """Create a trailing stop order."""
        side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
        
        return Order(
            id="",  # Will be assigned by broker
            symbol=position.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            metadata={'order_type': 'trailing_stop', 'position_id': position.symbol}
        )
    
    def _calculate_kelly_position_size(self, price: float, account_equity: float, 
                                     volatility: Optional[float] = None) -> float:
        """Calculate position size using Kelly criterion."""
        # Simplified Kelly calculation
        # In practice, you'd want more sophisticated win rate and odds estimation
        if volatility is None:
            volatility = 0.02  # Default 2% volatility
        
        # Kelly fraction = (bp - q) / b
        # where b = odds received, p = probability of win, q = probability of loss
        # For simplicity, using a basic formula
        kelly_fraction = self.config.kelly_fraction * volatility
        
        position_value = account_equity * kelly_fraction
        shares = position_value / price
        
        return int(shares)
    
    def _calculate_fixed_position_size(self, price: float, account_equity: float) -> float:
        """Calculate position size using fixed percentage."""
        position_value = account_equity * 0.02  # 2% of equity
        shares = position_value / price
        
        return int(shares)
    
    def _calculate_volatility_position_size(self, price: float, account_equity: float, 
                                          volatility: Optional[float] = None) -> float:
        """Calculate position size based on volatility."""
        if volatility is None:
            volatility = 0.02  # Default 2% volatility
        
        # Higher volatility = smaller position size
        position_value = account_equity * (0.02 / volatility)  # Adjust for volatility
        shares = position_value / price
        
        return int(shares)
    
    def __str__(self) -> str:
        return f"PositionManager(positions={len(self.managed_positions)}, method={self.config.position_sizing_method})" 