"""
Risk management system for live trading.

This module provides comprehensive risk controls including position sizing,
drawdown limits, exposure limits, and portfolio-level risk management.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

from evo.core.exceptions import RiskError
from evo.core.logging import get_logger
from evo.core.config import Config
from ..brokers.base_broker import Account, Position, Order, OrderSide, OrderType


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    portfolio_exposure: float = 0.0
    largest_position: float = 0.0
    correlation_exposure: float = 0.0
    orders_today: int = 0
    last_reset_date: Optional[datetime] = None


class RiskManager:
    """
    Risk management system for live trading.
    
    This class implements comprehensive risk controls including position sizing,
    drawdown limits, exposure limits, and portfolio-level risk management.
    """
    
    def __init__(self, config: Config):
        """
        Initialize risk manager.
        
        Args:
            config: Configuration object containing risk limits
        """
        self.logger = get_logger(__name__)
        self.limits = config.execution.risk_limits
        self.metrics = RiskMetrics()
        self.initial_capital = 0.0
        self.peak_capital = 0.0
        self.daily_start_capital = 0.0
        
        # Track positions and orders
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.daily_orders: List[Order] = []
        
        self.logger.info(f"Risk manager initialized with limits: {self.limits}")
    
    def set_initial_capital(self, capital: float) -> None:
        """Set initial capital for risk calculations."""
        self.initial_capital = capital
        self.peak_capital = capital
        self.daily_start_capital = capital
        self.metrics.last_reset_date = datetime.now()
        self.logger.info(f"Initial capital set to: ${capital:,.2f}")
    
    def update_account(self, account: Account) -> None:
        """Update risk metrics based on current account state."""
        current_capital = account.equity
        
        # Update peak capital
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # Calculate drawdown
        if self.peak_capital > 0:
            self.metrics.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        # Calculate daily P&L
        if self.daily_start_capital > 0:
            self.metrics.daily_pnl = (current_capital - self.daily_start_capital) / self.daily_start_capital
        
        # Check if we need to reset daily metrics
        self._check_daily_reset()
    
    def update_positions(self, positions: List[Position]) -> None:
        """Update position tracking and calculate exposure metrics."""
        self.positions = {pos.symbol: pos for pos in positions}
        
        if not self.positions:
            self.metrics.portfolio_exposure = 0.0
            self.metrics.largest_position = 0.0
            return
        
        # Calculate portfolio exposure
        total_position_value = sum(
            abs(pos.market_value or 0) for pos in self.positions.values()
        )
        
        if self.initial_capital > 0:
            self.metrics.portfolio_exposure = total_position_value / self.initial_capital
        
        # Find largest position
        if self.initial_capital > 0:
            largest_position_pct = max(
                abs(pos.market_value or 0) / self.initial_capital 
                for pos in self.positions.values()
            )
            self.metrics.largest_position = largest_position_pct
    
    def validate_order(self, order: Order, account: Account) -> Tuple[bool, str]:
        """
        Validate if an order meets risk requirements.
        
        Args:
            order: Order to validate
            account: Current account state
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check if market is open (basic check)
            if not hasattr(account, 'trading_blocked') or account.trading_blocked:
                return False, "Trading is blocked"
            
            # Check daily order limit
            if self.metrics.orders_today >= self.limits.max_orders_per_day:
                return False, f"Daily order limit exceeded ({self.limits.max_orders_per_day})"
            
            # Check order size limits
            order_value = order.quantity * (order.price or 0)
            if order_value < self.limits.min_order_size:
                return False, f"Order size too small: ${order_value:.2f} < ${self.limits.min_order_size}"
            
            if order_value > self.limits.max_order_size:
                return False, f"Order size too large: ${order_value:.2f} > ${self.limits.max_order_size}"
            
            # Check position size limit
            if self.initial_capital > 0:
                position_size_pct = order_value / self.initial_capital
                if position_size_pct > self.limits.max_position_size:
                    return False, f"Position size too large: {position_size_pct:.1%} > {self.limits.max_position_size:.1%}"
            
            # Check portfolio exposure limit
            if self.metrics.portfolio_exposure + (order_value / self.initial_capital) > self.limits.max_portfolio_exposure:
                return False, f"Portfolio exposure limit exceeded: {self.metrics.portfolio_exposure:.1%} + {order_value / self.initial_capital:.1%} > {self.limits.max_portfolio_exposure:.1%}"
            
            # Check drawdown limit
            if self.metrics.current_drawdown > self.limits.max_drawdown:
                return False, f"Drawdown limit exceeded: {self.metrics.current_drawdown:.1%} > {self.limits.max_drawdown:.1%}"
            
            # Check daily loss limit
            if self.metrics.daily_pnl < -self.limits.max_daily_loss:
                return False, f"Daily loss limit exceeded: {self.metrics.daily_pnl:.1%} < -{self.limits.max_daily_loss:.1%}"
            
            return True, "Order validated successfully"
            
        except Exception as e:
            self.logger.error(f"Error validating order: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def calculate_position_size(self, symbol: str, price: float, account: Account) -> float:
        """
        Calculate appropriate position size based on risk limits.
        
        Args:
            symbol: Trading symbol
            price: Current price
            account: Current account state
            
        Returns:
            Recommended position size in shares
        """
        try:
            # Base position size on risk limits
            max_position_value = self.initial_capital * self.limits.max_position_size
            
            # Check current position
            current_position = self.positions.get(symbol)
            if current_position:
                current_value = abs(current_position.market_value or 0)
                remaining_capacity = max_position_value - current_value
                if remaining_capacity <= 0:
                    return 0.0
                max_position_value = remaining_capacity
            
            # Check portfolio exposure
            available_capital = account.buying_power
            max_by_capital = available_capital * self.limits.max_position_size
            
            # Use the smaller of the two limits
            position_value = min(max_position_value, max_by_capital)
            
            # Convert to shares
            shares = position_value / price
            
            # Round down to whole shares
            shares = int(shares)
            
            self.logger.debug(f"Calculated position size for {symbol}: {shares} shares (${position_value:.2f})")
            return shares
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def add_order(self, order: Order) -> None:
        """Track a new order for risk management."""
        self.pending_orders[order.id] = order
        self.daily_orders.append(order)
        self.metrics.orders_today += 1
        
        self.logger.debug(f"Order tracked: {order.id} for {order.symbol}")
    
    def remove_order(self, order_id: str) -> None:
        """Remove a completed/cancelled order from tracking."""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
    
    def should_stop_trading(self) -> Tuple[bool, str]:
        """
        Check if trading should be stopped due to risk limits.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check drawdown limit
        if self.metrics.current_drawdown > self.limits.max_drawdown:
            return True, f"Drawdown limit exceeded: {self.metrics.current_drawdown:.1%}"
        
        # Check daily loss limit
        if self.metrics.daily_pnl < -self.limits.max_daily_loss:
            return True, f"Daily loss limit exceeded: {self.metrics.daily_pnl:.1%}"
        
        # Check if account is blocked
        # This would be checked in the account update
        
        return False, "Trading allowed"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk metrics summary."""
        return {
            'current_drawdown': self.metrics.current_drawdown,
            'daily_pnl': self.metrics.daily_pnl,
            'portfolio_exposure': self.metrics.portfolio_exposure,
            'largest_position': self.metrics.largest_position,
            'orders_today': self.metrics.orders_today,
            'pending_orders': len(self.pending_orders),
            'total_positions': len(self.positions),
            'risk_limits': {
                'max_drawdown': self.limits.max_drawdown,
                'max_daily_loss': self.limits.max_daily_loss,
                'max_position_size': self.limits.max_position_size,
                'max_portfolio_exposure': self.limits.max_portfolio_exposure,
                'max_orders_per_day': self.limits.max_orders_per_day
            }
        }
    
    def _check_daily_reset(self) -> None:
        """Check if daily metrics should be reset."""
        if not self.metrics.last_reset_date:
            return
        
        current_date = datetime.now().date()
        last_reset_date = self.metrics.last_reset_date.date()
        
        if current_date > last_reset_date:
            # Reset daily metrics
            self.metrics.daily_pnl = 0.0
            self.metrics.orders_today = 0
            self.daily_orders.clear()
            # Update daily start capital to current peak capital for new day
            self.daily_start_capital = self.peak_capital
            self.metrics.last_reset_date = datetime.now()
            
            self.logger.info("Daily risk metrics reset")
    
    def __str__(self) -> str:
        return f"RiskManager(drawdown={self.metrics.current_drawdown:.1%}, daily_pnl={self.metrics.daily_pnl:.1%}, limits={self.limits})" 