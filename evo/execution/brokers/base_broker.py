"""
Abstract base class for broker implementations.

This module defines the interface that all broker implementations must follow
to provide consistent order execution across different trading platforms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from evo.core.exceptions import BrokerError
from evo.core.config import BrokerConfig


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    average_entry_price: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pl: Optional[float] = None
    realized_pl: Optional[float] = None
    side: str = "long"  # long or short


@dataclass
class Account:
    """Account information."""
    account_id: str
    cash: float
    buying_power: float
    equity: float
    market_value: float
    day_trade_count: int = 0
    pattern_day_trader: bool = False
    account_blocked: bool = False
    trading_blocked: bool = False
    transfers_blocked: bool = False


class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.
    
    All broker implementations must implement these methods to provide
    consistent order execution interfaces across different platforms.
    """
    
    def __init__(self, config: BrokerConfig):
        """
        Initialize the broker with configuration.
        
        Args:
            config: Broker configuration object containing broker-specific settings
        """
        self.config = config
        self._validate_config()
        self._initialize()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the broker configuration.
        
        Raises:
            BrokerError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the broker connection and resources.
        
        Raises:
            BrokerError: If initialization fails
        """
        pass
    
    @abstractmethod
    async def get_account(self) -> Account:
        """
        Get current account information.
        
        Returns:
            Account information
            
        Raises:
            BrokerError: If account information cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            List of current positions
            
        Raises:
            BrokerError: If positions cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position for the symbol or None if no position exists
            
        Raises:
            BrokerError: If position cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
            
        Returns:
            Updated order with broker-assigned ID and status
            
        Raises:
            BrokerError: If order submission fails
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Broker-assigned order ID
            
        Returns:
            Order information or None if not found
            
        Raises:
            BrokerError: If order cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Broker-assigned order ID
            
        Returns:
            True if cancellation was successful
            
        Raises:
            BrokerError: If cancellation fails
        """
        pass
    
    @abstractmethod
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get list of orders.
        
        Args:
            status: Filter by order status (optional)
            
        Returns:
            List of orders matching the criteria
            
        Raises:
            BrokerError: If orders cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest price or None if not available
            
        Raises:
            BrokerError: If price cannot be retrieved
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
        Perform a health check on the broker.
        
        Returns:
            True if broker is healthy, False otherwise
        """
        try:
            await self.get_account()
            return True
        except Exception as e:
            raise BrokerError(f"Health check failed: {str(e)}")
    
    async def close(self) -> None:
        """
        Close broker connection and cleanup resources.
        """
        # Default implementation - override in subclasses if needed
        pass
    
    def __str__(self) -> str:
        """String representation of the broker."""
        return f"{self.__class__.__name__}(config={self.config})" 