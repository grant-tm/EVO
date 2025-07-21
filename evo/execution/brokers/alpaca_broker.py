"""
Alpaca broker implementation for order execution.

This module provides a concrete implementation of the BaseBroker interface
for the Alpaca trading platform.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, OrderType as AlpacaOrderType, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.enums import Adjustment

from evo.core.exceptions import BrokerError
from evo.core.logging import get_logger
from evo.core.config import BrokerConfig
from .base_broker import (
    BaseBroker, Order, Position, Account, 
    OrderSide, OrderType, OrderStatus
)


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker implementation.
    
    This class provides order execution capabilities through the Alpaca API,
    supporting both paper and live trading.
    """
    
    def __init__(self, config: BrokerConfig):
        """
        Initialize Alpaca broker.
        
        Args:
            config: Broker configuration object containing Alpaca-specific settings
        """
        self.logger = get_logger(__name__)
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Validate Alpaca configuration."""
        if not self.config.alpaca.api_key or not self.config.alpaca.api_secret:
            raise BrokerError("Alpaca API key and secret cannot be empty")
    
    def _initialize(self) -> None:
        """Initialize Alpaca client connections."""
        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.api_secret,
                paper=self.config.alpaca.paper_trading
            )
            
            # Initialize data client for price lookups
            self.data_client = StockHistoricalDataClient(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.api_secret
            )
            
            self.logger.info(f"Alpaca broker initialized (paper_trading={self.config.alpaca.paper_trading})")
            
        except Exception as e:
            raise BrokerError(f"Failed to initialize Alpaca client: {str(e)}")
    
    async def get_account(self) -> Account:
        """Get current account information."""
        try:
            account = self.trading_client.get_account()
            
            return Account(
                account_id=account.id,
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                equity=float(account.equity),
                market_value=float(account.market_value),
                day_trade_count=account.daytrade_count,
                pattern_day_trader=account.pattern_day_trader,
                account_blocked=account.account_blocked,
                trading_blocked=account.trading_blocked,
                transfers_blocked=account.transfers_blocked
            )
        except Exception as e:
            raise BrokerError(f"Failed to get account information: {str(e)}")
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for pos in positions:
                result.append(Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    average_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price) if pos.current_price else None,
                    market_value=float(pos.market_value) if pos.market_value else None,
                    unrealized_pl=float(pos.unrealized_pl) if pos.unrealized_pl else None,
                    realized_pl=float(pos.realized_pl) if pos.realized_pl else None,
                    side="long" if float(pos.qty) > 0 else "short"
                ))
            
            return result
        except Exception as e:
            raise BrokerError(f"Failed to get positions: {str(e)}")
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        try:
            position = self.trading_client.get_position(symbol)
            
            return Position(
                symbol=position.symbol,
                quantity=float(position.qty),
                average_entry_price=float(position.avg_entry_price),
                current_price=float(position.current_price) if position.current_price else None,
                market_value=float(position.market_value) if position.market_value else None,
                unrealized_pl=float(position.unrealized_pl) if position.unrealized_pl else None,
                realized_pl=float(position.realized_pl) if position.realized_pl else None,
                side="long" if float(position.qty) > 0 else "short"
            )
        except Exception as e:
            # Position doesn't exist
            if "position does not exist" in str(e).lower():
                return None
            raise BrokerError(f"Failed to get position for {symbol}: {str(e)}")
    
    async def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca."""
        try:
            # Convert our order types to Alpaca order types
            alpaca_side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
            
            # Create order request based on order type
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=alpaca_side,
                    client_order_id=order.client_order_id or str(uuid.uuid4()),
                    time_in_force=TimeInForce.DAY
                )
            elif order.order_type == OrderType.LIMIT:
                if not order.price:
                    raise BrokerError("Limit orders require a price")
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=alpaca_side,
                    limit_price=order.price,
                    client_order_id=order.client_order_id or str(uuid.uuid4()),
                    time_in_force=TimeInForce.DAY
                )
            elif order.order_type == OrderType.STOP:
                if not order.stop_price:
                    raise BrokerError("Stop orders require a stop price")
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=alpaca_side,
                    stop_price=order.stop_price,
                    client_order_id=order.client_order_id or str(uuid.uuid4()),
                    time_in_force=TimeInForce.DAY
                )
            else:
                raise BrokerError(f"Unsupported order type: {order.order_type}")
            
            # Submit the order
            alpaca_order = self.trading_client.submit_order(request)
            
            # Update our order with Alpaca's response
            order.id = alpaca_order.id
            order.status = OrderStatus(alpaca_order.status.value)
            order.created_at = alpaca_order.created_at.replace(tzinfo=timezone.utc)
            order.updated_at = alpaca_order.updated_at.replace(tzinfo=timezone.utc) if alpaca_order.updated_at else None
            
            self.logger.info(f"Order submitted: {order.id} for {order.symbol}")
            return order
            
        except Exception as e:
            raise BrokerError(f"Failed to submit order: {str(e)}")
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        try:
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            
            return Order(
                id=alpaca_order.id,
                symbol=alpaca_order.symbol,
                side=OrderSide.BUY if alpaca_order.side == AlpacaOrderSide.BUY else OrderSide.SELL,
                order_type=OrderType(alpaca_order.type.value),
                quantity=float(alpaca_order.qty),
                price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                status=OrderStatus(alpaca_order.status.value),
                filled_quantity=float(alpaca_order.filled_qty),
                average_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                created_at=alpaca_order.created_at.replace(tzinfo=timezone.utc),
                updated_at=alpaca_order.updated_at.replace(tzinfo=timezone.utc) if alpaca_order.updated_at else None,
                client_order_id=alpaca_order.client_order_id
            )
        except Exception as e:
            if "order not found" in str(e).lower():
                return None
            raise BrokerError(f"Failed to get order {order_id}: {str(e)}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            raise BrokerError(f"Failed to cancel order {order_id}: {str(e)}")
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get list of orders."""
        try:
            # Convert status filter if provided
            status_filter = None
            if status:
                status_filter = status.value
            
            alpaca_orders = self.trading_client.get_orders(status=status_filter)
            
            result = []
            for alpaca_order in alpaca_orders:
                result.append(Order(
                    id=alpaca_order.id,
                    symbol=alpaca_order.symbol,
                    side=OrderSide.BUY if alpaca_order.side == AlpacaOrderSide.BUY else OrderSide.SELL,
                    order_type=OrderType(alpaca_order.type.value),
                    quantity=float(alpaca_order.qty),
                    price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                    stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                    status=OrderStatus(alpaca_order.status.value),
                    filled_quantity=float(alpaca_order.filled_qty),
                    average_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                    created_at=alpaca_order.created_at.replace(tzinfo=timezone.utc),
                    updated_at=alpaca_order.updated_at.replace(tzinfo=timezone.utc) if alpaca_order.updated_at else None,
                    client_order_id=alpaca_order.client_order_id
                ))
            
            return result
        except Exception as e:
            raise BrokerError(f"Failed to get orders: {str(e)}")
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            response = self.data_client.get_stock_latest_quote(request)
            
            if symbol in response:
                quote = response[symbol]
                return float(quote.ask_price)  # Use ask price as latest
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get latest price for {symbol}: {str(e)}")
            return None
    
    async def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            self.logger.warning(f"Failed to check market status: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close broker connections."""
        # Alpaca clients don't need explicit cleanup
        pass 