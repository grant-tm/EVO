"""
Tests for broker implementations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import uuid

from evo.execution.brokers.base_broker import (
    BaseBroker, Order, Position, Account, 
    OrderSide, OrderType, OrderStatus
)
from evo.execution.brokers.alpaca_broker import AlpacaBroker
from evo.core.exceptions import BrokerError
from evo.core.config import BrokerConfig, AlpacaConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.execution,
    pytest.mark.brokers
]

class TestBaseBroker:
    """Test base broker functionality."""
    
    def test_order_creation(self):
        """Test order creation."""
        order = Order(
            id="test_id",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
    
    def test_position_creation(self):
        """Test position creation."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            average_entry_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            unrealized_pl=500.0
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.average_entry_price == 150.0
        assert position.current_price == 155.0
        assert position.unrealized_pl == 500.0
    
    def test_account_creation(self):
        """Test account creation."""
        account = Account(
            account_id="test_account",
            cash=50000.0,
            buying_power=100000.0,
            equity=100000.0,
            market_value=50000.0
        )
        
        assert account.account_id == "test_account"
        assert account.cash == 50000.0
        assert account.buying_power == 100000.0
        assert account.equity == 100000.0


class TestAlpacaBroker:
    """Test Alpaca broker implementation."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = BrokerConfig()
        config.alpaca.api_key = 'test_key'
        config.alpaca.api_secret = 'test_secret'
        config.alpaca.paper_trading = True
        return config
    
    @pytest.fixture
    def mock_broker(self, mock_config):
        """Create a mock Alpaca broker for testing."""
        with patch('evo.execution.brokers.alpaca_broker.TradingClient') as mock_trading_client, \
             patch('evo.execution.brokers.alpaca_broker.StockHistoricalDataClient') as mock_data_client:
            
            broker = AlpacaBroker(mock_config)
            broker.trading_client = mock_trading_client.return_value
            broker.data_client = mock_data_client.return_value
            return broker
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test missing API key
        with pytest.raises(BrokerError, match="Alpaca API key and secret cannot be empty"):
            AlpacaBroker(BrokerConfig())
        
        # Test empty API key
        config = BrokerConfig()
        config.alpaca.api_key = ""
        config.alpaca.api_secret = ""
        with pytest.raises(BrokerError, match="Alpaca API key and secret cannot be empty"):
            AlpacaBroker(config)
        
        # Test valid config
        config = BrokerConfig()
        config.alpaca.api_key = 'test_key'
        config.alpaca.api_secret = 'test_secret'
        config.alpaca.paper_trading = True
        
        # Mock the Alpaca clients to avoid actual API calls
        with pytest.MonkeyPatch().context() as m:
            m.setattr('evo.execution.brokers.alpaca_broker.TradingClient', Mock())
            m.setattr('evo.execution.brokers.alpaca_broker.StockHistoricalDataClient', Mock())
            
            broker = AlpacaBroker(config)
            assert broker.config == config
    
    def test_initialization_failure(self, mock_config):
        """Test initialization failure."""
        with patch('evo.execution.brokers.alpaca_broker.TradingClient') as mock_trading_client:
            mock_trading_client.side_effect = Exception("Connection failed")
            
            with pytest.raises(BrokerError, match="Failed to initialize Alpaca client"):
                AlpacaBroker(mock_config)
    
    @pytest.mark.asyncio
    async def test_get_account_success(self, mock_broker):
        """Test successful account retrieval."""
        # Mock account response
        mock_account = Mock()
        mock_account.id = "test_account"
        mock_account.cash = "50000.0"
        mock_account.buying_power = "100000.0"
        mock_account.equity = "100000.0"
        mock_account.market_value = "50000.0"
        mock_account.daytrade_count = 0
        mock_account.pattern_day_trader = False
        mock_account.account_blocked = False
        mock_account.trading_blocked = False
        mock_account.transfers_blocked = False
        
        mock_broker.trading_client.get_account.return_value = mock_account
        
        account = await mock_broker.get_account()
        
        assert account.account_id == "test_account"
        assert account.cash == 50000.0
        assert account.buying_power == 100000.0
        assert account.equity == 100000.0
        assert account.market_value == 50000.0
        assert account.day_trade_count == 0
        assert account.pattern_day_trader is False
        assert account.account_blocked is False
        assert account.trading_blocked is False
        assert account.transfers_blocked is False
    
    @pytest.mark.asyncio
    async def test_get_account_failure(self, mock_broker):
        """Test account retrieval failure."""
        mock_broker.trading_client.get_account.side_effect = Exception("API Error")
        
        with pytest.raises(BrokerError, match="Failed to get account information"):
            await mock_broker.get_account()
    
    @pytest.mark.asyncio
    async def test_get_positions_success(self, mock_broker):
        """Test successful positions retrieval."""
        # Mock positions response
        mock_position1 = Mock()
        mock_position1.symbol = "AAPL"
        mock_position1.qty = "100"
        mock_position1.avg_entry_price = "150.0"
        mock_position1.current_price = "155.0"
        mock_position1.market_value = "15500.0"
        mock_position1.unrealized_pl = "500.0"
        mock_position1.realized_pl = "0.0"
        
        mock_position2 = Mock()
        mock_position2.symbol = "TSLA"
        mock_position2.qty = "-50"
        mock_position2.avg_entry_price = "200.0"
        mock_position2.current_price = "190.0"
        mock_position2.market_value = "9500.0"
        mock_position2.unrealized_pl = "500.0"
        mock_position2.realized_pl = "0.0"
        
        mock_broker.trading_client.get_all_positions.return_value = [mock_position1, mock_position2]
        
        positions = await mock_broker.get_positions()
        
        assert len(positions) == 2
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 100.0
        assert positions[0].side == "long"
        assert positions[1].symbol == "TSLA"
        assert positions[1].quantity == -50.0
        assert positions[1].side == "short"
    
    @pytest.mark.asyncio
    async def test_get_positions_failure(self, mock_broker):
        """Test positions retrieval failure."""
        mock_broker.trading_client.get_all_positions.side_effect = Exception("API Error")
        
        with pytest.raises(BrokerError, match="Failed to get positions"):
            await mock_broker.get_positions()
    
    @pytest.mark.asyncio
    async def test_get_position_success(self, mock_broker):
        """Test successful single position retrieval."""
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.0"
        mock_position.current_price = "155.0"
        mock_position.market_value = "15500.0"
        mock_position.unrealized_pl = "500.0"
        mock_position.realized_pl = "0.0"
        
        mock_broker.trading_client.get_position.return_value = mock_position
        
        position = await mock_broker.get_position("AAPL")
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100.0
        assert position.average_entry_price == 150.0
        assert position.current_price == 155.0
        assert position.side == "long"
    
    @pytest.mark.asyncio
    async def test_get_position_not_found(self, mock_broker):
        """Test position retrieval when position doesn't exist."""
        mock_broker.trading_client.get_position.side_effect = Exception("position does not exist")
        
        position = await mock_broker.get_position("INVALID")
        
        assert position is None
    
    @pytest.mark.asyncio
    async def test_get_position_failure(self, mock_broker):
        """Test position retrieval failure."""
        mock_broker.trading_client.get_position.side_effect = Exception("API Error")
        
        with pytest.raises(BrokerError, match="Failed to get position for AAPL"):
            await mock_broker.get_position("AAPL")
    
    @pytest.mark.asyncio
    async def test_submit_market_order_success(self, mock_broker):
        """Test successful market order submission."""
        # Mock order response
        mock_alpaca_order = Mock()
        mock_alpaca_order.id = "alpaca_order_id"
        mock_alpaca_order.status.value = "submitted"
        mock_alpaca_order.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_alpaca_order.updated_at = datetime(2023, 1, 1, 12, 0, 1)
        
        mock_broker.trading_client.submit_order.return_value = mock_alpaca_order
        
        order = Order(
            id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        result = await mock_broker.submit_order(order)
        
        assert result.id == "alpaca_order_id"
        assert result.status == OrderStatus.SUBMITTED
        assert result.created_at is not None
        assert result.updated_at is not None
        mock_broker.trading_client.submit_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_limit_order_success(self, mock_broker):
        """Test successful limit order submission."""
        mock_alpaca_order = Mock()
        mock_alpaca_order.id = "alpaca_order_id"
        mock_alpaca_order.status.value = "submitted"
        mock_alpaca_order.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_alpaca_order.updated_at = datetime(2023, 1, 1, 12, 0, 1)
        
        mock_broker.trading_client.submit_order.return_value = mock_alpaca_order
        
        order = Order(
            id="",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )
        
        result = await mock_broker.submit_order(order)
        
        assert result.id == "alpaca_order_id"
        assert result.status == OrderStatus.SUBMITTED
        mock_broker.trading_client.submit_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_stop_order_success(self, mock_broker):
        """Test successful stop order submission."""
        mock_alpaca_order = Mock()
        mock_alpaca_order.id = "alpaca_order_id"
        mock_alpaca_order.status.value = "submitted"
        mock_alpaca_order.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_alpaca_order.updated_at = datetime(2023, 1, 1, 12, 0, 1)
        
        mock_broker.trading_client.submit_order.return_value = mock_alpaca_order
        
        order = Order(
            id="",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=100,
            stop_price=140.0
        )
        
        result = await mock_broker.submit_order(order)
        
        assert result.id == "alpaca_order_id"
        assert result.status == OrderStatus.SUBMITTED
        mock_broker.trading_client.submit_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_limit_order_missing_price(self, mock_broker):
        """Test limit order submission without price."""
        order = Order(
            id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100
        )
        
        with pytest.raises(BrokerError, match="Limit orders require a price"):
            await mock_broker.submit_order(order)
    
    @pytest.mark.asyncio
    async def test_submit_stop_order_missing_stop_price(self, mock_broker):
        """Test stop order submission without stop price."""
        order = Order(
            id="",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=100
        )
        
        with pytest.raises(BrokerError, match="Stop orders require a stop price"):
            await mock_broker.submit_order(order)
    
    @pytest.mark.asyncio
    async def test_submit_unsupported_order_type(self, mock_broker):
        """Test submission of unsupported order type."""
        order = Order(
            id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            quantity=100
        )
        
        with pytest.raises(BrokerError, match="Unsupported order type"):
            await mock_broker.submit_order(order)
    
    @pytest.mark.asyncio
    async def test_submit_order_failure(self, mock_broker):
        """Test order submission failure."""
        mock_broker.trading_client.submit_order.side_effect = Exception("API Error")
        
        order = Order(
            id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        with pytest.raises(BrokerError, match="Failed to submit order"):
            await mock_broker.submit_order(order)
    
    @pytest.mark.asyncio
    async def test_get_order_success(self, mock_broker):
        """Test successful order retrieval."""
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        
        mock_alpaca_order = Mock()
        mock_alpaca_order.id = "order_id"
        mock_alpaca_order.symbol = "AAPL"
        mock_alpaca_order.side = AlpacaOrderSide.BUY
        mock_alpaca_order.type.value = "market"
        mock_alpaca_order.qty = "100"
        mock_alpaca_order.limit_price = None
        mock_alpaca_order.stop_price = None
        mock_alpaca_order.status.value = "filled"
        mock_alpaca_order.filled_qty = "100"
        mock_alpaca_order.filled_avg_price = "155.0"
        mock_alpaca_order.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_alpaca_order.updated_at = datetime(2023, 1, 1, 12, 0, 1)
        mock_alpaca_order.client_order_id = "client_id"
        
        mock_broker.trading_client.get_order_by_id.return_value = mock_alpaca_order
        
        order = await mock_broker.get_order("order_id")
        
        assert order.id == "order_id"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100.0
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100.0
        assert order.average_fill_price == 155.0
        assert order.client_order_id == "client_id"
    
    @pytest.mark.asyncio
    async def test_get_order_not_found(self, mock_broker):
        """Test order retrieval when order doesn't exist."""
        mock_broker.trading_client.get_order_by_id.side_effect = Exception("order not found")
        
        order = await mock_broker.get_order("invalid_id")
        
        assert order is None
    
    @pytest.mark.asyncio
    async def test_get_order_failure(self, mock_broker):
        """Test order retrieval failure."""
        mock_broker.trading_client.get_order_by_id.side_effect = Exception("API Error")
        
        with pytest.raises(BrokerError, match="Failed to get order order_id"):
            await mock_broker.get_order("order_id")
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, mock_broker):
        """Test successful order cancellation."""
        result = await mock_broker.cancel_order("order_id")
        
        assert result is True
        mock_broker.trading_client.cancel_order_by_id.assert_called_once_with("order_id")
    
    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, mock_broker):
        """Test order cancellation failure."""
        mock_broker.trading_client.cancel_order_by_id.side_effect = Exception("API Error")
        
        with pytest.raises(BrokerError, match="Failed to cancel order order_id"):
            await mock_broker.cancel_order("order_id")
    
    @pytest.mark.asyncio
    async def test_get_orders_success(self, mock_broker):
        """Test successful orders retrieval."""
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        
        mock_alpaca_order1 = Mock()
        mock_alpaca_order1.id = "order1"
        mock_alpaca_order1.symbol = "AAPL"
        mock_alpaca_order1.side = AlpacaOrderSide.BUY
        mock_alpaca_order1.type.value = "market"
        mock_alpaca_order1.qty = "100"
        mock_alpaca_order1.limit_price = None
        mock_alpaca_order1.stop_price = None
        mock_alpaca_order1.status.value = "filled"
        mock_alpaca_order1.filled_qty = "100"
        mock_alpaca_order1.filled_avg_price = "155.0"
        mock_alpaca_order1.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_alpaca_order1.updated_at = datetime(2023, 1, 1, 12, 0, 1)
        mock_alpaca_order1.client_order_id = "client1"
        
        mock_alpaca_order2 = Mock()
        mock_alpaca_order2.id = "order2"
        mock_alpaca_order2.symbol = "TSLA"
        mock_alpaca_order2.side = AlpacaOrderSide.SELL
        mock_alpaca_order2.type.value = "limit"
        mock_alpaca_order2.qty = "50"
        mock_alpaca_order2.limit_price = "200.0"
        mock_alpaca_order2.stop_price = None
        mock_alpaca_order2.status.value = "submitted"
        mock_alpaca_order2.filled_qty = "0"
        mock_alpaca_order2.filled_avg_price = None
        mock_alpaca_order2.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_alpaca_order2.updated_at = datetime(2023, 1, 1, 12, 0, 1)
        mock_alpaca_order2.client_order_id = "client2"
        
        mock_broker.trading_client.get_orders.return_value = [mock_alpaca_order1, mock_alpaca_order2]
        
        orders = await mock_broker.get_orders()
        
        assert len(orders) == 2
        assert orders[0].id == "order1"
        assert orders[0].symbol == "AAPL"
        assert orders[0].status == OrderStatus.FILLED
        assert orders[1].id == "order2"
        assert orders[1].symbol == "TSLA"
        assert orders[1].status == OrderStatus.SUBMITTED
        assert orders[1].price == 200.0
    
    @pytest.mark.asyncio
    async def test_get_orders_with_status_filter(self, mock_broker):
        """Test orders retrieval with status filter."""
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        
        mock_alpaca_order = Mock()
        mock_alpaca_order.id = "order1"
        mock_alpaca_order.symbol = "AAPL"
        mock_alpaca_order.side = AlpacaOrderSide.BUY
        mock_alpaca_order.type.value = "market"
        mock_alpaca_order.qty = "100"
        mock_alpaca_order.limit_price = None
        mock_alpaca_order.stop_price = None
        mock_alpaca_order.status.value = "filled"
        mock_alpaca_order.filled_qty = "100"
        mock_alpaca_order.filled_avg_price = "155.0"
        mock_alpaca_order.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_alpaca_order.updated_at = datetime(2023, 1, 1, 12, 0, 1)
        mock_alpaca_order.client_order_id = "client1"
        
        mock_broker.trading_client.get_orders.return_value = [mock_alpaca_order]
        
        orders = await mock_broker.get_orders(status=OrderStatus.FILLED)
        
        assert len(orders) == 1
        assert orders[0].status == OrderStatus.FILLED
        mock_broker.trading_client.get_orders.assert_called_once_with(status="filled")
    
    @pytest.mark.asyncio
    async def test_get_orders_failure(self, mock_broker):
        """Test orders retrieval failure."""
        mock_broker.trading_client.get_orders.side_effect = Exception("API Error")
        
        with pytest.raises(BrokerError, match="Failed to get orders"):
            await mock_broker.get_orders()
    
    @pytest.mark.asyncio
    async def test_get_latest_price_success(self, mock_broker):
        """Test successful latest price retrieval."""
        mock_quote = Mock()
        mock_quote.ask_price = "155.50"
        
        mock_response = {"AAPL": mock_quote}
        mock_broker.data_client.get_stock_latest_quote.return_value = mock_response
        
        price = await mock_broker.get_latest_price("AAPL")
        
        assert price == 155.50
        mock_broker.data_client.get_stock_latest_quote.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_latest_price_symbol_not_found(self, mock_broker):
        """Test latest price retrieval when symbol not found."""
        mock_response = {}
        mock_broker.data_client.get_stock_latest_quote.return_value = mock_response
        
        price = await mock_broker.get_latest_price("INVALID")
        
        assert price is None
    
    @pytest.mark.asyncio
    async def test_get_latest_price_failure(self, mock_broker):
        """Test latest price retrieval failure."""
        mock_broker.data_client.get_stock_latest_quote.side_effect = Exception("API Error")
        
        price = await mock_broker.get_latest_price("AAPL")
        
        assert price is None
    
    @pytest.mark.asyncio
    async def test_is_market_open_true(self, mock_broker):
        """Test market open check when market is open."""
        mock_clock = Mock()
        mock_clock.is_open = True
        mock_broker.trading_client.get_clock.return_value = mock_clock
        
        is_open = await mock_broker.is_market_open()
        
        assert is_open is True
        mock_broker.trading_client.get_clock.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_is_market_open_false(self, mock_broker):
        """Test market open check when market is closed."""
        mock_clock = Mock()
        mock_clock.is_open = False
        mock_broker.trading_client.get_clock.return_value = mock_clock
        
        is_open = await mock_broker.is_market_open()
        
        assert is_open is False
    
    @pytest.mark.asyncio
    async def test_is_market_open_failure(self, mock_broker):
        """Test market open check failure."""
        mock_broker.trading_client.get_clock.side_effect = Exception("API Error")
        
        is_open = await mock_broker.is_market_open()
        
        assert is_open is False
    
    @pytest.mark.asyncio
    async def test_close(self, mock_broker):
        """Test broker close method."""
        # Should not raise any exception
        await mock_broker.close()
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_broker):
        """Test health check functionality."""
        # Mock successful account retrieval with proper attributes
        mock_account = Mock()
        mock_account.id = "test_account"
        mock_account.cash = "50000.0"
        mock_account.buying_power = "100000.0"
        mock_account.equity = "100000.0"
        mock_account.market_value = "50000.0"
        mock_account.daytrade_count = 0
        mock_account.pattern_day_trader = False
        mock_account.account_blocked = False
        mock_account.trading_blocked = False
        mock_account.transfers_blocked = False
        
        mock_broker.trading_client.get_account.return_value = mock_account
        
        # Health check should pass
        assert await mock_broker.health_check() is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_broker):
        """Test health check failure."""
        # Mock failed account retrieval
        mock_broker.trading_client.get_account.side_effect = Exception("API Error")
        
        # Health check should raise exception
        with pytest.raises(BrokerError, match="Health check failed"):
            await mock_broker.health_check()


class TestOrderTypes:
    """Test order type functionality."""
    
    def test_order_side_enum(self):
        """Test order side enumeration."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_type_enum(self):
        """Test order type enumeration."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
    
    def test_order_status_enum(self):
        """Test order status enumeration."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIAL.value == "partial"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired" 