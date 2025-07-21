"""
Tests for risk management module.

This module tests the risk management system including
the RiskManager and PositionManager classes.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np

from evo.execution.risk.risk_manager import RiskManager, RiskMetrics
from evo.execution.risk.position_manager import PositionManager, ManagedPosition
from evo.core.exceptions import RiskError
from evo.core.config import Config, RiskLimitsConfig, PositionConfig
from evo.execution.brokers.base_broker import Account, Position, Order, OrderSide, OrderType, OrderStatus

pytestmark = [
    pytest.mark.unit,
    pytest.mark.execution,
    pytest.mark.risk
]


class TestRiskLimitsConfig:
    """Test RiskLimitsConfig dataclass."""
    
    def test_risk_limits_config_creation(self):
        """Test creating a RiskLimitsConfig instance."""
        limits = RiskLimitsConfig(
            max_position_size=0.1,
            max_portfolio_exposure=0.5,
            max_drawdown=0.15,
            max_daily_loss=0.05,
            max_correlation_exposure=0.3,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            max_orders_per_day=50,
            min_order_size=100.0,
            max_order_size=10000.0
        )
        
        assert limits.max_position_size == 0.1
        assert limits.max_portfolio_exposure == 0.5
        assert limits.max_drawdown == 0.15
        assert limits.max_daily_loss == 0.05
        assert limits.max_correlation_exposure == 0.3
        assert limits.stop_loss_pct == 0.02
        assert limits.take_profit_pct == 0.04
        assert limits.max_orders_per_day == 50
        assert limits.min_order_size == 100.0
        assert limits.max_order_size == 10000.0
    
    def test_risk_limits_config_defaults(self):
        """Test RiskLimitsConfig creation with defaults."""
        limits = RiskLimitsConfig()
        
        assert limits.max_position_size == 0.1
        assert limits.max_portfolio_exposure == 0.5
        assert limits.max_drawdown == 0.15
        assert limits.max_daily_loss == 0.05
        assert limits.max_correlation_exposure == 0.3
        assert limits.stop_loss_pct == 0.02
        assert limits.take_profit_pct == 0.04
        assert limits.max_orders_per_day == 50
        assert limits.min_order_size == 100.0
        assert limits.max_order_size == 10000.0


class TestRiskMetrics:
    """Test RiskMetrics dataclass."""
    
    def test_risk_metrics_creation(self):
        """Test creating a RiskMetrics instance."""
        now = datetime.now()
        metrics = RiskMetrics(
            current_drawdown=0.05,
            daily_pnl=0.02,
            portfolio_exposure=0.3,
            largest_position=0.1,
            correlation_exposure=0.2,
            orders_today=25,
            last_reset_date=now
        )
        
        assert metrics.current_drawdown == 0.05
        assert metrics.daily_pnl == 0.02
        assert metrics.portfolio_exposure == 0.3
        assert metrics.largest_position == 0.1
        assert metrics.correlation_exposure == 0.2
        assert metrics.orders_today == 25
        assert metrics.last_reset_date == now
    
    def test_risk_metrics_defaults(self):
        """Test RiskMetrics creation with defaults."""
        metrics = RiskMetrics()
        
        assert metrics.current_drawdown == 0.0
        assert metrics.daily_pnl == 0.0
        assert metrics.portfolio_exposure == 0.0
        assert metrics.largest_position == 0.0
        assert metrics.correlation_exposure == 0.0
        assert metrics.orders_today == 0
        assert metrics.last_reset_date is None


class TestRiskManager:
    """Test RiskManager class."""
    
    @pytest.fixture
    def risk_config(self):
        """Create risk configuration."""
        config = Config()
        config.execution.risk_limits = RiskLimitsConfig(
            max_position_size=0.1,
            max_portfolio_exposure=0.5,
            max_drawdown=0.15,
            max_daily_loss=0.05,
            max_correlation_exposure=0.3,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            max_orders_per_day=50,
            min_order_size=100.0,
            max_order_size=10000.0
        )
        return config
    
    @pytest.fixture
    def risk_manager(self, risk_config):
        """Create a RiskManager instance for testing."""
        return RiskManager(risk_config)
    
    def test_risk_manager_initialization(self, risk_manager, risk_config):
        """Test RiskManager initialization."""
        assert risk_manager.limits.max_position_size == 0.1
        assert risk_manager.limits.max_portfolio_exposure == 0.5
        assert risk_manager.limits.max_drawdown == 0.15
        assert risk_manager.initial_capital == 0.0
        assert risk_manager.peak_capital == 0.0
        assert len(risk_manager.positions) == 0
        assert len(risk_manager.pending_orders) == 0
        assert len(risk_manager.daily_orders) == 0
    
    def test_set_initial_capital(self, risk_manager):
        """Test setting initial capital."""
        risk_manager.set_initial_capital(50000.0)
        
        assert risk_manager.initial_capital == 50000.0
        assert risk_manager.peak_capital == 50000.0
        assert risk_manager.daily_start_capital == 50000.0
        assert risk_manager.metrics.last_reset_date is not None
    
    def test_update_account(self, risk_manager):
        """Test updating account state."""
        risk_manager.set_initial_capital(50000.0)
        
        # Mock account with higher equity
        account = Mock(spec=Account)
        account.equity = 55000.0
        
        risk_manager.update_account(account)
        
        assert risk_manager.peak_capital == 55000.0
        assert risk_manager.metrics.current_drawdown == 0.0
        assert risk_manager.metrics.daily_pnl == 0.1  # 10% gain
    
    def test_update_account_drawdown(self, risk_manager):
        """Test updating account with drawdown."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.peak_capital = 60000.0  # Set peak higher
        
        # Mock account with lower equity
        account = Mock(spec=Account)
        account.equity = 51000.0
        
        risk_manager.update_account(account)
        
        # Drawdown should be (60000 - 51000) / 60000 = 0.15
        assert risk_manager.metrics.current_drawdown == 0.15
    
    def test_update_positions(self, risk_manager):
        """Test updating positions."""
        risk_manager.set_initial_capital(50000.0)
        
        # Mock positions
        position1 = Mock(spec=Position)
        position1.symbol = "AAPL"
        position1.market_value = 5000.0
        
        position2 = Mock(spec=Position)
        position2.symbol = "TSLA"
        position2.market_value = 3000.0
        
        positions = [position1, position2]
        
        risk_manager.update_positions(positions)
        
        assert len(risk_manager.positions) == 2
        assert risk_manager.positions["AAPL"] == position1
        assert risk_manager.positions["TSLA"] == position2
        assert risk_manager.metrics.portfolio_exposure == 0.16  # 8000/50000
        assert risk_manager.metrics.largest_position == 0.1  # 5000/50000
    
    def test_update_positions_empty(self, risk_manager):
        """Test updating with empty positions."""
        risk_manager.set_initial_capital(50000.0)
        
        risk_manager.update_positions([])
        
        assert len(risk_manager.positions) == 0
        assert risk_manager.metrics.portfolio_exposure == 0.0
        assert risk_manager.metrics.largest_position == 0.0
    
    def test_validate_order_success(self, risk_manager):
        """Test successful order validation."""
        risk_manager.set_initial_capital(50000.0)
        
        # Mock order
        order = Mock(spec=Order)
        order.quantity = 10
        order.price = 150.0
        
        # Mock account
        account = Mock(spec=Account)
        account.trading_blocked = False
        
        is_valid, reason = risk_manager.validate_order(order, account)
        
        assert is_valid is True
        assert "validated successfully" in reason
    
    def test_validate_order_trading_blocked(self, risk_manager):
        """Test order validation when trading is blocked."""
        risk_manager.set_initial_capital(50000.0)
        
        order = Mock(spec=Order)
        order.quantity = 10
        order.price = 150.0
        
        account = Mock(spec=Account)
        account.trading_blocked = True
        
        is_valid, reason = risk_manager.validate_order(order, account)
        
        assert is_valid is False
        assert "Trading is blocked" in reason
    
    def test_validate_order_daily_limit_exceeded(self, risk_manager):
        """Test order validation when daily limit is exceeded."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.metrics.orders_today = 50  # At limit
        
        order = Mock(spec=Order)
        order.quantity = 10
        order.price = 150.0
        
        account = Mock(spec=Account)
        account.trading_blocked = False
        
        is_valid, reason = risk_manager.validate_order(order, account)
        
        assert is_valid is False
        assert "Daily order limit exceeded" in reason
    
    def test_validate_order_size_too_small(self, risk_manager):
        """Test order validation with order size too small."""
        risk_manager.set_initial_capital(50000.0)
        
        order = Mock(spec=Order)
        order.quantity = 1
        order.price = 50.0  # Order value = 50, below min of 100
        
        account = Mock(spec=Account)
        account.trading_blocked = False
        
        is_valid, reason = risk_manager.validate_order(order, account)
        
        assert is_valid is False
        assert "Order size too small" in reason
    
    def test_validate_order_size_too_large(self, risk_manager):
        """Test order validation with order size too large."""
        risk_manager.set_initial_capital(50000.0)
        
        order = Mock(spec=Order)
        order.quantity = 1000
        order.price = 150.0  # Order value = 150000, above max of 10000
        
        account = Mock(spec=Account)
        account.trading_blocked = False
        
        is_valid, reason = risk_manager.validate_order(order, account)
        
        assert is_valid is False
        assert "Order size too large" in reason
    
    def test_validate_order_position_size_too_large(self, risk_manager):
        """Test order validation with position size too large."""
        risk_manager.set_initial_capital(50000.0)
        
        order = Mock(spec=Order)
        order.quantity = 100
        order.price = 150.0  # Order value = 15000, 30% of capital
        
        account = Mock(spec=Account)
        account.trading_blocked = False
        
        is_valid, reason = risk_manager.validate_order(order, account)
        
        assert is_valid is False
        assert "Order size too large" in reason
    
    def test_validate_order_drawdown_limit_exceeded(self, risk_manager):
        """Test order validation when drawdown limit is exceeded."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.metrics.current_drawdown = 0.2  # Above 15% limit
        
        order = Mock(spec=Order)
        order.quantity = 10
        order.price = 150.0
        
        account = Mock(spec=Account)
        account.trading_blocked = False
        
        is_valid, reason = risk_manager.validate_order(order, account)
        
        assert is_valid is False
        assert "Drawdown limit exceeded" in reason
    
    def test_validate_order_daily_loss_limit_exceeded(self, risk_manager):
        """Test order validation when daily loss limit is exceeded."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.metrics.daily_pnl = -0.1  # -10% daily loss
        
        order = Mock(spec=Order)
        order.quantity = 10
        order.price = 150.0
        
        account = Mock(spec=Account)
        account.trading_blocked = False
        
        is_valid, reason = risk_manager.validate_order(order, account)
        
        assert is_valid is False
        assert "Daily loss limit exceeded" in reason
    
    def test_calculate_position_size(self, risk_manager):
        """Test position size calculation."""
        risk_manager.set_initial_capital(50000.0)
        
        # Mock account
        account = Mock(spec=Account)
        account.equity = 50000.0
        account.buying_power = 50000.0
        
        position_size = risk_manager.calculate_position_size("AAPL", 150.0, account)
        
        # Should be based on max position size (10% of 50000 = 5000)
        # Divided by price 150 = 33.33 shares, rounded down to 33
        expected_size = int((50000.0 * 0.1) / 150.0)
        assert position_size == expected_size
    
    def test_calculate_position_size_with_existing_position(self, risk_manager):
        """Test position size calculation with existing position."""
        risk_manager.set_initial_capital(50000.0)
        
        # Add existing position
        existing_position = Mock(spec=Position)
        existing_position.symbol = "AAPL"
        existing_position.market_value = 3000.0
        risk_manager.positions["AAPL"] = existing_position
        
        account = Mock(spec=Account)
        account.equity = 50000.0
        account.buying_power = 50000.0
        
        position_size = risk_manager.calculate_position_size("AAPL", 150.0, account)
        
        # Remaining capacity: 5000 - 3000 = 2000
        # Position size: 2000 / 150 = 13.33 shares, rounded down to 13
        expected_size = int((50000.0 * 0.1 - 3000.0) / 150.0)
        assert position_size == expected_size
    
    def test_calculate_position_size_no_capacity(self, risk_manager):
        """Test position size calculation when no capacity remains."""
        risk_manager.set_initial_capital(50000.0)
        
        # Add existing position at max capacity
        existing_position = Mock(spec=Position)
        existing_position.symbol = "AAPL"
        existing_position.market_value = 5000.0  # At max
        risk_manager.positions["AAPL"] = existing_position
        
        account = Mock(spec=Account)
        account.equity = 50000.0
        account.buying_power = 50000.0
        
        position_size = risk_manager.calculate_position_size("AAPL", 150.0, account)
        
        assert position_size == 0.0
    
    def test_add_order(self, risk_manager):
        """Test adding an order."""
        order = Mock(spec=Order)
        order.id = "order_123"
        order.symbol = "AAPL"
        
        risk_manager.add_order(order)
        
        assert "order_123" in risk_manager.pending_orders
        assert risk_manager.pending_orders["order_123"] == order
        assert len(risk_manager.daily_orders) == 1
        assert risk_manager.metrics.orders_today == 1
    
    def test_remove_order(self, risk_manager):
        """Test removing an order."""
        order = Mock(spec=Order)
        order.id = "order_123"
        order.symbol = "AAPL"
        
        risk_manager.add_order(order)
        risk_manager.remove_order("order_123")
        
        assert "order_123" not in risk_manager.pending_orders
    
    def test_should_stop_trading_false(self, risk_manager):
        """Test should_stop_trading when trading should continue."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.metrics.current_drawdown = 0.1  # Below limit
        risk_manager.metrics.daily_pnl = -0.02  # Above limit
        
        should_stop, reason = risk_manager.should_stop_trading()
        
        assert should_stop is False
        assert "Trading allowed" in reason
    
    def test_should_stop_trading_drawdown(self, risk_manager):
        """Test should_stop_trading when drawdown limit exceeded."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.metrics.current_drawdown = 0.2  # Above 15% limit
        
        should_stop, reason = risk_manager.should_stop_trading()
        
        assert should_stop is True
        assert "Drawdown limit exceeded" in reason
    
    def test_should_stop_trading_daily_loss(self, risk_manager):
        """Test should_stop_trading when daily loss limit exceeded."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.metrics.daily_pnl = -0.1  # Below -5% limit
        
        should_stop, reason = risk_manager.should_stop_trading()
        
        assert should_stop is True
        assert "Daily loss limit exceeded" in reason
    
    def test_get_risk_summary(self, risk_manager):
        """Test getting risk summary."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.metrics.current_drawdown = 0.05
        risk_manager.metrics.daily_pnl = 0.02
        risk_manager.metrics.portfolio_exposure = 0.3
        
        summary = risk_manager.get_risk_summary()
        
        assert summary['current_drawdown'] == 0.05
        assert summary['daily_pnl'] == 0.02
        assert summary['portfolio_exposure'] == 0.3
        assert 'largest_position' in summary
        assert 'orders_today' in summary
        assert 'pending_orders' in summary
        assert 'total_positions' in summary
        assert 'risk_limits' in summary
        assert 'max_drawdown' in summary['risk_limits']
        assert 'max_daily_loss' in summary['risk_limits']
        assert 'max_position_size' in summary['risk_limits']
        assert 'max_portfolio_exposure' in summary['risk_limits']
        assert 'max_orders_per_day' in summary['risk_limits']
    
    def test_check_daily_reset(self, risk_manager):
        """Test daily reset check."""
        risk_manager.set_initial_capital(50000.0)
        risk_manager.metrics.last_reset_date = datetime.now() - timedelta(days=2)
        risk_manager.metrics.orders_today = 25
        risk_manager.daily_start_capital = 50000.0
        
        # Mock account with new equity
        account = Mock(spec=Account)
        account.equity = 55000.0
        
        risk_manager.update_account(account)
        
        # Should reset daily metrics
        assert risk_manager.metrics.orders_today == 0
        assert risk_manager.daily_start_capital == 55000.0
    
    def test_string_representation(self, risk_manager):
        """Test string representation."""
        string_repr = str(risk_manager)
        
        assert "RiskManager" in string_repr
        assert "limits" in string_repr


class TestPositionConfig:
    """Test PositionConfig dataclass."""
    
    def test_position_config_creation(self):
        """Test creating a PositionConfig instance."""
        config = PositionConfig(
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.04,
            trailing_stop_pct=0.01,
            max_positions=10,
            position_sizing_method="kelly",
            kelly_fraction=0.25,
            volatility_lookback=20
        )
        
        assert config.default_stop_loss_pct == 0.02
        assert config.default_take_profit_pct == 0.04
        assert config.trailing_stop_pct == 0.01
        assert config.max_positions == 10
        assert config.position_sizing_method == "kelly"
        assert config.kelly_fraction == 0.25
        assert config.volatility_lookback == 20
    
    def test_position_config_defaults(self):
        """Test PositionConfig creation with defaults."""
        config = PositionConfig()
        
        assert config.default_stop_loss_pct == 0.02
        assert config.default_take_profit_pct == 0.04
        assert config.trailing_stop_pct == 0.01
        assert config.max_positions == 10
        assert config.position_sizing_method == "kelly"
        assert config.kelly_fraction == 0.25
        assert config.volatility_lookback == 20


class TestManagedPosition:
    """Test ManagedPosition dataclass."""
    
    def test_managed_position_creation(self):
        """Test creating a ManagedPosition instance."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        now = datetime.now()
        managed_pos = ManagedPosition(
            position=position,
            entry_price=150.0,
            stop_loss_price=147.0,
            take_profit_price=156.0,
            trailing_stop_price=148.5,
            entry_time=now,
            last_update=now
        )
        
        assert managed_pos.position == position
        assert managed_pos.entry_price == 150.0
        assert managed_pos.stop_loss_price == 147.0
        assert managed_pos.take_profit_price == 156.0
        assert managed_pos.trailing_stop_price == 148.5
        assert managed_pos.entry_time == now
        assert managed_pos.last_update == now
        assert managed_pos.stop_loss_order_id is None
        assert managed_pos.take_profit_order_id is None
    
    def test_managed_position_post_init(self):
        """Test ManagedPosition post-init with defaults."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        
        managed_pos = ManagedPosition(
            position=position,
            entry_price=150.0
        )
        
        assert managed_pos.entry_time is not None
        assert managed_pos.last_update is not None
        assert isinstance(managed_pos.entry_time, datetime)
        assert isinstance(managed_pos.last_update, datetime)


class TestPositionManager:
    """Test PositionManager class."""
    
    @pytest.fixture
    def position_config(self):
        """Create position configuration."""
        config = Config()
        config.execution.position_config = PositionConfig(
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.04,
            trailing_stop_pct=0.01,
            max_positions=10,
            position_sizing_method='kelly',
            kelly_fraction=0.25,
            volatility_lookback=20
        )
        return config
    
    @pytest.fixture
    def position_manager(self, position_config):
        """Create a PositionManager instance for testing."""
        return PositionManager(position_config)
    
    def test_position_manager_initialization(self, position_manager, position_config):
        """Test PositionManager initialization."""
        assert position_manager.config.default_stop_loss_pct == 0.02
        assert position_manager.config.default_take_profit_pct == 0.04
        assert position_manager.config.trailing_stop_pct == 0.01
        assert position_manager.config.max_positions == 10
        assert len(position_manager.managed_positions) == 0
    
    def test_add_position_success(self, position_manager):
        """Test successfully adding a position."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        managed_pos = position_manager.add_position(position, 150.0)
        
        assert "AAPL" in position_manager.managed_positions
        assert managed_pos.position == position
        assert managed_pos.entry_price == 150.0
        assert managed_pos.stop_loss_price == 147.0  # 150 * 0.98
        assert managed_pos.take_profit_price == 156.0  # 150 * 1.04
        assert managed_pos.trailing_stop_price == 148.5  # 150 * 0.99
    
    def test_add_position_short(self, position_manager):
        """Test adding a short position."""
        position = Mock(spec=Position)
        position.symbol = "TSLA"
        position.side = "short"
        position.quantity = 50
        
        managed_pos = position_manager.add_position(position, 200.0)
        
        assert managed_pos.stop_loss_price == 204.0  # 200 * 1.02
        assert managed_pos.take_profit_price == 192.0  # 200 * 0.96
        assert managed_pos.trailing_stop_price == 202.0  # 200 * 1.01
    
    def test_add_position_max_limit_reached(self, position_manager):
        """Test adding position when max limit is reached."""
        position_manager.config.max_positions = 1
        
        # Add first position
        position1 = Mock(spec=Position)
        position1.symbol = "AAPL"
        position1.side = "long"
        position1.quantity = 100
        position_manager.add_position(position1, 150.0)
        
        # Try to add second position
        position2 = Mock(spec=Position)
        position2.symbol = "TSLA"
        position2.side = "long"
        position2.quantity = 50
        
        with pytest.raises(RiskError, match="Maximum positions limit reached"):
            position_manager.add_position(position2, 200.0)
    
    def test_update_position_no_triggers(self, position_manager):
        """Test updating position with no triggers."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        managed_pos = position_manager.add_position(position, 150.0)
        
        # Update with price that doesn't trigger anything
        orders = position_manager.update_position("AAPL", 152.0)
        
        assert len(orders) == 0
        assert managed_pos.last_update is not None
    
    def test_update_position_stop_loss_trigger(self, position_manager):
        """Test updating position with stop loss trigger."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        managed_pos = position_manager.add_position(position, 150.0)
        
        # Update with price below stop loss
        orders = position_manager.update_position("AAPL", 146.0)
        
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].order_type == OrderType.MARKET
    
    def test_update_position_take_profit_trigger(self, position_manager):
        """Test updating position with take profit trigger."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        managed_pos = position_manager.add_position(position, 150.0)
        
        # Update with price above take profit
        orders = position_manager.update_position("AAPL", 157.0)
        
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].order_type == OrderType.MARKET
    
    def test_update_position_trailing_stop_trigger(self, position_manager):
        """Test updating position with trailing stop trigger."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        managed_pos = position_manager.add_position(position, 150.0)
        
        # First update to move price up and update trailing stop
        position_manager.update_position("AAPL", 160.0)
        
        # Then update with price below trailing stop
        orders = position_manager.update_position("AAPL", 158.0)
        
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].order_type == OrderType.MARKET
    
    def test_update_position_nonexistent(self, position_manager):
        """Test updating non-existent position."""
        orders = position_manager.update_position("NONEXISTENT", 150.0)
        
        assert len(orders) == 0
    
    def test_remove_position(self, position_manager):
        """Test removing a position."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        position_manager.add_position(position, 150.0)
        assert "AAPL" in position_manager.managed_positions
        
        position_manager.remove_position("AAPL")
        assert "AAPL" not in position_manager.managed_positions
    
    def test_get_position_summary(self, position_manager):
        """Test getting position summary."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        position.current_price = 155.0
        
        position_manager.add_position(position, 150.0)
        
        summary = position_manager.get_position_summary()
        
        assert summary['total_positions'] == 1
        assert "AAPL" in summary['positions']
        assert summary['positions']['AAPL']['side'] == "long"
        assert summary['positions']['AAPL']['quantity'] == 100
        assert summary['positions']['AAPL']['entry_price'] == 150.0
        assert summary['positions']['AAPL']['current_price'] == 155.0
        assert summary['positions']['AAPL']['unrealized_pnl'] == 500.0  # (155-150)*100
    
    def test_calculate_position_size_kelly(self, position_manager):
        """Test Kelly criterion position sizing."""
        position_manager.config.position_sizing_method = "kelly"
        
        size = position_manager.calculate_position_size("AAPL", 150.0, 50000.0, volatility=0.2)
        
        # Kelly fraction should be applied
        assert size > 0
        assert size <= (50000.0 * 0.25) / 150.0  # Max Kelly size
    
    def test_calculate_position_size_fixed(self, position_manager):
        """Test fixed position sizing."""
        position_manager.config.position_sizing_method = "fixed"
        
        size = position_manager.calculate_position_size("AAPL", 150.0, 50000.0)
        
        # Should be a fixed percentage of account, rounded down to whole shares
        expected_size = int((50000.0 * 0.02) / 150.0)  # 2% of account (from implementation)
        assert size == expected_size
    
    def test_calculate_position_size_volatility(self, position_manager):
        """Test volatility-based position sizing."""
        position_manager.config.position_sizing_method = "volatility"
        
        size = position_manager.calculate_position_size("AAPL", 150.0, 50000.0, volatility=0.2)
        
        # Should be inversely proportional to volatility
        assert size > 0
        assert size < (50000.0 * 0.1) / 150.0  # Less than fixed size due to volatility
    
    def test_should_trigger_stop_loss(self, position_manager):
        """Test stop loss trigger logic."""
        position = Mock(spec=Position)
        position.side = "long"
        
        managed_pos = ManagedPosition(
            position=position,
            entry_price=150.0,
            stop_loss_price=147.0
        )
        
        # Price below stop loss
        assert position_manager._should_trigger_stop_loss(managed_pos, 146.0) is True
        
        # Price above stop loss
        assert position_manager._should_trigger_stop_loss(managed_pos, 148.0) is False
    
    def test_should_trigger_take_profit(self, position_manager):
        """Test take profit trigger logic."""
        position = Mock(spec=Position)
        position.side = "long"
        
        managed_pos = ManagedPosition(
            position=position,
            entry_price=150.0,
            take_profit_price=156.0
        )
        
        # Price above take profit
        assert position_manager._should_trigger_take_profit(managed_pos, 157.0) is True
        
        # Price below take profit
        assert position_manager._should_trigger_take_profit(managed_pos, 155.0) is False
    
    def test_should_trigger_trailing_stop(self, position_manager):
        """Test trailing stop trigger logic."""
        position = Mock(spec=Position)
        position.side = "long"
        
        managed_pos = ManagedPosition(
            position=position,
            entry_price=150.0,
            trailing_stop_price=148.5
        )
        
        # Price below trailing stop
        assert position_manager._should_trigger_trailing_stop(managed_pos, 148.0) is True
        
        # Price above trailing stop
        assert position_manager._should_trigger_trailing_stop(managed_pos, 149.0) is False
    
    def test_create_stop_loss_order(self, position_manager):
        """Test creating stop loss order."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        order = position_manager._create_stop_loss_order(position, 146.0)
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100
    
    def test_create_take_profit_order(self, position_manager):
        """Test creating take profit order."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        order = position_manager._create_take_profit_order(position, 157.0)
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100
    
    def test_create_trailing_stop_order(self, position_manager):
        """Test creating trailing stop order."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.side = "long"
        position.quantity = 100
        
        order = position_manager._create_trailing_stop_order(position, 148.0)
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100
    
    def test_string_representation(self, position_manager):
        """Test string representation."""
        string_repr = str(position_manager)
        
        assert "PositionManager" in string_repr
        assert "positions" in string_repr 