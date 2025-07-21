"""
Trading monitoring and alerting system.

This module provides real-time monitoring, metrics collection, and alerting
for live trading operations.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time

from evo.core.logging import get_logger


@dataclass
class Alert:
    """Trading alert."""
    id: str
    type: str  # 'info', 'warning', 'error', 'critical'
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0


class TradingMonitor:
    """
    Trading monitoring and alerting system.
    
    This class provides real-time monitoring, metrics collection, and alerting
    for live trading operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trading monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Monitoring state
        self.is_running = False
        self.start_time = None
        self.last_update = None
        
        # Metrics tracking
        self.performance_metrics = PerformanceMetrics()
        self.trade_history: List[Dict[str, Any]] = []
        self.price_history: List[Dict[str, Any]] = []
        self.alert_history: List[Alert] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_drawdown': 0.15,  # 15% max drawdown
            'daily_loss': 0.05,    # 5% daily loss
            'position_size': 0.2,  # 20% max position size
            'order_failure_rate': 0.1,  # 10% order failure rate
            'data_latency': 30,    # 30 seconds max data latency
        }
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.metric_callbacks: List[Callable] = []
        
        # Monitoring thread
        self.monitor_thread = None
        self.monitor_interval = self.config.get('monitor_interval', 30)  # 30 seconds
        
        self.logger.info("Trading monitor initialized")
    
    def start(self) -> None:
        """Start monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Trading monitor started")
    
    def stop(self) -> None:
        """Stop monitoring."""
        self.is_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Trading monitor stopped")
    
    def update_state(self, state: Any) -> None:
        """
        Update monitoring state with current trading state.
        
        Args:
            state: Current trading state object
        """
        try:
            self.last_update = datetime.now()
            
            # Update price history
            if hasattr(state, 'current_price') and state.current_price:
                self.price_history.append({
                    'timestamp': datetime.now(),
                    'price': state.current_price,
                    'position': getattr(state, 'current_position', 0)
                })
            
            # Keep only recent price history
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]
            
            # Check for alerts
            self._check_alerts(state)
            
        except Exception as e:
            self.logger.error(f"Error updating monitor state: {str(e)}")
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Add a trade to the trade history.
        
        Args:
            trade_data: Trade information
        """
        try:
            trade_data['timestamp'] = datetime.now()
            self.trade_history.append(trade_data)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Check for trade-related alerts
            self._check_trade_alerts(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error adding trade: {str(e)}")
    
    def add_alert(self, alert_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new alert.
        
        Args:
            alert_type: Type of alert ('info', 'warning', 'error', 'critical')
            message: Alert message
            data: Additional alert data
        """
        try:
            alert = Alert(
                id=str(len(self.alert_history) + 1),
                type=alert_type,
                message=message,
                timestamp=datetime.now(),
                data=data or {}
            )
            
            self.alert_history.append(alert)
            
            # Keep only recent alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {str(e)}")
            
            self.logger.info(f"Alert: {alert_type.upper()} - {message}")
            
        except Exception as e:
            self.logger.error(f"Error adding alert: {str(e)}")
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def register_metric_callback(self, callback: Callable) -> None:
        """Register a metric callback function."""
        self.metric_callbacks.append(callback)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'performance_metrics': self.performance_metrics.__dict__,
            'total_trades': len(self.trade_history),
            'total_alerts': len(self.alert_history),
            'unacknowledged_alerts': len([a for a in self.alert_history if not a.acknowledged]),
            'price_history_count': len(self.price_history),
            'alert_thresholds': self.alert_thresholds
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        recent_alerts = self.alert_history[-limit:] if self.alert_history else []
        return [
            {
                'id': alert.id,
                'type': alert.type,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'acknowledged': alert.acknowledged,
                'data': alert.data
            }
            for alert in recent_alerts
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.__dict__
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def save_monitoring_data(self, file_path: Path) -> None:
        """Save monitoring data to file."""
        try:
            data = {
                'summary': self.get_summary(),
                'trade_history': self.trade_history,
                'alert_history': [
                    {
                        'id': alert.id,
                        'type': alert.type,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'acknowledged': alert.acknowledged,
                        'data': alert.data
                    }
                    for alert in self.alert_history
                ],
                'price_history': self.price_history
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Monitoring data saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving monitoring data: {str(e)}")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Check for stale data
                if self.last_update:
                    time_since_update = (datetime.now() - self.last_update).total_seconds()
                    if time_since_update > self.alert_thresholds['data_latency']:
                        self.add_alert(
                            'warning',
                            f"Data latency high: {time_since_update:.1f} seconds",
                            {'latency_seconds': time_since_update}
                        )
                
                # Call metric callbacks
                for callback in self.metric_callbacks:
                    try:
                        callback(self.get_summary())
                    except Exception as e:
                        self.logger.error(f"Error in metric callback: {str(e)}")
                
                # Sleep
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _check_alerts(self, state: Any) -> None:
        """Check for various alert conditions."""
        try:
            # Check for high position size
            if hasattr(state, 'current_position') and hasattr(state, 'account_equity'):
                if state.account_equity > 0:
                    position_size_pct = abs(state.current_position * state.current_price / state.account_equity)
                    if position_size_pct > self.alert_thresholds['position_size']:
                        self.add_alert(
                            'warning',
                            f"Large position size: {position_size_pct:.1%}",
                            {'position_size_pct': position_size_pct}
                        )
            
            # Check for data staleness
            if self.last_update:
                time_since_update = (datetime.now() - self.last_update).total_seconds()
                if time_since_update > self.alert_thresholds['data_latency']:
                    self.add_alert(
                        'warning',
                        f"Data may be stale: {time_since_update:.1f} seconds since last update",
                        {'latency_seconds': time_since_update}
                    )
            
        except Exception as e:
            self.logger.error(f"Error checking alerts: {str(e)}")
    
    def _check_trade_alerts(self, trade_data: Dict[str, Any]) -> None:
        """Check for trade-related alerts."""
        try:
            # Check for large trades
            if 'quantity' in trade_data and 'price' in trade_data:
                trade_value = trade_data['quantity'] * trade_data['price']
                if trade_value > 10000:  # $10k threshold
                    self.add_alert(
                        'info',
                        f"Large trade executed: ${trade_value:,.2f}",
                        {'trade_value': trade_value}
                    )
            
            # Check for order failures
            if trade_data.get('status') == 'rejected':
                self.add_alert(
                    'error',
                    f"Order rejected: {trade_data.get('reason', 'Unknown reason')}",
                    trade_data
                )
            
        except Exception as e:
            self.logger.error(f"Error checking trade alerts: {str(e)}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics from trade history."""
        try:
            if not self.trade_history:
                return
            
            # Calculate basic metrics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
            
            # Calculate P&L metrics
            total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
            winning_pnl = sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0)
            losing_pnl = sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0)
            
            # Update metrics
            self.performance_metrics.total_trades = total_trades
            self.performance_metrics.winning_trades = winning_trades
            self.performance_metrics.losing_trades = losing_trades
            self.performance_metrics.win_rate = winning_trades / total_trades if total_trades > 0 else 0
            self.performance_metrics.total_return = total_pnl
            
            if winning_trades > 0:
                self.performance_metrics.average_win = winning_pnl / winning_trades
            if losing_trades > 0:
                self.performance_metrics.average_loss = abs(losing_pnl) / losing_trades
            
            if abs(losing_pnl) > 0:
                self.performance_metrics.profit_factor = winning_pnl / abs(losing_pnl)
            
            # Calculate max drawdown from price history
            if self.price_history:
                self._calculate_drawdown()
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _calculate_drawdown(self) -> None:
        """Calculate maximum drawdown from price history."""
        try:
            if not self.price_history:
                return
            
            # Extract equity values (simplified - in practice you'd track actual equity)
            equity_values = []
            current_equity = 100000  # Starting equity
            
            for price_data in self.price_history:
                # Simplified equity calculation
                if 'position' in price_data and price_data['position'] != 0:
                    # Assume some P&L based on position
                    current_equity += price_data['position'] * 0.01  # Simplified
                equity_values.append(current_equity)
            
            if not equity_values:
                return
            
            # Calculate drawdown
            peak = equity_values[0]
            max_drawdown = 0
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            self.performance_metrics.max_drawdown = max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {str(e)}")
    
    def __str__(self) -> str:
        return f"TradingMonitor(running={self.is_running}, trades={len(self.trade_history)}, alerts={len(self.alert_history)})" 