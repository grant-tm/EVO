"""
Tests for performance metrics and calculations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from evo.optimization.backtesting.metrics import (
    MetricType,
    PerformanceMetrics,
    PerformanceCalculator,
    MetricsAggregator
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.optimization,
    pytest.mark.backtesting
]


class TestMetricType:
    """Test MetricType enum."""
    
    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.RETURN.value == "return"
        assert MetricType.RISK.value == "risk"
        assert MetricType.RATIO.value == "ratio"
        assert MetricType.DRAWDOWN.value == "drawdown"
        assert MetricType.TRADE.value == "trade"
    
    def test_metric_type_names(self):
        """Test MetricType enum names."""
        assert MetricType.RETURN.name == "RETURN"
        assert MetricType.RISK.name == "RISK"
        assert MetricType.RATIO.name == "RATIO"
        assert MetricType.DRAWDOWN.name == "DRAWDOWN"
        assert MetricType.TRADE.name == "TRADE"


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test creating a PerformanceMetrics instance."""
        daily_returns = np.array([0.01, -0.005, 0.02, -0.01])
        
        metrics = PerformanceMetrics(
            total_return=0.015,
            annualized_return=0.12,
            daily_returns=daily_returns,
            volatility=0.15,
            max_drawdown=-0.05,
            var_95=-0.02,
            cvar_95=-0.025,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.4,
            information_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.5,
            average_win=0.02,
            average_loss=-0.015,
            max_consecutive_wins=3,
            max_consecutive_losses=2,
            total_trades=10,
            beta=1.1,
            alpha=0.02,
            treynor_ratio=1.1
        )
        
        assert metrics.total_return == 0.015
        assert metrics.annualized_return == 0.12
        assert np.array_equal(metrics.daily_returns, daily_returns)
        assert metrics.volatility == 0.15
        assert metrics.max_drawdown == -0.05
        assert metrics.sharpe_ratio == 1.2
        assert metrics.win_rate == 0.6
        assert metrics.total_trades == 10
    
    def test_performance_metrics_post_init(self):
        """Test post-init validation and conversion."""
        daily_returns = [0.01, -0.005, 0.02, -0.01]  # List instead of array
        
        metrics = PerformanceMetrics(
            total_return=0.015,
            annualized_return=0.12,
            daily_returns=daily_returns,
            volatility=0.15,
            max_drawdown=-0.05,
            var_95=-0.02,
            cvar_95=-0.025,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.4,
            information_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.5,
            average_win=0.02,
            average_loss=-0.015,
            max_consecutive_wins=3,
            max_consecutive_losses=2,
            total_trades=10,
            beta=1.1,
            alpha=0.02,
            treynor_ratio=1.1
        )
        
        # Should be converted to numpy array
        assert isinstance(metrics.daily_returns, np.ndarray)
        assert np.array_equal(metrics.daily_returns, np.array(daily_returns))
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        daily_returns = np.array([0.01, -0.005, 0.02, -0.01])
        
        metrics = PerformanceMetrics(
            total_return=0.015,
            annualized_return=0.12,
            daily_returns=daily_returns,
            volatility=0.15,
            max_drawdown=-0.05,
            var_95=-0.02,
            cvar_95=-0.025,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.4,
            information_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.5,
            average_win=0.02,
            average_loss=-0.015,
            max_consecutive_wins=3,
            max_consecutive_losses=2,
            total_trades=10,
            beta=1.1,
            alpha=0.02,
            treynor_ratio=1.1
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['total_return'] == 0.015
        assert metrics_dict['annualized_return'] == 0.12
        assert metrics_dict['volatility'] == 0.15
        assert metrics_dict['sharpe_ratio'] == 1.2
        assert metrics_dict['win_rate'] == 0.6
        assert metrics_dict['total_trades'] == 10


class TestPerformanceCalculator:
    """Test PerformanceCalculator class."""
    
    def test_performance_calculator_creation(self):
        """Test creating a PerformanceCalculator instance."""
        calculator = PerformanceCalculator(risk_free_rate=0.03)
        
        assert calculator.risk_free_rate == 0.03
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        
        # Create sample returns
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.008])
        
        metrics = calculator.calculate_metrics(returns)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > 0  # Should be positive for these returns
        assert metrics.volatility > 0
        assert metrics.max_drawdown < 0
        assert metrics.sharpe_ratio is not None
        assert metrics.total_trades > 0
    
    def test_calculate_metrics_with_trades(self):
        """Test metrics calculation with trade data."""
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        
        returns = np.array([0.01, -0.005, 0.02, -0.01])
        trades = [
            {'entry_price': 100, 'exit_price': 110, 'position_size': 100},  # Win
            {'entry_price': 100, 'exit_price': 90, 'position_size': 100},   # Loss
            {'entry_price': 100, 'exit_price': 105, 'position_size': 100},  # Win
        ]
        
        metrics = calculator.calculate_metrics(returns, trades=trades)
        
        assert metrics.win_rate == 2/3  # 2 wins out of 3 trades
        assert metrics.total_trades == 3
        assert metrics.average_win > 0
        assert metrics.average_loss < 0
    
    def test_calculate_metrics_with_benchmark(self):
        """Test metrics calculation with benchmark returns."""
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        
        returns = np.array([0.01, -0.005, 0.02, -0.01])
        benchmark_returns = np.array([0.008, -0.003, 0.015, -0.008])
        
        metrics = calculator.calculate_metrics(returns, benchmark_returns=benchmark_returns)
        
        assert metrics.information_ratio is not None
        assert metrics.beta is not None
        assert metrics.alpha is not None
        assert metrics.treynor_ratio is not None
    
    def test_calculate_total_return(self):
        """Test total return calculation."""
        calculator = PerformanceCalculator()
        
        returns = np.array([0.1, 0.2, -0.1])
        total_return = calculator._calculate_total_return(returns)
        
        expected_return = (1 + 0.1) * (1 + 0.2) * (1 - 0.1) - 1
        assert np.isclose(total_return, expected_return)
    
    def test_calculate_annualized_return(self):
        """Test annualized return calculation."""
        calculator = PerformanceCalculator()
        
        returns = np.array([0.01] * 252)  # 1% daily return for 252 days
        annualized_return = calculator._calculate_annualized_return(returns)
        
        # Should be approximately (1 + 0.01)^252 - 1
        expected_return = (1 + 0.01) ** 252 - 1
        assert np.isclose(annualized_return, expected_return, rtol=1e-2)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        calculator = PerformanceCalculator()
        
        returns = np.array([0.01, -0.005, 0.02, -0.01])
        volatility = calculator._calculate_volatility(returns)
        
        # Annualized volatility = std * sqrt(252)
        expected_volatility = np.std(returns) * np.sqrt(252)
        assert np.isclose(volatility, expected_volatility)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        calculator = PerformanceCalculator()
        
        # Create returns that will produce a drawdown
        returns = np.array([0.1, -0.2, 0.15, -0.1, 0.05])
        max_drawdown = calculator._calculate_max_drawdown(returns)
        
        assert max_drawdown < 0  # Should be negative
        assert max_drawdown >= -1  # Should not be less than -100%
    
    def test_calculate_var_cvar(self):
        """Test VaR and CVaR calculation."""
        calculator = PerformanceCalculator()
        
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005, -0.03])
        var_95, cvar_95 = calculator._calculate_var_cvar(returns)
        
        # VaR should be the 5th percentile (since confidence=0.95)
        expected_var = np.percentile(returns, 5)
        assert np.isclose(var_95, expected_var)
        
        # CVaR should be the mean of returns below VaR
        expected_cvar = np.mean(returns[returns <= var_95])
        assert np.isclose(cvar_95, expected_cvar)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        
        returns = np.array([0.01, -0.005, 0.02, -0.01])
        annualized_return = 0.12
        volatility = 0.15
        
        sharpe_ratio = calculator._calculate_sharpe_ratio(returns, annualized_return, volatility)
        
        expected_sharpe = (annualized_return - 0.02) / volatility
        assert np.isclose(sharpe_ratio, expected_sharpe)
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        
        returns = np.array([0.01, -0.005, 0.02, -0.01])
        annualized_return = 0.12
        
        sortino_ratio = calculator._calculate_sortino_ratio(returns, annualized_return)
        
        # Should be positive for positive excess return
        assert sortino_ratio > 0
    
    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calculator = PerformanceCalculator()
        
        annualized_return = 0.12
        max_drawdown = -0.05
        
        calmar_ratio = calculator._calculate_calmar_ratio(annualized_return, max_drawdown)
        
        expected_calmar = annualized_return / abs(max_drawdown)
        assert np.isclose(calmar_ratio, expected_calmar)
    
    def test_calculate_trade_metrics(self):
        """Test trade metrics calculation."""
        calculator = PerformanceCalculator()
        
        trades = [
            {'entry_price': 100, 'exit_price': 110, 'position_size': 100},  # Win: 10%
            {'entry_price': 100, 'exit_price': 90, 'position_size': 100},   # Loss: -10%
            {'entry_price': 100, 'exit_price': 105, 'position_size': 100},  # Win: 5%
            {'entry_price': 100, 'exit_price': 95, 'position_size': 100},   # Loss: -5%
        ]
        
        trade_metrics = calculator._calculate_trade_metrics(trades)
        
        assert trade_metrics['win_rate'] == 0.5  # 2 wins out of 4 trades
        assert trade_metrics['total_trades'] == 4
        assert trade_metrics['average_win'] > 0
        assert trade_metrics['average_loss'] < 0
        assert trade_metrics['profit_factor'] > 0
    
    def test_estimate_trade_metrics_from_returns(self):
        """Test estimating trade metrics from returns."""
        calculator = PerformanceCalculator()
        
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.008])
        
        trade_metrics = calculator._estimate_trade_metrics_from_returns(returns)
        
        assert trade_metrics['total_trades'] > 0
        assert 0 <= trade_metrics['win_rate'] <= 1
        assert trade_metrics['profit_factor'] > 0
    
    def test_calculate_max_consecutive(self):
        """Test maximum consecutive wins/losses calculation."""
        calculator = PerformanceCalculator()
        
        # Test consecutive wins
        values = np.array([1, 1, 0, 1, 1, 1, 0, 1])
        max_consecutive_wins = calculator._calculate_max_consecutive(values, positive=True)
        assert max_consecutive_wins == 3
        
        # Test consecutive losses
        values = np.array([-1, -1, 5, -1, -1, 5, -1])
        max_consecutive_losses = calculator._calculate_max_consecutive(values, positive=False)
        assert max_consecutive_losses == 2
    
    def test_calculate_benchmark_metrics(self):
        """Test benchmark metrics calculation."""
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        
        returns = np.array([0.01, -0.005, 0.02, -0.01])
        benchmark_returns = np.array([0.008, -0.003, 0.015, -0.008])
        
        benchmark_metrics = calculator._calculate_benchmark_metrics(returns, benchmark_returns)
        
        assert 'information_ratio' in benchmark_metrics
        assert 'beta' in benchmark_metrics
        assert 'alpha' in benchmark_metrics
        assert 'treynor_ratio' in benchmark_metrics
        
        # Beta should be positive for positively correlated returns
        assert benchmark_metrics['beta'] > 0


class TestMetricsAggregator:
    """Test MetricsAggregator class."""
    
    def test_aggregate_metrics(self):
        """Test aggregating multiple metrics."""
        # Create sample metrics
        metrics1 = PerformanceMetrics(
            total_return=0.1,
            annualized_return=0.12,
            daily_returns=np.array([0.01, 0.02]),
            volatility=0.15,
            max_drawdown=-0.05,
            var_95=-0.02,
            cvar_95=-0.025,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.4,
            information_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.5,
            average_win=0.02,
            average_loss=-0.015,
            max_consecutive_wins=3,
            max_consecutive_losses=2,
            total_trades=10,
            beta=1.1,
            alpha=0.02,
            treynor_ratio=1.1
        )
        
        metrics2 = PerformanceMetrics(
            total_return=0.2,
            annualized_return=0.24,
            daily_returns=np.array([0.02, 0.03]),
            volatility=0.20,
            max_drawdown=-0.08,
            var_95=-0.03,
            cvar_95=-0.035,
            sharpe_ratio=1.8,
            sortino_ratio=2.0,
            calmar_ratio=3.0,
            information_ratio=1.2,
            win_rate=0.7,
            profit_factor=2.0,
            average_win=0.025,
            average_loss=-0.012,
            max_consecutive_wins=4,
            max_consecutive_losses=1,
            total_trades=15,
            beta=1.2,
            alpha=0.03,
            treynor_ratio=1.3
        )
        
        aggregated = MetricsAggregator.aggregate_metrics([metrics1, metrics2])
        
        assert isinstance(aggregated, dict)
        assert np.isclose(aggregated['total_return_median'], 0.15)  # Average of 0.1 and 0.2
        assert np.isclose(aggregated['sharpe_ratio_median'], 1.5)   # Average of 1.2 and 1.8
        assert np.isclose(aggregated['win_rate_median'], 0.65)      # Average of 0.6 and 0.7
        assert np.isclose(aggregated['total_trades_mean'], 12.5)  # Average of 10 and 15
    
    def test_calculate_robust_fitness(self):
        """Test calculating robust fitness from multiple metrics."""
        # Create sample metrics
        metrics1 = Mock(spec=PerformanceMetrics)
        metrics1.sharpe_ratio = 1.5
        
        metrics2 = Mock(spec=PerformanceMetrics)
        metrics2.sharpe_ratio = 2.0
        
        metrics3 = Mock(spec=PerformanceMetrics)
        metrics3.sharpe_ratio = 1.8
        
        fitness = MetricsAggregator.calculate_robust_fitness(
            [metrics1, metrics2, metrics3], 
            primary_metric="sharpe_ratio"
        )
        
        # Should be some combination of the metrics (not just average)
        assert isinstance(fitness, float)
        assert fitness > 0
    
    def test_calculate_robust_fitness_different_metric(self):
        """Test calculating robust fitness with different primary metric."""
        # Create sample metrics
        metrics1 = Mock(spec=PerformanceMetrics)
        metrics1.total_return = 0.1
        
        metrics2 = Mock(spec=PerformanceMetrics)
        metrics2.total_return = 0.2
        
        fitness = MetricsAggregator.calculate_robust_fitness(
            [metrics1, metrics2], 
            primary_metric="total_return"
        )
        
        assert isinstance(fitness, float)
        assert fitness > 0
    
    def test_aggregate_metrics_empty_list(self):
        """Test aggregating empty metrics list."""
        aggregated = MetricsAggregator.aggregate_metrics([])
        
        assert isinstance(aggregated, dict)
        assert len(aggregated) == 0
    
    def test_aggregate_metrics_single_metric(self):
        """Test aggregating single metric."""
        metrics = PerformanceMetrics(
            total_return=0.1,
            annualized_return=0.12,
            daily_returns=np.array([0.01, 0.02]),
            volatility=0.15,
            max_drawdown=-0.05,
            var_95=-0.02,
            cvar_95=-0.025,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=2.4,
            information_ratio=0.8,
            win_rate=0.6,
            profit_factor=1.5,
            average_win=0.02,
            average_loss=-0.015,
            max_consecutive_wins=3,
            max_consecutive_losses=2,
            total_trades=10,
            beta=1.1,
            alpha=0.02,
            treynor_ratio=1.1
        )
        
        aggregated = MetricsAggregator.aggregate_metrics([metrics])
        
        assert aggregated['total_return_median'] == 0.1
        assert aggregated['sharpe_ratio_median'] == 1.2
        assert aggregated['win_rate_median'] == 0.6 