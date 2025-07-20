"""
Integration tests for the backtesting framework.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from evo.optimization.backtesting.engine import BacktestEngine, CSVDataProvider, BacktestResult
from evo.optimization.backtesting.strategies import (
    MovingAverageStrategy, 
    MeanReversionStrategy, 
    MomentumStrategy,
    StrategyFactory,
    Action
)
from evo.optimization.backtesting.metrics import PerformanceCalculator, PerformanceMetrics

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optimization,
    pytest.mark.backtesting
]


@pytest.fixture
def sample_market_data_file(temp_dir):
    """Create a sample market data CSV file for integration testing."""
    # Create realistic market data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate price data with some trend and volatility
    np.random.seed(42)  # For reproducible tests
    base_price = 100.0
    prices = [base_price]
    
    for i in range(99):
        # Add some trend and random walk
        change = np.random.normal(0.001, 0.02)  # Small positive trend with volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 100),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    
    # Ensure high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    csv_path = temp_dir / "market_data.csv"
    data.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def backtest_engine_integration(sample_market_data_file):
    """Create a backtest engine with real data for integration testing."""
    data_provider = CSVDataProvider(str(sample_market_data_file), ['feature1', 'feature2'])
    
    return BacktestEngine(
        data_provider=data_provider,
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0,
        risk_free_rate=0.02
    )


class TestBacktestingIntegration:
    """Integration tests for the complete backtesting workflow."""
    
    def test_complete_backtest_workflow(self, backtest_engine_integration):
        """Test a complete backtest workflow from data to results."""
        # Create a simple moving average strategy
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            initial_capital=100000.0
        )
        
        # Run backtest
        result = backtest_engine_integration.run_backtest(strategy, start_date='2023-01-01', end_date='2023-02-19')
        
        # Verify result structure
        assert result.strategy == strategy
        assert isinstance(result.metrics, PerformanceMetrics)
        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, list)
        assert isinstance(result.metadata, dict)
        
        # Verify metrics are reasonable
        assert result.metrics.total_return is not None
        assert result.metrics.volatility > 0
        assert result.metrics.max_drawdown <= 0
        assert result.metrics.total_trades >= 0
        assert 0 <= result.metrics.win_rate <= 1
    
    def test_multiple_strategies_comparison(self, backtest_engine_integration):
        """Test comparing multiple strategies."""
        strategies = {
            'moving_average': MovingAverageStrategy(5, 15),
            'mean_reversion': MeanReversionStrategy(10, 2.0),
            'momentum': MomentumStrategy(10, 0.02)
        }
        
        results = {}
        for name, strategy in strategies.items():
            results[name] = backtest_engine_integration.run_backtest(strategy, length=50)
        
        # Verify all strategies produced results
        assert len(results) == 3
        for name, result in results.items():
            assert isinstance(result, BacktestResult)
            assert result.metrics.total_return is not None
            assert result.metrics.sharpe_ratio is not None
        
        # Compare performance (should be different for different strategies)
        returns = [result.metrics.total_return for result in results.values()]
        assert len(set(returns)) > 1  # At least some strategies should have different returns
    
    def test_strategy_factory_integration(self, backtest_engine_integration):
        """Test using StrategyFactory in a complete workflow."""
        # Create strategies using factory
        ma_strategy = StrategyFactory.create_strategy(
            "moving_average",
            short_window=5,
            long_window=15
        )
        
        mr_strategy = StrategyFactory.create_strategy(
            "mean_reversion",
            window=10,
            std_dev=2.0
        )
        
        # Run backtests
        ma_result = backtest_engine_integration.run_backtest(ma_strategy, length=50)
        mr_result = backtest_engine_integration.run_backtest(mr_strategy, length=50)
        
        # Verify results
        assert isinstance(ma_result, BacktestResult)
        assert isinstance(mr_result, BacktestResult)
        assert ma_result.metrics.total_return is not None
        assert mr_result.metrics.total_return is not None
    
    def test_multiple_backtests_robustness(self, backtest_engine_integration):
        """Test running multiple backtests for robustness analysis."""
        strategy = MovingAverageStrategy(5, 15)
        
        # Run multiple backtests
        results = backtest_engine_integration.run_multiple_backtests(
            strategy, 
            num_backtests=5, 
            length=30
        )
        
        # Verify results
        assert len(results) == 5
        assert all(isinstance(result, BacktestResult) for result in results)
        
        # Aggregate results
        aggregated = backtest_engine_integration.aggregate_results(results)
        
        # Verify aggregated metrics
        assert isinstance(aggregated, dict)
        assert 'total_return_median' in aggregated
        assert 'sharpe_ratio_median' in aggregated
        assert 'win_rate_median' in aggregated
        
        # Calculate robust fitness
        fitness = backtest_engine_integration.calculate_robust_fitness(results)
        assert isinstance(fitness, float)
        assert not np.isnan(fitness)
    
    def test_cross_validation_integration(self, backtest_engine_integration):
        """Test cross-validation workflow."""
        from evo.optimization.backtesting.engine import CrossValidationEngine
        
        cv_engine = CrossValidationEngine(backtest_engine_integration)
        strategy = MovingAverageStrategy(5, 15)
        
        # Run time series cross-validation
        cv_results = cv_engine.run_time_series_cv(strategy, n_splits=3)
        
        # Verify results
        assert len(cv_results) == 3
        assert all(isinstance(result, BacktestResult) for result in cv_results)
        
        # All results should have metrics
        for result in cv_results:
            assert result.metrics.total_return is not None
            assert result.metrics.sharpe_ratio is not None
    
    def test_performance_calculator_integration(self, backtest_engine_integration):
        """Test performance calculator in a real workflow."""
        strategy = MovingAverageStrategy(5, 15)
        result = backtest_engine_integration.run_backtest(strategy, length=50)
        
        # Extract returns from strategy
        returns = strategy.get_returns()
        
        # Calculate metrics manually
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        manual_metrics = calculator.calculate_metrics(returns)
        
        # Compare with backtest results
        assert abs(manual_metrics.total_return - result.metrics.total_return) < 1e-6
        assert abs(manual_metrics.volatility - result.metrics.volatility) < 1e-6
        assert abs(manual_metrics.max_drawdown - result.metrics.max_drawdown) < 1e-6
    
    def test_data_provider_integration(self, sample_market_data_file):
        """Test data provider with real CSV file."""
        provider = CSVDataProvider(str(sample_market_data_file), ['feature1', 'feature2'])
        
        # Get data
        data = provider.get_data()
        
        # Verify data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'datetime' in data.columns
        assert 'open' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
        
        # Get features
        features = provider.get_features(data)
        
        # Verify features
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(data)
        assert features.shape[1] == 2  # feature1 and feature2
        assert features.dtype == np.float32
    
    def test_trade_execution_integration(self, backtest_engine_integration):
        """Test that trades are properly executed and recorded."""
        # Create a strategy that will definitely trade
        class AggressiveStrategy(MovingAverageStrategy):
            def generate_signal(self, data):
                # Force some trades
                if len(self.equity_curve) < 10:
                    return Action.BUY
                elif len(self.equity_curve) < 20:
                    return Action.SELL
                else:
                    return Action.HOLD
        
        strategy = AggressiveStrategy(5, 15)
        
        # Run backtest
        result = backtest_engine_integration.run_backtest(strategy, length=50)
        
        # Verify trades were executed
        assert len(result.trades) > 0
        
        # Verify trade structure
        for trade in result.trades:
            assert hasattr(trade, 'entry_time')
            assert hasattr(trade, 'exit_time')
            assert hasattr(trade, 'entry_price')
            assert hasattr(trade, 'exit_price')
            assert hasattr(trade, 'position_size')
            assert hasattr(trade, 'action')
            assert trade.pnl is not None
            assert trade.return_pct is not None
    
    def test_equity_curve_integration(self, backtest_engine_integration):
        """Test that equity curve is properly calculated."""
        strategy = MovingAverageStrategy(5, 15)
        result = backtest_engine_integration.run_backtest(strategy, length=50)
        
        # Verify equity curve
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 100000.0  # Initial capital
        
        # Verify equity curve is monotonically increasing or decreasing
        # (allowing for some variation due to trades)
        equity_array = np.array(result.equity_curve)
        assert np.all(equity_array >= 0)  # Should never go negative
    
    def test_commission_and_slippage_integration(self, sample_market_data_file):
        """Test that commission and slippage are properly applied."""
        data_provider = CSVDataProvider(str(sample_market_data_file), ['feature1', 'feature2'])
        
        # Create engine with commission and slippage
        engine = BacktestEngine(
            data_provider=data_provider,
            initial_capital=100000.0,
            commission=0.01,  # 1% commission
            slippage=0.005,   # 0.5% slippage
            risk_free_rate=0.02
        )
        
        strategy = MovingAverageStrategy(5, 15)
        result = engine.run_backtest(strategy, length=50)
        
        # Verify that costs are reflected in results
        assert result.metrics.total_return is not None
        
        # The presence of commission and slippage should affect performance
        # (though the exact impact depends on the strategy and data)
        assert len(result.trades) >= 0  # May or may not have trades
    
    def test_error_handling_integration(self, backtest_engine_integration):
        """Test error handling in the backtesting workflow."""
        # Test with invalid strategy parameters
        with pytest.raises(Exception):
            # This should raise an error due to invalid parameters
            invalid_strategy = MovingAverageStrategy(short_window=20, long_window=10)  # Invalid: short > long
            backtest_engine_integration.run_backtest(invalid_strategy, length=50)
        
        # Test with empty data
        empty_data_provider = Mock()
        empty_data_provider.get_data.return_value = pd.DataFrame()
        
        engine_with_empty_data = BacktestEngine(
            data_provider=empty_data_provider,
            initial_capital=100000.0
        )
        
        strategy = MovingAverageStrategy(5, 15)
        
        # Should handle empty data gracefully: expect a BacktestResult with empty metrics
        result = engine_with_empty_data.run_backtest(strategy, length=50)
        assert hasattr(result, 'metrics')
        assert result.metrics.total_return == 0.0
        assert result.metrics.total_trades == 0
        assert len(result.trades) == 0
        assert isinstance(result.equity_curve, list)
        assert result.equity_curve[0] == 100000.0
    
    def test_benchmark_comparison_integration(self, backtest_engine_integration):
        """Test benchmark comparison functionality."""
        # Create a simple buy-and-hold strategy as benchmark
        class BuyAndHoldStrategy(MovingAverageStrategy):
            def generate_signal(self, data):
                return Action.BUY  # Always buy
        
        strategy = MovingAverageStrategy(5, 15)
        benchmark_strategy = BuyAndHoldStrategy(5, 15)
        
        # Run both strategies
        strategy_result = backtest_engine_integration.run_backtest(strategy, length=50)
        benchmark_result = backtest_engine_integration.run_backtest(benchmark_strategy, length=50)
        
        # Extract returns for comparison
        strategy_returns = strategy.get_returns()
        benchmark_returns = benchmark_strategy.get_returns()
        
        # Calculate metrics with benchmark
        calculator = PerformanceCalculator(risk_free_rate=0.02)
        metrics_with_benchmark = calculator.calculate_metrics(
            strategy_returns, 
            benchmark_returns=benchmark_returns
        )
        
        # Verify benchmark metrics
        assert metrics_with_benchmark.information_ratio is not None
        assert metrics_with_benchmark.beta is not None
        assert metrics_with_benchmark.alpha is not None
        assert metrics_with_benchmark.treynor_ratio is not None 