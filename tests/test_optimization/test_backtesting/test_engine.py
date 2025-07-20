"""
Tests for backtesting engine and related components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Union

from evo.optimization.backtesting.engine import (
    BacktestEngine, 
    BacktestResult, 
    DataProvider, 
    CSVDataProvider,
    CrossValidationEngine
)
from evo.optimization.backtesting.strategies import TradingStrategy, Action
from evo.optimization.backtesting.metrics import PerformanceMetrics

pytestmark = [
    pytest.mark.unit,
    pytest.mark.optimization,
    pytest.mark.backtesting
]


class MockDataProvider(DataProvider):
    """Mock data provider for testing."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def get_data(self, start_date=None, end_date=None):
        return self.data.copy()
    
    def get_features(self, data):
        return data[['open', 'close', 'volume']].values.astype(np.float32)


class MockStrategy(TradingStrategy):
    """Mock trading strategy for testing."""
    
    def __init__(self, actions=None, initial_capital=100000.0):
        super().__init__(initial_capital)
        self.actions = actions or [Action.HOLD] * 100
        self.action_index = 0
    
    def generate_signal(self, data):
        if self.action_index < len(self.actions):
            action = self.actions[self.action_index]
            self.action_index += 1
            return action
        return Action.HOLD


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    return pd.DataFrame({
        'datetime': dates,
        'timestamp': dates,
        'open': np.random.uniform(100, 110, 500),
        'high': np.random.uniform(105, 115, 500),
        'low': np.random.uniform(95, 105, 500),
        'close': np.random.uniform(100, 110, 500),
        'volume': np.random.uniform(1000, 2000, 500),
        'feature1': np.random.randn(500),
        'feature2': np.random.randn(500)
    })


@pytest.fixture
def mock_data_provider(sample_market_data):
    """Create a mock data provider."""
    return MockDataProvider(sample_market_data)


@pytest.fixture
def backtest_engine(mock_data_provider):
    """Create a backtest engine instance."""
    return BacktestEngine(
        data_provider=mock_data_provider,
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0,
        risk_free_rate=0.02
    )


class TestBacktestResult:
    """Test BacktestResult dataclass."""
    
    def test_backtest_result_creation(self):
        """Test creating a BacktestResult instance."""
        strategy = Mock(spec=TradingStrategy)
        metrics = Mock(spec=PerformanceMetrics)
        trades = [{'entry': 100, 'exit': 110}]
        equity_curve = [100000, 101000, 102000]
        metadata = {'test': True}
        
        result = BacktestResult(
            strategy=strategy,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            metadata=metadata
        )
        
        assert result.strategy == strategy
        assert result.metrics == metrics
        assert result.trades == trades
        assert result.equity_curve == equity_curve
        assert result.metadata == metadata


class TestDataProvider:
    """Test DataProvider abstract base class."""
    
    def test_data_provider_abstract(self):
        """Test that DataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProvider()


class TestCSVDataProvider:
    """Test CSVDataProvider class."""
    
    def test_csv_data_provider_creation(self, temp_dir):
        """Test creating a CSVDataProvider instance."""
        # Create test CSV file
        csv_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 2000, 10),
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        csv_path = temp_dir / "test_data.csv"
        csv_data.to_csv(csv_path, index=False)
        
        provider = CSVDataProvider(str(csv_path), ['feature1', 'feature2'])
        
        assert provider.data_path == str(csv_path)
        assert provider.feature_columns == ['feature1', 'feature2']
        assert provider._data is None
    
    def test_get_data(self, temp_dir):
        """Test getting data from CSV file."""
        # Create test CSV file
        csv_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 2000, 10),
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        csv_path = temp_dir / "test_data.csv"
        csv_data.to_csv(csv_path, index=False)
        
        provider = CSVDataProvider(str(csv_path), ['feature1', 'feature2'])
        data = provider.get_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert 'datetime' in data.columns
        assert 'open' in data.columns
        assert 'close' in data.columns
    
    def test_get_data_with_date_filter(self, temp_dir):
        """Test getting data with date filtering."""
        # Create test CSV file
        csv_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(105, 115, 10),
            'low': np.random.uniform(95, 105, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000, 2000, 10),
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        csv_path = temp_dir / "test_data.csv"
        csv_data.to_csv(csv_path, index=False)
        
        provider = CSVDataProvider(str(csv_path), ['feature1', 'feature2'])
        data = provider.get_data(start_date='2023-01-05', end_date='2023-01-08')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) <= 10  # Should be filtered
    
    def test_get_features(self, temp_dir):
        """Test extracting features from data."""
        # Create test CSV file
        csv_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [99, 100, 101, 102, 103],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        csv_path = temp_dir / "test_data.csv"
        csv_data.to_csv(csv_path, index=False)
        
        provider = CSVDataProvider(str(csv_path), ['feature1', 'feature2'])
        data = provider.get_data()
        features = provider.get_features(data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (5, 2)
        assert features.dtype == np.float32
        assert np.array_equal(features[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(features[:, 1], np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=features.dtype))


class TestBacktestEngine:
    """Test BacktestEngine class."""
    
    def test_backtest_engine_creation(self, mock_data_provider):
        """Test creating a BacktestEngine instance."""
        engine = BacktestEngine(
            data_provider=mock_data_provider,
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0,
            risk_free_rate=0.02
        )
        
        assert engine.data_provider == mock_data_provider
        assert engine.initial_capital == 100000.0
        assert engine.commission == 0.001
        assert engine.slippage == 0.0
        assert engine.risk_free_rate == 0.02
        assert engine.performance_calculator is not None
    
    def test_run_backtest_with_strategy_object(self, backtest_engine, sample_market_data):
        """Test running backtest with a strategy object."""
        # Create a simple strategy that buys on first day, sells on last day
        actions = [Action.BUY] + [Action.HOLD] * 98 + [Action.SELL]
        strategy = MockStrategy(actions=actions)
        
        result = backtest_engine.run_backtest(strategy, length=50)
        
        assert isinstance(result, BacktestResult)
        assert result.strategy == strategy
        assert isinstance(result.metrics, PerformanceMetrics)
        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, list)
        assert isinstance(result.metadata, dict)
    
    def test_run_backtest_with_strategy_string(self, backtest_engine):
        """Test running backtest with a strategy type string."""
        with patch('evo.optimization.backtesting.engine.StrategyFactory') as mock_factory:
            mock_strategy = MockStrategy()
            mock_factory.create_strategy.return_value = mock_strategy
            
            result = backtest_engine.run_backtest("moving_average", length=50)
            
            mock_factory.create_strategy.assert_called_once()
            assert isinstance(result, BacktestResult)
    
    def test_run_backtest_with_date_range(self, backtest_engine):
        """Test running backtest with date range."""
        strategy = MockStrategy()
        
        result = backtest_engine.run_backtest(
            strategy, 
            start_date='2023-01-01', 
            end_date='2023-01-31'
        )
        
        assert isinstance(result, BacktestResult)
    
    def test_run_backtest_with_model(self, backtest_engine, temp_dir):
        """Test running backtest with a trained model."""
        # Create a mock model file
        model_path = temp_dir / "test_model.zip"
        model_path.touch()
        
        with patch('evo.optimization.backtesting.engine.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.load.return_value = mock_model
            
            with patch('evo.optimization.backtesting.engine.PPOStrategy') as mock_ppo_strategy:
                mock_strategy_instance = MockStrategy()
                mock_ppo_strategy.return_value = mock_strategy_instance
                
                result = backtest_engine.run_backtest_with_model(str(model_path), length=50)
                
                assert isinstance(result, BacktestResult)
                mock_ppo.load.assert_called_once_with(str(model_path))
    
    def test_run_backtest_with_model_file_not_found(self, backtest_engine):
        """Test running backtest with non-existent model file."""
        with pytest.raises(FileNotFoundError):
            backtest_engine.run_backtest_with_model("nonexistent_model.zip")
    
    def test_run_multiple_backtests(self, backtest_engine):
        """Test running multiple backtests."""
        strategy = MockStrategy()
        
        results = backtest_engine.run_multiple_backtests(
            strategy, 
            num_backtests=3, 
            length=50
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, BacktestResult) for result in results)
    
    def test_aggregate_results(self, backtest_engine):
        """Test aggregating multiple backtest results."""
        # Create mock results
        mock_metrics1 = Mock(spec=PerformanceMetrics)
        mock_metrics1.sharpe_ratio = 1.5
        mock_metrics1.total_return = 0.1
        mock_metrics1.to_dict.return_value = {'sharpe_ratio': 1.5, 'total_return': 0.1}

        mock_metrics2 = Mock(spec=PerformanceMetrics)
        mock_metrics2.sharpe_ratio = 2.0
        mock_metrics2.total_return = 0.2
        mock_metrics2.to_dict.return_value = {'sharpe_ratio': 2.0, 'total_return': 0.2}
        
        result1 = BacktestResult(
            strategy=Mock(),
            metrics=mock_metrics1,
            trades=[],
            equity_curve=[],
            metadata={}
        )
        
        result2 = BacktestResult(
            strategy=Mock(),
            metrics=mock_metrics2,
            trades=[],
            equity_curve=[],
            metadata={}
        )
        
        aggregated = backtest_engine.aggregate_results([result1, result2])
        
        assert isinstance(aggregated, dict)
        assert 'sharpe_ratio_mean' in aggregated
        assert 'total_return_mean' in aggregated
        assert aggregated['sharpe_ratio_mean'] == 1.75  # Average of 1.5 and 2.0
        assert aggregated['sharpe_ratio_max'] == 2.0
        assert aggregated['sharpe_ratio_min'] == 1.5
    
    def test_calculate_robust_fitness(self, backtest_engine):
        """Test calculating robust fitness from multiple results."""
        # Create mock results
        mock_metrics1 = Mock(spec=PerformanceMetrics)
        mock_metrics1.sharpe_ratio = 1.5
        
        mock_metrics2 = Mock(spec=PerformanceMetrics)
        mock_metrics2.sharpe_ratio = 2.0
        
        result1 = BacktestResult(
            strategy=Mock(),
            metrics=mock_metrics1,
            trades=[],
            equity_curve=[],
            metadata={}
        )
        
        result2 = BacktestResult(
            strategy=Mock(),
            metrics=mock_metrics2,
            trades=[],
            equity_curve=[],
            metadata={}
        )
        
        fitness = backtest_engine.calculate_robust_fitness([result1, result2])
        
        assert isinstance(fitness, float)
        assert fitness > 0  # Should be positive for good results


class TestCrossValidationEngine:
    """Test CrossValidationEngine class."""
    
    def test_cross_validation_engine_creation(self, backtest_engine):
        """Test creating a CrossValidationEngine instance."""
        cv_engine = CrossValidationEngine(backtest_engine)
        
        assert cv_engine.backtest_engine == backtest_engine
    
    def test_run_time_series_cv(self, backtest_engine):
        """Test running time series cross-validation."""
        strategy = MockStrategy()
        
        cv_engine = CrossValidationEngine(backtest_engine)
        with patch.object(backtest_engine, 'run_backtest') as mock_run:
            mock_result = Mock(spec=BacktestResult)
            mock_run.return_value = mock_result
            
            results = cv_engine.run_time_series_cv(strategy, n_splits=3)
            
            assert isinstance(results, list)
            assert len(results) == 3
            assert all(isinstance(result, BacktestResult) for result in results)

    def test_run_walk_forward_cv(self, backtest_engine):
        """Test running walk-forward cross-validation."""
        strategy = MockStrategy()
        
        cv_engine = CrossValidationEngine(backtest_engine)
        with patch.object(backtest_engine, 'run_backtest') as mock_run:
            mock_result = Mock(spec=BacktestResult)
            mock_run.return_value = mock_result
            
            results = cv_engine.run_walk_forward_cv(
                strategy, 
                window_size=100, 
                step_size=50
            )
            
            assert isinstance(results, list)
            assert all(isinstance(result, BacktestResult) for result in results) 