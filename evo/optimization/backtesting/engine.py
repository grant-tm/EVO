"""
Backtesting engine for evaluating trading strategies.

This module provides a comprehensive backtesting framework for evaluating
trading strategies on historical data.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from stable_baselines3 import PPO
from evo.core.logging import get_logger
from .strategies import TradingStrategy, PPOStrategy, StrategyFactory
from .metrics import PerformanceMetrics, PerformanceCalculator, MetricsAggregator

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    
    strategy: TradingStrategy
    metrics: PerformanceMetrics
    trades: List[Any]
    equity_curve: List[float]
    metadata: Dict[str, Any]


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_data(self, start_date: Optional[str] = None, 
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Get historical data for backtesting."""
        pass
    
    @abstractmethod
    def get_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from data for model prediction."""
        pass


class CSVDataProvider(DataProvider):
    """Data provider that reads from CSV files."""
    
    def __init__(self, data_path: str, feature_columns: Optional[List[str]] = None):
        """
        Initialize CSV data provider.
        
        Args:
            data_path: Path to CSV data file
            feature_columns: List of column names to use as features (optional)
        """
        self.data_path = data_path
        self.feature_columns = feature_columns or []
        self._data: Optional[pd.DataFrame] = None
    
    def get_data(self, start_date: Optional[str] = None, 
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Load and filter data from CSV."""
        if self._data is None:
            self._data = pd.read_csv(self.data_path)
            
            # Convert timestamp column
            if 'timestamp' in self._data.columns:
                self._data['datetime'] = pd.to_datetime(self._data['timestamp'])
                self._data = self._data.sort_values('datetime').drop_duplicates(subset=['datetime'])
            
            # Clean data
            required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            if self.feature_columns:
                required_columns += self.feature_columns
            self._data = self._data.dropna(subset=required_columns).reset_index(drop=True)
        
        data = self._data.copy()
        
        # Filter by date range if provided
        if start_date:
            data = data[data['datetime'] >= start_date]
        if end_date:
            data = data[data['datetime'] <= end_date]
        
        return data.reset_index(drop=True)
    
    def get_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from data."""
        if not self.feature_columns:
            # Return empty array with shape (len(data), 0) if no features
            return np.empty((len(data), 0), dtype=np.float32)
        features = data[self.feature_columns].values.astype(np.float32)
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        return features


class BacktestEngine:
    """Main backtesting engine for evaluating trading strategies."""
    
    def __init__(
        self,
        data_provider: DataProvider,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize backtesting engine.
        
        Args:
            data_provider: Provider for historical data
            initial_capital: Initial capital for trading
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            risk_free_rate: Risk-free rate for performance calculations
        """
        self.data_provider = data_provider
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.performance_calculator = PerformanceCalculator(risk_free_rate)
    
    def run_backtest(
        self,
        strategy: Union[TradingStrategy, str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        length: Optional[int] = None,
        **strategy_kwargs
    ) -> BacktestResult:
        """
        Run a backtest with the specified strategy.
        
        Args:
            strategy: Trading strategy or strategy type string
            start_date: Start date for backtest
            end_date: End date for backtest
            length: Number of periods to backtest (if not using date range)
            **strategy_kwargs: Additional strategy parameters
            
        Returns:
            Backtest result
        """
        # Get data
        data = self.data_provider.get_data(start_date, end_date)
        
        # Apply length filter if specified
        if length is not None and len(data) > length:
            # Randomly select a slice of data
            max_start = len(data) - length
            start_idx = np.random.randint(0, max_start)
            data = data.iloc[start_idx:start_idx + length].reset_index(drop=True)
        
        # Create strategy if string provided
        if isinstance(strategy, str):
            strategy = StrategyFactory.create_strategy(
                strategy, 
                initial_capital=self.initial_capital,
                **strategy_kwargs
            )
        
        # Run backtest
        return self._execute_backtest(strategy, data)
    
    def run_backtest_with_model(
        self,
        model_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        length: Optional[int] = None
    ) -> BacktestResult:
        """
        Run backtest using a trained PPO model.
        
        Args:
            model_path: Path to trained PPO model
            start_date: Start date for backtest
            end_date: End date for backtest
            length: Number of periods to backtest
            
        Returns:
            Backtest result
        """
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = PPO.load(model_path)
        
        # Create PPO strategy
        strategy = PPOStrategy(model, initial_capital=self.initial_capital)
        
        # Run backtest
        return self.run_backtest(strategy, start_date, end_date, length)
    
    def run_multiple_backtests(
        self,
        strategy: Union[TradingStrategy, str],
        num_backtests: int = 10,
        length: int = 1000,
        **strategy_kwargs
    ) -> List[BacktestResult]:
        """
        Run multiple backtests with different data slices.
        
        Args:
            strategy: Trading strategy or strategy type
            num_backtests: Number of backtests to run
            length: Length of each backtest
            **strategy_kwargs: Additional strategy parameters
            
        Returns:
            List of backtest results
        """
        results = []
        
        for i in range(num_backtests):
            logger.info(f"Running backtest {i + 1}/{num_backtests}")
            
            try:
                result = self.run_backtest(strategy, length=length, **strategy_kwargs)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Backtest {i + 1} failed: {e}")
                # Continue with other backtests
        
        return results
    
    def _execute_backtest(self, strategy: TradingStrategy, data: pd.DataFrame) -> BacktestResult:
        """Execute a single backtest."""
        # Get features for model-based strategies
        features = self.data_provider.get_features(data)
        
        # Initialize strategy
        strategy.initial_capital = self.initial_capital
        strategy.current_capital = self.initial_capital
        strategy.position = None
        strategy.trades = []
        strategy.equity_curve = [self.initial_capital]
        
        # Run backtest
        for i, row in data.iterrows():
            # Prepare data for strategy
            data_point = {
                'time': i,
                'datetime': row.get('datetime', i),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'features': features[i] if i < len(features) else []
            }
            
            # Update strategy
            completed_trade = strategy.update(data_point)
            
            # Apply transaction costs if trade completed
            if completed_trade is not None:
                self._apply_transaction_costs(completed_trade, row['close'])
        
        # Close any remaining position at the end
        if strategy.position is not None:
            final_price = data.iloc[-1]['close']
            final_time = len(data) - 1
            strategy._close_position(final_time, final_price)
        
        # Calculate performance metrics
        returns = strategy.get_returns()
        if len(returns) == 0:
            # No trades, create empty metrics
            metrics = self._create_empty_metrics()
        else:
            metrics = self.performance_calculator.calculate_metrics(returns)
        
        # Create result
        result = BacktestResult(
            strategy=strategy,
            metrics=metrics,
            trades=strategy.trades,
            equity_curve=strategy.equity_curve,
            metadata={
                'data_length': len(data),
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage
            }
        )
        
        return result
    
    def _apply_transaction_costs(self, trade: Any, price: float) -> None:
        """Apply commission and slippage to trade."""
        if hasattr(trade, 'pnl') and trade.pnl is not None:
            # Calculate transaction costs
            transaction_value = abs(trade.position_size * price)
            commission_cost = transaction_value * self.commission
            slippage_cost = transaction_value * self.slippage
            total_cost = commission_cost + slippage_cost
            
            # Adjust PnL
            trade.pnl -= total_cost
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty performance metrics for no-trade scenarios."""
        empty_returns = np.array([])
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            daily_returns=empty_returns,
            volatility=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            total_trades=0,
            beta=1.0,
            alpha=0.0,
            treynor_ratio=0.0
        )
    
    def aggregate_results(self, results: List[BacktestResult]) -> Dict[str, float]:
        """
        Aggregate results from multiple backtests.
        
        Args:
            results: List of backtest results
            
        Returns:
            Aggregated metrics
        """
        if not results:
            return {}
        
        metrics_list = [result.metrics for result in results]
        return MetricsAggregator.aggregate_metrics(metrics_list)
    
    def calculate_robust_fitness(
        self, 
        results: List[BacktestResult], 
        primary_metric: str = "sharpe_ratio"
    ) -> float:
        """
        Calculate robust fitness score from multiple backtests.
        
        Args:
            results: List of backtest results
            primary_metric: Primary metric to use for fitness
            
        Returns:
            Robust fitness score
        """
        if not results:
            return float('-inf')
        
        metrics_list = [result.metrics for result in results]
        return MetricsAggregator.calculate_robust_fitness(metrics_list, primary_metric)


class CrossValidationEngine:
    """Engine for cross-validation of trading strategies."""
    
    def __init__(self, backtest_engine: BacktestEngine):
        """
        Initialize cross-validation engine.
        
        Args:
            backtest_engine: Backtesting engine to use
        """
        self.backtest_engine = backtest_engine
    
    def run_time_series_cv(
        self,
        strategy: Union[TradingStrategy, str],
        n_splits: int = 5,
        test_size: float = 0.2,
        **strategy_kwargs
    ) -> List[BacktestResult]:
        """
        Run time series cross-validation.
        
        Args:
            strategy: Trading strategy or strategy type
            n_splits: Number of CV splits
            test_size: Size of test set as fraction of total data
            **strategy_kwargs: Additional strategy parameters
            
        Returns:
            List of backtest results for each fold
        """
        # Get full dataset
        full_data = self.backtest_engine.data_provider.get_data()
        total_length = len(full_data)
        test_length = int(total_length * test_size)
        
        results = []
        
        for i in range(n_splits):
            # Calculate split indices
            test_start = total_length - (n_splits - i) * test_length
            test_end = test_start + test_length
            
            if test_start < 0:
                continue
            
            # Split data
            train_data = full_data.iloc[:test_start]
            test_data = full_data.iloc[test_start:test_end]
            
            if len(test_data) < 20:  # Minimum test size
                continue
            
            logger.info(f"Running CV fold {i + 1}/{n_splits}")
            
            try:
                # Run backtest on test set
                result = self.backtest_engine._execute_backtest(
                    strategy, test_data
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"CV fold {i + 1} failed: {e}")
        
        return results
    
    def run_walk_forward_cv(
        self,
        strategy: Union[TradingStrategy, str],
        window_size: int = 1000,
        step_size: int = 200,
        **strategy_kwargs
    ) -> List[BacktestResult]:
        """
        Run walk-forward cross-validation.
        
        Args:
            strategy: Trading strategy or strategy type
            window_size: Size of training window
            step_size: Step size for moving window
            **strategy_kwargs: Additional strategy parameters
            
        Returns:
            List of backtest results for each window
        """
        # Get full dataset
        full_data = self.backtest_engine.data_provider.get_data()
        total_length = len(full_data)
        
        results = []
        
        for start_idx in range(0, total_length - window_size, step_size):
            end_idx = start_idx + window_size
            
            if end_idx > total_length:
                break
            
            # Extract window data
            window_data = full_data.iloc[start_idx:end_idx].reset_index(drop=True)
            
            logger.info(f"Running walk-forward window {start_idx}-{end_idx}")
            
            try:
                # Run backtest on window
                result = self.backtest_engine._execute_backtest(
                    strategy, window_data
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Walk-forward window {start_idx}-{end_idx} failed: {e}")
        
        return results 