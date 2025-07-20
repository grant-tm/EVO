"""
Performance metrics for trading strategy evaluation.

This module provides comprehensive metrics for evaluating trading strategy
performance, including risk-adjusted returns, drawdown analysis, and
statistical measures.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Types of performance metrics."""
    RETURN = "return"
    RISK = "risk"
    RATIO = "ratio"
    DRAWDOWN = "drawdown"
    TRADE = "trade"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for trading strategies."""
    
    # Basic return metrics
    total_return: float
    annualized_return: float
    daily_returns: np.ndarray
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    
    # Risk-adjusted return metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trade metrics
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_trades: int
    
    # Additional metrics
    beta: float
    alpha: float
    treynor_ratio: float
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if not isinstance(self.daily_returns, np.ndarray):
            self.daily_returns = np.array(self.daily_returns)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_trades": self.total_trades,
            "beta": self.beta,
            "alpha": self.alpha,
            "treynor_ratio": self.treynor_ratio
        }


class PerformanceCalculator:
    """Calculator for comprehensive trading performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(
        self, 
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Array of daily returns
            benchmark_returns: Array of benchmark returns (optional)
            trades: List of trade dictionaries (optional)
            
        Returns:
            Performance metrics object
        """
        returns = np.array(returns)
        
        # Basic return metrics
        total_return = self._calculate_total_return(returns)
        annualized_return = self._calculate_annualized_return(returns)
        
        # Risk metrics
        volatility = self._calculate_volatility(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        var_95, cvar_95 = self._calculate_var_cvar(returns)
        
        # Risk-adjusted return metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns, annualized_return, volatility)
        sortino_ratio = self._calculate_sortino_ratio(returns, annualized_return)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
        
        # Trade metrics
        if trades is not None:
            trade_metrics = self._calculate_trade_metrics(trades)
        else:
            trade_metrics = self._estimate_trade_metrics_from_returns(returns)
        
        # Benchmark metrics
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)
        else:
            benchmark_metrics = {
                "information_ratio": 0.0,
                "beta": 1.0,
                "alpha": 0.0,
                "treynor_ratio": sharpe_ratio
            }
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns=returns,
            volatility=volatility,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=benchmark_metrics["information_ratio"],
            win_rate=trade_metrics["win_rate"],
            profit_factor=trade_metrics["profit_factor"],
            average_win=trade_metrics["average_win"],
            average_loss=trade_metrics["average_loss"],
            max_consecutive_wins=trade_metrics["max_consecutive_wins"],
            max_consecutive_losses=trade_metrics["max_consecutive_losses"],
            total_trades=trade_metrics["total_trades"],
            beta=benchmark_metrics["beta"],
            alpha=benchmark_metrics["alpha"],
            treynor_ratio=benchmark_metrics["treynor_ratio"]
        )
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return from daily returns."""
        return np.prod(1 + returns) - 1
    
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        return (1 + total_return) ** (252 / n_periods) - 1
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        return np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_var_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = np.mean(returns[returns <= var])
        return var, cvar
    
    def _calculate_sharpe_ratio(
        self, 
        returns: np.ndarray, 
        annualized_return: float, 
        volatility: float
    ) -> float:
        """Calculate Sharpe ratio."""
        if volatility == 0:
            return 0.0
        return (annualized_return - self.risk_free_rate) / volatility
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, annualized_return: float) -> float:
        """Calculate Sortino ratio."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(negative_returns) * np.sqrt(252)
        if downside_deviation == 0:
            return 0.0
        
        return (annualized_return - self.risk_free_rate) / downside_deviation
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        return annualized_return / abs(max_drawdown)
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics from trade data."""
        if not trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "total_trades": 0
            }
        
        # Extract pr calculate trade returns
        trade_returns = []
        for trade in trades:
            if "return" in trade:
                trade_returns.append(trade["return"])
            elif "entry_price" in trade and "exit_price" in trade:
                entry = trade["entry_price"]
                exit = trade["exit_price"]
                trade_returns.append((exit - entry) / entry)
            else:
                trade_returns.append(0.0)
        trade_returns = np.array(trade_returns)
        
        # Identify winning and losing trades
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        # Calculate metrics
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        total_profit = sum(winning_trades) if winning_trades else 0.0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        average_win = np.mean(winning_trades) if winning_trades else 0.0
        average_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        # Calculate consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(trade_returns, positive=True)
        max_consecutive_losses = self._calculate_max_consecutive(trade_returns, positive=False)
        
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "total_trades": total_trades
        }
    
    def _estimate_trade_metrics_from_returns(self, returns: np.ndarray) -> Dict[str, float]:
        """Estimate trade metrics from daily returns."""
        # Simple estimation: treat positive days as wins, negative as losses
        positive_days = returns[returns > 0]
        negative_days = returns[returns < 0]
        
        total_days = len(returns)
        win_rate = len(positive_days) / total_days if total_days > 0 else 0.0
        
        total_profit = np.sum(positive_days) if len(positive_days) > 0 else 0.0
        total_loss = abs(np.sum(negative_days)) if len(negative_days) > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        average_win = np.mean(positive_days) if len(positive_days) > 0 else 0.0
        average_loss = np.mean(negative_days) if len(negative_days) > 0 else 0.0
        
        max_consecutive_wins = self._calculate_max_consecutive(returns, positive=True)
        max_consecutive_losses = self._calculate_max_consecutive(returns, positive=False)
        
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "total_trades": total_days
        }
    
    def _calculate_max_consecutive(self, values: np.ndarray, positive: bool = True) -> int:
        """Calculate maximum consecutive positive or negative values."""
        if positive:
            condition = values > 0
        else:
            condition = values < 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for val in condition:
            if val:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_benchmark_metrics(
        self, 
        returns: np.ndarray, 
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate benchmark-relative metrics."""
        # Ensure same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        # Calculate excess returns
        excess_returns = returns - benchmark_returns
        
        # Information ratio
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
        
        # Beta and Alpha
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
        
        # Treynor ratio
        benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
        treynor_ratio = (np.mean(returns) * 252 - self.risk_free_rate) / beta if beta != 0 else 0.0
        
        return {
            "information_ratio": information_ratio,
            "beta": beta,
            "alpha": alpha,
            "treynor_ratio": treynor_ratio
        }


class MetricsAggregator:
    """Aggregator for combining multiple performance metrics."""
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[PerformanceMetrics]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple backtests.
        
        Args:
            metrics_list: List of performance metrics
            
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        metric_names = list(metrics_list[0].to_dict().keys())
        
        for metric_name in metric_names:
            values = [getattr(metrics, metric_name) for metrics in metrics_list]
            
            # Handle different aggregation methods based on metric type
            if metric_name in ["total_return", "annualized_return", "sharpe_ratio", 
                             "sortino_ratio", "calmar_ratio", "information_ratio",
                             "win_rate", "profit_factor", "beta", "alpha", "treynor_ratio"]:
                # Use median for ratio and return metrics (robust to outliers)
                aggregated[f"{metric_name}_median"] = np.median(values)
                aggregated[f"{metric_name}_mean"] = np.mean(values)
                aggregated[f"{metric_name}_std"] = np.std(values)
                aggregated[f"{metric_name}_min"] = np.min(values)
                aggregated[f"{metric_name}_max"] = np.max(values)
            
            elif metric_name in ["volatility", "max_drawdown", "var_95", "cvar_95"]:
                # Use mean for risk metrics
                aggregated[f"{metric_name}_mean"] = np.mean(values)
                aggregated[f"{metric_name}_std"] = np.std(values)
                aggregated[f"{metric_name}_max"] = np.max(values)
            
            else:
                # Default aggregation
                aggregated[f"{metric_name}_mean"] = np.mean(values)
                aggregated[f"{metric_name}_std"] = np.std(values)
        
        return aggregated
    
    @staticmethod
    def calculate_robust_fitness(metrics_list: List[PerformanceMetrics], 
                               primary_metric: str = "sharpe_ratio") -> float:
        """
        Calculate robust fitness score from multiple backtests.
        
        Args:
            metrics_list: List of performance metrics
            primary_metric: Primary metric to use for fitness
            
        Returns:
            Robust fitness score
        """
        if not metrics_list:
            return float('-inf')
        
        # Extract primary metric values
        primary_values = [getattr(metrics, primary_metric) for metrics in metrics_list]
        
        # Calculate robust fitness (median to handle outliers)
        fitness_score = np.median(primary_values)
        
        # Apply penalty for high variance (unstable performance)
        variance = np.var(primary_values)
        stability_penalty = min(variance * 0.1, 0.5)  # Cap penalty at 0.5
        
        return fitness_score - stability_penalty 