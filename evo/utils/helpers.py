"""
Helper utilities for the EVO trading system.

This module contains common helper functions used throughout the trading system.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import logging

from ..core.logging import get_logger


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
    
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.05 for 5%)
        decimal_places: Number of decimal places to show
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def normalize_data(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data array
        method: Normalization method ("zscore", "minmax", "robust")
    
    Returns:
        Normalized data array
    """
    if method == "zscore":
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-8)
    
    elif method == "minmax":
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "robust":
        median = np.median(data, axis=0, keepdims=True)
        mad = np.median(np.abs(data - median), axis=0, keepdims=True)
        return (data - median) / (mad + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate percentage returns from price data.
    
    Args:
        prices: Array of prices
    
    Returns:
        Array of returns
    """
    if len(prices) < 2:
        return np.array([])
    
    returns = np.diff(prices) / prices[:-1]
    return returns


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    
    if std_return == 0:
        return 0.0
    
    return mean_return / std_return * np.sqrt(252)  # Annualized


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of portfolio values
    
    Returns:
        Maximum drawdown as a percentage
    """
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    
    return max_dd


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate win rate from list of trades.
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
    
    Returns:
        Win rate as a percentage
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    return winning_trades / len(trades)


def format_timestamp(timestamp: Union[str, pd.Timestamp]) -> str:
    """
    Format timestamp consistently.
    
    Args:
        timestamp: Timestamp to format
    
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in MB
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0.0
    
    return file_path.stat().st_size / (1024 * 1024)


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Clean a DataFrame by removing nulls, duplicates, and infinite values.
    
    Args:
        df: Input DataFrame
        drop_duplicates: Whether to drop duplicate rows
    
    Returns:
        Cleaned DataFrame
    """
    logger = get_logger(__name__)
    
    original_rows = len(df)
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove null values
    df = df.dropna()
    
    # Remove duplicates
    if drop_duplicates:
        df = df.drop_duplicates()
    
    final_rows = len(df)
    removed_rows = original_rows - final_rows
    
    if removed_rows > 0:
        logger.info(f"Cleaned DataFrame: removed {removed_rows} rows ({removed_rows/original_rows:.1%})")
    
    return df


def create_sliding_windows(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Input time series data
        window_size: Size of each window
        step: Step size between windows
    
    Returns:
        Array of sliding windows
    """
    if len(data) < window_size:
        return np.array([])
    
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators for a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with additional technical indicators
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std()
    
    # Returns
    df['return'] = df['close'].pct_change()
    
    return df


def get_memory_usage_mb() -> float:
    """
    Get current memory usage in megabytes.
    
    Returns:
        Memory usage in MB
    """
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(logger: Optional[logging.Logger] = None):
    """
    Log current memory usage.
    
    Args:
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger(__name__)
    
    memory_mb = get_memory_usage_mb()
    logger.info(f"Current memory usage: {memory_mb:.1f} MB") 