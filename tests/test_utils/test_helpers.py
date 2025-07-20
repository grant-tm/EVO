"""
Tests for EVO helper utilities.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from evo.utils.helpers import (
    ensure_directory, safe_divide, format_percentage, normalize_data,
    calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_win_rate, format_timestamp, get_file_size_mb,
    clean_dataframe, create_sliding_windows, calculate_technical_indicators,
    get_memory_usage_mb, log_memory_usage
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.utils,
    pytest.mark.helpers
]

class TestDirectoryHelpers:
    """Test directory-related helper functions."""
    
    def test_ensure_directory_new(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_directory"
        result = ensure_directory(new_dir)
        
        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_ensure_directory_exists(self, temp_dir):
        """Test ensuring directory that already exists."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory(existing_dir)
        
        assert result == existing_dir
        assert existing_dir.exists()
    
    def test_ensure_directory_nested(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "parent" / "child" / "grandchild"
        result = ensure_directory(nested_dir)
        
        assert result == nested_dir
        assert nested_dir.exists()
        assert (temp_dir / "parent").exists()
        assert (temp_dir / "parent" / "child").exists()


class TestMathHelpers:
    """Test mathematical helper functions."""
    
    def test_safe_divide_normal(self):
        """Test normal division."""
        result = safe_divide(10, 2)
        assert result == 5.0
    
    def test_safe_divide_zero_denominator(self):
        """Test division by zero returns default."""
        result = safe_divide(10, 0)
        assert result == 0.0
    
    def test_safe_divide_custom_default(self):
        """Test division by zero with custom default."""
        result = safe_divide(10, 0, default=42.0)
        assert result == 42.0
    
    def test_safe_divide_float_result(self):
        """Test division resulting in float."""
        result = safe_divide(5, 2)
        assert result == 2.5
    
    def test_format_percentage_basic(self):
        """Test basic percentage formatting."""
        result = format_percentage(0.05)
        assert result == "5.00%"
    
    def test_format_percentage_custom_decimals(self):
        """Test percentage formatting with custom decimal places."""
        result = format_percentage(0.12345, decimal_places=3)
        assert result == "12.345%"
    
    def test_format_percentage_zero(self):
        """Test percentage formatting for zero."""
        result = format_percentage(0.0)
        assert result == "0.00%"
    
    def test_format_percentage_negative(self):
        """Test percentage formatting for negative values."""
        result = format_percentage(-0.05)
        assert result == "-5.00%"


class TestDataNormalization:
    """Test data normalization functions."""
    
    def test_normalize_data_zscore(self):
        """Test z-score normalization."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = normalize_data(data, method="zscore")
        
        # Check that result has mean close to 0 and std close to 1
        assert np.allclose(np.mean(result, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(result, axis=0), 1, atol=1e-10)
    
    def test_normalize_data_minmax(self):
        """Test min-max normalization."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = normalize_data(data, method="minmax")
        
        # Check that result is between 0 and 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        # Check that min is 0 and max is 1
        assert np.allclose(np.min(result, axis=0), 0, atol=1e-10)
        assert np.allclose(np.max(result, axis=0), 1, atol=1e-10)
    
    def test_normalize_data_robust(self):
        """Test robust normalization."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = normalize_data(data, method="robust")
        
        # Check that result has median close to 0
        assert np.allclose(np.median(result, axis=0), 0, atol=1e-10)
    
    def test_normalize_data_invalid_method(self):
        """Test normalization with invalid method."""
        data = np.array([[1, 2, 3]])
        with pytest.raises(ValueError) as exc_info:
            normalize_data(data, method="invalid")
        assert "Unknown normalization method" in str(exc_info.value)
    
    def test_normalize_data_constant_column(self):
        """Test normalization with constant column (zero std)."""
        data = np.array([[1, 2, 5], [1, 3, 6], [1, 4, 7]])  # First column is constant
        result = normalize_data(data, method="zscore")
        
        # Should handle constant column gracefully
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestFinancialCalculations:
    """Test financial calculation functions."""
    
    def test_calculate_returns_basic(self):
        """Test basic return calculation."""
        prices = np.array([100, 110, 105, 115])
        returns = calculate_returns(prices)
        
        expected = np.array([0.1, -0.04545, 0.09524])  # Manual calculation
        assert np.allclose(returns, expected, atol=1e-4)
    
    def test_calculate_returns_single_price(self):
        """Test return calculation with single price."""
        prices = np.array([100])
        returns = calculate_returns(prices)
        
        assert len(returns) == 0
    
    def test_calculate_returns_empty(self):
        """Test return calculation with empty array."""
        prices = np.array([])
        returns = calculate_returns(prices)
        
        assert len(returns) == 0
    
    def test_calculate_sharpe_ratio_basic(self):
        """Test basic Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        
        # Should be a reasonable positive value for these returns
        assert sharpe > 0
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_calculate_sharpe_ratio_zero_returns(self):
        """Test Sharpe ratio with zero returns."""
        returns = np.array([0.0, 0.0, 0.0])
        sharpe = calculate_sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_calculate_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty returns."""
        returns = np.array([])
        sharpe = calculate_sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_calculate_sharpe_ratio_with_risk_free_rate(self):
        """Test Sharpe ratio with risk-free rate."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        # Should be lower than without risk-free rate
        sharpe_no_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe < sharpe_no_rf
    
    def test_calculate_max_drawdown_basic(self):
        """Test basic maximum drawdown calculation."""
        equity_curve = np.array([100, 110, 105, 115, 108, 120])
        max_dd = calculate_max_drawdown(equity_curve)
        
        # Manual calculation: peak at 115, low at 108, drawdown = (115-108)/115 = 0.0609
        expected = 0.0609
        assert abs(max_dd - expected) < 0.001
    
    def test_calculate_max_drawdown_always_rising(self):
        """Test max drawdown with always rising equity curve."""
        equity_curve = np.array([100, 110, 120, 130, 140])
        max_dd = calculate_max_drawdown(equity_curve)
        
        assert max_dd == 0.0
    
    def test_calculate_max_drawdown_single_value(self):
        """Test max drawdown with single value."""
        equity_curve = np.array([100])
        max_dd = calculate_max_drawdown(equity_curve)
        
        assert max_dd == 0.0
    
    def test_calculate_win_rate_basic(self):
        """Test basic win rate calculation."""
        trades = [
            {'pnl': 100},   # Win
            {'pnl': -50},   # Loss
            {'pnl': 200},   # Win
            {'pnl': -30},   # Loss
            {'pnl': 150}    # Win
        ]
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 0.6  # 3 wins out of 5 trades

    def test_calculate_win_rate_all_wins(self):
        """Test win rate with all winning trades."""
        trades = [{'pnl': 100}, {'pnl': 200}, {'pnl': 150}]
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 1.0
    
    def test_calculate_win_rate_all_losses(self):
        """Test win rate with all losing trades."""
        trades = [{'pnl': -100}, {'pnl': -200}, {'pnl': -150}]
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 0.0
    
    def test_calculate_win_rate_empty(self):
        """Test win rate with empty trades list."""
        trades = []
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 0.0
    
    def test_calculate_win_rate_missing_pnl(self):
        """Test win rate with trades missing pnl key."""
        trades = [{'pnl': 100}, {}, {'pnl': -50}]
        win_rate = calculate_win_rate(trades)
        
        # Trade 1: pnl=100 > 0 → win
        # Trade 2: pnl=0 (default) ≤ 0 → loss  
        # Trade 3: pnl=-50 ≤ 0 → loss
        # Result: 1 win out of 3 trades = 0.333...
        assert win_rate == 1/3


class TestDataProcessing:
    """Test data processing helper functions."""
    
    def test_clean_dataframe_basic(self):
        """Test basic DataFrame cleaning."""
        df = pd.DataFrame({
            'A': [1, 2, 2, 3, 3, 3],  # Duplicates
            'B': [1, 2, 3, 4, 5, 6],
            'C': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]  # NaN values
        })
        
        result = clean_dataframe(df, drop_duplicates=True)
        
        # Should remove duplicates and NaN values
        assert len(result) < len(df)
        assert not result.duplicated().any()
        assert not result.isna().any().any()
    
    def test_clean_dataframe_no_duplicates(self):
        """Test DataFrame cleaning without dropping duplicates."""
        df = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [1, 2, 3, 4]
        })
        
        result = clean_dataframe(df, drop_duplicates=False)
        
        # Should keep duplicates
        assert len(result) == len(df)
    
    def test_create_sliding_windows_basic(self):
        """Test basic sliding window creation."""
        data = np.array([1, 2, 3, 4, 5, 6])
        windows = create_sliding_windows(data, window_size=3, step=1)
        
        expected = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]
        ])
        assert np.array_equal(windows, expected)
    
    def test_create_sliding_windows_step_2(self):
        """Test sliding windows with step=2."""
        data = np.array([1, 2, 3, 4, 5, 6])
        windows = create_sliding_windows(data, window_size=3, step=2)
        
        expected = np.array([
            [1, 2, 3],
            [3, 4, 5]
        ])
        assert np.array_equal(windows, expected)
    
    def test_create_sliding_windows_2d(self):
        """Test sliding windows with 2D data."""
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        windows = create_sliding_windows(data, window_size=2, step=1)
        
        expected = np.array([
            [[1, 10], [2, 20]],
            [[2, 20], [3, 30]],
            [[3, 30], [4, 40]]
        ])
        assert np.array_equal(windows, expected)
    
    def test_calculate_technical_indicators_basic(self):
        """Test basic technical indicator calculation."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        })
        
        result = calculate_technical_indicators(df)
        
        # Should add technical indicators
        assert 'sma_5' in result.columns
        assert 'sma_20' in result.columns
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert len(result) == len(df)


class TestUtilityFunctions:
    """Test utility helper functions."""
    
    def test_format_timestamp_string(self):
        """Test timestamp formatting from string."""
        timestamp_str = "2023-01-15 14:30:00"
        result = format_timestamp(timestamp_str)
        
        assert result == "2023-01-15 14:30:00"
    
    def test_format_timestamp_pandas(self):
        """Test timestamp formatting from pandas Timestamp."""
        timestamp = pd.Timestamp("2023-01-15 14:30:00")
        result = format_timestamp(timestamp)
        
        assert result == "2023-01-15 14:30:00"
    
    def test_get_file_size_mb(self, temp_dir):
        """Test getting file size in MB."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")
        
        size_mb = get_file_size_mb(test_file)
        
        # Should be a small positive number
        assert size_mb > 0
        assert size_mb < 1  # Should be less than 1 MB
    
    def test_get_file_size_mb_nonexistent(self):
        """Test getting file size for nonexistent file."""
        nonexistent_file = Path("nonexistent_file.txt")
        
        with pytest.raises(FileNotFoundError):
            get_file_size_mb(nonexistent_file)
    
    def test_get_memory_usage_mb(self):
        """Test getting memory usage in MB."""
        # Since psutil is imported inside the function, we need to patch it at the import level
        with patch('builtins.__import__') as mock_import:
            # Create mock psutil
            mock_psutil = MagicMock()
            mock_process = MagicMock()
            mock_memory_info = MagicMock()
            mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB in bytes
            mock_process.memory_info.return_value = mock_memory_info
            mock_psutil.Process.return_value = mock_process
            
            # Make __import__ return our mock psutil when 'psutil' is requested
            def side_effect(name, *args, **kwargs):
                if name == 'psutil':
                    return mock_psutil
                # For other imports, use the real __import__
                import builtins
                return builtins.__import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            memory_mb = get_memory_usage_mb()
            
            assert memory_mb == 100.0
    
    @patch('evo.utils.helpers.get_memory_usage_mb')
    @patch('evo.utils.helpers.get_logger')
    def test_log_memory_usage(self, mock_get_logger, mock_get_memory):
        """Test memory usage logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_memory.return_value = 150.5
        
        log_memory_usage()
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "150.5 MB" in call_args
    
    @patch('evo.utils.helpers.get_memory_usage_mb')
    def test_log_memory_usage_custom_logger(self, mock_get_memory):
        """Test memory usage logging with custom logger."""
        mock_logger = MagicMock()
        mock_get_memory.return_value = 200.0
        
        log_memory_usage(logger=mock_logger)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "200.0 MB" in call_args 