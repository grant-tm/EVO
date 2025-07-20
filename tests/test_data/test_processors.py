"""
Tests for data processors.

This module contains tests for the feature engineering and normalization components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from evo.data.processors.feature_engineer import (
    FeatureEngineer, SMAIndicator, EMAIndicator, RSIIndicator,
    MACDIndicator, VolatilityIndicator, BollingerBandsIndicator
)
from evo.data.processors.normalizer import DataNormalizer, RollingNormalizer

pytestmark = [
    pytest.mark.unit,
    pytest.mark.data,
    pytest.mark.processors
]

class TestTechnicalIndicators:
    """Test technical indicators."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.01, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.002, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 100))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_sma_indicator(self, sample_data):
        """Test Simple Moving Average indicator."""
        indicator = SMAIndicator(window=5)
        result = indicator.calculate(sample_data)
        
        assert len(result) == len(sample_data)
        assert indicator.name == "sma_5"
        assert not result.isna().all()
    
    def test_ema_indicator(self, sample_data):
        """Test Exponential Moving Average indicator."""
        indicator = EMAIndicator(span=12)
        result = indicator.calculate(sample_data)
        
        assert len(result) == len(sample_data)
        assert indicator.name == "ema_12"
        assert not result.isna().all()
    
    def test_rsi_indicator(self, sample_data):
        """Test Relative Strength Index indicator."""
        indicator = RSIIndicator(period=14)
        result = indicator.calculate(sample_data)
        
        assert len(result) == len(sample_data)
        assert indicator.name == "rsi_14"
        assert not result.isna().all()
        # RSI should be between 0 and 100
        assert result.dropna().min() >= 0
        assert result.dropna().max() <= 100
    
    def test_macd_indicator(self, sample_data):
        """Test MACD indicator."""
        indicator = MACDIndicator()
        result = indicator.calculate(sample_data)
        
        assert len(result) == len(sample_data)
        assert indicator.name == "macd_12_26_9"
        assert not result.isna().all()
    
    def test_volatility_indicator(self, sample_data):
        """Test Volatility indicator."""
        indicator = VolatilityIndicator(window=10)
        result = indicator.calculate(sample_data)
        
        assert len(result) == len(sample_data)
        assert indicator.name == "volatility_10"
        assert not result.isna().all()
        # Volatility should be non-negative
        assert result.dropna().min() >= 0
    
    def test_bollinger_bands_indicator(self, sample_data):
        """Test Bollinger Bands indicator."""
        indicator = BollingerBandsIndicator()
        result = indicator.calculate(sample_data)
        
        assert len(result) == len(sample_data)
        assert indicator.name == "bb_20_2.0"
        assert not result.isna().all()


class TestFeatureEngineer:
    """Test the feature engineer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1min')
        np.random.seed(42)
        
        returns = np.random.normal(0, 0.01, 50)
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.002, 50)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 50))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 50))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 50)
        })
    
    @pytest.fixture
    def feature_engineer(self):
        """Create a feature engineer instance."""
        return FeatureEngineer()
    
    def test_init_default_indicators(self, feature_engineer):
        """Test initialization with default indicators."""
        indicators = feature_engineer.get_available_indicators()
        assert len(indicators) > 0
        assert "sma_5" in indicators
        assert "sma_20" in indicators
        assert "rsi_14" in indicators
        assert "macd_12_26_9" in indicators
    
    def test_add_indicator(self, feature_engineer):
        """Test adding a custom indicator."""
        custom_indicator = SMAIndicator(50)
        feature_engineer.add_indicator(custom_indicator)
        
        indicators = feature_engineer.get_available_indicators()
        assert "sma_50" in indicators
    
    def test_remove_indicator(self, feature_engineer):
        """Test removing an indicator."""
        feature_engineer.remove_indicator("sma_5")
        
        indicators = feature_engineer.get_available_indicators()
        assert "sma_5" not in indicators
    
    def test_calculate_features(self, feature_engineer, sample_data):
        """Test feature calculation."""
        result = feature_engineer.calculate_features(sample_data)
        
        assert len(result) == len(sample_data)
        assert len(result.columns) > len(sample_data.columns)
        
        # Check that basic features were added
        assert "return" in result.columns
        assert "log_return" in result.columns
        assert "high_low_ratio" in result.columns
        assert "close_open_ratio" in result.columns
    
    def test_calculate_features_specific_indicators(self, feature_engineer, sample_data):
        """Test feature calculation with specific indicators."""
        indicators = ["sma_5", "rsi_14"]
        result = feature_engineer.calculate_features(sample_data, indicators=indicators)
        
        assert "sma_5" in result.columns
        assert "rsi_14" in result.columns
        assert "sma_20" not in result.columns  # Not requested
    
    def test_calculate_features_empty_data(self, feature_engineer):
        """Test feature calculation with empty data."""
        empty_data = pd.DataFrame()
        result = feature_engineer.calculate_features(empty_data)
        
        assert result.empty
    
    def test_get_feature_names(self, feature_engineer):
        """Test getting feature names."""
        names = feature_engineer.get_feature_names()
        assert len(names) > 0
        
        # Test without basic features
        names_no_basic = feature_engineer.get_feature_names(include_basic=False)
        assert len(names_no_basic) < len(names)
    
    def test_validate_data_valid(self, feature_engineer, sample_data):
        """Test data validation with valid data."""
        assert feature_engineer.validate_data(sample_data) is True
    
    def test_validate_data_missing_columns(self, feature_engineer):
        """Test data validation with missing columns."""
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0]
            # Missing high, low, close, volume
        })
        
        assert feature_engineer.validate_data(invalid_data) is False
    
    def test_validate_data_empty(self, feature_engineer):
        """Test data validation with empty data."""
        empty_data = pd.DataFrame()
        assert feature_engineer.validate_data(empty_data) is False


class TestDataNormalizer:
    """Test the data normalizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1min'),
            'feature1': np.random.normal(100, 10, 100),
            'feature2': np.random.normal(50, 5, 100),
            'feature3': np.random.normal(0, 1, 100),
            'categorical': ['A', 'B'] * 50
        })
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        normalizer = DataNormalizer()
        assert normalizer.method == "standard"
        assert normalizer.features is None
        assert normalizer.exclude_features == []
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            "method": "minmax",
            "features": ["feature1", "feature2"],
            "exclude_features": ["feature3"]
        }
        normalizer = DataNormalizer(config)
        assert normalizer.method == "minmax"
        assert normalizer.features == ["feature1", "feature2"]
        assert normalizer.exclude_features == ["feature3"]
    
    def test_init_unsupported_method(self):
        """Test initialization with unsupported method."""
        config = {"method": "unsupported"}
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            DataNormalizer(config)
    
    def test_fit_transform_standard(self, sample_data):
        """Test fit and transform with standard scaling."""
        normalizer = DataNormalizer({"method": "standard"})
        result = normalizer.fit_transform(sample_data)
        
        assert len(result) == len(sample_data)
        assert normalizer.is_fitted()
        
        # Check that numeric features were normalized
        assert "feature1" in result.columns
        assert "feature2" in result.columns
        assert "feature3" in result.columns
        
        # Check that non-numeric features were not normalized
        assert "categorical" in result.columns
    
    def test_fit_transform_minmax(self, sample_data):
        """Test fit and transform with minmax scaling."""
        normalizer = DataNormalizer({"method": "minmax"})
        result = normalizer.fit_transform(sample_data)
        
        assert len(result) == len(sample_data)
        assert normalizer.is_fitted()
    
    def test_fit_transform_robust(self, sample_data):
        """Test fit and transform with robust scaling."""
        normalizer = DataNormalizer({"method": "robust"})
        result = normalizer.fit_transform(sample_data)
        
        assert len(result) == len(sample_data)
        assert normalizer.is_fitted()
    
    def test_fit_empty_data(self):
        """Test fitting with empty data."""
        normalizer = DataNormalizer()
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot fit normalizer on empty data"):
            normalizer.fit(empty_data)
    
    def test_transform_without_fit(self, sample_data):
        """Test transform without fitting."""
        normalizer = DataNormalizer()
        
        with pytest.raises(ValueError, match="Normalizer must be fitted before transforming"):
            normalizer.transform(sample_data)
    
    def test_inverse_transform(self, sample_data):
        """Test inverse transform."""
        normalizer = DataNormalizer()
        fitted_data = normalizer.fit_transform(sample_data)
        
        # Inverse transform
        inverse_data = normalizer.inverse_transform(fitted_data)
        
        # Check that numeric features are approximately restored
        for col in ["feature1", "feature2", "feature3"]:
            np.testing.assert_array_almost_equal(
                sample_data[col].values,
                inverse_data[col].values,
                decimal=10
            )
    
    def test_get_feature_names(self, sample_data):
        """Test getting feature names."""
        normalizer = DataNormalizer()
        normalizer.fit(sample_data)
        
        feature_names = normalizer.get_feature_names()
        assert len(feature_names) > 0
        assert "feature1" in feature_names
        assert "feature2" in feature_names
        assert "feature3" in feature_names
        assert "categorical" not in feature_names  # Non-numeric
    
    def test_save_load(self, sample_data, tmp_path):
        """Test saving and loading normalizer."""
        normalizer = DataNormalizer()
        normalizer.fit(sample_data)
        
        # Save
        filepath = tmp_path / "normalizer.pkl"
        normalizer.save(filepath)
        
        # Load
        loaded_normalizer = DataNormalizer.load(filepath)
        
        assert loaded_normalizer.is_fitted()
        assert loaded_normalizer.method == normalizer.method
        assert loaded_normalizer.get_feature_names() == normalizer.get_feature_names()
    
    def test_validate_data_valid(self, sample_data):
        """Test data validation with valid data."""
        normalizer = DataNormalizer()
        assert normalizer.validate_data(sample_data) is True
    
    def test_validate_data_empty(self):
        """Test data validation with empty data."""
        normalizer = DataNormalizer()
        empty_data = pd.DataFrame()
        assert normalizer.validate_data(empty_data) is False
    
    def test_validate_data_missing_features(self, sample_data):
        """Test data validation with missing features."""
        normalizer = DataNormalizer()
        normalizer.fit(sample_data)
        
        # Remove a feature
        incomplete_data = sample_data.drop(columns=["feature1"])
        assert normalizer.validate_data(incomplete_data) is False


class TestRollingNormalizer:
    """Test the rolling normalizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1min'),
            'feature1': np.random.normal(100, 10, 100),
            'feature2': np.random.normal(50, 5, 100)
        })
    
    def test_init(self):
        """Test initialization."""
        config = {"window_size": 50, "method": "standard"}
        normalizer = RollingNormalizer(config)
        
        assert normalizer.window_size == 50
        assert normalizer.method == "standard"
    
    def test_update_window(self, sample_data):
        """Test updating the rolling window."""
        normalizer = RollingNormalizer({"window_size": 10})
        normalizer.update_window(sample_data)
        
        assert "feature1" in normalizer._window_data
        assert "feature2" in normalizer._window_data
        assert len(normalizer._window_data["feature1"]) <= 10
    
    def test_transform_rolling(self, sample_data):
        """Test rolling transformation."""
        normalizer = RollingNormalizer({"window_size": 20})
        
        # Update window with first 20 rows
        normalizer.update_window(sample_data.iloc[:20])
        
        # Transform all data
        result = normalizer.transform_rolling(sample_data)
        
        assert len(result) == len(sample_data)
        assert "feature1" in result.columns
        assert "feature2" in result.columns 