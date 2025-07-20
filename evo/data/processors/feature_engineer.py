"""
Feature engineering for market data.

This module provides technical indicators and feature engineering capabilities
for transforming raw market data into features suitable for machine learning.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from evo.core.logging import get_logger

logger = get_logger(__name__)


class TechnicalIndicator(ABC):
    """Abstract base class for technical indicators."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the technical indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series containing the calculated indicator values
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the indicator."""
        pass


class SMAIndicator(TechnicalIndicator):
    """Simple Moving Average indicator."""
    
    def __init__(self, window: int):
        self.window = window
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data["close"].rolling(window=self.window).mean()
    
    @property
    def name(self) -> str:
        return f"sma_{self.window}"


class EMAIndicator(TechnicalIndicator):
    """Exponential Moving Average indicator."""
    
    def __init__(self, span: int):
        self.span = span
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data["close"].ewm(span=self.span, adjust=False).mean()
    
    @property
    def name(self) -> str:
        return f"ema_{self.span}"


class RSIIndicator(TechnicalIndicator):
    """Relative Strength Index indicator."""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / (loss + 1e-9)  # Add small value to avoid division by zero
        return 100 - (100 / (1 + rs))
    
    @property
    def name(self) -> str:
        return f"rsi_{self.period}"


class MACDIndicator(TechnicalIndicator):
    """MACD (Moving Average Convergence Divergence) indicator."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate MACD."""
        ema_fast = data["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        return macd_line - signal_line
    
    @property
    def name(self) -> str:
        return f"macd_{self.fast}_{self.slow}_{self.signal}"


class VolatilityIndicator(TechnicalIndicator):
    """Volatility indicator based on price changes."""
    
    def __init__(self, window: int = 10):
        self.window = window
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility."""
        return data["close"].pct_change().rolling(window=self.window).std()
    
    @property
    def name(self) -> str:
        return f"volatility_{self.window}"


class BollingerBandsIndicator(TechnicalIndicator):
    """Bollinger Bands indicator."""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Bands (returns the bandwidth)."""
        sma = data["close"].rolling(window=self.window).mean()
        std = data["close"].rolling(window=self.window).std()
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        bandwidth = (upper_band - lower_band) / sma
        return bandwidth
    
    @property
    def name(self) -> str:
        return f"bb_{self.window}_{self.num_std}"


class FeatureEngineer:
    """
    Feature engineering for market data.
    
    Provides methods to calculate technical indicators and create
    features suitable for machine learning models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary containing indicator settings
        """
        self.config = config or {}
        self._indicators: Dict[str, TechnicalIndicator] = {}
        self._setup_default_indicators()
    
    def _setup_default_indicators(self) -> None:
        """Setup default technical indicators."""
        default_indicators = [
            SMAIndicator(5),
            SMAIndicator(20),
            EMAIndicator(12),
            EMAIndicator(26),
            RSIIndicator(14),
            MACDIndicator(),
            VolatilityIndicator(10),
            BollingerBandsIndicator()
        ]
        
        for indicator in default_indicators:
            self.add_indicator(indicator)
    
    def add_indicator(self, indicator: TechnicalIndicator) -> None:
        """
        Add a technical indicator.
        
        Args:
            indicator: Technical indicator instance
        """
        self._indicators[indicator.name] = indicator
        logger.info(f"Added indicator: {indicator.name}")
    
    def remove_indicator(self, name: str) -> None:
        """
        Remove a technical indicator.
        
        Args:
            name: Name of the indicator to remove
        """
        if name in self._indicators:
            del self._indicators[name]
            logger.info(f"Removed indicator: {name}")
    
    def get_available_indicators(self) -> List[str]:
        """
        Get list of available indicators.
        
        Returns:
            List of indicator names
        """
        return list(self._indicators.keys())
    
    def calculate_features(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate features for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            indicators: List of indicator names to calculate (None for all)
            
        Returns:
            DataFrame with original data plus calculated features
        """
        if data.empty:
            logger.warning("Empty data provided for feature calculation")
            return data
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Determine which indicators to calculate
        if indicators is None:
            indicators = list(self._indicators.keys())
        
        logger.info(f"Calculating {len(indicators)} indicators")
        
        # Calculate each indicator
        for indicator_name in indicators:
            if indicator_name not in self._indicators:
                logger.warning(f"Indicator not found: {indicator_name}")
                continue
            
            try:
                indicator = self._indicators[indicator_name]
                result[indicator_name] = indicator.calculate(data)
                logger.debug(f"Calculated {indicator_name}")
            except Exception as e:
                logger.error(f"Failed to calculate {indicator_name}: {str(e)}")
        
        # Add basic features
        result = self._add_basic_features(result)
        
        logger.info(f"Feature engineering completed. Shape: {result.shape}")
        return result
    
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic features to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with basic features added
        """
        # Price changes
        data["return"] = data["close"].pct_change()
        data["log_return"] = np.log(data["close"] / data["close"].shift(1))
        
        # Price ranges
        data["high_low_ratio"] = data["high"] / data["low"]
        data["close_open_ratio"] = data["close"] / data["open"]
        
        # Volume features
        data["volume_sma"] = data["volume"].rolling(window=10).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma"]
        
        # Time-based features
        if "timestamp" in data.columns:
            data["hour"] = pd.to_datetime(data["timestamp"]).dt.hour
            data["day_of_week"] = pd.to_datetime(data["timestamp"]).dt.dayofweek
        
        return data
    
    def get_feature_names(self, include_basic: bool = True) -> List[str]:
        """
        Get list of feature names.
        
        Args:
            include_basic: Whether to include basic features
            
        Returns:
            List of feature names
        """
        features = list(self._indicators.keys())
        
        if include_basic:
            basic_features = [
                "return", "log_return", "high_low_ratio", "close_open_ratio",
                "volume_sma", "volume_ratio", "hour", "day_of_week"
            ]
            features.extend(basic_features)
        
        return features
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns for feature calculation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if data.empty:
            logger.error("Data is empty")
            return False
        
        return True 