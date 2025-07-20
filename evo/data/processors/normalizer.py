"""
Data normalization for machine learning.

This module provides data normalization and scaling capabilities
for preparing market data for machine learning models.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import os

from evo.core.logging import get_logger

logger = get_logger(__name__)


class DataNormalizer:
    """
    Data normalizer for market data.
    
    Provides various normalization and scaling methods for preparing
    data for machine learning models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data normalizer.
        
        Args:
            config: Configuration dictionary containing:
                - method: Normalization method ('standard', 'minmax', 'robust')
                - features: List of features to normalize (None for all numeric)
                - exclude_features: List of features to exclude from normalization
                - fit_on_training: Whether to fit scaler on training data only
        """
        self.config = config or {}
        self.method = self.config.get("method", "standard")
        self.features = self.config.get("features")
        self.exclude_features = self.config.get("exclude_features", [])
        self.fit_on_training = self.config.get("fit_on_training", True)
        
        # Initialize scaler
        self._scaler = self._create_scaler()
        self._is_fitted = False
        self._feature_names = []
        
        logger.info(f"Initialized {self.method} normalizer")
    
    def _create_scaler(self) -> BaseEstimator:
        """Create the appropriate scaler based on method."""
        if self.method == "standard":
            return StandardScaler()
        elif self.method == "minmax":
            return MinMaxScaler()
        elif self.method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")
    
    def fit(self, data: pd.DataFrame) -> 'DataNormalizer':
        """
        Fit the normalizer on the data.
        
        Args:
            data: DataFrame to fit the normalizer on
            
        Returns:
            Self for method chaining
        """
        if data.empty:
            raise ValueError("Cannot fit normalizer on empty data")
        
        # Determine features to normalize
        features_to_normalize = self._get_features_to_normalize(data)
        
        if not features_to_normalize:
            logger.warning("No features to normalize")
            return self
        
        # Fit the scaler
        self._scaler.fit(data[features_to_normalize])
        self._feature_names = features_to_normalize
        self._is_fitted = True
        
        logger.info(f"Fitted normalizer on {len(features_to_normalize)} features")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted normalizer.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Normalizer must be fitted before transforming")
        
        if data.empty:
            logger.warning("Empty data provided for transformation")
            return data
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Transform features
        if self._feature_names:
            try:
                transformed_features = self._scaler.transform(data[self._feature_names])
                result[self._feature_names] = transformed_features
                logger.debug(f"Transformed {len(self._feature_names)} features")
            except Exception as e:
                logger.error(f"Failed to transform features: {str(e)}")
                raise
        
        return result
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the normalizer and transform the data.
        
        Args:
            data: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data.
        
        Args:
            data: DataFrame to inverse transform
            
        Returns:
            Inverse transformed DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming")
        
        if data.empty:
            logger.warning("Empty data provided for inverse transformation")
            return data
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Inverse transform features
        if self._feature_names:
            try:
                inverse_transformed_features = self._scaler.inverse_transform(data[self._feature_names])
                result[self._feature_names] = inverse_transformed_features
                logger.debug(f"Inverse transformed {len(self._feature_names)} features")
            except Exception as e:
                logger.error(f"Failed to inverse transform features: {str(e)}")
                raise
        
        return result
    
    def _get_features_to_normalize(self, data: pd.DataFrame) -> List[str]:
        """
        Get list of features to normalize.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of feature names to normalize
        """
        # Get numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter by specified features
        if self.features is not None:
            numeric_columns = [col for col in numeric_columns if col in self.features]
        
        # Exclude specified features
        numeric_columns = [col for col in numeric_columns if col not in self.exclude_features]
        
        # Remove timestamp and other non-feature columns
        exclude_patterns = ['timestamp', 'date', 'time', 'symbol']
        for pattern in exclude_patterns:
            numeric_columns = [col for col in numeric_columns if pattern not in col.lower()]
        
        return numeric_columns
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of normalized feature names.
        
        Returns:
            List of feature names
        """
        return self._feature_names.copy()
    
    def get_scaler_params(self) -> Dict[str, Any]:
        """
        Get scaler parameters.
        
        Returns:
            Dictionary containing scaler parameters
        """
        if not self._is_fitted:
            return {}
        
        params = {
            "method": self.method,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted
        }
        
        # Add scaler-specific parameters
        if hasattr(self._scaler, 'mean_'):
            params["mean"] = self._scaler.mean_.tolist()
        if hasattr(self._scaler, 'scale_'):
            params["scale"] = self._scaler.scale_.tolist()
        if hasattr(self._scaler, 'min_'):
            params["min"] = self._scaler.min_.tolist()
        if hasattr(self._scaler, 'data_min_'):
            params["data_min"] = self._scaler.data_min_.tolist()
        if hasattr(self._scaler, 'data_max_'):
            params["data_max"] = self._scaler.data_max_.tolist()
        
        return params
    
    def save(self, filepath: str) -> None:
        """
        Save the normalizer to a file.
        
        Args:
            filepath: Path to save the normalizer
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Saved normalizer to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save normalizer: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'DataNormalizer':
        """
        Load a normalizer from a file.
        
        Args:
            filepath: Path to load the normalizer from
            
        Returns:
            Loaded normalizer instance
        """
        try:
            with open(filepath, 'rb') as f:
                normalizer = pickle.load(f)
            logger.info(f"Loaded normalizer from {filepath}")
            return normalizer
        except Exception as e:
            logger.error(f"Failed to load normalizer: {str(e)}")
            raise
    
    def is_fitted(self) -> bool:
        """
        Check if the normalizer is fitted.
        
        Returns:
            True if fitted, False otherwise
        """
        return self._is_fitted
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required features for normalization.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            logger.error("Data is empty")
            return False
        
        if self._is_fitted:
            # Check if all fitted features are present
            missing_features = [f for f in self._feature_names if f not in data.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                return False
        
        return True


class RollingNormalizer(DataNormalizer):
    """
    Rolling window normalizer for time series data.
    
    Provides normalization using a rolling window approach,
    useful for online learning and real-time applications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize rolling normalizer.
        
        Args:
            config: Configuration dictionary containing:
                - window_size: Size of the rolling window
                - method: Normalization method
                - features: List of features to normalize
        """
        super().__init__(config)
        self.window_size = self.config.get("window_size", 100)
        self._window_data = {}
        
        logger.info(f"Initialized rolling normalizer with window size {self.window_size}")
    
    def update_window(self, data: pd.DataFrame) -> None:
        """
        Update the rolling window with new data.
        
        Args:
            data: New data to add to the window
        """
        features_to_normalize = self._get_features_to_normalize(data)
        
        for feature in features_to_normalize:
            if feature not in self._window_data:
                self._window_data[feature] = []
            
            # Add new values
            self._window_data[feature].extend(data[feature].dropna().tolist())
            
            # Keep only the last window_size values
            if len(self._window_data[feature]) > self.window_size:
                self._window_data[feature] = self._window_data[feature][-self.window_size:]
    
    def transform_rolling(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using rolling window statistics.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if data.empty:
            return data
        
        result = data.copy()
        features_to_normalize = self._get_features_to_normalize(data)
        
        for feature in features_to_normalize:
            if feature not in self._window_data or not self._window_data[feature]:
                continue
            
            window_values = np.array(self._window_data[feature])
            
            if self.method == "standard":
                mean = np.mean(window_values)
                std = np.std(window_values)
                if std > 0:
                    result[feature] = (data[feature] - mean) / std
            elif self.method == "minmax":
                min_val = np.min(window_values)
                max_val = np.max(window_values)
                if max_val > min_val:
                    result[feature] = (data[feature] - min_val) / (max_val - min_val)
            elif self.method == "robust":
                median = np.median(window_values)
                mad = np.median(np.abs(window_values - median))
                if mad > 0:
                    result[feature] = (data[feature] - median) / mad
        
        return result 