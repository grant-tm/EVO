"""
Validation utilities for the EVO trading system.

This module provides validation functions for data, model paths, and configuration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Any, Dict
import logging

from ..core.exceptions import ValidationError, ConfigurationError
from ..core.logging import get_logger


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    check_nulls: bool = True,
    check_infinite: bool = True
) -> bool:
    """
    Validate a pandas DataFrame for trading data.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        check_nulls: Whether to check for null values
        check_infinite: Whether to check for infinite values
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If validation fails
    """
    logger = get_logger(__name__)
    
    if df is None:
        raise ValidationError("DataFrame cannot be None")
    
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected pandas DataFrame, got {type(df)}")
    
    if len(df) < min_rows:
        raise ValidationError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
    
    if check_nulls:
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")
    
    if check_infinite:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if np.isinf(df[col]).any():
                raise ValidationError(f"Found infinite values in column: {col}")
    
    logger.debug(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def validate_model_path(model_path: Path, must_exist: bool = False) -> bool:
    """
    Validate a model file path.
    
    Args:
        model_path: Path to model file
        must_exist: Whether the file must exist
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If validation fails
    """
    logger = get_logger(__name__)
    
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    
    if not model_path.parent.exists():
        raise ValidationError(f"Model directory does not exist: {model_path.parent}")
    
    if must_exist and not model_path.exists():
        raise ValidationError(f"Model file does not exist: {model_path}")
    
    # Check file extension
    valid_extensions = {'.zip', '.pkl', '.pth', '.h5', '.onnx'}
    if model_path.suffix.lower() not in valid_extensions:
        logger.warning(f"Model file has unusual extension: {model_path.suffix}")
    
    logger.debug(f"Model path validation passed: {model_path}")
    return True


def validate_config(config: Any) -> bool:
    """
    Validate configuration object.
    
    Args:
        config: Configuration object to validate
    
    Returns:
        True if validation passes
    
    Raises:
        ConfigurationError: If validation fails
    """
    logger = get_logger(__name__)
    
    if config is None:
        raise ConfigurationError("Configuration cannot be None")
    
    # Check for required attributes
    required_attrs = ['trading', 'data', 'training', 'reward', 'optimization', 'alpaca']
    missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
    
    if missing_attrs:
        raise ConfigurationError(f"Configuration missing required attributes: {missing_attrs}")
    
    # Validate trading configuration
    if config.trading.trade_qty <= 0:
        raise ConfigurationError("Trade quantity must be positive")
    
    if config.trading.sim_speed <= 0:
        raise ConfigurationError("Simulation speed must be positive")
    
    # Validate data configuration
    if config.data.seq_len <= 0:
        raise ConfigurationError("Sequence length must be positive")
    
    if not config.data.features:
        raise ConfigurationError("Features list cannot be empty")
    
    # Validate training configuration
    if config.training.training_steps <= 0:
        raise ConfigurationError("Training steps must be positive")
    
    if config.training.learning_rate <= 0:
        raise ConfigurationError("Learning rate must be positive")
    
    # Validate reward configuration
    if config.reward.tp_pct <= 0 or config.reward.sl_pct <= 0:
        raise ConfigurationError("Take profit and stop loss percentages must be positive")
    
    if config.reward.max_episode_steps <= 0:
        raise ConfigurationError("Max episode steps must be positive")
    
    # Validate optimization configuration
    if not 0 <= config.optimization.mutation_rate <= 1:
        raise ConfigurationError("Mutation rate must be between 0 and 1")
    
    if not 0 <= config.optimization.elite_proportion <= 1:
        raise ConfigurationError("Elite proportion must be between 0 and 1")
    
    logger.debug("Configuration validation passed")
    return True


def validate_hyperparameters(hyperparams: Dict[str, Any]) -> bool:
    """
    Validate hyperparameters for training.
    
    Args:
        hyperparams: Dictionary of hyperparameters
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If validation fails
    """
    logger = get_logger(__name__)
    
    if not isinstance(hyperparams, dict):
        raise ValidationError("Hyperparameters must be a dictionary")
    
    # Validate learning rate
    if 'learning_rate' in hyperparams:
        lr = hyperparams['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValidationError("Learning rate must be a positive number")
    
    # Validate batch size
    if 'batch_size' in hyperparams:
        bs = hyperparams['batch_size']
        if not isinstance(bs, int) or bs <= 0:
            raise ValidationError("Batch size must be a positive integer")
    
    # Validate clip range
    if 'clip_range' in hyperparams:
        cr = hyperparams['clip_range']
        if not isinstance(cr, (int, float)) or cr <= 0 or cr > 1:
            raise ValidationError("Clip range must be between 0 and 1")
    
    # Validate entropy coefficients
    for key in ['entropy_coef_init', 'entropy_coef_final']:
        if key in hyperparams:
            ec = hyperparams[key]
            if not isinstance(ec, (int, float)) or ec < 0:
                raise ValidationError(f"{key} must be a non-negative number")
    
    # Validate GAE lambda and gamma
    for key in ['gae_lambda', 'gamma']:
        if key in hyperparams:
            val = hyperparams[key]
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                raise ValidationError(f"{key} must be between 0 and 1")
    
    logger.debug("Hyperparameters validation passed")
    return True 