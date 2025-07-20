"""
Custom exceptions for the EVO trading system.

This module defines a hierarchy of exceptions that provide specific error handling
for different components of the trading system.
"""

from typing import Optional, Any

class EVOException(Exception):
    """Base exception for errors."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(EVOException):
    """Raised when there are issues with configuration settings."""
    pass


class DataError(EVOException):
    """Raised when there are issues with data processing or retrieval."""
    pass


class DataProviderError(DataError):
    """Raised when there are issues with data providers."""
    pass


class TrainingError(EVOException):
    """Raised when there are issues during model training."""
    pass


class ValidationError(EVOException):
    """Raised when data or parameters fail validation."""
    pass


class BrokerError(EVOException):
    """Raised when there are issues with broker operations."""
    pass


class RiskError(EVOException):
    """Raised when risk management rules are violated."""
    pass


class OptimizationError(EVOException):
    """Raised when there are issues during genetic optimization."""
    pass


class BacktestError(EVOException):
    """Raised when there are issues during backtesting."""
    pass 