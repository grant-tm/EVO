"""
Tests for EVO exception system.
"""

import pytest
from evo.core.exceptions import (
    EVOException, ConfigurationError, DataError, TrainingError,
    ValidationError, BrokerError, RiskError, OptimizationError, BacktestError
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.exceptions
]

class TestEVOException:
    """Test the base EVOException class."""
    
    def test_base_exception_creation(self):
        """Test creating a base EVOException."""
        exception = EVOException("Test message")
        assert str(exception) == "Test message"
        assert exception.message == "Test message"
        assert exception.details is None
    
    def test_base_exception_with_details(self):
        """Test creating a base EVOException with details."""
        details = {"key": "value", "number": 42}
        exception = EVOException("Test message", details)
        assert exception.message == "Test message"
        assert exception.details == details
        assert "Details: {'key': 'value', 'number': 42}" in str(exception)
    
    def test_base_exception_inheritance(self):
        """Test that EVOException inherits from Exception."""
        exception = EVOException("Test")
        assert isinstance(exception, Exception)


class TestConfigurationError:
    """Test the ConfigurationError class."""
    
    @pytest.mark.config
    def test_configuration_error_creation(self):
        """Test creating a ConfigurationError."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, EVOException)
        assert error.message == "Invalid config"
    
    @pytest.mark.config
    def test_configuration_error_with_details(self):
        """Test creating a ConfigurationError with details."""
        details = ["Missing API key", "Invalid symbol"]
        error = ConfigurationError("Config validation failed", details)
        assert error.details == details


class TestDataError:
    """Test the DataError class."""
    
    @pytest.mark.data
    def test_data_error_creation(self):
        """Test creating a DataError."""
        error = DataError("Data processing failed")
        assert isinstance(error, EVOException)
        assert error.message == "Data processing failed"
    
    @pytest.mark.data
    def test_data_error_with_details(self):
        """Test creating a DataError with details."""
        details = {"file": "data.csv", "line": 42}
        error = DataError("Invalid data format", details)
        assert error.details == details


class TestTrainingError:
    """Test the TrainingError class."""
    
    def test_training_error_creation(self):
        """Test creating a TrainingError."""
        error = TrainingError("Model training failed")
        assert isinstance(error, EVOException)
        assert error.message == "Model training failed"
    
    def test_training_error_with_details(self):
        """Test creating a TrainingError with details."""
        details = {"epoch": 100, "loss": 0.5}
        error = TrainingError("Training diverged", details)
        assert error.details == details


class TestValidationError:
    """Test the ValidationError class."""
    
    def test_validation_error_creation(self):
        """Test creating a ValidationError."""
        error = ValidationError("Input validation failed")
        assert isinstance(error, EVOException)
        assert error.message == "Input validation failed"


class TestBrokerError:
    """Test the BrokerError class."""
    
    def test_broker_error_creation(self):
        """Test creating a BrokerError."""
        error = BrokerError("Order execution failed")
        assert isinstance(error, EVOException)
        assert error.message == "Order execution failed"


class TestRiskError:
    """Test the RiskError class."""
    
    def test_risk_error_creation(self):
        """Test creating a RiskError."""
        error = RiskError("Risk limit exceeded")
        assert isinstance(error, EVOException)
        assert error.message == "Risk limit exceeded"


class TestOptimizationError:
    """Test the OptimizationError class."""
    
    def test_optimization_error_creation(self):
        """Test creating an OptimizationError."""
        error = OptimizationError("Genetic optimization failed")
        assert isinstance(error, EVOException)
        assert error.message == "Genetic optimization failed"


class TestBacktestError:
    """Test the BacktestError class."""
    
    def test_backtest_error_creation(self):
        """Test creating a BacktestError."""
        error = BacktestError("Backtest execution failed")
        assert isinstance(error, EVOException)
        assert error.message == "Backtest execution failed"


class TestExceptionHierarchy:
    """Test the exception hierarchy and inheritance."""
    
    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from EVOException."""
        exceptions = [
            ConfigurationError("test"),
            DataError("test"),
            TrainingError("test"),
            ValidationError("test"),
            BrokerError("test"),
            RiskError("test"),
            OptimizationError("test"),
            BacktestError("test")
        ]
        
        for exception in exceptions:
            assert isinstance(exception, EVOException)
            assert isinstance(exception, Exception)
    
    def test_exception_string_representation(self):
        """Test string representation of exceptions."""
        error = ConfigurationError("Config failed", ["error1", "error2"])
        error_str = str(error)
        assert "Config failed" in error_str
        assert "Details:" in error_str
        assert "error1" in error_str
        assert "error2" in error_str 