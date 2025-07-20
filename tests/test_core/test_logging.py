"""
Tests for EVO logging system.
"""

import pytest
import logging
import logging.handlers
import tempfile
import json
from pathlib import Path
from evo.core.logging import (
    setup_logging, get_logger, set_correlation_id, 
    generate_correlation_id, log_with_correlation,
    CorrelationFilter, StructuredFormatter
)

pytestmark = [
    pytest.mark.core,
    pytest.mark.logging
]

class TestLoggingSetup:
    """Test logging setup and configuration."""
    
    @pytest.mark.unit
    def test_setup_logging_console_only(self):
        """Test setting up logging with console output only."""
        logger = setup_logging(level="DEBUG", enable_file=False, enable_console=True)
        
        assert logger is not None
        assert logger.level == logging.DEBUG
        
        # Check that console handler is present
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) > 0
    
    @pytest.mark.unit
    def test_setup_logging_file_only(self, temp_dir):
        """Test setting up logging with file output only."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="INFO", 
            log_file=log_file, 
            enable_file=True, 
            enable_console=False
        )
        
        assert logger is not None
        assert logger.level == logging.INFO
        
        # Check that file handler is present
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) > 0
    
    @pytest.mark.unit
    def test_setup_logging_both(self, temp_dir):
        """Test setting up logging with both console and file output."""
        log_file = temp_dir / "test.log"
        logger = setup_logging(
            level="WARNING",
            log_file=log_file,
            enable_file=True,
            enable_console=True
        )
        
        assert logger is not None
        assert logger.level == logging.WARNING
        
        # Check that both handlers are present
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        
        assert len(console_handlers) > 0
        assert len(file_handlers) > 0
    
    @pytest.mark.unit
    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert logger.name == "test_module"
        assert isinstance(logger, logging.Logger)


class TestCorrelationFilter:
    """Test correlation ID filtering."""
    
    @pytest.mark.unit
    def test_correlation_filter_creation(self):
        """Test creating a correlation filter."""
        filter_obj = CorrelationFilter()
        
        assert filter_obj.correlation_id is None
    
    @pytest.mark.unit
    def test_correlation_filter_set_id(self):
        """Test setting correlation ID."""
        filter_obj = CorrelationFilter()
        correlation_id = "test-123"
        
        filter_obj.set_correlation_id(correlation_id)
        assert filter_obj.correlation_id == correlation_id
    
    @pytest.mark.unit
    def test_correlation_filter_filtering(self):
        """Test that correlation filter adds ID to records."""
        filter_obj = CorrelationFilter()
        correlation_id = "test-456"
        filter_obj.set_correlation_id(correlation_id)
        
        # Create a mock log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Apply filter
        result = filter_obj.filter(record)
        
        assert result is True
        assert hasattr(record, 'correlation_id')
        assert record.correlation_id == correlation_id
    
    @pytest.mark.unit
    def test_correlation_filter_no_id(self):
        """Test correlation filter when no ID is set."""
        filter_obj = CorrelationFilter()
        
        # Create a mock log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Apply filter
        result = filter_obj.filter(record)
        
        assert result is True
        assert not hasattr(record, 'correlation_id')


class TestStructuredFormatter:
    """Test structured JSON formatter."""
    
    @pytest.mark.unit
    def test_structured_formatter_basic(self):
        """Test basic structured formatting."""
        formatter = StructuredFormatter()
        
        # Create a mock log record
        record = logging.LogRecord(
            name="test_module",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_structured_formatter_basic"
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse JSON
        log_entry = json.loads(formatted)
        
        assert log_entry["level"] == "INFO"
        assert log_entry["logger"] == "test_module"
        assert log_entry["message"] == "Test message"
        assert log_entry["module"] == "test"
        assert log_entry["function"] == "test_structured_formatter_basic"
        assert log_entry["line"] == 42
        assert "timestamp" in log_entry
    
    @pytest.mark.unit
    def test_structured_formatter_with_correlation_id(self):
        """Test structured formatting with correlation ID."""
        formatter = StructuredFormatter()
        
        # Create a mock log record with correlation ID
        record = logging.LogRecord(
            name="test_module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_structured_formatter_with_correlation_id"
        record.correlation_id = "test-789"        
        # Format the record
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry["correlation_id"] == "test-789"
    
    @pytest.mark.unit
    def test_structured_formatter_with_extra_fields(self):
        """Test structured formatting with extra fields."""
        formatter = StructuredFormatter()
        
        # Create a mock log record with extra fields
        record = logging.LogRecord(
            name="test_module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_structured_formatter_with_extra_fields"
        record.extra_fields = {"user_id": "123", "action": "login"}
        
        # Format the record
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry["user_id"] == "123"
        assert log_entry["action"] == "login"
    
    @pytest.mark.unit
    def test_structured_formatter_with_exception(self):
        """Test structured formatting with exception info."""
        formatter = StructuredFormatter()
        
        # Create a mock log record with exception
        record = logging.LogRecord(
            name="test_module",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test error",
            args=(),
            exc_info=(ValueError, ValueError("Test error"), None)
        )
        record.funcName = "test_structured_formatter_with_exception"
        
        # Format the record
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert "exception" in log_entry
        assert "ValueError" in log_entry["exception"]


class TestCorrelationIDFunctions:
    """Test correlation ID utility functions."""
    
    @pytest.mark.unit
    def test_generate_correlation_id(self):
        """Test generating correlation IDs."""
        id1 = generate_correlation_id()
        id2 = generate_correlation_id()
        
        assert id1 != id2  # Should be unique
        assert len(id1) == 36  # UUID4 length (32 hex chars + 4 hyphens)
        assert isinstance(id1, str)
    
    @pytest.mark.unit
    def test_set_correlation_id(self):
        """Test setting correlation ID globally."""
        # Set up logging first
        setup_logging(level="DEBUG", enable_file=False, enable_console=False)
        
        correlation_id = "test-123"
        set_correlation_id(correlation_id)
        
        # Get the root logger and check if filter is set
        root_logger = logging.getLogger()
        correlation_filters = [f for f in root_logger.filters if isinstance(f, CorrelationFilter)]
        
        assert len(correlation_filters) > 0
        assert correlation_filters[0].correlation_id == correlation_id


class TestLoggingDecorator:
    """Test the log_with_correlation decorator."""
    
    @pytest.mark.unit
    def test_log_with_correlation_success(self, caplog):
        """Test successful function execution with correlation logging."""
        # Set up logging and ensure it works with pytest's caplog
        caplog.set_level("DEBUG")
        
        # Ensure the root logger has the right level
        logging.getLogger().setLevel("DEBUG")
        
        @log_with_correlation
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert "Starting test_function" in caplog.text
        assert "Completed test_function" in caplog.text
    
    @pytest.mark.unit
    def test_log_with_correlation_failure(self, caplog):
        """Test function failure with correlation logging."""
        # Set up logging and ensure it works with pytest's caplog
        caplog.set_level("DEBUG")
        
        # Ensure the root logger has the right level
        logging.getLogger().setLevel("DEBUG")
        
        @log_with_correlation
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        assert "Starting test_function" in caplog.text
        assert "Error in test_function" in caplog.text
        assert "Test error" in caplog.text


class TestLoggingIntegration:
    """Integration tests for logging system."""
    
    @pytest.mark.integration
    def test_logging_with_correlation_id(self, temp_dir, caplog):
        """Test complete logging flow with correlation IDs."""
        log_file = temp_dir / "integration.log"
        
        # Set up logging and preserve pytest's caplog
        caplog.set_level("DEBUG")
        logging.getLogger().setLevel("DEBUG")
        
        # Set up file logging without clearing handlers
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler manually to preserve caplog
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
        
        # Add correlation filter
        correlation_filter = CorrelationFilter()
        logging.getLogger().addFilter(correlation_filter)
        
        logger = get_logger("integration_test")
        
        # Set correlation ID
        correlation_id = "integration-123"
        set_correlation_id(correlation_id)
        
        # Log messages
        logger.info("Test message 1")
        logger.warning("Test message 2")
        logger.error("Test message 3")
        
        # Check console output
        assert "Test message 1" in caplog.text
        assert "Test message 2" in caplog.text
        assert "Test message 3" in caplog.text
        
        # Check file output
        assert log_file.exists()
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert "Test message 1" in log_content
            assert "Test message 2" in log_content
            assert "Test message 3" in log_content
    
    @pytest.mark.integration
    def test_log_rotation(self, temp_dir):
        """Test log file rotation."""
        log_file = temp_dir / "rotation.log"
        setup_logging(
            level="DEBUG",
            log_file=log_file,
            enable_file=True,
            enable_console=False,
            max_file_size=100,  # Small size to trigger rotation
            backup_count=2
        )
        
        logger = get_logger("rotation_test")
        
        # Write enough logs to trigger rotation
        for i in range(50):
            logger.info(f"Log message {i} with some additional content to make it longer")
        
        # Check that rotation files exist
        assert log_file.exists()
        rotated_files = list(temp_dir.glob("rotation.log.*"))
        assert len(rotated_files) > 0 