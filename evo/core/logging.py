"""
Centralized logging for the EVO trading system.

This module provides structured logging with correlation IDs for tracking
operations across the trading system.
"""

import logging
import logging.handlers
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class CorrelationFilter(logging.Filter):
    """Adds correlation ID to log records for request tracing."""
    
    def __init__(self):
        super().__init__()
        self.correlation_id = None
    
    def filter(self, record):
        if self.correlation_id:
            record.correlation_id = self.correlation_id
        return True
    
    def set_correlation_id(self, correlation_id: str):
        """Set the correlation ID for the current context."""
        self.correlation_id = correlation_id


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured JSON for better parsing."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up centralized logging for the EVO system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Whether to log to console
        enable_file: Whether to log to file
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured root logger
    """
    # Create logs directory if it doesn't exist
    if log_file is None:
        log_file = Path("logs") / "evo.log"
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = StructuredFormatter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Add correlation filter
    correlation_filter = CorrelationFilter()
    root_logger.addFilter(correlation_filter)
    
    # Log startup message
    logger = get_logger(__name__)
    logger.info("EVO logging system initialized", extra={
        "extra_fields": {
            "log_level": level,
            "log_file": str(log_file),
            "enable_console": enable_console,
            "enable_file": enable_file
        }
    })
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger
    """
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str):
    """
    Set the correlation ID for the current logging context.
    
    Args:
        correlation_id: Unique identifier for tracking operations
    """
    root_logger = logging.getLogger()
    for filter_obj in root_logger.filters:
        if isinstance(filter_obj, CorrelationFilter):
            filter_obj.set_correlation_id(correlation_id)
            break


def generate_correlation_id() -> str:
    """
    Generate a new correlation ID.
    
    Returns:
        Unique correlation ID string
    """
    return str(uuid.uuid4())


def log_with_correlation(func):
    """
    Decorator to automatically add correlation ID to function calls.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)
        
        logger = get_logger(func.__module__)
        logger.debug(f"Starting {func.__name__}", extra={
            "extra_fields": {
                "correlation_id": correlation_id,
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            }
        })
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completed {func.__name__}", extra={
                "extra_fields": {
                    "correlation_id": correlation_id,
                    "function": func.__name__,
                    "result_type": type(result).__name__
                }
            })
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", extra={
                "extra_fields": {
                    "correlation_id": correlation_id,
                    "function": func.__name__,
                    "error_type": type(e).__name__
                }
            }, exc_info=True)
            raise
    
    return wrapper 