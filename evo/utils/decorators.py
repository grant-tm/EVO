"""
Decorators for the EVO trading system.

This module provides decorators for retry logic, timeouts, caching, and other
common patterns used throughout the trading system.
"""

import time
import functools
import asyncio
from typing import Any, Callable, Optional, Type, Union, Tuple
import logging

from ..core.logging import get_logger


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    logger: Optional[logging.Logger] = None
):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Multiplier for delay after each attempt
        exceptions: Exception types to catch and retry
        logger: Logger instance for retry messages
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                logger_instance = get_logger(func.__module__)
            else:
                logger_instance = logger
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger_instance.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger_instance.error(
                            f"All {max_attempts} attempts failed. Last error: {str(e)}"
                        )
            
            raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if logger is None:
                logger_instance = get_logger(func.__module__)
            else:
                logger_instance = logger
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger_instance.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger_instance.error(
                            f"All {max_attempts} attempts failed. Last error: {str(e)}"
                        )
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def timeout(seconds: float):
    """
    Decorator to add timeout to function execution.
    
    Args:
        seconds: Timeout in seconds
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore signal handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def cache_result(max_size: int = 128, ttl: Optional[float] = None):
    """
    Decorator to cache function results.
    
    Args:
        max_size: Maximum number of cached results
        ttl: Time to live for cached results in seconds (None for no expiration)
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            # Check if result is cached and not expired
            if key in cache:
                if ttl is None or time.time() - cache_times[key] < ttl:
                    return cache[key]
                else:
                    # Remove expired entry
                    del cache[key]
                    del cache_times[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Add to cache
            cache[key] = result
            cache_times[key] = time.time()
            
            # Remove oldest entries if cache is full
            if len(cache) > max_size:
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            return result
        
        return wrapper
    
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance for timing messages
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                logger_instance = get_logger(func.__module__)
            else:
                logger_instance = logger
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger_instance.debug(
                    f"{func.__name__} completed in {execution_time:.4f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger_instance.error(
                    f"{func.__name__} failed after {execution_time:.4f}s: {str(e)}"
                )
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if logger is None:
                logger_instance = get_logger(func.__module__)
            else:
                logger_instance = logger
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger_instance.debug(
                    f"{func.__name__} completed in {execution_time:.4f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger_instance.error(
                    f"{func.__name__} failed after {execution_time:.4f}s: {str(e)}"
                )
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def validate_inputs(*validators: Callable):
    """
    Decorator to validate function inputs.
    
    Args:
        *validators: Validation functions to apply to arguments
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply validators to arguments
            for i, validator in enumerate(validators):
                if i < len(args):
                    validator(args[i])
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator 