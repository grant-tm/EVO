"""
Tests for EVO decorators.
"""

import pytest
import asyncio
import time
import signal
from unittest.mock import patch, MagicMock, call
from evo.utils.decorators import (
    retry, timeout, cache_result, log_execution_time, validate_inputs
)


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_retry_success_first_try(self):
        """Test retry decorator when function succeeds on first try."""
        @retry(max_attempts=3)
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_retry_success_after_failures(self):
        """Test retry decorator when function succeeds after some failures."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_retry_all_attempts_fail(self):
        """Test retry decorator when all attempts fail."""
        @retry(max_attempts=3, delay=0.01)
        def test_function():
            raise ValueError("Persistent failure")
        
        with pytest.raises(ValueError) as exc_info:
            test_function()
        assert "Persistent failure" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_retry_custom_exceptions(self):
        """Test retry decorator with custom exception types."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01, exceptions=(ValueError, TypeError))
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retryable error")
            return "success"
        
        result = test_function()
        assert result == "success"
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_retry_non_retryable_exception(self):
        """Test retry decorator with non-retryable exception."""
        @retry(max_attempts=3, delay=0.01, exceptions=ValueError)
        def test_function():
            raise TypeError("Non-retryable error")
        
        with pytest.raises(TypeError) as exc_info:
            test_function()
        assert "Non-retryable error" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_retry_backoff(self):
        """Test retry decorator with exponential backoff."""
        call_count = 0
        start_times = []
        
        @retry(max_attempts=3, delay=0.1, backoff=2.0)
        def test_function():
            nonlocal call_count
            start_times.append(time.time())
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = test_function()
        assert result == "success"
        
        # Check that delays increase with backoff
        if len(start_times) >= 3:
            delay1 = start_times[1] - start_times[0]
            delay2 = start_times[2] - start_times[1]
            assert delay2 > delay1  # Second delay should be longer
    
    @pytest.mark.unit
    @pytest.mark.utils
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Test retry decorator with async function."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.unit
    @pytest.mark.utils
    @pytest.mark.asyncio
    async def test_retry_async_all_fail(self):
        """Test retry decorator with async function that always fails."""
        @retry(max_attempts=3, delay=0.01)
        async def test_function():
            raise ValueError("Persistent failure")
        
        with pytest.raises(ValueError) as exc_info:
            await test_function()
        assert "Persistent failure" in str(exc_info.value)


class TestTimeoutDecorator:
    """Test timeout decorator functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_timeout_function_completes(self):
        """Test timeout decorator when function completes in time."""
        @timeout(seconds=1.0)
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_timeout_function_times_out(self):
        """Test timeout decorator when function times out."""
        @timeout(seconds=0.1)
        def test_function():
            time.sleep(0.2)  # Sleep longer than timeout
            return "success"
        
        with pytest.raises(TimeoutError) as exc_info:
            test_function()
        assert "timed out after 0.1 seconds" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    @pytest.mark.asyncio
    async def test_timeout_async_completes(self):
        """Test timeout decorator with async function that completes."""
        @timeout(seconds=1.0)
        async def test_function():
            await asyncio.sleep(0.01)
            return "success"
        
        result = await test_function()
        assert result == "success"
    
    @pytest.mark.unit
    @pytest.mark.utils
    @pytest.mark.asyncio
    async def test_timeout_async_times_out(self):
        """Test timeout decorator with async function that times out."""
        @timeout(seconds=0.1)
        async def test_function():
            await asyncio.sleep(0.2)  # Sleep longer than timeout
            return "success"
        
        with pytest.raises(TimeoutError) as exc_info:
            await test_function()
        assert "timed out after 0.1 seconds" in str(exc_info.value)


class TestCacheResultDecorator:
    """Test cache_result decorator functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_cache_result_basic(self):
        """Test basic caching functionality."""
        call_count = 0
        
        @cache_result()
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = test_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = test_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different args should execute function again
        result3 = test_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_cache_result_with_kwargs(self):
        """Test caching with keyword arguments."""
        call_count = 0
        
        @cache_result()
        def test_function(x, y=0):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # Test with different kwarg combinations
        result1 = test_function(1, y=2)
        result2 = test_function(1, y=2)
        result3 = test_function(1)  # Different args
        
        assert result1 == 3
        assert result2 == 3
        assert result3 == 1
        assert call_count == 2  # Only 2 unique calls
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_cache_result_ttl_expiration(self):
        """Test cache expiration with TTL."""
        call_count = 0
        
        @cache_result(ttl=0.1)  # 100ms TTL
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = test_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call within TTL should use cache
        result2 = test_function(5)
        assert result2 == 10
        assert call_count == 1
        
        # Wait for TTL to expire
        time.sleep(0.15)
        
        # Third call after TTL should execute function again
        result3 = test_function(5)
        assert result3 == 10
        assert call_count == 2
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_cache_result_max_size(self):
        """Test cache size limit."""
        call_count = 0
        
        @cache_result(max_size=2)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # Fill cache to max size
        test_function(1)  # call_count = 1
        test_function(2)  # call_count = 2
        
        # These should use cache
        test_function(1)  # call_count = 2
        test_function(2)  # call_count = 2
        
        # This should evict oldest entry and execute function
        test_function(3)  # call_count = 3
        
        # This should execute function again (1 was evicted)
        test_function(1)  # call_count = 4
        
        assert call_count == 4


class TestLogExecutionTimeDecorator:
    """Test log_execution_time decorator functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    @patch('evo.utils.decorators.get_logger')
    def test_log_execution_time_sync(self, mock_get_logger):
        """Test execution time logging for sync function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        @log_execution_time()
        def test_function():
            time.sleep(0.01)  # Small delay to measure
            return "success"
        
        result = test_function()
        
        assert result == "success"
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_function" in call_args
        assert "completed in" in call_args
    
    @pytest.mark.unit
    @pytest.mark.utils
    @pytest.mark.asyncio
    @patch('evo.utils.decorators.get_logger')
    async def test_log_execution_time_async(self, mock_get_logger):
        """Test execution time logging for async function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        @log_execution_time()
        async def test_function():
            await asyncio.sleep(0.01)  # Small delay to measure
            return "success"
        
        result = await test_function()
        
        assert result == "success"
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_function" in call_args
        assert "completed in" in call_args
    
    @pytest.mark.unit
    @pytest.mark.utils
    @patch('evo.utils.decorators.get_logger')
    def test_log_execution_time_custom_logger(self, mock_get_logger):
        """Test execution time logging with custom logger."""
        custom_logger = MagicMock()
        
        @log_execution_time(logger=custom_logger)
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result == "success"
        custom_logger.info.assert_called_once()
        # Should not use the default logger
        mock_get_logger.assert_not_called()


class TestValidateInputsDecorator:
    """Test validate_inputs decorator functionality."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_inputs_success(self):
        """Test input validation when all validators pass."""
        def is_positive(x):
            return x > 0
        
        def is_even(x):
            return x % 2 == 0
        
        @validate_inputs(is_positive, is_even)
        def test_function(x):
            return x * 2
        
        result = test_function(4)  # Positive and even
        assert result == 8
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_inputs_failure(self):
        """Test input validation when validator fails."""
        def is_positive(x):
            return x > 0
        
        @validate_inputs(is_positive)
        def test_function(x):
            return x * 2
        
        with pytest.raises(ValueError) as exc_info:
            test_function(-1)  # Not positive
        assert "Input validation failed" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_inputs_multiple_args(self):
        """Test input validation with multiple arguments."""
        def is_positive(x):
            return x > 0
        
        def is_string(x):
            return isinstance(x, str)
        
        @validate_inputs(is_positive, is_string)
        def test_function(x, y):
            return f"{x} {y}"
        
        # Should pass validation
        result = test_function(5, "hello")
        assert result == "5 hello"
        
        # Should fail validation
        with pytest.raises(ValueError):
            test_function(-1, "hello")  # First validator fails
        
        with pytest.raises(ValueError):
            test_function(5, 123)  # Second validator fails
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_inputs_no_validators(self):
        """Test input validation with no validators."""
        @validate_inputs()
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        assert result == 10
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_inputs_custom_error_message(self):
        """Test input validation with custom error messages."""
        def is_positive(x):
            if x <= 0:
                raise ValueError("Value must be positive")
            return True
        
        @validate_inputs(is_positive)
        def test_function(x):
            return x * 2
        
        with pytest.raises(ValueError) as exc_info:
            test_function(-1)
        assert "Value must be positive" in str(exc_info.value)


class TestDecoratorIntegration:
    """Test integration between multiple decorators."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_retry_with_timeout(self):
        """Test combining retry and timeout decorators."""
        call_count = 0
        
        @retry(max_attempts=2, delay=0.01)
        @timeout(seconds=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(0.2)  # Timeout on first attempt
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_cache_with_logging(self):
        """Test combining cache and logging decorators."""
        @cache_result()
        @log_execution_time()
        def test_function(x):
            return x * 2
        
        # First call should log execution time
        result1 = test_function(5)
        assert result1 == 10
        
        # Second call should use cache (no additional logging)
        result2 = test_function(5)
        assert result2 == 10
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_with_retry(self):
        """Test combining validation and retry decorators."""
        def is_positive(x):
            return x > 0
        
        call_count = 0
        
        @retry(max_attempts=2, delay=0.01)
        @validate_inputs(is_positive)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated failure on first attempt")
            return x * 2
        
        # Should fail on first attempt, retry, then succeed
        result = test_function(5)
        assert result == 10
        assert call_count == 2 