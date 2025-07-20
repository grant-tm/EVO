# EVO Testing Framework

This directory contains a comprehensive testing framework for the EVO trading system, built with **pytest**.

## 🏗️ Framework Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_core/               # Core functionality tests
│   ├── test_exceptions.py   # Exception system tests
│   ├── test_config.py       # Configuration system tests
│   └── test_logging.py      # Logging system tests
├── test_utils/              # Utility function tests
│   └── test_validators.py   # Validation utility tests
└── README.md               # This file
```

## 🚀 Quick Start

### Running Tests

```bash
# Run all unit tests (fast)
python run_tests.py --type unit

# Run all tests
python run_tests.py --type all

# Run specific test category
python run_tests.py --type config
python run_tests.py --type logging
python run_tests.py --type utils

# Run with coverage
python run_tests.py --type all --coverage

# Run specific test file
python run_tests.py --file tests/test_core/test_config.py

# Run specific test function
python run_tests.py --function test_default_config_creation
```

### Direct pytest Usage

```bash
# Run all tests
pytest

# Run specific markers
pytest -m unit
pytest -m integration
pytest -m core

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=evo --cov-report=html
```

## 🏷️ Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **Fast, isolated tests** that don't depend on external systems
- Test individual functions and classes
- Mock external dependencies
- Should run in < 1 second each

### Integration Tests (`@pytest.mark.integration`)
- **Slower tests** that test component interactions
- May use real files, databases, or external APIs
- Test end-to-end workflows
- Can take several seconds each

### Core Tests (`@pytest.mark.core`)
- Tests for core EVO functionality
- Exception handling, configuration, logging

### Configuration Tests (`@pytest.mark.config`)
- Tests for configuration management
- File loading, validation, environment variables

### Logging Tests (`@pytest.mark.logging`)
- Tests for logging system
- Correlation IDs, formatters, handlers

### Utility Tests (`@pytest.mark.utils`)
- Tests for utility functions
- Validators, decorators, helpers

### Slow Tests (`@pytest.mark.slow`)
- Tests that take longer to run
- Excluded by default, use `--slow` to include

## 🔧 Test Fixtures

Common fixtures available in `conftest.py`:

### Data Fixtures
- `sample_dataframe`: Sample pandas DataFrame for testing
- `sample_numpy_array`: Sample numpy array for testing
- `sample_config_data`: Sample configuration data

### Environment Fixtures
- `temp_dir`: Temporary directory for test files
- `mock_env_vars`: Mocked environment variables
- `config_file`: Temporary configuration file
- `config_with_mock_keys`: Config instance with mocked API keys

### Logging Fixtures
- `test_logger`: Configured logger for tests

## 📝 Writing Tests

### Test Naming Convention
```python
def test_function_name_scenario():
    """Test description."""
    # Test implementation
```

### Test Class Structure
```python
class TestComponentName:
    """Test the ComponentName class."""
    
    @pytest.mark.unit
    @pytest.mark.core
    def test_method_name_success(self):
        """Test successful method execution."""
        # Arrange
        component = Component()
        
        # Act
        result = component.method()
        
        # Assert
        assert result == expected_value
    
    @pytest.mark.unit
    @pytest.mark.core
    def test_method_name_failure(self):
        """Test method failure handling."""
        # Arrange
        component = Component()
        
        # Act & Assert
        with pytest.raises(ExpectedException):
            component.method()
```

### Using Fixtures
```python
def test_with_fixtures(sample_dataframe, temp_dir):
    """Test using fixtures."""
    # Use sample_dataframe and temp_dir
    result = process_dataframe(sample_dataframe)
    assert result is not None
```

### Testing Exceptions
```python
def test_exception_handling():
    """Test that exceptions are raised correctly."""
    with pytest.raises(ValidationError) as exc_info:
        validate_input(invalid_data)
    
    assert "Expected error message" in str(exc_info.value)
```

### Testing Logging
```python
def test_logging_output(caplog):
    """Test logging output."""
    logger = get_logger("test")
    logger.info("Test message")
    
    assert "Test message" in caplog.text
```

## 🎯 Test Coverage

### Current Coverage
- ✅ **Exceptions**: All exception types and hierarchy
- ✅ **Configuration**: File loading, validation, API keys
- ✅ **Logging**: Setup, correlation IDs, formatters
- ✅ **Validators**: DataFrame, model paths, config, hyperparameters

### Planned Coverage (Future Phases)
- 🔄 **Data Layer**: Data providers, streamers, processors
- 🔄 **Model Layer**: Agents, environments, training
- 🔄 **Optimization**: Genetic search, backtesting
- 🔄 **Execution**: Brokers, risk management, live trading

## 🛠️ Test Configuration

### pytest.ini
- Test discovery patterns
- Output formatting
- Marker definitions
- Warning filters

### Environment Setup
Tests automatically:
- Set up temporary directories
- Mock environment variables
- Configure logging for tests
- Clean up after completion

## 📊 Running Coverage

```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
python run_tests.py --coverage

# Generate HTML report
pytest --cov=evo --cov-report=html
```

Coverage reports will show:
- Line coverage percentage
- Missing lines
- Branch coverage
- HTML report in `htmlcov/` directory

## 🔍 Debugging Tests

### Verbose Output
```bash
python run_tests.py --verbose
pytest -vv
```

### Single Test Debugging
```bash
# Run single test with full output
pytest tests/test_core/test_config.py::TestConfigCreation::test_default_config_creation -vv -s

# Run with debugger
pytest --pdb tests/test_core/test_config.py::TestConfigCreation::test_default_config_creation
```

### Test Discovery
```bash
# List all tests
pytest --collect-only

# List tests with markers
pytest --collect-only -m unit
```

## 🚨 Best Practices

### Test Design
1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion per test**: Test one thing at a time
3. **Descriptive names**: Test names should explain what's being tested
4. **Independent tests**: Tests should not depend on each other

### Test Data
1. **Use fixtures**: Reuse common test data
2. **Minimal data**: Use the smallest dataset that tests the functionality
3. **Realistic data**: Use data that resembles real usage

### Error Testing
1. **Test edge cases**: Boundary conditions, invalid inputs
2. **Test exceptions**: Verify correct error handling
3. **Test error messages**: Ensure helpful error messages

### Performance
1. **Fast unit tests**: Keep unit tests under 1 second
2. **Mock external calls**: Don't make real API calls in unit tests
3. **Use appropriate markers**: Mark slow tests appropriately

## 🔄 Continuous Integration

The testing framework is designed to work with CI/CD systems:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    python run_tests.py --type unit
    python run_tests.py --type integration --slow
    python run_tests.py --coverage
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://realpython.com/python-testing/) 