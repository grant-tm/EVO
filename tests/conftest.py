"""
Pytest configuration and common fixtures for EVO testing.

This file contains shared fixtures and configuration that can be used
across all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from evo.core.config import get_config, set_config
from evo.core.exceptions import DataError

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evo.core import Config, setup_logging
from evo.core.logging import get_logger


@pytest.fixture(scope="session")
def test_logger():
    """Set up logging for tests."""
    # Configure logging for tests (console only, no files)
    setup_logging(level="DEBUG", enable_file=False, enable_console=True)
    return get_logger("test")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "trading": {
            "symbol": "AAPL",
            "trade_qty": 10,
            "use_simulation": True,
            "sim_speed": 1.0
        },
        "data": {
            "data_path": "test_data.csv",
            "seq_len": 20,
            "features": ["open", "close", "volume"]
        },
        "training": {
            "model_dir": "test_models",
            "model_name": "test_model.zip",
            "training_steps": 1000,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        "reward": {
            "tp_pct": 0.02,
            "sl_pct": 0.01,
            "max_episode_steps": 1000
        },
        "optimization": {
            "mutation_rate": 0.15,
            "elite_proportion": 0.25,
            "num_backtests": 100
        },
        "alpaca": {
            "paper_trading": True
        }
    }


@pytest.fixture
def config_file(temp_dir, sample_config_data):
    """Create a temporary config file for testing."""
    import json
    config_path = temp_dir / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config_data, f, indent=2)
    return config_path


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [99, 100, 101, 102, 103],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'return': [0.02, 0.01, 0.01, 0.01, 0.01],
        'volatility': [0.01, 0.01, 0.01, 0.01, 0.01]
    })


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array for testing."""
    return np.array([
        [100, 105, 99, 102, 1000],
        [101, 106, 100, 103, 1100],
        [102, 107, 101, 104, 1200],
        [103, 108, 102, 105, 1300],
        [104, 109, 103, 106, 1400]
    ], dtype=np.float32)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("ALPACA_API_KEY", "test_api_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret_key")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for test model files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def patch_logging():
    """Patch logging setup to avoid file creation during tests."""
    with patch('evo.core.logging.setup_logging') as mock_setup:
        with patch('evo.core.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            yield mock_setup, mock_get_logger, mock_logger


@pytest.fixture
def config_with_mock_keys(mock_env_vars):
    """Create a config instance with mocked API keys."""
    return Config()


@pytest.fixture
def config_from_file(config_file, mock_env_vars):
    """Create a config instance from a test file."""
    return Config(config_file=config_file)