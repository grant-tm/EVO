"""
Tests for EVO configuration system.
"""

import pytest
import json
from pathlib import Path
from evo.core.config import Config, get_config, set_config
from evo.core.exceptions import ConfigurationError


class TestConfigCreation:
    """Test configuration creation and initialization."""
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_default_config_creation(self, mock_env_vars):
        """Test creating a config with default values."""
        config = Config()
        
        # Test default values
        assert config.trading.symbol == "SPY"
        assert config.trading.trade_qty == 1
        assert config.trading.use_simulation is True
        assert config.data.seq_len == 15
        assert config.training.training_steps == 1_000_000
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_config_from_file(self, config_from_file, sample_config_data):
        """Test creating a config from a JSON file."""
        config = config_from_file
        
        # Test values from file
        assert config.trading.symbol == sample_config_data["trading"]["symbol"]
        assert config.trading.trade_qty == sample_config_data["trading"]["trade_qty"]
        assert config.data.seq_len == sample_config_data["data"]["seq_len"]
        assert config.training.learning_rate == sample_config_data["training"]["learning_rate"]
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_api_keys_loaded(self, config_with_mock_keys):
        """Test that API keys are loaded from environment."""
        config = config_with_mock_keys
        assert config.alpaca.api_key == "test_api_key"
        assert config.alpaca.api_secret == "test_secret_key"
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_config_without_api_keys(self, monkeypatch):
        """Test config creation without API keys."""
        # Clear environment variables
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        
        # Pass a non-existent .env file to prevent loading the actual .env file
        config = Config(env_file=Path("nonexistent.env"))
        assert config.alpaca.api_key is None
        assert config.alpaca.api_secret is None


class TestConfigValidation:
    """Test configuration validation."""
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_valid_config_passes_validation(self, config_with_mock_keys):
        """Test that valid config passes validation."""
        config = config_with_mock_keys
        # Should not raise any exception
        assert config is not None
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_invalid_trade_qty(self, temp_dir, mock_env_vars):
        """Test validation fails with invalid trade quantity."""
        invalid_config = {
            "trading": {"trade_qty": 0}  # Invalid: must be > 0
        }
        
        config_file = temp_dir / "invalid_config.json"
        with open(config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        with pytest.raises(ConfigurationError) as exc_info:
            Config(config_file=config_file)
        
        assert "Trade quantity must be positive" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_invalid_learning_rate(self, temp_dir, mock_env_vars):
        """Test validation fails with invalid learning rate."""
        invalid_config = {
            "training": {"learning_rate": 0}  # Invalid: must be > 0
        }
        
        config_file = temp_dir / "invalid_config.json"
        with open(config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        with pytest.raises(ConfigurationError) as exc_info:
            Config(config_file=config_file)
        
        assert "Learning rate must be positive" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_invalid_mutation_rate(self, temp_dir, mock_env_vars):
        """Test validation fails with invalid mutation rate."""
        invalid_config = {
            "optimization": {"mutation_rate": 1.5}  # Invalid: must be <= 1
        }
        
        config_file = temp_dir / "invalid_config.json"
        with open(config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        with pytest.raises(ConfigurationError) as exc_info:
            Config(config_file=config_file)
        
        assert "Mutation rate must be between 0 and 1" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_live_trading_without_api_keys(self, temp_dir, monkeypatch):
        """Test validation fails for live trading without API keys."""
        # Clear environment variables
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        
        live_config = {
            "trading": {"use_simulation": False}  # Live trading
        }
        
        config_file = temp_dir / "live_config.json"
        with open(config_file, 'w') as f:
            json.dump(live_config, f)
        
        with pytest.raises(ConfigurationError) as exc_info:
            Config(config_file=config_file, env_file=Path("nonexistent.env"))
        
        assert "Alpaca API key is required for live trading" in str(exc_info.value)


class TestConfigMethods:
    """Test configuration methods."""
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_to_dict(self, config_with_mock_keys):
        """Test converting config to dictionary."""
        config = config_with_mock_keys
        config_dict = config.to_dict()
        
        assert "trading" in config_dict
        assert "data" in config_dict
        assert "training" in config_dict
        assert "reward" in config_dict
        assert "optimization" in config_dict
        assert "alpaca" in config_dict
        
        # Check that API keys are masked
        assert config_dict["alpaca"]["api_key"] == "***"
        assert config_dict["alpaca"]["api_secret"] == "***"
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_save_and_load(self, temp_dir, config_with_mock_keys):
        """Test saving and loading configuration."""
        config = config_with_mock_keys
        
        # Save config
        config_file = temp_dir / "saved_config.json"
        config.save(config_file)
        
        # Load config
        loaded_config = Config(config_file=config_file)
        
        # Compare values
        assert loaded_config.trading.symbol == config.trading.symbol
        assert loaded_config.trading.trade_qty == config.trading.trade_qty
        assert loaded_config.data.seq_len == config.data.seq_len
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_get_model_path(self, config_with_mock_keys):
        """Test getting model path."""
        config = config_with_mock_keys
        model_path = config.get_model_path()
        
        expected_path = Path(config.training.model_dir) / config.training.model_name
        assert model_path == expected_path
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_get_data_path(self, config_with_mock_keys):
        """Test getting data path."""
        config = config_with_mock_keys
        data_path = config.get_data_path()
        
        expected_path = Path(config.data.data_path)
        assert data_path == expected_path
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_repr(self, config_with_mock_keys):
        """Test string representation."""
        config = config_with_mock_keys
        config_str = repr(config)
        
        assert "Config" in config_str
        assert config.trading.symbol in config_str
        assert str(config.trading.use_simulation) in config_str


class TestGlobalConfig:
    """Test global configuration functions."""
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_get_config_singleton(self, mock_env_vars):
        """Test that get_config returns a singleton."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_set_config(self, mock_env_vars):
        """Test setting global configuration."""
        original_config = get_config()
        
        # Create new config
        new_config = Config()
        new_config.trading.symbol = "AAPL"
        
        # Set new config
        set_config(new_config)
        
        # Get config and verify it's the new one
        retrieved_config = get_config()
        assert retrieved_config is new_config
        assert retrieved_config.trading.symbol == "AAPL"
        
        # Restore original config
        set_config(original_config)


class TestConfigFileHandling:
    """Test configuration file handling."""
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_nonexistent_config_file(self, mock_env_vars):
        """Test handling of nonexistent config file."""
        nonexistent_file = Path("nonexistent_config.json")
        config = Config(config_file=nonexistent_file)
        
        # Should use defaults
        assert config.trading.symbol == "SPY"
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_invalid_json_file(self, temp_dir, mock_env_vars):
        """Test handling of invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(ConfigurationError) as exc_info:
            Config(config_file=invalid_file)
        
        assert "Failed to load config file" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.config
    def test_partial_config_file(self, temp_dir, mock_env_vars):
        """Test loading partial configuration file."""
        partial_config = {
            "trading": {"symbol": "TSLA"},
            "training": {"learning_rate": 0.001}
        }
        
        config_file = temp_dir / "partial_config.json"
        with open(config_file, 'w') as f:
            json.dump(partial_config, f)
        
        config = Config(config_file=config_file)
        
        # Should use file values where provided
        assert config.trading.symbol == "TSLA"
        assert config.training.learning_rate == 0.001
        
        # Should use defaults for missing values
        assert config.trading.trade_qty == 1  # default
        assert config.data.seq_len == 15  # default 