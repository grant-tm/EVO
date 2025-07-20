"""
Tests for EVO validation utilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from evo.utils.validators import (
    validate_dataframe, validate_model_path, validate_config, validate_hyperparameters
)
from evo.core.exceptions import ValidationError, ConfigurationError


class TestDataFrameValidation:
    """Test DataFrame validation functions."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_valid(self, sample_dataframe):
        """Test validation of valid DataFrame."""
        result = validate_dataframe(sample_dataframe)
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_none(self):
        """Test validation fails with None DataFrame."""
        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(None)
        assert "DataFrame cannot be None" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_wrong_type(self):
        """Test validation fails with wrong type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe("not a dataframe")
        assert "Expected pandas DataFrame" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_empty(self):
        """Test validation fails with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(empty_df, min_rows=1)
        assert "DataFrame must have at least 1 rows" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_missing_columns(self):
        """Test validation fails with missing required columns."""
        df = pd.DataFrame({'open': [100], 'close': [101]})
        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(df, required_columns=['open', 'close', 'volume'])
        assert "Missing required columns" in str(exc_info.value)
        assert "volume" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_with_nulls(self, caplog):
        """Test validation with null values (should warn but not fail)."""
        df = pd.DataFrame({
            'open': [100, 101, np.nan],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        
        result = validate_dataframe(df, check_nulls=True)
        assert result is True
        assert "Found null values" in caplog.text
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_with_infinite(self):
        """Test validation fails with infinite values."""
        df = pd.DataFrame({
            'open': [100, 101, np.inf],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        
        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(df, check_infinite=True)
        assert "Found infinite values in column" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_min_rows_ok(self):
        """Test validation passes with sufficient rows."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = validate_dataframe(df, min_rows=3)
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_dataframe_no_checks(self):
        """Test validation with checks disabled."""
        df = pd.DataFrame({
            'open': [100, np.nan, np.inf],
            'close': [102, 103, 104]
        })
        
        # Should pass when checks are disabled
        result = validate_dataframe(df, check_nulls=False, check_infinite=False)
        assert result is True


class TestModelPathValidation:
    """Test model path validation functions."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_model_path_valid(self, temp_dir):
        """Test validation of valid model path."""
        model_path = temp_dir / "model.zip"
        result = validate_model_path(model_path)
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_model_path_string(self, temp_dir):
        """Test validation with string path."""
        # Create the models directory and file
        models_dir = temp_dir / "models"
        models_dir.mkdir()
        model_file = models_dir / "model.pkl"
        model_file.touch()  # Create empty file
        
        # Test with relative path from temp_dir
        result = validate_model_path(str(model_file))
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_model_path_nonexistent_dir(self):
        """Test validation fails with nonexistent directory."""
        model_path = Path("nonexistent_dir/model.zip")
        with pytest.raises(ValidationError) as exc_info:
            validate_model_path(model_path)
        assert "Model directory does not exist" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_model_path_must_exist(self, temp_dir):
        """Test validation fails when file must exist but doesn't."""
        model_path = temp_dir / "nonexistent_model.zip"
        with pytest.raises(ValidationError) as exc_info:
            validate_model_path(model_path, must_exist=True)
        assert "Model file does not exist" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_model_path_unusual_extension(self, temp_dir, caplog):
        """Test validation with unusual file extension."""
        model_path = temp_dir / "model.txt"
        result = validate_model_path(model_path)
        assert result is True
        assert "Model file has unusual extension" in caplog.text
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_model_path_valid_extensions(self, temp_dir):
        """Test validation with valid file extensions."""
        valid_extensions = ['.zip', '.pkl', '.pth', '.h5', '.onnx']
        
        for ext in valid_extensions:
            model_path = temp_dir / f"model{ext}"
            result = validate_model_path(model_path)
            assert result is True


class TestConfigValidation:
    """Test configuration validation functions."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_config_valid(self, config_with_mock_keys):
        """Test validation of valid configuration."""
        result = validate_config(config_with_mock_keys)
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_config_none(self):
        """Test validation fails with None config."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(None)
        assert "Configuration cannot be None" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_config_missing_attributes(self):
        """Test validation fails with missing required attributes."""
        class InvalidConfig:
            def __init__(self):
                self.trading = None
                # Missing other required attributes
        
        invalid_config = InvalidConfig()
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(invalid_config)
        assert "Configuration missing required attributes" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_config_invalid_trading(self, config_with_mock_keys):
        """Test validation fails with invalid trading config."""
        config = config_with_mock_keys
        config.trading.trade_qty = 0  # Invalid
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Trade quantity must be positive" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_config_invalid_data(self, config_with_mock_keys):
        """Test validation fails with invalid data config."""
        config = config_with_mock_keys
        config.data.seq_len = 0  # Invalid
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Sequence length must be positive" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_config_invalid_training(self, config_with_mock_keys):
        """Test validation fails with invalid training config."""
        config = config_with_mock_keys
        config.training.learning_rate = 0  # Invalid
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Learning rate must be positive" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_config_invalid_reward(self, config_with_mock_keys):
        """Test validation fails with invalid reward config."""
        config = config_with_mock_keys
        config.reward.tp_pct = 0  # Invalid
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Take profit and stop loss percentages must be positive" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_config_invalid_optimization(self, config_with_mock_keys):
        """Test validation fails with invalid optimization config."""
        config = config_with_mock_keys
        config.optimization.mutation_rate = 1.5  # Invalid
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Mutation rate must be between 0 and 1" in str(exc_info.value)


class TestHyperparameterValidation:
    """Test hyperparameter validation functions."""
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_valid(self):
        """Test validation of valid hyperparameters."""
        valid_params = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "clip_range": 0.2,
            "entropy_coef_init": 0.1,
            "gae_lambda": 0.95
        }
        
        result = validate_hyperparameters(valid_params)
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_not_dict(self):
        """Test validation fails with non-dict input."""
        with pytest.raises(ValidationError) as exc_info:
            validate_hyperparameters("not a dict")
        assert "Hyperparameters must be a dictionary" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_invalid_learning_rate(self):
        """Test validation fails with invalid learning rate."""
        invalid_params = {"learning_rate": 0}  # Invalid
        
        with pytest.raises(ValidationError) as exc_info:
            validate_hyperparameters(invalid_params)
        assert "Learning rate must be a positive number" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_invalid_batch_size(self):
        """Test validation fails with invalid batch size."""
        invalid_params = {"batch_size": 0}  # Invalid
        
        with pytest.raises(ValidationError) as exc_info:
            validate_hyperparameters(invalid_params)
        assert "Batch size must be a positive integer" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_invalid_clip_range(self):
        """Test validation fails with invalid clip range."""
        invalid_params = {"clip_range": 1.5}  # Invalid
        
        with pytest.raises(ValidationError) as exc_info:
            validate_hyperparameters(invalid_params)
        assert "Clip range must be between 0 and 1" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_invalid_entropy_coef(self):
        """Test validation fails with invalid entropy coefficient."""
        invalid_params = {"entropy_coef_init": -0.1}  # Invalid
        
        with pytest.raises(ValidationError) as exc_info:
            validate_hyperparameters(invalid_params)
        assert "entropy_coef_init must be a non-negative number" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_invalid_gae_lambda(self):
        """Test validation fails with invalid GAE lambda."""
        invalid_params = {"gae_lambda": 1.5}  # Invalid
        
        with pytest.raises(ValidationError) as exc_info:
            validate_hyperparameters(invalid_params)
        assert "gae_lambda must be between 0 and 1" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_partial(self):
        """Test validation with partial hyperparameters."""
        partial_params = {"learning_rate": 0.001}  # Only one parameter
        
        result = validate_hyperparameters(partial_params)
        assert result is True
    
    @pytest.mark.unit
    @pytest.mark.utils
    def test_validate_hyperparameters_empty(self):
        """Test validation with empty hyperparameters."""
        result = validate_hyperparameters({})
        assert result is True 