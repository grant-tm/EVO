"""
Tests for training components.

This module contains tests for the training orchestration, hyperparameters,
and callbacks.
"""

import pytest
import numpy as np
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from evo.models.training.trainer import Trainer
from evo.models.training.hyperparameters import TrainingHyperparameters, HyperparameterManager
from evo.models.training.callbacks import (
    TrainingCallback, EntropyAnnealingCallback, RewardLoggerCallback,
    EarlyStoppingCallback, ModelCheckpointCallback
)
from evo.models.agents.ppo_agent import PPOAgent
from evo.models.environments.trading_env import TradingEnv

pytestmark = [
    pytest.mark.unit,
    pytest.mark.models,
    pytest.mark.training
]


class TestTrainingHyperparameters:
    """Test the TrainingHyperparameters dataclass."""
    
    def test_default_initialization(self):
        """Test initialization with default values."""
        hyperparams = TrainingHyperparameters()
        
        # Check default values
        assert hyperparams.total_timesteps == 1_000_000
        assert hyperparams.batch_size == 64
        assert hyperparams.learning_rate == 3e-4
        assert hyperparams.n_steps == 512
        assert hyperparams.n_epochs == 10
        assert hyperparams.gamma == 0.99
        assert hyperparams.gae_lambda == 0.95
        assert hyperparams.clip_range == 0.2
        assert hyperparams.ent_coef == 0.01
        assert hyperparams.vf_coef == 0.5
        assert hyperparams.max_grad_norm == 0.5
        assert hyperparams.policy == "MlpPolicy"
        assert hyperparams.seq_len == 20
        assert hyperparams.tp_pct == 0.02
        assert hyperparams.sl_pct == 0.01
        assert hyperparams.idle_penalty == 0.001
        assert hyperparams.max_episode_steps == 1000
    
    def test_custom_initialization(self):
        """Test initialization with custom values."""
        custom_params = {
            'total_timesteps': 500_000,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'n_steps': 1024,
            'n_epochs': 20,
            'gamma': 0.95,
            'ent_coef': 0.02,
            'policy': 'CnnPolicy',
            'seq_len': 30,
            'tp_pct': 0.03,
            'sl_pct': 0.015
        }
        
        hyperparams = TrainingHyperparameters(**custom_params)
        
        for key, value in custom_params.items():
            assert getattr(hyperparams, key) == value
    
    def test_validation_valid_params(self):
        """Test validation with valid parameters."""
        hyperparams = TrainingHyperparameters()
        errors = hyperparams.validate()
        assert len(errors) == 0
    
    def test_validation_invalid_params(self):
        """Test validation with invalid parameters."""
        invalid_params = {
            'total_timesteps': 0,  # Invalid: must be positive
            'batch_size': -1,      # Invalid: must be positive
            'learning_rate': 0,    # Invalid: must be positive
            'gamma': 1.5,          # Invalid: must be <= 1
            'ent_coef': -0.1,      # Invalid: must be non-negative
            'seq_len': 0,          # Invalid: must be positive
            'tp_pct': -0.01,       # Invalid: must be positive
            'max_episode_steps': 0 # Invalid: must be positive
        }
        
        hyperparams = TrainingHyperparameters(**invalid_params)
        errors = hyperparams.validate()
        
        # Should have validation errors
        assert len(errors) > 0
        
        # Check specific error messages
        error_messages = [error.lower() for error in errors]
        assert any('total_timesteps' in msg for msg in error_messages)
        assert any('batch_size' in msg for msg in error_messages)
        assert any('learning_rate' in msg for msg in error_messages)
        assert any('gamma' in msg for msg in error_messages)
        assert any('ent_coef' in msg for msg in error_messages)
        assert any('seq_len' in msg for msg in error_messages)
        assert any('tp_pct' in msg for msg in error_messages)
        assert any('max_episode_steps' in msg for msg in error_messages)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        hyperparams = TrainingHyperparameters()
        config_dict = hyperparams.to_dict()
        
        # Check that all expected keys are present
        expected_keys = [
            'total_timesteps', 'batch_size', 'learning_rate', 'n_steps',
            'n_epochs', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef',
            'vf_coef', 'max_grad_norm', 'policy', 'seq_len', 'tp_pct',
            'sl_pct', 'idle_penalty', 'max_episode_steps'
        ]
        
        for key in expected_keys:
            assert key in config_dict
        
        # Check that values match
        for key in expected_keys:
            assert config_dict[key] == getattr(hyperparams, key)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'total_timesteps': 500_000,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'policy': 'CnnPolicy',
            'seq_len': 30
        }
        
        hyperparams = TrainingHyperparameters.from_dict(config_dict)
        
        for key, value in config_dict.items():
            assert getattr(hyperparams, key) == value


class TestHyperparameterManager:
    """Test the HyperparameterManager class."""
    
    @pytest.fixture
    def temp_config_dir(self, temp_dir):
        """Create a temporary config directory."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        return config_dir
    
    @pytest.fixture
    def hyperparam_manager(self, temp_config_dir):
        """Create a hyperparameter manager for testing."""
        return HyperparameterManager(str(temp_config_dir))
    
    def test_init(self, temp_config_dir):
        """Test initialization."""
        manager = HyperparameterManager(str(temp_config_dir))
        assert manager.config_dir == temp_config_dir
    
    def test_save_hyperparameters(self, hyperparam_manager, temp_config_dir):
        """Test saving hyperparameters."""
        hyperparams = TrainingHyperparameters(
            total_timesteps=500_000,
            batch_size=128,
            policy='CnnPolicy'
        )
        
        hyperparam_manager.save_hyperparameters(hyperparams, "test_config")
        
        # Check that file was created
        config_file = temp_config_dir / "test_config.json"
        assert config_file.exists()
        
        # Check file contents
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config['total_timesteps'] == 500_000
        assert saved_config['batch_size'] == 128
        assert saved_config['policy'] == 'CnnPolicy'
    
    def test_load_hyperparameters(self, hyperparam_manager, temp_config_dir):
        """Test loading hyperparameters."""
        # Create a config file
        config_data = {
            'total_timesteps': 500_000,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'policy': 'CnnPolicy'
        }
        
        config_file = temp_config_dir / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load hyperparameters
        hyperparams = hyperparam_manager.load_hyperparameters("test_config")
        
        assert hyperparams.total_timesteps == 500_000
        assert hyperparams.batch_size == 128
        assert hyperparams.learning_rate == 1e-4
        assert hyperparams.policy == 'CnnPolicy'
    
    def test_load_nonexistent_config(self, hyperparam_manager):
        """Test loading nonexistent configuration raises error."""
        with pytest.raises(FileNotFoundError):
            hyperparam_manager.load_hyperparameters("nonexistent")
    
    def test_list_configurations(self, hyperparam_manager, temp_config_dir):
        """Test listing available configurations."""
        # Create some config files
        configs = ["config1", "config2", "config3"]
        for config_name in configs:
            config_file = temp_config_dir / f"{config_name}.json"
            with open(config_file, 'w') as f:
                json.dump({'total_timesteps': 1000}, f)
        
        # List configurations
        available_configs = hyperparam_manager.list_configurations()
        
        # Should return all config names (without .json extension)
        assert set(available_configs) == set(configs)
    
    def test_get_default_hyperparameters(self, hyperparam_manager):
        """Test getting default hyperparameters."""
        default_params = hyperparam_manager.get_default_hyperparameters()
        
        assert isinstance(default_params, TrainingHyperparameters)
        assert default_params.total_timesteps == 1_000_000
        assert default_params.batch_size == 64
    
    def test_create_hyperparameter_grid(self, hyperparam_manager):
        """Test creating hyperparameter grid."""
        param_ranges = {
            'batch_size': [32, 64, 128],
            'learning_rate': [1e-4, 3e-4],
            'policy': ['MlpPolicy', 'CnnPolicy']
        }
        
        grid = hyperparam_manager.create_hyperparameter_grid(param_ranges)
        
        # Should create 3 * 2 * 2 = 12 combinations
        assert len(grid) == 12
        
        # Check that all combinations are unique
        import json
        configs = [json.dumps(hp.to_dict(), sort_keys=True) for hp in grid]
        assert len(set(configs)) == 12
        
        # Check that all values from ranges are used
        batch_sizes = [hp.batch_size for hp in grid]
        learning_rates = [hp.learning_rate for hp in grid]
        policies = [hp.policy for hp in grid]
        
        assert set(batch_sizes) == set(param_ranges['batch_size'])
        assert set(learning_rates) == set(param_ranges['learning_rate'])
        assert set(policies) == set(param_ranges['policy'])


class TestTrainingCallback:
    """Test the abstract training callback."""
    
    def test_abstract_methods(self):
        """Test that TrainingCallback cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrainingCallback()
    
    def test_abstract_methods_required(self):
        """Test that all abstract methods are properly defined."""
        assert hasattr(TrainingCallback, 'on_step')


class TestEntropyAnnealingCallback:
    """Test the entropy annealing callback."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.ent_coef = 0.01
        return model
    
    @pytest.fixture
    def callback(self, mock_model):
        """Create an entropy annealing callback for testing."""
        with patch('stable_baselines3.common.callbacks.BaseCallback.__init__'):
            callback = EntropyAnnealingCallback(
                initial_coef=0.01,
                final_coef=0.001,
                total_timesteps=1000,
                verbose=1
            )
            callback.model = mock_model
            callback.verbose = 1  # Manually set verbose since __init__ is patched
            return callback
    
    def test_init(self):
        """Test callback initialization."""
        with patch('stable_baselines3.common.callbacks.BaseCallback.__init__'):
            callback = EntropyAnnealingCallback(
                initial_coef=0.01,
                final_coef=0.001,
                total_timesteps=1000,
                verbose=1
            )
            
            assert callback.initial_coef == 0.01
            assert callback.final_coef == 0.001
            assert callback.total_timesteps == 1000
    
    def test_on_step_beginning(self, callback, mock_model):
        """Test callback at beginning of training."""
        callback.num_timesteps = 0
        
        result = callback._on_step()
        
        assert result is True
        assert mock_model.ent_coef == 0.01  # Should be initial value
    
    def test_on_step_middle(self, callback, mock_model):
        """Test callback in middle of training."""
        callback.num_timesteps = 500  # Halfway through
        
        result = callback._on_step()
        
        assert result is True
        # Should be halfway between initial and final
        expected_coef = 0.01 + 0.5 * (0.001 - 0.01)
        assert abs(mock_model.ent_coef - expected_coef) < 1e-6
    
    def test_on_step_end(self, callback, mock_model):
        """Test callback at end of training."""
        callback.num_timesteps = 1000  # End of training
        
        result = callback._on_step()
        
        assert result is True
        assert np.isclose(mock_model.ent_coef, 0.001)


class TestRewardLoggerCallback:
    """Test the reward logger callback."""
    
    @pytest.fixture
    def callback(self):
        """Create a reward logger callback for testing."""
        with patch('stable_baselines3.common.callbacks.BaseCallback.__init__'):
            callback = RewardLoggerCallback(verbose=1)
            # Mock the locals dict that stable_baselines3 would provide
            callback.locals = {}
            return callback
    
    def test_init(self):
        """Test callback initialization."""
        with patch('stable_baselines3.common.callbacks.BaseCallback.__init__'):
            callback = RewardLoggerCallback(verbose=1)
            assert callback.reward_history == []
            assert callback.episode_rewards == []
    
    @patch('evo.models.training.callbacks.RewardLoggerCallback.logger')
    def test_on_step_with_reward(self, mock_logger, callback):
        """Test on_step method with reward in locals."""
        callback.num_timesteps = 100
        callback.locals = {"reward": 0.5}
        
        result = callback._on_step()
        
        assert result is True
        assert len(callback.reward_history) == 1
        assert callback.reward_history[0] == 0.5
        mock_logger.record.assert_called_with("rollout/raw_reward", 0.5)
    
    @patch('evo.models.training.callbacks.RewardLoggerCallback.logger')
    def test_on_step_without_reward(self, mock_logger, callback):
        """Test on_step method without reward in locals."""
        callback.num_timesteps = 100
        callback.locals = {}
        
        result = callback._on_step()
        
        assert result is True
        assert len(callback.reward_history) == 0
        mock_logger.record.assert_not_called()
    
    @patch('evo.models.training.callbacks.RewardLoggerCallback.logger')
    def test_on_step_with_moving_average(self, mock_logger, callback):
        """Test on_step method with enough rewards for moving average."""
        callback.num_timesteps = 100
        # Add 99 rewards first
        for i in range(99):
            callback.reward_history.append(i * 0.01)
        
        callback.locals = {"reward": 1.0}
        
        result = callback._on_step()
        
        assert result is True
        assert len(callback.reward_history) == 100
        # Should call record twice - once for raw reward, once for moving average
        assert mock_logger.record.call_count == 2
        # Check the moving average call
        moving_avg_call = mock_logger.record.call_args_list[1]
        assert moving_avg_call[0][0] == "rollout/reward_moving_avg"
    
    @patch('evo.models.training.callbacks.RewardLoggerCallback.logger')
    def test_on_rollout_end_with_episode_rewards(self, mock_logger, callback):
        """Test on_rollout_end method with episode rewards."""
        callback.episode_rewards = [0.1, 0.2, 0.3]
        
        callback._on_rollout_end()
        
        # Should record the mean episode reward
        args, kwargs = mock_logger.record.call_args
        assert args[0] == "rollout/episode_reward"
        assert np.isclose(args[1], 0.2)
        # Episode rewards should be cleared
        assert callback.episode_rewards == []
    
    @patch('evo.models.training.callbacks.RewardLoggerCallback.logger')
    def test_on_rollout_end_without_episode_rewards(self, mock_logger, callback):
        """Test on_rollout_end method without episode rewards."""
        callback.episode_rewards = []
        
        callback._on_rollout_end()
        
        # Should not call record since no episode rewards
        mock_logger.record.assert_not_called()


class TestEarlyStoppingCallback:
    """Test the early stopping callback."""
    
    @pytest.fixture
    def callback(self):
        """Create an early stopping callback for testing."""
        with patch('stable_baselines3.common.callbacks.BaseCallback.__init__'):
            callback = EarlyStoppingCallback(
                eval_freq=100,
                patience=3,
                min_evals=2,
                metric="eval_mean_reward",
                verbose=1
            )
            callback.verbose = 1  # Manually set verbose since __init__ is patched
            return callback
    
    def test_init(self):
        """Test callback initialization."""
        with patch('stable_baselines3.common.callbacks.BaseCallback.__init__'):
            callback = EarlyStoppingCallback(
                eval_freq=100,
                patience=3,
                min_evals=2,
                metric="eval_mean_reward",
                verbose=1
            )
            callback.verbose = 1  # Manually set verbose since __init__ is patched
            assert callback.eval_freq == 100
            assert callback.patience == 3
            assert callback.min_evals == 2
            assert callback.metric == "eval_mean_reward"
            assert callback.verbose == 1
            assert callback.best_score == -np.inf
            assert callback.no_improvement_count == 0
            assert callback.eval_count == 0
    
    @patch('evo.models.training.callbacks.EarlyStoppingCallback.logger')
    def test_on_step_no_eval(self, mock_logger, callback):
        """Test on_step when no evaluation is due."""
        callback.num_timesteps = 50  # Not at eval_freq
        callback.verbose = 1  # Ensure verbose is set
        mock_logger.name_to_value = {}
        
        result = callback._on_step()
        
        assert result is True  # Should continue training
    
    @patch('evo.models.training.callbacks.EarlyStoppingCallback.logger')
    def test_on_step_with_eval_improving(self, mock_logger, callback):
        """Test on_step when evaluation shows improvement."""
        callback.num_timesteps = 100  # At eval_freq
        callback.best_score = 0.5
        callback.verbose = 1  # Ensure verbose is set
        mock_logger.name_to_value = {"eval_mean_reward": 0.6}  # Better than best
        
        result = callback._on_step()
        
        assert result is True  # Should continue training
        assert callback.best_score == 0.6
        assert callback.no_improvement_count == 0
        assert callback.eval_count == 1
    
    @patch('evo.models.training.callbacks.EarlyStoppingCallback.logger')
    def test_on_step_with_eval_no_improvement(self, mock_logger, callback):
        """Test on_step when evaluation shows no improvement."""
        callback.num_timesteps = 100  # At eval_freq
        callback.best_score = 0.6
        callback.verbose = 1  # Ensure verbose is set
        mock_logger.name_to_value = {"eval_mean_reward": 0.5}  # Worse than best
        callback.no_improvement_count = 2  # Close to patience limit
        callback.eval_count = 2  # Above min_evals
        
        result = callback._on_step()
        
        assert result is False  # Should stop training
    
    @patch('evo.models.training.callbacks.EarlyStoppingCallback.logger')
    def test_on_step_with_eval_no_improvement_but_below_min_evals(self, mock_logger, callback):
        """Test on_step when evaluation shows no improvement but below min_evals."""
        callback.num_timesteps = 100  # At eval_freq
        callback.best_score = 0.6
        callback.verbose = 1  # Ensure verbose is set
        mock_logger.name_to_value = {"eval_mean_reward": 0.5}  # Worse than best
        callback.no_improvement_count = 3  # At patience limit
        callback.eval_count = 0  # Below min_evals (will become 1 after _on_step)
        
        result = callback._on_step()
        
        assert result is True  # Should continue training because below min_evals
    
    @patch('evo.models.training.callbacks.EarlyStoppingCallback.logger')
    def test_on_step_with_metric_not_found(self, mock_logger, callback):
        """Test on_step when metric is not found in logger."""
        callback.num_timesteps = 100  # At eval_freq
        callback.best_score = 0.5
        callback.verbose = 1  # Ensure verbose is set
        mock_logger.name_to_value = {}  # Metric not found
        
        result = callback._on_step()
        
        assert result is True  # Should continue training
        assert callback.best_score == 0.5  # Should not change
        assert callback.no_improvement_count == 1  # Should increment


class TestModelCheckpointCallback:
    """Test the model checkpoint callback."""
    
    @pytest.fixture
    def temp_save_dir(self, temp_dir):
        """Create a temporary save directory."""
        save_dir = temp_dir / "models"
        save_dir.mkdir()
        return save_dir
    
    @pytest.fixture
    def callback(self, temp_save_dir):
        """Create a model checkpoint callback for testing."""
        with patch('stable_baselines3.common.callbacks.BaseCallback.__init__'):
            callback = ModelCheckpointCallback(
                save_freq=100,
                save_path=str(temp_save_dir),
                name_prefix="test_model",
                save_best_only=True,
                metric="eval_mean_reward",
                verbose=1
            )
            # Mock the model properly
            mock_model = Mock()
            callback.model = mock_model
            callback.verbose = 1  # Manually set verbose since __init__ is patched
            return callback
    
    def test_init(self, temp_save_dir):
        """Test callback initialization."""
        with patch('stable_baselines3.common.callbacks.BaseCallback.__init__'):
            callback = ModelCheckpointCallback(
                save_freq=100,
                save_path=str(temp_save_dir),
                name_prefix="test_model",
                save_best_only=True,
                metric="eval_mean_reward",
                verbose=1
            )
            
            assert callback.save_freq == 100
            assert callback.save_path == str(temp_save_dir)
            assert callback.name_prefix == "test_model"
            assert callback.save_best_only is True
            assert callback.metric == "eval_mean_reward"
    
    @patch('evo.models.training.callbacks.ModelCheckpointCallback.logger')
    def test_on_step_no_save(self, mock_logger, callback):
        """Test on_step when no save is due."""
        callback.num_timesteps = 50  # Not at save_freq
        mock_logger.name_to_value = {}
        
        result = callback._on_step()
        
        assert result is True  # Should continue training
        callback.model.save.assert_not_called()
    
    @patch('evo.models.training.callbacks.ModelCheckpointCallback.logger')
    def test_on_step_save_always(self, mock_logger, callback, temp_save_dir):
        """Test on_step when save is due and save_best_only=False."""
        callback.save_best_only = False
        callback.num_timesteps = 100  # At save_freq
        mock_logger.name_to_value = {}
        
        result = callback._on_step()
        
        assert result is True  # Should continue training
        callback.model.save.assert_called_once()
    
    @patch('evo.models.training.callbacks.ModelCheckpointCallback.logger')
    def test_on_step_save_best_only_improving(self, mock_logger, callback, temp_save_dir):
        """Test on_step when save is due and model is improving."""
        callback.save_best_only = True
        callback.num_timesteps = 100  # At save_freq
        callback.best_score = 0.5
        mock_logger.name_to_value = {"eval_mean_reward": 0.6}  # Better than best
        
        result = callback._on_step()
        
        assert result is True  # Should continue training
        callback.model.save.assert_called_once()  # Should save improved model
        assert callback.best_score == 0.6
    
    @patch('evo.models.training.callbacks.ModelCheckpointCallback.logger')
    def test_on_step_save_best_only_not_improving(self, mock_logger, callback, temp_save_dir):
        """Test on_step when save is due but model is not improving."""
        callback.save_best_only = True
        callback.num_timesteps = 100  # At save_freq
        callback.best_score = 0.6
        mock_logger.name_to_value = {"eval_mean_reward": 0.5}  # Worse than best
        
        result = callback._on_step()
        
        assert result is True  # Should continue training
        callback.model.save.assert_not_called()  # Should not save


class TestTrainer:
    """Test the main trainer class."""
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """Create sample data file for testing."""
        data = pd.DataFrame({
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [99 + i for i in range(100)],
            'close': [102 + i for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })
        
        data_path = temp_dir / "test_data.csv"
        data.to_csv(data_path, index=False)
        return str(data_path)
    
    @pytest.fixture
    def trainer(self, sample_data, temp_dir):
        """Create a trainer instance for testing."""
        model_dir = temp_dir / "models"
        config_dir = temp_dir / "config"
        tensorboard_log = temp_dir / "tensorboard"
        
        return Trainer(
            data_path=sample_data,
            model_dir=str(model_dir),
            config_dir=str(config_dir),
            tensorboard_log=str(tensorboard_log)
        )
    
    def test_init(self, sample_data, temp_dir):
        """Test trainer initialization."""
        model_dir = temp_dir / "models"
        config_dir = temp_dir / "config"
        tensorboard_log = temp_dir / "tensorboard"
        
        trainer = Trainer(
            data_path=sample_data,
            model_dir=str(model_dir),
            config_dir=str(config_dir),
            tensorboard_log=str(tensorboard_log)
        )
        
        assert trainer.data_path == sample_data
        assert trainer.model_dir == model_dir
        assert trainer.tensorboard_log == str(tensorboard_log)
        assert trainer.hyperparam_manager is not None
        
        # Check that directories were created
        assert model_dir.exists()
        assert tensorboard_log.exists()
    
    def test_load_data(self, trainer):
        """Test data loading."""
        trainer.load_data()
        
        assert trainer.data is not None
        assert len(trainer.data) == 100
        assert list(trainer.data.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    def test_load_data_with_features(self, trainer):
        """Test data loading with specific features."""
        features = ['open', 'close']
        trainer.load_data(features=features)
        
        assert trainer.data is not None
        assert list(trainer.data.columns) == features
    
    def test_load_data_missing_features(self, trainer):
        """Test data loading with missing features raises error."""
        with pytest.raises(ValueError, match="Missing features in data"):
            trainer.load_data(features=['open', 'nonexistent'])
    
    def test_create_environments(self, trainer):
        """Test environment creation."""
        trainer.load_data()
        
        hyperparams = TrainingHyperparameters(
            seq_len=10,
            max_episode_steps=50
        )
        
        trainer.create_environments(hyperparams, train_split=0.8)
        
        assert trainer.train_env is not None
        assert trainer.eval_env is not None
        
        # Check that environments have correct data splits
        assert len(trainer.train_env.envs[0].env.data) == 80  # 80% of 100
        assert len(trainer.eval_env.envs[0].env.data) == 20   # 20% of 100
    
    def test_create_agent(self, trainer):
        """Test agent creation."""
        trainer.load_data()
        # Use a smaller seq_len to ensure eval set is large enough
        trainer.create_environments(TrainingHyperparameters(seq_len=5))
        
        with patch('evo.models.agents.ppo_agent.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            trainer.create_agent(TrainingHyperparameters(seq_len=5))
            
            assert trainer.agent is not None
            assert isinstance(trainer.agent, PPOAgent)
    
    def test_create_callbacks(self, trainer):
        """Test callback creation."""
        hyperparams = TrainingHyperparameters(
            eval_freq=1000,
            save_freq=5000,
            entropy_coef_init=0.01,
            entropy_coef_final=0.001,
            seq_len=2
        )
        trainer.load_data()
        trainer.create_environments(hyperparams)
        callbacks = trainer.create_callbacks(hyperparams, "test_model")
        assert len(callbacks) > 0
        # Check that we have the expected callback types
        callback_types = [type(callback) for callback in callbacks]
        assert EntropyAnnealingCallback in callback_types
    
    @patch('evo.models.agents.PPOAgent')
    @pytest.mark.slow
    def test_train(self, mock_ppo_agent, trainer):
        """Test training process."""
        # Setup mocks
        mock_agent = Mock()
        mock_ppo_agent.return_value = mock_agent
        
        trainer.load_data()
        # Use a smaller seq_len to ensure eval set is large enough
        minimal_hyperparams = TrainingHyperparameters(
            total_timesteps=1,
            batch_size=2,  # <-- change this to 2
            n_epochs=1,
            seq_len=1,
            max_episode_steps=1
        )
        trainer.create_environments(minimal_hyperparams)
        trainer.create_agent(minimal_hyperparams)
        
        # Run training with minimal hyperparams
        results = trainer.train("test_model", hyperparams=minimal_hyperparams)
        
        # Check that training was called
        mock_agent.learn.assert_called_once()
        
        # Check results
        assert 'model_path' in results
        assert 'training_time' in results
        assert 'final_reward' in results
    
    @pytest.mark.slow
    def test_load_trained_model(self, trainer, temp_dir):
        """Test loading a trained model."""
        # Create the expected directory and file
        model_dir = temp_dir / "models" / "test_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "final_model.zip"
        model_file.touch()  # Create an empty file to simulate the saved model

        trainer.load_data()  # Ensure data is loaded for the environment

        with patch('evo.models.agents.ppo_agent.PPO.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            agent = trainer.load_trained_model("test_model")

            assert agent is not None
            assert isinstance(agent, PPOAgent)
            mock_load.assert_called_once()
    
    @pytest.mark.slow
    def test_evaluate_model(self, trainer):
        trainer.load_data()
        trainer.create_environments(TrainingHyperparameters(seq_len=5, max_episode_steps=1))
        
        with patch('evo.models.agents.ppo_agent.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            trainer.create_agent(TrainingHyperparameters())
            trainer.agent.is_trained = True
            
            # Mock the evaluation environment
            mock_env = Mock()
            mock_env.step.return_value = (np.random.randn(10, 5), 0.1, True, False, {})
            trainer.eval_env = mock_env
            
            # Mock agent prediction
            trainer.agent.predict = Mock(return_value=(1, None))
            results = trainer.evaluate_model(trainer.agent, n_episodes=1)
            
            assert 'mean_reward' in results
            assert 'std_reward' in results
            assert 'min_reward' in results
            assert 'max_reward' in results
            assert results['n_episodes'] == 1 