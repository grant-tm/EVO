"""
Integration tests for the models module.

This module contains integration tests that test the interaction between
different components of the models module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from evo.models.agents.ppo_agent import PPOAgent
from evo.models.environments.trading_env import TradingEnv
from evo.models.training.trainer import Trainer
from evo.models.training.hyperparameters import TrainingHyperparameters

pytestmark = [
    pytest.mark.integration,
    pytest.mark.models,
    pytest.mark.slow
]


class TestModelsIntegration:
    """Integration tests for the models module components."""
    
    @pytest.fixture
    def large_sample_data(self):
        """Create larger sample data for integration testing."""
        np.random.seed(42)
        n_points = 200
        base_price = 100
        
        data = []
        for i in range(n_points):
            price_change = np.random.normal(0, 0.01)
            base_price *= (1 + price_change)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.005)))
            low = base_price * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.uniform(1000, 2000))
            
            data.append({
                'open': base_price,
                'high': high,
                'low': low,
                'close': base_price * (1 + np.random.normal(0, 0.002)),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_file(self, large_sample_data, temp_dir):
        """Create a sample data file for testing."""
        data_path = temp_dir / "integration_test_data.csv"
        large_sample_data.to_csv(data_path, index=False)
        return str(data_path)
    
    def test_agent_environment_compatibility(self, large_sample_data):
        """Test that PPO agent works correctly with trading environment."""
        # Create environment
        env = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=50
        )
        
        # Create agent
        with patch('evo.models.agents.ppo_agent.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            agent = PPOAgent(env, policy="MlpPolicy")
            
            # Test that agent can get spaces from environment
            assert agent.get_action_space() == env.action_space
            assert agent.get_observation_space() == env.observation_space
            
            # Test that agent can handle environment observations
            obs, _ = env.reset()
            assert obs.shape == env.observation_space.shape
            
            # Mock prediction on the model itself
            agent.is_trained = True
            # Mock the predict method on the actual model instance
            mock_model.predict.return_value = (1, None)
            
            action, info = agent.predict(obs)
            assert action in [0, 1, 2]  # Valid actions
            # Verify that the mock was called correctly
            mock_model.predict.assert_called_once()
            # The action should match the mocked return value
            assert action == 1  # Should match mocked return value
    
    def test_environment_episode_simulation(self, large_sample_data):
        """Test complete episode simulation with environment."""
        env = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=30
        )
        
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        
        # Simulate a trading episode
        while step_count < 30:
            # Simple strategy: alternate between actions
            action = step_count % 3  # 0=Hold, 1=Buy, 2=Sell
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Check that episode completed
        assert step_count > 0
        assert total_reward != 0  # Should have some reward/penalty
        
        # Check final state
        final_state = env.get_state()
        assert 'current_step' in final_state
        assert 'total_reward' in final_state
    
    def test_trainer_data_loading(self, sample_data_file, temp_dir):
        """Test trainer data loading functionality."""
        model_dir = temp_dir / "models"
        config_dir = temp_dir / "config"
        tensorboard_log = temp_dir / "tensorboard"
        
        trainer = Trainer(
            data_path=sample_data_file,
            model_dir=str(model_dir),
            config_dir=str(config_dir),
            tensorboard_log=str(tensorboard_log)
        )
        
        # Test data loading
        trainer.load_data()
        assert trainer.data is not None
        assert len(trainer.data) == 200
        
        # Test data loading with specific features
        trainer.load_data(features=['open', 'close'])
        assert list(trainer.data.columns) == ['open', 'close']
    
    def test_trainer_environment_creation(self, sample_data_file, temp_dir):
        """Test trainer environment creation."""
        model_dir = temp_dir / "models"
        config_dir = temp_dir / "config"
        tensorboard_log = temp_dir / "tensorboard"
        
        trainer = Trainer(
            data_path=sample_data_file,
            model_dir=str(model_dir),
            config_dir=str(config_dir),
            tensorboard_log=str(tensorboard_log)
        )
        
        trainer.load_data()
        
        hyperparams = TrainingHyperparameters(
            seq_len=10,
            max_episode_steps=30
        )
        
        trainer.create_environments(hyperparams, train_split=0.8)
        
        assert trainer.train_env is not None
        assert trainer.eval_env is not None
        
        # Check that environments have correct data splits
        # Access data through the underlying environment in the vectorized wrapper
        assert len(trainer.train_env.venv.envs[0].env.data) == 160  # 80% of 200
        assert len(trainer.eval_env.venv.envs[0].env.data) == 40    # 20% of 200
        
        # Check that environments have correct parameters
        assert trainer.train_env.venv.envs[0].env.seq_len == 10
        assert trainer.train_env.venv.envs[0].env.max_episode_steps == 30
        assert trainer.eval_env.venv.envs[0].env.seq_len == 10
        assert trainer.eval_env.venv.envs[0].env.max_episode_steps == 30
    
    def test_trainer_agent_creation(self, sample_data_file, temp_dir):
        """Test trainer agent creation."""
        model_dir = temp_dir / "models"
        config_dir = temp_dir / "config"
        tensorboard_log = temp_dir / "tensorboard"
        
        trainer = Trainer(
            data_path=sample_data_file,
            model_dir=str(model_dir),
            config_dir=str(config_dir),
            tensorboard_log=str(tensorboard_log)
        )
        
        trainer.load_data()
        
        # Use smaller seq_len for testing to ensure evaluation data is sufficient
        hyperparams = TrainingHyperparameters(seq_len=5, max_episode_steps=30)
        trainer.create_environments(hyperparams)
        
        with patch('evo.models.agents.ppo_agent.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            trainer.create_agent(hyperparams)
            
            assert trainer.agent is not None
            assert isinstance(trainer.agent, PPOAgent)
            
            # Check that agent has correct environment
            assert trainer.agent.env == trainer.train_env
    
    def test_hyperparameter_validation(self):
        """Test hyperparameter validation in real scenarios."""
        # Test valid hyperparameters
        valid_params = TrainingHyperparameters(
            total_timesteps=1000,
            batch_size=32,
            learning_rate=1e-4,
            gamma=0.99,
            ent_coef=0.01,
            seq_len=10,
            tp_pct=0.02,
            sl_pct=0.01,
            max_episode_steps=50
        )
        
        errors = valid_params.validate()
        assert len(errors) == 0
        
        # Test invalid hyperparameters
        invalid_params = TrainingHyperparameters(
            total_timesteps=0,      # Invalid
            batch_size=-1,          # Invalid
            learning_rate=0,        # Invalid
            gamma=1.5,              # Invalid
            ent_coef=-0.1,          # Invalid
            seq_len=0,              # Invalid
            tp_pct=-0.01,           # Invalid
            max_episode_steps=0     # Invalid
        )
        
        errors = invalid_params.validate()
        assert len(errors) > 0
        
        # Check specific error messages
        error_messages = [error.lower() for error in errors]
        assert any('total_timesteps' in msg for msg in error_messages)
        assert any('batch_size' in msg for msg in error_messages)
        assert any('learning_rate' in msg for msg in error_messages)
        assert any('gamma' in msg for msg in error_messages)
    
    def test_hyperparameter_serialization(self):
        """Test hyperparameter serialization and deserialization."""
        original_params = TrainingHyperparameters(
            total_timesteps=5000,
            batch_size=128,
            learning_rate=1e-4,
            policy='MlpPolicy',
            seq_len=15,
            tp_pct=0.03,
            sl_pct=0.015
        )
        
        # Convert to dictionary
        config_dict = original_params.to_dict()
        
        # Convert back to hyperparameters
        restored_params = TrainingHyperparameters.from_dict(config_dict)
        
        # Check that all values are preserved
        assert restored_params.total_timesteps == original_params.total_timesteps
        assert restored_params.batch_size == original_params.batch_size
        assert restored_params.learning_rate == original_params.learning_rate
        assert restored_params.policy == original_params.policy
        assert restored_params.seq_len == original_params.seq_len
        assert restored_params.tp_pct == original_params.tp_pct
        assert restored_params.sl_pct == original_params.sl_pct
    
    def test_environment_state_management(self, large_sample_data):
        """Test environment state saving and restoration."""
        env = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=30
        )
        
        # Reset and take some actions
        obs, _ = env.reset()
        env.step(1)  # Buy
        env.step(0)  # Hold
        env.step(0)  # Hold
        
        # Save state
        state = env.get_state()
        
        # Take more actions
        env.step(2)  # Sell
        env.step(0)  # Hold
        
        # Restore state
        env.set_state(state)
        
        # Check that environment is back to saved state
        restored_state = env.get_state()
        assert restored_state == state
        
        # Check specific values
        assert env.position == state['position']
        assert env.current_step == state['current_step']
        assert env.total_reward == state['total_reward']
    
    def test_agent_config_management(self, large_sample_data):
        """Test agent configuration management."""
        env = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=30
        )
        
        with patch('stable_baselines3.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            agent = PPOAgent(
                env,
                policy="MlpPolicy",
                learning_rate=1e-4,
                batch_size=128,
                n_steps=1024,
                gamma=0.95,
                ent_coef=0.02
            )
            
            # Get configuration
            config = agent.get_config()
            
            # Check that config contains expected keys
            expected_keys = [
                'policy', 'learning_rate', 'n_steps', 'batch_size',
                'n_epochs', 'gamma', 'gae_lambda', 'clip_range',
                'ent_coef', 'vf_coef', 'max_grad_norm'
            ]
            
            for key in expected_keys:
                assert key in config
            
            # Check specific values
            assert config['policy'] == "MlpPolicy"
            assert config['learning_rate'] == 1e-4
            assert config['batch_size'] == 128
            assert config['n_steps'] == 1024
            assert config['gamma'] == 0.95
            assert config['ent_coef'] == 0.02
            
            # Check that config is a copy
            config['test_key'] = 'test_value'
            assert 'test_key' not in agent.config
    
    def test_end_to_end_training_simulation(self, sample_data_file, temp_dir):
        """Test end-to-end training simulation (mocked)."""
        model_dir = temp_dir / "models"
        config_dir = temp_dir / "config"
        tensorboard_log = temp_dir / "tensorboard"
        
        trainer = Trainer(
            data_path=sample_data_file,
            model_dir=str(model_dir),
            config_dir=str(config_dir),
            tensorboard_log=str(tensorboard_log)
        )
        
        # Setup training
        trainer.load_data()
        
        # Use smaller seq_len for testing to ensure evaluation data is sufficient
        hyperparams = TrainingHyperparameters(seq_len=5, max_episode_steps=30)
        trainer.create_environments(hyperparams)
        
        with patch('evo.models.agents.ppo_agent.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            trainer.create_agent(hyperparams)
            
            # Mock the training process
            mock_model.learn.return_value = None
            
            # Run training with the same hyperparameters
            results = trainer.train("test_model", hyperparams=hyperparams)
            
            # Check that training was called
            mock_model.learn.assert_called_once()
            
            # Check results
            assert 'model_path' in results
            assert 'model_name' in results
            assert 'hyperparameters' in results
            assert 'total_timesteps' in results
            
            # Check that model directory was created
            assert model_dir.exists()
    
    def test_model_persistence(self, large_sample_data, temp_dir):
        """Test model saving and loading."""
        env = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=30
        )
        
        with patch('evo.models.agents.ppo_agent.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            agent = PPOAgent(env, policy="MlpPolicy")
            agent.is_trained = True
            
            # Test saving
            save_path = temp_dir / "test_model.zip"
            agent.save(str(save_path))
            
            mock_model.save.assert_called_once_with(str(save_path))
            
            # Test loading - mock PPO.load to return the same mock model
            mock_ppo.load.return_value = mock_model
            load_path = temp_dir / "test_model.zip"
            agent.load(str(load_path))
            
            # Verify PPO.load was called with the correct path
            mock_ppo.load.assert_called_once_with(str(load_path))
            assert agent.is_trained is True
    
    def test_environment_reproducibility(self, large_sample_data):
        """Test that environment behavior is reproducible with seeds."""
        env1 = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=30
        )
        
        env2 = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=30
        )
        
        # Reset both environments with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Observations should be identical
        np.testing.assert_array_equal(obs1, obs2)
        
        # Take same actions and check results
        for i in range(5):
            action = i % 3
            
            obs1, reward1, terminated1, truncated1, info1 = env1.step(action)
            obs2, reward2, terminated2, truncated2, info2 = env2.step(action)
            
            # Results should be identical
            np.testing.assert_array_equal(obs1, obs2)
            assert reward1 == reward2
            assert terminated1 == terminated2
            assert truncated1 == truncated2
            assert info1 == info2 