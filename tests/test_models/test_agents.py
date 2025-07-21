"""
Tests for trading agents.

This module contains tests for the agent implementations.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from evo.models.agents.base_agent import BaseAgent
from evo.models.agents.ppo_agent import PPOAgent
from evo.models.environments.trading_env import TradingEnv

pytestmark = [
    pytest.mark.unit,
    pytest.mark.models,
    pytest.mark.agents
]


class TestBaseAgent:
    """Test the abstract base agent."""
    
    def test_abstract_methods(self):
        """Test that BaseAgent cannot be instantiated directly."""
        mock_env = Mock()
        with pytest.raises(TypeError):
            BaseAgent(mock_env)
    
    def test_abstract_methods_required(self):
        """Test that all abstract methods are properly defined."""
        # Check that BaseAgent has the required abstract methods
        assert hasattr(BaseAgent, 'predict')
        assert hasattr(BaseAgent, 'learn')
        assert hasattr(BaseAgent, 'save')
        assert hasattr(BaseAgent, 'load')


class TestPPOAgent:
    """Test the PPO agent implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
    
    @pytest.fixture
    def trading_env(self, sample_data):
        """Create a trading environment for testing."""
        return TradingEnv(
            data=sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=5,
            max_episode_steps=10
        )
    
    @pytest.fixture
    def ppo_agent(self, trading_env):
        """Create a PPO agent for testing."""
        with patch('stable_baselines3.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            # Create agent with _init_setup_model=False to prevent automatic model setup
            agent = PPOAgent(trading_env, policy="MlpPolicy", _init_setup_model=False)
            # Manually set the mock model
            agent.model = mock_model
            # Set up the required attributes that would normally be set by _setup_model
            agent._original_env = trading_env
            agent._vec_env = Mock()
            return agent
    
    def test_init_default_parameters(self, trading_env):
        """Test PPO agent initialization with default parameters."""
        with patch('stable_baselines3.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            agent = PPOAgent(trading_env, _init_setup_model=False)
            
            assert agent.env == trading_env
            assert agent.policy == "MlpPolicy"
            assert agent.learning_rate == 3e-4
            assert agent.n_steps == 512
            assert agent.batch_size == 64
            assert agent.n_epochs == 10
            assert agent.gamma == 0.99
            assert agent.gae_lambda == 0.95
            assert agent.clip_range == 0.2
            assert agent.ent_coef == 0.01
            assert agent.vf_coef == 0.5
            assert agent.max_grad_norm == 0.5
            assert agent.is_trained is False
    
    def test_init_custom_parameters(self, trading_env):
        """Test PPO agent initialization with custom parameters."""
        with patch('stable_baselines3.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            custom_params = {
                'policy': 'MlpPolicy',
                'learning_rate': 1e-4,
                'n_steps': 1024,
                'batch_size': 128,
                'n_epochs': 20,
                'gamma': 0.95,
                'ent_coef': 0.02,
                '_init_setup_model': False
            }
            
            agent = PPOAgent(trading_env, **custom_params)
            
            assert agent.policy == 'MlpPolicy'
            assert agent.learning_rate == 1e-4
            assert agent.n_steps == 1024
            assert agent.batch_size == 128
            assert agent.n_epochs == 20
            assert agent.gamma == 0.95
            assert agent.ent_coef == 0.02
    
    def test_predict_not_trained(self, ppo_agent):
        """Test prediction when agent is not trained."""
        observation = np.random.randn(5, 5).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="Agent must be trained before making predictions"):
            ppo_agent.predict(observation)
    
    def test_predict_trained(self, ppo_agent):
        """Test prediction when agent is trained."""
        ppo_agent.is_trained = True
        observation = np.random.randn(5, 5).astype(np.float32)
        
        # Mock the model's predict method
        ppo_agent.model.predict.return_value = (1, None)
        
        action, info = ppo_agent.predict(observation)
        
        assert action == 1
        assert info == {"states": None}
        ppo_agent.model.predict.assert_called_once_with(observation, deterministic=True)
    
    def test_predict_deterministic_false(self, ppo_agent):
        """Test prediction with deterministic=False."""
        ppo_agent.is_trained = True
        observation = np.random.randn(5, 5).astype(np.float32)
        
        # Mock the model's predict method
        ppo_agent.model.predict.return_value = (2, {'log_prob': 0.5})
        
        action, info = ppo_agent.predict(observation, deterministic=False)
        
        assert action == 2
        assert info == {"states": {'log_prob': 0.5}}
        ppo_agent.model.predict.assert_called_once_with(observation, deterministic=False)
    
    def test_learn(self, ppo_agent):
        """Test agent learning."""
        total_timesteps = 1000
        mock_callback = Mock()
        
        ppo_agent.learn(total_timesteps, callback=mock_callback)
        
        ppo_agent.model.learn.assert_called_once_with(
            total_timesteps=total_timesteps,
            callback=mock_callback
        )
        assert ppo_agent.is_trained is True
    
    def test_save(self, ppo_agent, temp_dir):
        """Test saving the agent."""
        save_path = temp_dir / "test_model.zip"
        
        ppo_agent.save(str(save_path))
        
        ppo_agent.model.save.assert_called_once_with(str(save_path))
    
    def test_load(self, ppo_agent, temp_dir):
        """Test loading the agent."""
        load_path = temp_dir / "test_model.zip"
        
        with patch('stable_baselines3.PPO.load') as mock_ppo_load:
            mock_model = Mock()
            mock_ppo_load.return_value = mock_model
            
            ppo_agent.load(str(load_path))
            
            mock_ppo_load.assert_called_once_with(str(load_path))
            assert ppo_agent.model == mock_model
            assert ppo_agent.is_trained is True
    
    def test_get_model(self, ppo_agent):
        """Test getting the underlying model."""
        model = ppo_agent.get_model()
        assert model == ppo_agent.model
    
    def test_set_entropy_coefficient(self, ppo_agent):
        """Test setting entropy coefficient."""
        new_ent_coef = 0.02
        ppo_agent.set_entropy_coefficient(new_ent_coef)
        
        assert ppo_agent.ent_coef == new_ent_coef
        # Check that the model's entropy coefficient was set correctly
        assert ppo_agent.model.ent_coef == new_ent_coef
    
    def test_get_entropy_coefficient(self, ppo_agent):
        """Test getting entropy coefficient."""
        # Set the mock model's entropy coefficient to match the agent's
        ppo_agent.model.ent_coef = ppo_agent.ent_coef
        ent_coef = ppo_agent.get_entropy_coefficient()
        assert ent_coef == ppo_agent.ent_coef
    
    def test_get_action_space(self, ppo_agent):
        """Test getting action space."""
        action_space = ppo_agent.get_action_space()
        assert action_space == ppo_agent.env.action_space
    
    def test_get_observation_space(self, ppo_agent):
        """Test getting observation space."""
        observation_space = ppo_agent.get_observation_space()
        assert observation_space == ppo_agent.env.observation_space
    
    def test_is_ready(self, ppo_agent):
        """Test is_ready method."""
        # Initially not ready
        assert ppo_agent.is_ready() is False
        
        # After training
        ppo_agent.is_trained = True
        assert ppo_agent.is_ready() is True
    
    def test_get_config(self, ppo_agent):
        """Test getting agent configuration."""
        config = ppo_agent.get_config()
        
        # Check that config contains expected keys
        expected_keys = ['policy', 'learning_rate', 'n_steps', 'batch_size', 
                        'n_epochs', 'gamma', 'gae_lambda', 'clip_range', 
                        'ent_coef', 'vf_coef', 'max_grad_norm']
        
        for key in expected_keys:
            assert key in config
        
        # Check that config is a copy
        config['test_key'] = 'test_value'
        assert 'test_key' not in ppo_agent.config


class TestPPOAgentIntegration:
    """Integration tests for PPO agent with real environment."""
    
    @pytest.fixture
    def large_sample_data(self):
        """Create larger sample data for integration testing."""
        np.random.seed(42)
        n_points = 100
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
    
    @pytest.mark.integration
    def test_agent_environment_compatibility(self, large_sample_data):
        """Test that PPO agent works correctly with trading environment."""
        env = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=50
        )
        
        with patch('stable_baselines3.PPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            agent = PPOAgent(env, policy="MlpPolicy", _init_setup_model=False)
            # Set up the required attributes that would normally be set by _setup_model
            agent._original_env = env
            agent._vec_env = Mock()
            agent.model = mock_model  # Set the model attribute manually
            
            # Test that agent can get spaces from environment
            assert agent.get_action_space() == env.action_space
            assert agent.get_observation_space() == env.observation_space
            
            # Test that agent can handle environment observations
            obs, _ = env.reset()
            assert obs.shape == env.observation_space.shape
            
            # Mock prediction
            agent.is_trained = True
            agent.model.predict.return_value = (1, None)
            
            action, info = agent.predict(obs)
            assert action in [0, 1, 2]  # Valid actions
            assert action == 1  # Should match mocked return value
            assert info == {"states": None} 