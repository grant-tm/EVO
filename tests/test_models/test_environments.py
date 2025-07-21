"""
Tests for trading environments.

This module contains tests for the environment implementations.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from evo.models.environments.base_env import BaseEnv
from evo.models.environments.trading_env import TradingEnv

pytestmark = [
    pytest.mark.unit,
    pytest.mark.models,
    pytest.mark.environments
]


class TestBaseEnv:
    """Test the abstract base environment."""
    
    def test_abstract_methods(self):
        """Test that BaseEnv cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEnv()
    
    def test_abstract_methods_required(self):
        """Test that all abstract methods are properly defined."""
        # Check that BaseEnv has the required abstract methods
        assert hasattr(BaseEnv, 'reset')
        assert hasattr(BaseEnv, 'step')
        assert hasattr(BaseEnv, 'render')
        assert hasattr(BaseEnv, 'close')


class TestTradingEnv:
    """Test the trading environment implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]
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
    
    def test_init_default_parameters(self, sample_data):
        """Test trading environment initialization with default parameters."""
        env = TradingEnv(data=sample_data)
        
        assert env.data.equals(sample_data)
        assert env.features == ['open', 'high', 'low', 'close', 'volume']
        assert env.seq_len == 20
        assert env.tp_pct == 0.02
        assert env.sl_pct == 0.01
        assert env.idle_penalty == 0.001
        assert env.sl_penalty_coef == 1.0
        assert env.tp_reward_coef == 1.0
        assert env.timeout_duration == 100
        assert env.timeout_reward_coef == 0.5
        assert env.ongoing_reward_coef == 0.1
        assert env.reward_clip_range == (-1.0, 1.0)
        assert env.max_episode_steps == 1000
        
        # Check action and observation spaces
        assert env.action_space.n == 3  # Hold, Buy, Sell
        assert env.observation_space.shape == (20, 5)  # seq_len, n_features
    
    def test_init_custom_parameters(self, sample_data):
        """Test trading environment initialization with custom parameters."""
        custom_params = {
            'features': ['open', 'close'],
            'seq_len': 10,
            'tp_pct': 0.03,
            'sl_pct': 0.015,
            'idle_penalty': 0.002,
            'max_episode_steps': 50
        }
        
        env = TradingEnv(data=sample_data, **custom_params)
        
        assert env.features == ['open', 'close']
        assert env.seq_len == 10
        assert env.tp_pct == 0.03
        assert env.sl_pct == 0.015
        assert env.idle_penalty == 0.002
        assert env.max_episode_steps == 50
        
        # Check that observation space matches custom parameters
        assert env.observation_space.shape == (10, 2)  # seq_len, n_features
    
    def test_init_empty_data(self):
        """Test initialization with empty data raises error."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            TradingEnv(data=empty_df)
    
    def test_init_missing_features(self, sample_data):
        """Test initialization with missing features raises error."""
        with pytest.raises(ValueError, match="Missing features in data"):
            TradingEnv(data=sample_data, features=['open', 'nonexistent'])
    
    def test_reset(self, trading_env):
        """Test environment reset."""
        obs, info = trading_env.reset()
        
        # Check observation shape
        assert obs.shape == trading_env.observation_space.shape
        assert obs.dtype == np.float32
        
        # Check that environment state is reset
        assert trading_env.episode_step_count == 0
        assert trading_env.position == 0  # No position
        assert trading_env.entry_price is None
        assert trading_env.entry_index is None
        assert trading_env.total_reward == 0
        assert trading_env.index == trading_env.seq_len
        
        # Check info
        assert 'step_count' in info
        assert 'position' in info
        assert 'total_reward' in info
    
    def test_reset_with_seed(self, trading_env):
        """Test environment reset with seed for reproducibility."""
        obs1, _ = trading_env.reset(seed=42)
        obs2, _ = trading_env.reset(seed=42)
        
        # With same seed, observations should be identical
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_step_hold_action(self, trading_env):
        """Test step with hold action."""
        obs, _ = trading_env.reset()
        
        # Take hold action
        next_obs, reward, terminated, truncated, info = trading_env.step(0)
        
        # Check observation
        assert next_obs.shape == trading_env.observation_space.shape
        assert next_obs.dtype == np.float32
        
        # Check reward (should be idle penalty)
        assert reward == -trading_env.idle_penalty
        
        # Check termination
        assert not terminated
        assert not truncated
        
        # Check info
        assert 'position' in info
        assert 'total_reward' in info
    
    def test_step_buy_action(self, trading_env):
        """Test step with buy action."""
        obs, _ = trading_env.reset()
        
        # Take buy action
        next_obs, reward, terminated, truncated, info = trading_env.step(1)
        
        # Check that position is opened
        assert trading_env.position == 1
        assert trading_env.entry_price > 0
        assert trading_env.entry_index == 0
        assert trading_env.episode_step_count == 1
        
        # Check reward (should be ongoing reward)
        assert reward == trading_env.ongoing_reward_coef
        
        # Check info
        assert info['action'] == 1
        assert info['position'] == 1
        assert 'entry_price' in info
    
    def test_step_sell_action(self, trading_env):
        """Test step with sell action."""
        obs, _ = trading_env.reset()
        
        # Take sell action
        next_obs, reward, terminated, truncated, info = trading_env.step(2)
        
        # Check that position is opened (short)
        assert trading_env.position == -1
        assert trading_env.entry_price > 0
        assert trading_env.entry_index == 0
        assert trading_env.episode_step_count == 1
        
        # Check reward (should be ongoing reward)
        assert reward == trading_env.ongoing_reward_coef
        
        # Check info
        assert info['action'] == 2
        assert info['position'] == -1
        assert 'entry_price' in info
    
    def test_step_invalid_action(self, trading_env):
        """Test step with invalid action raises error."""
        trading_env.reset()
        
        with pytest.raises(ValueError):
            trading_env.step(3)  # Invalid action
    
    def test_position_management(self, trading_env):
        """Test position opening and closing logic."""
        obs, _ = trading_env.reset()
        
        # Open long position
        trading_env.step(1)
        assert trading_env.position == 1
        assert trading_env.entry_price > 0
        
        # Try to open another long position (should close current one and reopen)
        trading_env.step(1)
        assert trading_env.position == 1  # Still long, but entry price updated
        
        # Switch from long to short position with sell
        trading_env.step(2)
        assert trading_env.position == -1  # Now short position
        assert trading_env.entry_price > 0  # New entry price for short position
        assert trading_env.entry_index is not None  # New entry index for short position
    
    def test_take_profit_logic(self, trading_env):
        """Test take profit logic."""
        # Create data with clear upward trend
        trend_data = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [99] * 20,
            'close': [100 + i * 0.5 for i in range(20)],  # Steady increase
            'volume': [1000] * 20
        })
        
        env = TradingEnv(
            data=trend_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=5,
            tp_pct=0.02,  # 2% take profit
            max_episode_steps=20
        )
        
        obs, _ = env.reset()
        
        # Open long position
        env.step(1)
        entry_price = env.entry_price
        
        # Simulate price increase to trigger take profit
        # Move forward until take profit is hit
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold
            
            if terminated:
                # Check that take profit was triggered
                current_price = env._get_current_price()
                expected_reward = (current_price - entry_price) / entry_price * env.tp_reward_coef
                assert reward == expected_reward
                assert info.get('exit_reason') == 'take_profit'
                break
    
    def test_stop_loss_logic(self, trading_env):
        """Test stop loss logic."""
        # Create data with clear downward trend
        trend_data = pd.DataFrame({
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [99] * 20,
            'close': [100 - i * 0.5 for i in range(20)],  # Steady decrease
            'volume': [1000] * 20
        })
        
        env = TradingEnv(
            data=trend_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=5,
            sl_pct=0.01,  # 1% stop loss
            max_episode_steps=20
        )
        
        obs, _ = env.reset()
        
        # Open long position
        env.step(1)
        entry_price = env.entry_price
        
        # Simulate price decrease to trigger stop loss
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold
            
            if terminated:
                # Check that stop loss was triggered
                current_price = env._get_current_price()
                expected_reward = (current_price - entry_price) / entry_price * env.sl_penalty_coef
                assert reward == expected_reward
                assert info.get('exit_reason') == 'stop_loss'
                break
    
    def test_timeout_logic(self, trading_env):
        """Test position timeout logic."""
        env = TradingEnv(
            data=trading_env.data,
            features=trading_env.features,
            seq_len=trading_env.seq_len,
            timeout_duration=3,  # Short timeout for testing
            max_episode_steps=10
        )
        
        obs, _ = env.reset()
        
        # Open position
        env.step(1)
        
        # Hold for timeout duration
        for i in range(3):
            obs, reward, terminated, truncated, info = env.step(0)
            
            if terminated:
                assert info.get('exit_reason') == 'timeout'
                break
    
    def test_episode_termination(self, trading_env):
        """Test episode termination conditions."""
        env = TradingEnv(
            data=trading_env.data,
            features=trading_env.features,
            seq_len=trading_env.seq_len,
            max_episode_steps=5  # Short episode for testing
        )
        
        obs, _ = env.reset()
        
        # Take steps until episode ends
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            
            if truncated:
                assert i == 4  # Should terminate after 5 steps
                break
    
    def test_reward_clipping(self, trading_env):
        """Test that rewards are properly clipped."""
        # Create environment with wide reward range
        env = TradingEnv(
            data=trading_env.data,
            features=trading_env.features,
            seq_len=trading_env.seq_len,
            reward_clip_range=(-0.5, 0.5)  # Narrow clip range
        )
        
        obs, _ = env.reset()
        
        # Take some actions and check that rewards are clipped
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            assert -0.5 <= reward <= 0.5
    
    def test_get_current_price(self, trading_env):
        """Test getting current price."""
        obs, _ = trading_env.reset()
        
        current_price = trading_env._get_current_price()
        assert isinstance(current_price, float)
        assert current_price > 0
    
    def test_get_observation(self, trading_env):
        """Test getting current observation."""
        obs, _ = trading_env.reset()
        
        observation = trading_env._get_observation()
        assert observation.shape == trading_env.observation_space.shape
        assert observation.dtype == np.float32
    
    def test_render(self, trading_env):
        """Test environment rendering."""
        obs, _ = trading_env.reset()
        
        # Test human mode
        rendered = trading_env.render(mode="human")
        assert rendered is None  # Human mode returns None
        
        # Test invalid mode
        with pytest.raises(ValueError):
            trading_env.render(mode="invalid")
    
    def test_close(self, trading_env):
        """Test environment closing."""
        # Should not raise any exception
        trading_env.close()
    
    def test_get_state(self, trading_env):
        """Test getting environment state."""
        obs, _ = trading_env.reset()
        trading_env.step(1)  # Open position
        
        state = trading_env.get_state()
        
        # Check that state contains all necessary information
        expected_keys = [
            'current_step', 'position', 'entry_price', 'entry_step',
            'total_reward', 'steps_in_position'
        ]
        
        for key in expected_keys:
            assert key in state
    
    def test_set_state(self, trading_env):
        """Test setting environment state."""
        obs, _ = trading_env.reset()
        
        # Get initial state
        initial_state = trading_env.get_state()
        
        # Modify environment
        trading_env.step(1)
        trading_env.step(0)
        
        # Set back to initial state
        trading_env.set_state(initial_state)
        
        # Check that environment is back to initial state
        current_state = trading_env.get_state()
        assert current_state == initial_state
    
    def test_get_action_space(self, trading_env):
        """Test getting action space."""
        action_space = trading_env.get_action_space()
        assert action_space == trading_env.action_space
    
    def test_get_observation_space(self, trading_env):
        """Test getting observation space."""
        observation_space = trading_env.get_observation_space()
        assert observation_space == trading_env.observation_space
    
    def test_get_config(self, trading_env):
        """Test getting environment configuration."""
        config = trading_env.get_config()
        
        # Check that config contains expected keys
        expected_keys = [
            'features', 'seq_len', 'tp_pct', 'sl_pct', 'idle_penalty',
            'sl_penalty_coef', 'tp_reward_coef', 'timeout_duration',
            'timeout_reward_coef', 'ongoing_reward_coef', 'reward_clip_range',
            'max_episode_steps'
        ]
        
        for key in expected_keys:
            assert key in config
        
        # Check that config is a copy
        config['test_key'] = 'test_value'
        assert 'test_key' not in trading_env.config


class TestTradingEnvIntegration:
    """Integration tests for trading environment."""
    
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
    def test_full_episode_simulation(self, large_sample_data):
        """Test a complete episode simulation."""
        env = TradingEnv(
            data=large_sample_data,
            features=['open', 'high', 'low', 'close', 'volume'],
            seq_len=10,
            max_episode_steps=50
        )
        
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        
        # Simulate a trading episode
        while step_count < 50:
            # Simple strategy: buy if price is rising, sell if falling
            current_price = env._get_current_price()
            
            if step_count == 0:
                action = 1  # Start with buy
            elif step_count % 10 == 0:
                action = 2  # Sell every 10 steps
            else:
                action = 0  # Hold
            
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