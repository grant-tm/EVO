"""
Trading environment for reinforcement learning agents.

This module provides a Gymnasium-compatible trading environment that
simulates trading with various reward shaping mechanisms.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple, Union

from .base_env import BaseEnv


class TradingEnv(BaseEnv):
    """
    Trading environment for reinforcement learning.
    
    This environment simulates trading with configurable reward shaping
    and supports various trading strategies.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[list] = None,
        seq_len: int = 20,
        tp_pct: float = 0.02,
        sl_pct: float = 0.01,
        idle_penalty: float = 0.001,
        sl_penalty_coef: float = 1.0,
        tp_reward_coef: float = 1.0,
        timeout_duration: int = 100,
        timeout_reward_coef: float = 0.5,
        ongoing_reward_coef: float = 0.1,
        reward_clip_range: Tuple[float, float] = (-1.0, 1.0),
        max_episode_steps: int = 1000,
        **kwargs
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame containing OHLCV data
            features: List of feature columns to use
            seq_len: Length of observation sequence
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            idle_penalty: Penalty for being idle
            sl_penalty_coef: Stop loss penalty coefficient
            tp_reward_coef: Take profit reward coefficient
            timeout_duration: Maximum steps to hold a position
            timeout_reward_coef: Timeout reward coefficient
            ongoing_reward_coef: Ongoing position reward coefficient
            reward_clip_range: Range to clip rewards
            max_episode_steps: Maximum steps per episode
            **kwargs: Additional configuration parameters
        """
        # Store configuration in config dict for base class compatibility
        config = {
            'features': features or ["open", "high", "low", "close", "volume"],
            'seq_len': seq_len,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'idle_penalty': idle_penalty,
            'sl_penalty_coef': sl_penalty_coef,
            'tp_reward_coef': tp_reward_coef,
            'timeout_duration': timeout_duration,
            'timeout_reward_coef': timeout_reward_coef,
            'ongoing_reward_coef': ongoing_reward_coef,
            'reward_clip_range': reward_clip_range,
            'max_episode_steps': max_episode_steps,
            **kwargs
        }
        super().__init__(**config)
        
        # Store data and configuration
        self.data = data.reset_index(drop=True)
        self.features = self.config['features']
        self.seq_len = self.config['seq_len']
        self.tp_pct = self.config['tp_pct']
        self.sl_pct = self.config['sl_pct']
        self.idle_penalty = self.config['idle_penalty']
        self.sl_penalty_coef = self.config['sl_penalty_coef']
        self.tp_reward_coef = self.config['tp_reward_coef']
        self.timeout_duration = self.config['timeout_duration']
        self.timeout_reward_coef = self.config['timeout_reward_coef']
        self.ongoing_reward_coef = self.config['ongoing_reward_coef']
        self.reward_clip_range = self.config['reward_clip_range']
        self.max_episode_steps = self.config['max_episode_steps']
        
        # Validate data
        self._validate_data()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        self.observation_space = spaces.Box(
            low=-1e10,
            high=1e10,
            shape=(self.seq_len, len(self.features)),
            dtype=np.float32,
        )
        
        # Initialize state variables
        self._reset_state()
    
    def _validate_data(self) -> None:
        """Validate the input data."""
        if self.data.empty:
            raise ValueError("Data cannot be empty")
        
        missing_features = [f for f in self.features if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        if len(self.data) < self.seq_len:
            raise ValueError(f"Data length ({len(self.data)}) must be >= seq_len ({self.seq_len})")
    
    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.index = self.seq_len
        self.episode_step_count = 0
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = None
        self.entry_index = None
        self.total_reward = 0.0
        self.done = False
        self.truncated = False
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        self._reset_state()
        
        info = {
            "position": self.position,
            "entry_price": self.entry_price,
            "total_reward": self.total_reward,
            "step_count": self.episode_step_count
        }
        
        return self._get_observation(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take (0=Hold, 1=Buy, 2=Sell)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done or self.truncated:
            raise RuntimeError("Environment is done, call reset() first")
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")
        
        current_price = self._get_current_price()
        reward = 0.0
        
        # Handle position opening and switching
        position_opened = False
        if self.position == 0:
            if action == 1:  # Buy
                self._open_position(1, current_price)
                position_opened = True
            elif action == 2:  # Sell
                self._open_position(-1, current_price)
                position_opened = True
        else:
            # Handle position switching
            if action == 1 and self.position == -1:  # Switch from short to long
                self._close_position()
                self._open_position(1, current_price)
                position_opened = True
            elif action == 2 and self.position == 1:  # Switch from long to short
                self._close_position()
                self._open_position(-1, current_price)
                position_opened = True
            elif action == 1 and self.position == 1:  # Already long, close and reopen
                self._close_position()
                self._open_position(1, current_price)
                position_opened = True
            elif action == 2 and self.position == -1:  # Already short, close and reopen
                self._close_position()
                self._open_position(-1, current_price)
                position_opened = True
        
        # Apply penalty for being idle too long (encourage trading)
        if self.position == 0 and not position_opened:
            reward -= self.idle_penalty
        
        # Reward shaping while holding a position
        if self.position != 0:
            change = (current_price - self.entry_price) / self.entry_price
            step_duration = self.episode_step_count - self.entry_index
            direction = np.sign(self.position)
            unrealized = change * direction
            
            # Stop loss hit: close trade for a loss and apply penalty
            if unrealized <= -self.sl_pct:
                reward += unrealized * self.sl_penalty_coef
                self._close_position()
            
            # Take profit hit: close trade for a profit and apply reward
            elif unrealized >= self.tp_pct:
                reward += unrealized * self.tp_reward_coef
                self._close_position()
            
            # Trade timeout hit: close trade and apply scaled reward
            elif step_duration >= self.timeout_duration:
                reward += unrealized * self.timeout_reward_coef
                self._close_position()
            
            # Trade held: apply small reward/penalty for unrealized pnl
            else:
                reward += unrealized * self.ongoing_reward_coef
            
            # If position was just opened, give initial ongoing reward
            if position_opened:
                reward = self.ongoing_reward_coef
        
        # Normalize reward
        reward = np.clip(reward, self.reward_clip_range[0], self.reward_clip_range[1])
        
        # Update state
        self.index += 1
        self.episode_step_count += 1
        self.total_reward += reward
        
        # Check termination conditions
        self.done = self.index >= len(self.data)
        self.truncated = self.episode_step_count >= self.max_episode_steps
        
        # Prepare info
        info = {
            "action": action,
            "position": self.position,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "total_reward": self.total_reward,
            "step_count": self.episode_step_count,
            "done": self.done,
            "truncated": self.truncated
        }
        
        return self._get_observation(), reward, self.done, self.truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        window = self.data.iloc[self.index - self.seq_len: self.index]
        obs = window[self.features].to_numpy(dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e10, neginf=-1e10)
        return obs
    
    def _get_current_price(self) -> float:
        """Get the current price."""
        return float(self.data.loc[self.index, "close"])
    
    def _open_position(self, direction: int, price: float) -> None:
        """Open a new position."""
        self.position = direction
        self.entry_price = price
        self.entry_index = self.episode_step_count
    
    def _close_position(self) -> None:
        """Close the current position."""
        self.position = 0
        self.entry_price = None
        self.entry_index = None
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame or None
        """
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {mode}. Supported modes: {self.metadata['render_modes']}")
        
        if mode == "human":
            print(f"Step {self.index} | Position: {self.position} | "
                  f"Entry: {self.entry_price} | Total Reward: {self.total_reward:.4f}")
        return None
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        state = super().get_state()
        
        # Calculate steps in position
        steps_in_position = 0
        if self.position != 0 and self.entry_index is not None:
            steps_in_position = self.episode_step_count - self.entry_index
        
        state.update({
            "index": self.index,
            "episode_step_count": self.episode_step_count,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_index": self.entry_index,
            "total_reward": self.total_reward,
            "done": self.done,
            "truncated": self.truncated,
            # Add expected keys for compatibility with tests
            "current_step": self.index,
            "entry_step": self.entry_index,
            "steps_in_position": steps_in_position
        })
        return state
    
    @property
    def current_step(self) -> int:
        """Get the current step index."""
        return self.index
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the environment to a specific state."""
        super().set_state(state)
        
        # Handle both old and new key names for compatibility
        self.index = state.get("index", state.get("current_step", self.seq_len))
        self.episode_step_count = state.get("episode_step_count", 0)
        self.position = state.get("position", 0)
        self.entry_price = state.get("entry_price", None)
        self.entry_index = state.get("entry_index", state.get("entry_step", None))
        self.total_reward = state.get("total_reward", 0.0)
        self.done = state.get("done", False)
        self.truncated = state.get("truncated", False) 