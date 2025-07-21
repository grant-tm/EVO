"""
Abstract base class for trading environments.

This module defines the interface that all trading environments must implement,
ensuring consistency across different environment implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseEnv(gym.Env, ABC):
    """
    Abstract base class for trading environments.
    
    All trading environments must inherit from this class and implement
    the required abstract methods. This ensures consistency across
    different trading environment implementations.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the environment.
        
        Args:
            **kwargs: Configuration parameters for the environment
        """
        super().__init__()
        self.config = kwargs
    
    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame or None
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the environment and clean up resources."""
        pass
    
    def get_action_space(self) -> gym.Space:
        """Get the action space."""
        return self.action_space
    
    def get_observation_space(self) -> gym.Space:
        """Get the observation space."""
        return self.observation_space
    
    def get_config(self) -> Dict[str, Any]:
        """Get the environment configuration."""
        return self.config.copy()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        This method should return a dictionary containing all relevant
        state information that would be needed to restore the environment
        to its current state.
        """
        return {
            "config": self.config.copy(),
            "action_space": self.action_space,
            "observation_space": self.observation_space
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the environment to a specific state.
        
        Args:
            state: State dictionary returned by get_state()
        """
        self.config = state.get("config", {})
        self.action_space = state.get("action_space", self.action_space)
        self.observation_space = state.get("observation_space", self.observation_space) 