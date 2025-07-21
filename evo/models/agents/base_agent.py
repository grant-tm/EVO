"""
Abstract base class for trading agents.

This module defines the interface that all trading agents must implement,
ensuring consistency across different agent implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym


class BaseAgent(ABC):
    """
    Abstract base class for trading agents.
    
    All trading agents must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, env: gym.Env, **kwargs):
        """
        Initialize the agent.
        
        Args:
            env: The trading environment
            **kwargs: Additional configuration parameters
        """
        self.env = env
        self.config = kwargs
        self.is_trained = False
        
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Predict the next action given an observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        pass
    
    @abstractmethod
    def learn(self, total_timesteps: int, callback: Optional[Any] = None) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Number of timesteps to train for
            callback: Optional callback for monitoring training
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained agent to disk.
        
        Args:
            path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained agent from disk.
        
        Args:
            path: Path to the saved model
        """
        pass
    
    def get_action_space(self) -> gym.Space:
        """Get the action space of the environment."""
        return self.env.action_space
    
    def get_observation_space(self) -> gym.Space:
        """Get the observation space of the environment."""
        return self.env.observation_space
    
    def is_ready(self) -> bool:
        """Check if the agent is ready for prediction."""
        return self.is_trained
    
    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration."""
        return self.config.copy() 