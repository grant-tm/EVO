"""
PPO (Proximal Policy Optimization) trading agent implementation.

This module provides a PPO-based trading agent that wraps the stable-baselines3
implementation with additional trading-specific functionality.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from .base_agent import BaseAgent
from ..training.callbacks import EntropyAnnealingCallback


class PPOAgent(BaseAgent):
    """
    PPO-based trading agent.
    
    This agent uses Proximal Policy Optimization for learning trading strategies.
    It wraps the stable-baselines3 PPO implementation with additional
    trading-specific features.
    """
    
    def __init__(
        self,
        env: gym.Env,
        policy: str = "MlpPolicy",
        learning_rate: Union[float, str] = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
        **kwargs
    ):
        """
        Initialize the PPO agent.
        
        Args:
            env: The trading environment
            policy: Policy type (e.g., "MlpPolicy", "CnnPolicy")
            learning_rate: Learning rate for the optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for the value function
            clip_range_vf: Clipping parameter for the value function
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: Maximum norm for the gradient clipping
            use_sde: Whether to use generalized State Dependent Exploration
            sde_sample_freq: Sample a new noise matrix every n steps
            target_kl: Limit the KL divergence between updates
            tensorboard_log: Log directory for tensorboard
            policy_kwargs: Additional arguments to be passed to the policy
            verbose: Verbosity level
            seed: Random seed
            device: Device to use for training
            _init_setup_model: Whether to setup the model
            **kwargs: Additional arguments passed to BaseAgent
        """
        # Store PPO-specific configuration
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs
        self.verbose = verbose
        self.seed = seed
        self.device = device
        
        # Store configuration in parent class
        config = {
            'policy': policy,
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'clip_range_vf': clip_range_vf,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm,
            'use_sde': use_sde,
            'sde_sample_freq': sde_sample_freq,
            'target_kl': target_kl,
            'tensorboard_log': tensorboard_log,
            'policy_kwargs': policy_kwargs,
            'verbose': verbose,
            'seed': seed,
            'device': device,
            **kwargs
        }
        
        super().__init__(env, **config)
        
        # Initialize the PPO model
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        """Initialize the PPO model."""
        # Store original environment
        self._original_env = self.env
        
        # Check if environment is already a VecEnv (like VecNormalize)
        if hasattr(self.env, 'venv') or hasattr(self.env, 'envs'):
            # Environment is already a VecEnv, use it directly
            self._vec_env = self.env
        elif not isinstance(self.env, DummyVecEnv):
            # Environment is a raw gym.Env, wrap it
            self._vec_env = DummyVecEnv([lambda: Monitor(self._original_env)])
        else:
            # Environment is already a DummyVecEnv
            self._vec_env = self.env
        
        # Create PPO model
        self.model = PPO(
            policy=self.policy,
            env=self._vec_env,
            learning_rate=get_schedule_fn(self.learning_rate),
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            target_kl=self.target_kl,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs=self.policy_kwargs,
            verbose=self.verbose,
            seed=self.seed,
            device=self.device,
            _init_setup_model=True
        )
    
    def predict(
        self, 
        observation: np.ndarray, 
        deterministic: bool = True
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Predict the next action given an observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before making predictions")
        
        action, states = self.model.predict(observation, deterministic=deterministic)
        return action, {"states": states}
    
    def learn(self, total_timesteps: int, callback: Optional[Any] = None) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Number of timesteps to train for
            callback: Optional callback for monitoring training
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        self.is_trained = True
    
    def save(self, path: str) -> None:
        """
        Save the trained agent to disk.
        
        Args:
            path: Path where to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # Save normalization if using VecNormalize
        if isinstance(self._vec_env, VecNormalize):
            norm_path = path.replace(".zip", "_vecnormalize.pkl")
            self._vec_env.save(norm_path)
    
    def load(self, path: str) -> None:
        """
        Load a trained agent from disk.
        
        Args:
            path: Path to the saved model
        """
        self.model = PPO.load(path)
        
        # Load normalization if it exists
        norm_path = path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(norm_path) and isinstance(self._vec_env, VecNormalize):
            self._vec_env = VecNormalize.load(norm_path, self._vec_env)
        
        self.is_trained = True
    
    def get_model(self) -> PPO:
        """Get the underlying PPO model."""
        return self.model
    
    def set_entropy_coefficient(self, ent_coef: float) -> None:
        """Set the entropy coefficient for exploration."""
        self.ent_coef = ent_coef
        self.model.ent_coef = ent_coef
    
    def get_entropy_coefficient(self) -> float:
        """Get the current entropy coefficient."""
        return self.model.ent_coef
    
    def get_action_space(self) -> gym.Space:
        """Get the action space of the environment."""
        return self._original_env.action_space
    
    def get_observation_space(self) -> gym.Space:
        """Get the observation space of the environment."""
        return self._original_env.observation_space 