"""
Training callbacks for monitoring and controlling the training process.

This module provides various callbacks that can be used during training
to monitor progress, log metrics, and control training behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(ABC):
    """
    Abstract base class for training callbacks.
    
    All training callbacks must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str = "TrainingCallback"):
        """
        Initialize the callback.
        
        Args:
            name: Name of the callback for logging
        """
        self.name = name
        self.training_step = 0
    
    @abstractmethod
    def on_step(self, training_info: Dict[str, Any]) -> bool:
        """
        Called after each training step.
        
        Args:
            training_info: Dictionary containing training information
            
        Returns:
            True to continue training, False to stop
        """
        pass
    
    def on_episode_end(self, episode_info: Dict[str, Any]) -> None:
        """
        Called at the end of each episode.
        
        Args:
            episode_info: Dictionary containing episode information
        """
        pass
    
    def on_training_start(self) -> None:
        """Called at the start of training."""
        pass
    
    def on_training_end(self) -> None:
        """Called at the end of training."""
        pass


class EntropyAnnealingCallback(BaseCallback):
    """
    Callback for annealing the entropy coefficient during training.
    
    This callback gradually reduces the entropy coefficient from an initial
    value to a final value over the course of training, encouraging
    exploration early on and exploitation later.
    """
    
    def __init__(
        self,
        initial_coef: float,
        final_coef: float,
        total_timesteps: int,
        verbose: int = 0
    ):
        """
        Initialize the entropy annealing callback.
        
        Args:
            initial_coef: Initial entropy coefficient
            final_coef: Final entropy coefficient
            total_timesteps: Total number of training timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.total_timesteps = total_timesteps
    
    def _on_step(self) -> bool:
        """Update entropy coefficient based on training progress."""
        progress = self.num_timesteps / self.total_timesteps
        new_coef = self.initial_coef + progress * (self.final_coef - self.initial_coef)
        self.model.ent_coef = new_coef
        
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"Entropy coefficient: {new_coef:.4f}")
        
        return True


class RewardLoggerCallback(BaseCallback):
    """
    Callback for logging reward information during training.
    
    This callback logs various reward-related metrics to help monitor
    training progress and reward shaping effectiveness.
    """
    
    def __init__(self, verbose: int = 0):
        """
        Initialize the reward logger callback.
        
        Args:
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.reward_history = []
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        """Log reward information for each step."""
        if "reward" in self.locals:
            reward = self.locals["reward"]
            self.reward_history.append(reward)
            self.logger.record("rollout/raw_reward", reward)
            
            # Log moving average
            if len(self.reward_history) >= 100:
                moving_avg = np.mean(self.reward_history[-100:])
                self.logger.record("rollout/reward_moving_avg", moving_avg)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log episode-level reward information."""
        if hasattr(self, "episode_rewards") and self.episode_rewards:
            episode_reward = np.mean(self.episode_rewards)
            self.logger.record("rollout/episode_reward", episode_reward)
            self.episode_rewards = []


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on performance metrics.
    
    This callback monitors a specified metric and stops training if
    the metric doesn't improve for a specified number of evaluations.
    """
    
    def __init__(
        self,
        eval_freq: int,
        patience: int,
        min_evals: int = 3,
        metric: str = "eval_mean_reward",
        verbose: int = 0
    ):
        """
        Initialize the early stopping callback.
        
        Args:
            eval_freq: Frequency of evaluations
            patience: Number of evaluations without improvement before stopping
            min_evals: Minimum number of evaluations before stopping
            metric: Metric to monitor for improvement
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_evals = min_evals
        self.metric = metric
        self.best_score = -np.inf
        self.no_improvement_count = 0
        self.eval_count = 0
    
    def _on_step(self) -> bool:
        """Check if training should be stopped."""
        if self.num_timesteps % self.eval_freq == 0:
            self.eval_count += 1
            
            # Get current metric value
            current_score = self.logger.name_to_value.get(self.metric, -np.inf)
            
            if current_score > self.best_score:
                self.best_score = current_score
                self.no_improvement_count = 0
                if self.verbose > 0:
                    print(f"New best {self.metric}: {current_score:.4f}")
            else:
                self.no_improvement_count += 1
                if self.verbose > 0:
                    print(f"No improvement for {self.no_improvement_count} evaluations")
            
            # Check if we should stop
            if (self.eval_count >= self.min_evals and 
                self.no_improvement_count >= self.patience):
                if self.verbose > 0:
                    print(f"Early stopping after {self.eval_count} evaluations")
                return False
        
        return True


class ModelCheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints during training.
    
    This callback saves model checkpoints at regular intervals or when
    performance improves, allowing for model recovery and best model selection.
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "model",
        save_best_only: bool = True,
        metric: str = "eval_mean_reward",
        verbose: int = 0
    ):
        """
        Initialize the model checkpoint callback.
        
        Args:
            save_freq: Frequency of saves (in timesteps)
            save_path: Directory to save checkpoints
            name_prefix: Prefix for saved model names
            save_best_only: Whether to only save when performance improves
            metric: Metric to use for determining best model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_best_only = save_best_only
        self.metric = metric
        self.best_score = -np.inf
        self.last_save_step = 0
    
    def _on_step(self) -> bool:
        """Save model checkpoint if conditions are met."""
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            current_score = self.logger.name_to_value.get(self.metric, -np.inf)
            
            should_save = not self.save_best_only or current_score > self.best_score
            
            if should_save:
                model_path = f"{self.save_path}/{self.name_prefix}_{self.num_timesteps}"
                self.model.save(model_path)
                self.last_save_step = self.num_timesteps
                
                if current_score > self.best_score:
                    self.best_score = current_score
                
                if self.verbose > 0:
                    print(f"Model saved to {model_path}")
        
        return True 