"""
Hyperparameter management for training trading agents.

This module provides utilities for managing, validating, and optimizing
hyperparameters for training trading agents.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingHyperparameters:
    """
    Hyperparameters for training trading agents.
    
    This class encapsulates all hyperparameters needed for training
    and provides validation and serialization capabilities.
    """
    
    # Training parameters
    total_timesteps: int = 1_000_000
    batch_size: int = 64
    learning_rate: Union[float, str] = 3e-4
    n_steps: int = 512
    n_epochs: int = 10
    
    # PPO-specific parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Exploration parameters
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    
    # Policy parameters
    policy: str = "MlpPolicy"
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "net_arch": {"pi": [128, 128], "vf": [128, 128]}
    })
    
    # Environment parameters
    seq_len: int = 20
    tp_pct: float = 0.02
    sl_pct: float = 0.01
    idle_penalty: float = 0.001
    sl_penalty_coef: float = 1.0
    tp_reward_coef: float = 1.0
    timeout_duration: int = 100
    timeout_reward_coef: float = 0.5
    ongoing_reward_coef: float = 0.1
    reward_clip_range: tuple = (-1.0, 1.0)
    max_episode_steps: int = 1000
    
    # Callback parameters
    eval_freq: int = 25_000
    n_eval_episodes: int = 3
    save_freq: int = 100_000
    tensorboard_log: Optional[str] = "./tensorboard_logs/"
    
    # Entropy annealing
    entropy_coef_init: float = 0.01
    entropy_coef_final: float = 0.001
    
    def validate(self) -> List[str]:
        """
        Validate hyperparameters and return list of validation errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Training parameters
        if self.total_timesteps <= 0:
            errors.append("total_timesteps must be positive")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if isinstance(self.learning_rate, (int, float)) and self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.n_steps <= 0:
            errors.append("n_steps must be positive")
        
        if self.n_epochs <= 0:
            errors.append("n_epochs must be positive")
        
        # PPO parameters
        if not 0 <= self.gamma <= 1:
            errors.append("gamma must be between 0 and 1")
        
        if not 0 <= self.gae_lambda <= 1:
            errors.append("gae_lambda must be between 0 and 1")
        
        if self.clip_range <= 0:
            errors.append("clip_range must be positive")
        
        if self.ent_coef < 0:
            errors.append("ent_coef must be non-negative")
        
        if self.vf_coef < 0:
            errors.append("vf_coef must be non-negative")
        
        if self.max_grad_norm <= 0:
            errors.append("max_grad_norm must be positive")
        
        # Environment parameters
        if self.seq_len <= 0:
            errors.append("seq_len must be positive")
        
        if self.tp_pct <= 0:
            errors.append("tp_pct must be positive")
        
        if self.sl_pct <= 0:
            errors.append("sl_pct must be positive")
        
        if self.timeout_duration <= 0:
            errors.append("timeout_duration must be positive")
        
        if self.max_episode_steps <= 0:
            errors.append("max_episode_steps must be positive")
        
        # Callback parameters
        if self.eval_freq <= 0:
            errors.append("eval_freq must be positive")
        
        if self.n_eval_episodes <= 0:
            errors.append("n_eval_episodes must be positive")
        
        if self.save_freq <= 0:
            errors.append("save_freq must be positive")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hyperparameters to dictionary."""
        return {
            "total_timesteps": self.total_timesteps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "use_sde": self.use_sde,
            "sde_sample_freq": self.sde_sample_freq,
            "target_kl": self.target_kl,
            "policy": self.policy,
            "policy_kwargs": self.policy_kwargs,
            "seq_len": self.seq_len,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "idle_penalty": self.idle_penalty,
            "sl_penalty_coef": self.sl_penalty_coef,
            "tp_reward_coef": self.tp_reward_coef,
            "timeout_duration": self.timeout_duration,
            "timeout_reward_coef": self.timeout_reward_coef,
            "ongoing_reward_coef": self.ongoing_reward_coef,
            "reward_clip_range": self.reward_clip_range,
            "max_episode_steps": self.max_episode_steps,
            "eval_freq": self.eval_freq,
            "n_eval_episodes": self.n_eval_episodes,
            "save_freq": self.save_freq,
            "tensorboard_log": self.tensorboard_log,
            "entropy_coef_init": self.entropy_coef_init,
            "entropy_coef_final": self.entropy_coef_final
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingHyperparameters":
        """Create hyperparameters from dictionary."""
        return cls(**data)


class HyperparameterManager:
    """
    Manager for hyperparameter configuration and optimization.
    
    This class provides utilities for loading, saving, and managing
    hyperparameter configurations for training.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the hyperparameter manager.
        
        Args:
            config_dir: Directory to store hyperparameter configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_hyperparameters(
        self, 
        hyperparams: TrainingHyperparameters, 
        name: str
    ) -> None:
        """
        Save hyperparameters to a JSON file.
        
        Args:
            hyperparams: Hyperparameters to save
            name: Name for the configuration file
        """
        # Validate before saving
        errors = hyperparams.validate()
        if errors:
            raise ValueError(f"Invalid hyperparameters: {errors}")
        
        config_path = self.config_dir / f"{name}.json"
        with open(config_path, "w") as f:
            json.dump(hyperparams.to_dict(), f, indent=2)
    
    def load_hyperparameters(self, name: str) -> TrainingHyperparameters:
        """
        Load hyperparameters from a JSON file.
        
        Args:
            name: Name of the configuration file
            
        Returns:
            Loaded hyperparameters
        """
        config_path = self.config_dir / f"{name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            data = json.load(f)
        
        return TrainingHyperparameters.from_dict(data)
    
    def list_configurations(self) -> List[str]:
        """
        List all available hyperparameter configurations.
        
        Returns:
            List of configuration names
        """
        configs = []
        for config_file in self.config_dir.glob("*.json"):
            configs.append(config_file.stem)
        return sorted(configs)
    
    def get_default_hyperparameters(self) -> TrainingHyperparameters:
        """
        Get default hyperparameters for training.
        
        Returns:
            Default hyperparameters
        """
        return TrainingHyperparameters()
    
    def create_hyperparameter_grid(
        self, 
        param_ranges: Dict[str, List[Any]]
    ) -> List[TrainingHyperparameters]:
        """
        Create a grid of hyperparameters for hyperparameter search.
        
        Args:
            param_ranges: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of hyperparameter combinations
        """
        import itertools
        
        # Get base hyperparameters
        base_params = self.get_default_hyperparameters().to_dict()
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            params = base_params.copy()
            for name, value in zip(param_names, values):
                params[name] = value
            
            hyperparams = TrainingHyperparameters.from_dict(params)
            combinations.append(hyperparams)
        
        return combinations
    
    def optimize_hyperparameters(
        self,
        train_func,
        param_ranges: Dict[str, List[Any]],
        n_trials: int = 10,
        metric: str = "eval_mean_reward"
    ) -> Tuple[TrainingHyperparameters, Dict[str, Any]]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_func: Function that takes hyperparameters and returns metric
            param_ranges: Dictionary mapping parameter names to ranges
            n_trials: Number of optimization trials
            metric: Metric to optimize
            
        Returns:
            Tuple of (best_hyperparameters, optimization_results)
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range[0], str):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Create hyperparameters object
            base_params = self.get_default_hyperparameters().to_dict()
            base_params.update(params)
            hyperparams = TrainingHyperparameters.from_dict(base_params)
            
            # Train and evaluate
            result = train_func(hyperparams)
            return result.get(metric, -float('inf'))
        
        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best hyperparameters
        best_params = self.get_default_hyperparameters().to_dict()
        best_params.update(study.best_params)
        best_hyperparams = TrainingHyperparameters.from_dict(best_params)
        
        return best_hyperparams, {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
        } 