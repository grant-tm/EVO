"""
Training orchestration for trading agents.

This module provides the main Trainer class that orchestrates the training
process using the new model layer components.
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import time

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, StopTrainingOnNoModelImprovement

from ..agents import BaseAgent
from ..environments import TradingEnv
from .hyperparameters import TrainingHyperparameters, HyperparameterManager
from .callbacks import EntropyAnnealingCallback, RewardLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback




class Trainer:
    """
    Main trainer class for orchestrating the training process.
    
    This class provides a high-level interface for training trading agents
    with comprehensive configuration management and monitoring capabilities.
    """
    
    def __init__(
        self,
        data_path: str,
        model_dir: str = "trained_models",
        config_dir: str = "config",
        tensorboard_log: str = "./tensorboard_logs/"
    ):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the training data
            model_dir: Directory to save trained models
            config_dir: Directory for hyperparameter configurations
            tensorboard_log: Directory for tensorboard logs
        """
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.tensorboard_log = tensorboard_log
        
        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        Path(tensorboard_log).mkdir(exist_ok=True)
        
        # Initialize components
        self.hyperparam_manager = HyperparameterManager(config_dir)
        self.data = None
        self.train_env = None
        self.eval_env = None
        self.agent = None
        
    def load_data(self, features: Optional[List[str]] = None) -> None:
        """
        Load and prepare training data.
        
        Args:
            features: List of feature columns to use
        """
        print(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        self.data.dropna(inplace=True)
        
        if features:
            missing_features = [f for f in features if f not in self.data.columns]
            if missing_features:
                raise ValueError(f"Missing features in data: {missing_features}")
            
            # Filter data to only include requested features
            self.data = self.data[features]
        
        print(f"Loaded {len(self.data)} data points")
    
    def create_environments(
        self,
        hyperparams: TrainingHyperparameters,
        train_split: float = 0.95
    ) -> None:
        """
        Create training and evaluation environments.
        
        Args:
            hyperparams: Training hyperparameters
            train_split: Fraction of data to use for training
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded before creating environments")
        
        # Split data
        split_idx = int(len(self.data) * train_split)
        train_df = self.data.iloc[:split_idx].copy()
        eval_df = self.data.iloc[split_idx:].copy()
        
        print(f"Training data: {len(train_df)} points")
        print(f"Evaluation data: {len(eval_df)} points")
        
        # Create training environment
        train_env = TradingEnv(
            data=train_df,
            features=hyperparams.policy_kwargs.get("features", ["open", "high", "low", "close", "volume"]),
            seq_len=hyperparams.seq_len,
            tp_pct=hyperparams.tp_pct,
            sl_pct=hyperparams.sl_pct,
            idle_penalty=hyperparams.idle_penalty,
            sl_penalty_coef=hyperparams.sl_penalty_coef,
            tp_reward_coef=hyperparams.tp_reward_coef,
            timeout_duration=hyperparams.timeout_duration,
            timeout_reward_coef=hyperparams.timeout_reward_coef,
            ongoing_reward_coef=hyperparams.ongoing_reward_coef,
            reward_clip_range=hyperparams.reward_clip_range,
            max_episode_steps=hyperparams.max_episode_steps
        )
        
        # Wrap in VecEnv and normalize
        self.train_env = DummyVecEnv([lambda: Monitor(train_env)])
        self.train_env = VecNormalize(self.train_env, norm_obs=True, norm_reward=False)
        
        # Create evaluation environment
        eval_env = TradingEnv(
            data=eval_df,
            features=hyperparams.policy_kwargs.get("features", ["open", "high", "low", "close", "volume"]),
            seq_len=hyperparams.seq_len,
            tp_pct=hyperparams.tp_pct,
            sl_pct=hyperparams.sl_pct,
            idle_penalty=hyperparams.idle_penalty,
            sl_penalty_coef=hyperparams.sl_penalty_coef,
            tp_reward_coef=hyperparams.tp_reward_coef,
            timeout_duration=hyperparams.timeout_duration,
            timeout_reward_coef=hyperparams.timeout_reward_coef,
            ongoing_reward_coef=hyperparams.ongoing_reward_coef,
            reward_clip_range=hyperparams.reward_clip_range,
            max_episode_steps=hyperparams.max_episode_steps
        )
        
        self.eval_env = DummyVecEnv([lambda: Monitor(eval_env)])
        
        # Try to load existing VecNormalize, create new one if it doesn't exist
        vec_normalize_path = f"{self.model_dir}/vecnormalize.pkl"
        if os.path.exists(vec_normalize_path):
            self.eval_env = VecNormalize.load(vec_normalize_path, self.eval_env)
        else:
            self.eval_env = VecNormalize(self.eval_env, norm_obs=True, norm_reward=False)
            
        self.eval_env.training = False
        self.eval_env.norm_reward = False
        
        print("Environments created successfully")
    
    def create_agent(self, hyperparams: TrainingHyperparameters) -> None:
        """
        Create the training agent.
        
        Args:
            hyperparams: Training hyperparameters
        """
        if self.train_env is None:
            raise RuntimeError("Environments must be created before creating agent")
        
        # Import PPOAgent locally to avoid circular import
        from ..agents import PPOAgent
        
        self.agent = PPOAgent(
            env=self.train_env,
            policy=hyperparams.policy,
            learning_rate=hyperparams.learning_rate,
            n_steps=hyperparams.n_steps,
            batch_size=hyperparams.batch_size,
            n_epochs=hyperparams.n_epochs,
            gamma=hyperparams.gamma,
            gae_lambda=hyperparams.gae_lambda,
            clip_range=hyperparams.clip_range,
            clip_range_vf=hyperparams.clip_range_vf,
            ent_coef=hyperparams.ent_coef,
            vf_coef=hyperparams.vf_coef,
            max_grad_norm=hyperparams.max_grad_norm,
            use_sde=hyperparams.use_sde,
            sde_sample_freq=hyperparams.sde_sample_freq,
            target_kl=hyperparams.target_kl,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs=hyperparams.policy_kwargs,
            verbose=1
        )
        
        print("Agent created successfully")
    
    def create_callbacks(
        self,
        hyperparams: TrainingHyperparameters,
        model_name: str
    ) -> List[Any]:
        """
        Create training callbacks.
        
        Args:
            hyperparams: Training hyperparameters
            model_name: Name for the model
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Progress bar
        callbacks.append(ProgressBarCallback())
        
        # Entropy annealing
        callbacks.append(EntropyAnnealingCallback(
            initial_coef=hyperparams.entropy_coef_init,
            final_coef=hyperparams.entropy_coef_final,
            total_timesteps=hyperparams.total_timesteps
        ))
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{self.model_dir}/{model_name}/",
            log_path=f"{self.model_dir}/{model_name}/logs",
            eval_freq=hyperparams.eval_freq,
            n_eval_episodes=hyperparams.n_eval_episodes,
            deterministic=True,
            render=False,
            callback_after_eval=StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5,
                min_evals=3,
                verbose=1
            )
        )
        callbacks.append(eval_callback)
        
        # Model checkpointing
        callbacks.append(ModelCheckpointCallback(
            save_freq=hyperparams.save_freq,
            save_path=f"{self.model_dir}/{model_name}/checkpoints",
            name_prefix="model",
            save_best_only=True,
            metric="eval_mean_reward"
        ))
        
        return callbacks
    
    def train(
        self,
        model_name: str,
        hyperparams: Optional[TrainingHyperparameters] = None,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train a trading agent.
        
        Args:
            model_name: Name for the trained model
            hyperparams: Training hyperparameters (uses defaults if None)
            features: List of feature columns to use
            
        Returns:
            Training results
        """
        # Use default hyperparameters if none provided
        if hyperparams is None:
            hyperparams = self.hyperparam_manager.get_default_hyperparameters()
        
        # Validate hyperparameters
        errors = hyperparams.validate()
        if errors:
            raise ValueError(f"Invalid hyperparameters: {errors}")
        
        print(f"Starting training for model: {model_name}")
        print(f"Total timesteps: {hyperparams.total_timesteps}")
        
        # Load data
        self.load_data(features)
        
        # Create environments
        self.create_environments(hyperparams)
        
        # Create agent
        self.create_agent(hyperparams)
        
        # Create callbacks
        callbacks = self.create_callbacks(hyperparams, model_name)
        
        # Save normalization
        self.train_env.save(f"{self.model_dir}/vecnormalize.pkl")
        
        # Train the agent
        start_time = time.time()
        self.agent.learn(
            total_timesteps=hyperparams.total_timesteps,
            callback=callbacks
        )
        training_time = time.time() - start_time
        
        # Save final model
        model_path = f"{self.model_dir}/{model_name}/final_model"
        self.agent.save(model_path)
        
        # Save hyperparameters
        self.hyperparam_manager.save_hyperparameters(hyperparams, model_name)
        
        print(f"Training completed. Model saved to {model_path}")

        # Evaluate the final model to get final_reward
        try:
            eval_results = self.evaluate_model(self.agent, n_episodes=1)
            final_reward = eval_results.get("mean_reward", None)
        except Exception as e:
            final_reward = None

        return {
            "model_name": model_name,
            "model_path": model_path,
            "hyperparameters": hyperparams.to_dict(),
            "total_timesteps": hyperparams.total_timesteps,
            "training_time": training_time,
            "final_reward": final_reward
        }
    
    def train_with_genome(
        self,
        genome: Dict[str, Any],
        model_name: str,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train a model with parameters from a genome.
        
        Args:
            genome: Dictionary containing hyperparameters
            model_name: Name for the trained model
            features: List of feature columns to use
            
        Returns:
            Training results
        """
        # Create hyperparameters from genome
        base_hyperparams = self.hyperparam_manager.get_default_hyperparameters()
        genome_dict = base_hyperparams.to_dict()
        genome_dict.update(genome)
        hyperparams = TrainingHyperparameters.from_dict(genome_dict)
        
        return self.train(model_name, hyperparams, features)
    
    def load_trained_model(self, model_name: str) -> BaseAgent:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded agent
        """
        model_path = f"{self.model_dir}/{model_name}/final_model"
        
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create a dummy environment for loading
        dummy_env = TradingEnv(
            data=self.data.iloc[:100] if self.data is not None else pd.DataFrame(),
            seq_len=20
        )
        dummy_env = DummyVecEnv([lambda: Monitor(dummy_env)])
        
        # Load the agent
        from ..agents import PPOAgent
        agent = PPOAgent(dummy_env)
        agent.load(model_path)
        
        return agent
    
    def evaluate_model(
        self,
        agent: BaseAgent,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            agent: Trained agent to evaluate
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation results
        """
        if self.eval_env is None:
            raise RuntimeError("Evaluation environment not available")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                steps += 1
                
                if truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "n_episodes": n_episodes
        } 