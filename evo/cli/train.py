"""
Training CLI for the EVO trading system.

This module provides a command-line interface for training RL agents on trading environments.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..core.config import Config, get_config
from ..core.logging import setup_logging, get_logger
from ..models.training import Trainer, HyperparameterManager
# TODO: Import Trainer, agent, and environment classes as needed
# from ..models.training.trainer import Trainer
# from ..models.agents.ppo_agent import PPOAgent
# from ..models.environments.trading_env import TradingEnv

def _load_config(parsed_args: argparse.Namespace) -> Config:
    """
    Load and override configuration from command-line arguments.
    """
    config = get_config()
    if parsed_args.config:
        config = Config(config_file=parsed_args.config, env_file=parsed_args.env_file)
    
    if parsed_args.data_path:
        config.data.data_path = str(parsed_args.data_path)
    
    if parsed_args.features:
        config.data.features = parsed_args.features.split(',')
    
    return config

def train_command(args: Optional[list] = None) -> None:
    """
    Run training for RL agents on trading environments.
    """
    parser = argparse.ArgumentParser(
        description="Train RL agents on trading environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO agent
  python -m evo.cli.train ppo --env TradingEnv --epochs 100 --batch-size 64 --learning-rate 0.0003
        """
    )

    # -- Configuration ------------------------------------
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c',
        default='config/default_config.json',
        type=Path,
        help='Path to configuration file (JSON)'
    )
    config_group.add_argument(
        '--env-file', '-e',
        type=Path,
        help='Path to .env file for API keys'
    )

    # -- Data Parameters ----------------------------------
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        '--data-path',
        type=Path,
        help='Path to data file for training'
    )
    data_group.add_argument(
        '--features',
        default='open,high,low,close,volume',
        type=str,
        help='Comma-separated list of features to use'
    )

    # -- Output Parameters --------------------------------
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output-dir',
        type=Path,
        default=Path('trained_models'),
        help='Directory to save trained models'
    )
    output_group.add_argument(
        '--save-metrics',
        action='store_true',
        help='Save training metrics to file'
    )

    # -- Logging Parameters -------------------------------
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    # Subparsers for agents
    subparsers = parser.add_subparsers(dest='agent', required=True, title='Agents', description='Available RL agents')

    # -- PPO Agent ----------------------------------------
    ppo_parser = subparsers.add_parser('ppo', help='Proximal Policy Optimization (PPO) agent')
    ppo_parser.add_argument(
        '--env', 
        type=str, 
        required=True, 
        help='Environment class name'
    )
    ppo_parser.add_argument(
        '--max-timesteps',
        type=int,
        default=25000,
        help='Maximum number of training timesteps (default: 25000)'
    )
    ppo_parser.add_argument(
        '--epochs', 
        type=int, 
        default=10, 
        help='Number of training epochs (default: 10)'
    )
    ppo_parser.add_argument(
        '--batch-size', 
        type=int, 
        default=64, 
        help='Batch size for training (default: 64)'
    )
    ppo_parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.0003,
        help='Learning rate for optimizer (default: 0.0003)'
    )
    ppo_parser.add_argument(
        '--model-save-path',
        type=Path,
        help='Path to save trained model (default: trained_models/example.zip)'
    )
    ppo_parser.add_argument(
        '--model-load-path',
        type=Path,
        help='Path to load pre-trained model (optional)'
    )
    # TODO: Add remaining PPO-specific arguments

    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(level=parsed_args.log_level)
    logger = get_logger(__name__)

    try:
        # Load configuration
        config = _load_config(parsed_args)
        logger.info("Starting training")
        logger.info(f"Configuration: {config}")
        
        # Create output directory
        output_dir = parsed_args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fail if agent is not supported
        if parsed_args.agent not in ['ppo']:
            logger.error(f"Unknown agent: {parsed_args.agent}")
            sys.exit(1)
        
        logger.info("Training PPO agent...")
        
        # Instantiate Trainer
        trainer = Trainer(
            data_path=str(config.data.data_path),
            model_dir=str(output_dir),
            config_dir="config"
        )
        
        # Get default hyperparameters and override with CLI args
        hpm = HyperparameterManager("config")
        hyperparams = hpm.get_default_hyperparameters()
        
        # Override with CLI args if provided
        if parsed_args.epochs is not None:
            hyperparams.n_epochs = parsed_args.epochs
        if parsed_args.batch_size is not None:
            hyperparams.batch_size = parsed_args.batch_size
        if parsed_args.learning_rate is not None:
            hyperparams.learning_rate = parsed_args.learning_rate
        if hasattr(parsed_args, 'max_timesteps') and parsed_args.max_timesteps is not None:
            hyperparams.total_timesteps = parsed_args.max_timesteps
        
        # Train model
        model_name = parsed_args.model_save_path.stem if parsed_args.model_save_path else "ppo_model"
        features = config.data.features if hasattr(config.data, 'features') else None
        results = trainer.train(model_name=model_name, hyperparams=hyperparams, features=features)
        logger.info(f"Training results: {results}")
        
        # Save training metrics
        if parsed_args.save_metrics:
            import json
            metrics_file = output_dir / f"{model_name}_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file}")
        logger.info("PPO training completed.")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    train_command() 