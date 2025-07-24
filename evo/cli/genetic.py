"""
Genetic Optimization CLI for the EVO trading system.

This module provides a command-line interface for running genetic hyperparameter optimization.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..core.config import Config, get_config
from ..core.logging import setup_logging, get_logger
from ..optimization.genetic import GeneticSearch, GeneticSearchConfig, GenomeConfig, BacktestFitnessEvaluator
from ..optimization.backtesting import BacktestEngine, CSVDataProvider
from ..models.training import Trainer


def _load_config(parsed_args: argparse.Namespace) -> Config:
    config = get_config()
    if parsed_args.config:
        config = Config(config_file=parsed_args.config, env_file=parsed_args.env_file)
    if parsed_args.data_path:
        config.data.data_path = str(parsed_args.data_path)
    if parsed_args.features:
        config.data.features = parsed_args.features.split(',')
    return config


def genetic_command(args: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run genetic hyperparameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run genetic optimization with default settings
  python -m evo.cli.genetic --data-path data.csv --features open,high,low,close,volume

  # Customize population and generations
  python -m evo.cli.genetic --data-path --population-size 30 --max-generations 20 --mutation-rate 0.15
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
        required=True,
        help='Path to data file for training/backtesting'
    )
    data_group.add_argument(
        '--features',
        default='open,high,low,close,volume',
        type=str,
        help='Comma-separated list of features to use'
    )

    # -- Genetic Search Parameters ------------------------
    genetic_group = parser.add_argument_group('Genetic Search')
    genetic_group.add_argument('--population-size', type=int, default=50, help='Population size (default: 50)')
    genetic_group.add_argument('--max-generations', type=int, default=100, help='Max generations (default: 100)')
    genetic_group.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate (default: 0.1)')
    genetic_group.add_argument('--crossover-rate', type=float, default=0.8, help='Crossover rate (default: 0.8)')
    genetic_group.add_argument('--elite-fraction', type=float, default=0.2, help='Elite fraction (default: 0.2)')
    genetic_group.add_argument('--random-fraction', type=float, default=0.1, help='Random fraction (default: 0.1)')
    genetic_group.add_argument('--tournament-size', type=int, default=3, help='Tournament size (default: 3)')
    genetic_group.add_argument('--selection-pressure', type=float, default=1.5, help='Selection pressure (default: 1.5)')
    genetic_group.add_argument('--patience', type=int, default=20, help='Early stopping patience (default: 20)')
    genetic_group.add_argument('--min-improvement', type=float, default=0.001, help='Minimum improvement for early stopping (default: 0.001)')
    genetic_group.add_argument('--log-interval', type=int, default=5, help='Log interval (default: 5)')
    genetic_group.add_argument('--save-best-genome', action='store_true', help='Save best genome to file')
    genetic_group.add_argument('--no-save-best-genome', dest='save_best_genome', action='store_false')
    genetic_group.set_defaults(save_best_genome=True)
    genetic_group.add_argument('--save-history', action='store_true', help='Save search history to file')
    genetic_group.add_argument('--no-save-history', dest='save_history', action='store_false')
    genetic_group.set_defaults(save_history=True)

    # -- Fitness Evaluation Parameters --------------------
    fitness_group = parser.add_argument_group('Fitness Evaluation')
    fitness_group.add_argument('--num-backtests', type=int, default=10, help='Number of backtests per evaluation (default: 10)')
    fitness_group.add_argument('--backtest-length', type=int, default=1000, help='Length of each backtest (default: 1000)')
    fitness_group.add_argument('--fitness-metric', type=str, default='sharpe_ratio', help='Fitness metric (default: sharpe_ratio)')

    # -- Parallelism -------------------------------------
    parallel_group = parser.add_argument_group('Parallelism')
    parallel_group.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs (default: 1)')
    parallel_group.add_argument('--use-multiprocessing', action='store_true', help='Use multiprocessing for evaluation')

    # -- Output ------------------------------------------
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=Path, default=Path('checkpoints'), help='Directory to save results')
    output_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')

    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(level=parsed_args.log_level)
    logger = get_logger(__name__)

    # Load config
    config = _load_config(parsed_args)
    logger.info(f"Configuration: {config}")

    # Prepare data provider and backtest engine
    data_provider = CSVDataProvider(
        data_path=str(config.data.data_path),
        feature_columns=config.data.features
    )
    backtest_engine = BacktestEngine(
        data_provider=data_provider,
        initial_capital=getattr(config.trading, 'initial_capital', 100000.0),
        commission=getattr(config.trading, 'commission', 0.0),
        slippage=getattr(config.trading, 'slippage', 0.0),
        risk_free_rate=0.02
    )

    # Prepare model trainer (assume PPO agent for now)
    trainer = Trainer(
        data_path=str(config.data.data_path),
        model_dir=str(parsed_args.output_dir),
        config_dir="config"
    )

    def model_trainer_fn(genome):
        # Use genome.values for hyperparameters
        model_name = f"genome_{genome.hash()[:8]}"
        result = trainer.train_with_genome(genome.values, model_name, features=config.data.features)
        return result["model_path"] + ".zip"  # Assume model is saved as .zip

    # Prepare fitness evaluator
    fitness_evaluator = BacktestFitnessEvaluator(
        backtest_engine=backtest_engine,
        model_trainer=model_trainer_fn,
        model_dir=str(parsed_args.output_dir),
        num_backtests=parsed_args.num_backtests,
        backtest_length=parsed_args.backtest_length,
        fitness_metric=parsed_args.fitness_metric,
        cache_results=True
    )

    # Prepare genetic search config
    search_config = GeneticSearchConfig(
        population_size=parsed_args.population_size,
        max_generations=parsed_args.max_generations,
        mutation_rate=parsed_args.mutation_rate,
        crossover_rate=parsed_args.crossover_rate,
        elite_fraction=parsed_args.elite_fraction,
        random_fraction=parsed_args.random_fraction,
        tournament_size=parsed_args.tournament_size,
        selection_pressure=parsed_args.selection_pressure,
        num_backtests=parsed_args.num_backtests,
        backtest_length=parsed_args.backtest_length,
        fitness_metric=parsed_args.fitness_metric,
        n_jobs=parsed_args.n_jobs,
        use_multiprocessing=parsed_args.use_multiprocessing,
        patience=parsed_args.patience,
        min_improvement=parsed_args.min_improvement,
        log_interval=parsed_args.log_interval,
        save_best_genome=parsed_args.save_best_genome,
        save_history=parsed_args.save_history
    )

    # Run genetic search
    logger.info("Starting genetic optimization...")
    search = GeneticSearch(
        fitness_evaluator=fitness_evaluator,
        config=search_config,
        genome_config=GenomeConfig(),
        seed=42
    )
    best_genome, best_fitness = search.run()

    # Print summary
    print("\n=== Genetic Optimization Completed ===")
    print(f"Best fitness: {best_fitness:.4f}")
    print("Best genome parameters:")
    best_genome.print_values()
    print(f"Results saved to: {parsed_args.output_dir}")

if __name__ == "__main__":
    genetic_command() 