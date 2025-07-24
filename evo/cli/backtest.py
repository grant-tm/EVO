"""
Backtesting CLI for the EVO trading system.

This module provides command-line interface for backtesting trading strategies.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from ..core.config import Config, get_config
from ..core.logging import setup_logging, get_logger
from ..optimization.backtesting import BacktestEngine, CSVDataProvider, CrossValidationEngine


def _load_config(parsed_args: argparse.Namespace) -> Config:
    """
    Load and override configuration from command-line arguments.

    Args:
        parsed_args: Parsed command-line arguments.
    Returns:
        Config: The loaded and overridden configuration object.
    """
    config = get_config()
    if parsed_args.config:
        config = Config(config_file=parsed_args.config, env_file=parsed_args.env_file)
    if parsed_args.input:
        config.data.data_path = str(parsed_args.input)
    if parsed_args.features:
        config.data.features = parsed_args.features.split(',')
    if parsed_args.initial_capital:
        config.trading.initial_capital = parsed_args.initial_capital
    if parsed_args.commission:
        config.trading.commission = parsed_args.commission
    if parsed_args.slippage:
        config.trading.slippage = parsed_args.slippage
    return config


def _save_cv_results(
    cv_results: list,
    strategy: str,
    strategy_params: dict,
    parsed_args: argparse.Namespace,
    output_dir: Path,
    logger: Any
) -> None:
    """
    Save and log cross-validation results.

    Args:
        cv_results: List of cross-validation result objects.
        strategy: Name of the strategy.
        strategy_params: Parameters for the strategy.
        parsed_args: Parsed command-line arguments.
        output_dir: Directory to save results.
        logger: Logger instance.
    """
    import numpy as np
    import json
    avg_sharpe = np.mean([r.metrics.sharpe_ratio for r in cv_results])
    avg_return = np.mean([r.metrics.total_return for r in cv_results])
    avg_drawdown = np.mean([r.metrics.max_drawdown for r in cv_results])
    avg_win_rate = np.mean([r.metrics.win_rate for r in cv_results])
    logger.info({
        "event": "cv_summary",
        "average_sharpe_ratio": avg_sharpe,
        "average_total_return": avg_return,
        "average_max_drawdown": avg_drawdown,
        "average_win_rate": avg_win_rate
    })
    if parsed_args.save_results:
        cv_summary = {
            'strategy': strategy,
            'n_splits': parsed_args.n_splits,
            'test_size': parsed_args.test_size,
            'parameters': strategy_params,
            'average_metrics': {
                'sharpe_ratio': avg_sharpe,
                'total_return': avg_return,
                'max_drawdown': avg_drawdown,
                'win_rate': avg_win_rate
            },
            'fold_results': [
                {
                    'fold': i,
                    'sharpe_ratio': r.metrics.sharpe_ratio,
                    'total_return': r.metrics.total_return,
                    'max_drawdown': r.metrics.max_drawdown,
                    'win_rate': r.metrics.win_rate
                }
                for i, r in enumerate(cv_results)
            ]
        }
        cv_file = output_dir / f"cv_results_{strategy}.json"
        with open(cv_file, 'w', encoding='utf-8') as f:
            json.dump(cv_summary, f, indent=2)
        logger.info({"event": "cv_results_saved", "file": str(cv_file)})


def _save_backtest_result(
    result: Any,
    strategy: str,
    strategy_params: dict,
    parsed_args: argparse.Namespace,
    output_dir: Path,
    logger: Any
) -> None:
    """
    Save and log single backtest result.

    Args:
        result: Backtest result object.
        strategy: Name of the strategy.
        strategy_params: Parameters for the strategy.
        parsed_args: Parsed command-line arguments.
        output_dir: Directory to save results.
        logger: Logger instance.
    """
    import json
    result_data = {
        'strategy': strategy,
        'parameters': strategy_params,
        'metrics': {
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'total_return': result.metrics.total_return,
            'max_drawdown': result.metrics.max_drawdown,
            'win_rate': result.metrics.win_rate,
            'volatility': result.metrics.volatility,
            'calmar_ratio': result.metrics.calmar_ratio,
            'sortino_ratio': result.metrics.sortino_ratio
        },
        'trades': len(result.trades),
        'equity_curve': result.equity_curve.tolist() if hasattr(result.equity_curve, 'tolist') else result.equity_curve
    }
    result_file = output_dir / f"backtest_{strategy}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    logger.info({"event": "backtest_result_saved", "file": str(result_file)})


def _log_single_result(result: Any, logger: Any) -> None:
    """
    Log metrics for a single backtest result.

    Args:
        result: Backtest result object.
        logger: Logger instance.
    """
    logger.info({
        "event": "single_result",
        "sharpe_ratio": result.metrics.sharpe_ratio,
        "total_return": result.metrics.total_return,
        "max_drawdown": result.metrics.max_drawdown,
        "win_rate": result.metrics.win_rate
    })


def _log_comparison(results: dict, logger: Any) -> None:
    """
    Log a comparison table for multiple strategies.

    Args:
        results: Dictionary of strategy name to result object.
        logger: Logger instance.
    """
    comparison = []
    for strategy, result in results.items():
        if result:
            comparison.append({
                "strategy": strategy,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "total_return": result.metrics.total_return,
                "max_drawdown": result.metrics.max_drawdown,
                "win_rate": result.metrics.win_rate
            })
        else:
            comparison.append({
                "strategy": strategy,
                "sharpe_ratio": None,
                "total_return": None,
                "max_drawdown": None,
                "win_rate": None
            })
    logger.info({"event": "strategy_comparison", "results": comparison})


def extract_strategy_parameters(args: argparse.Namespace) -> dict:
    strategy_params = {}
    
    if args.strategy == 'moving_average':
        strategy_params['short_window'] = args.short_window
        strategy_params['long_window'] = args.long_window
    
    elif args.strategy == 'mean_reversion':
        strategy_params['window'] = args.window
        strategy_params['std_dev'] = args.std_dev
    
    elif args.strategy == 'momentum':
        strategy_params['lookback_period'] = args.lookback_period
        strategy_params['momentum_threshold'] = args.momentum_threshold
    
    elif args.strategy == 'ppo':
        strategy_params['model'] = args.model_path
    
    return strategy_params


def run_cross_validation(parsed_args, strategy, strategy_params, backtest_engine, output_dir, logger):
    """
    Run cross-validation for the given strategy.
    """
    logger.info(f"Running cross-validation for {strategy} strategy")
    cv_engine = CrossValidationEngine(backtest_engine)
    cv_results = cv_engine.run_time_series_cv(
        strategy=strategy,
        n_splits=parsed_args.n_splits,
        test_size=parsed_args.test_size,
        **strategy_params
    )
    if cv_results:
        logger.info(f"Cross-validation completed with {len(cv_results)} folds")
        _save_cv_results(cv_results, strategy, strategy_params, parsed_args, output_dir, logger)


def run_single_backtest(parsed_args, strategy, strategy_params, backtest_engine, output_dir, logger, results):
    """
    Run a single backtest for the given strategy.
    """
    logger.info(f"Testing {strategy} strategy...")
    try:
        result = backtest_engine.run_backtest(
            strategy=strategy,
            length=parsed_args.length or 1000,
            **strategy_params
        )
        results[strategy] = result
        _log_single_result(result, logger)
        if parsed_args.save_results:
            _save_backtest_result(result, strategy, strategy_params, parsed_args, output_dir, logger)
    except Exception as e:
        logger.error(f"Error testing {strategy}: {e}")
        results[strategy] = None
    if parsed_args.compare and len(results) > 1:
        _log_comparison(results, logger)


def backtest_command(args: Optional[list] = None) -> None:
    """
    Run backtesting on trading strategies.

    Args:
        args: Command line arguments (if None, uses sys.argv)
    """
    parser = argparse.ArgumentParser(
        description="Run backtesting on trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run moving average backtest
  python -m evo.cli.backtest moving_average --short-window 10 --long-window 30

  # Run mean reversion backtest
  python -m evo.cli.backtest mean_reversion --window 20 --std-dev 2.0

  # Run momentum backtest
  python -m evo.cli.backtest momentum --lookback-period 20 --momentum-threshold 0.02

  # Run PPO backtest
  python -m evo.cli.backtest ppo --model-path path/to/model.zip

  # Run random strategy backtest
  python -m evo.cli.backtest random
        """
    )

    # -- Configuration ------------------------------------
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c', 
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
        '--input', 
        type=Path, 
        required=True,
        help='Path to CSV file with data to backtest on'
    )
    data_group.add_argument(
        '--features', 
        type=str, 
        default='open,high,low,close,volume',
        help='Comma-separated list of features to use'
    )

    # -- Backtesting Parameters ---------------------------
    backtest_group = parser.add_argument_group('Backtesting')
    backtest_group.add_argument(
        '--length', '-l', 
        type=int, 
        help='Length of backtest in periods'
    )
    backtest_group.add_argument(
        '--initial-capital',
        type=float, 
        default=100000.0,
        help='Initial capital for backtesting'
    )
    backtest_group.add_argument(
        '--commission',
        type=float, 
        default=0.0,
        help='Commission rate for backtesting'
    )
    backtest_group.add_argument(
        '--slippage',
        type=float, 
        default=0.0,
        help='Slippage rate for backtesting'
    )

    # -- Cross-Validation ---------------------------------
    cv_group = parser.add_argument_group('Cross-Validation')
    cv_group.add_argument(
        '--cross-validation', '-cv',
        action='store_true', 
        help='Run cross-validation instead of single backtest'
    )
    cv_group.add_argument(
        '--n-splits', 
        type=int, 
        default=5, 
        help='Number of splits for cross-validation'
    )
    cv_group.add_argument(
        '--test-size', 
        type=float, 
        default=0.2, 
        help='Test size fraction for cross-validation'
    )

    # -- Output Parameters --------------------------------
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--compare', 
        action='store_true', 
        default=False,
        help='Compare multiple strategies side by side'
    )
    output_group.add_argument(
        '--output-dir', 
        type=Path, 
        default=Path('backtest_results'), 
        help='Directory to save backtest results'
    )
    output_group.add_argument(
        '--save-results', 
        action='store_true', 
        help='Save detailed results to files'
    )
    output_group.add_argument(
        '--plot', 
        action='store_true', 
        help='Generate performance plots'
    )

    # -- Logging Parameters -------------------------------
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO', 
        help='Logging level'
    )

    # Subparsers for strategies
    subparsers = parser.add_subparsers(dest='strategy', required=True, title='Strategies', description='Available trading strategies')

    # -- Moving Average -----------------------------------
    ma_parser = subparsers.add_parser(
        'moving_average', 
        help='Moving Average Crossover strategy'
    )
    ma_parser.add_argument(
        '--short-window', 
        type=int, 
        required=True, 
        help='Short window for moving average'
    )
    ma_parser.add_argument(
        '--long-window', 
        type=int, 
        required=True, 
        help='Long window for moving average'
    )

    # -- Mean Reversion -----------------------------------
    mr_parser = subparsers.add_parser('mean_reversion', help='Mean Reversion (Bollinger Bands) strategy')
    mr_parser.add_argument(
        '--window', 
        type=int, 
        required=True, 
        help='Window size for mean reversion'
    )
    mr_parser.add_argument(
        '--std-dev', 
        type=float, 
        required=True, 
        help='Standard deviation threshold for mean reversion'
    )

    # -- Momentum -----------------------------------------
    mom_parser = subparsers.add_parser('momentum', help='Momentum strategy')
    mom_parser.add_argument(
        '--lookback-period', 
        type=int, 
        required=True, 
        help='Lookback period for momentum'
    )
    mom_parser.add_argument(
        '--momentum-threshold', 
        type=float, 
        required=True, 
        help='Momentum threshold for momentum strategy'
    )

    # -- PPO ----------------------------------------------
    ppo_parser = subparsers.add_parser('ppo', help='PPO RL model strategy')
    ppo_parser.add_argument(
        '--model-path', 
        type=str, 
        required=True, 
        help='Path to trained PPO model'
    )

    # -- Random -------------------------------------------
    rand_parser = subparsers.add_parser(
        'random', 
        help='Random trading strategy (for testing/comparison)')
    

    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(level=parsed_args.log_level)
    logger = get_logger(__name__)

    try:
        # Load configuration
        config = _load_config(parsed_args)
        logger.info("Starting backtesting")
        logger.info(f"Configuration: {config}")
        
        # Create output directory
        output_dir = parsed_args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup data provider
        data_provider = CSVDataProvider(
            data_path=config.data.data_path,
            feature_columns=config.data.features
        )
        
        # Setup backtesting engine
        backtest_engine = BacktestEngine(
            data_provider=data_provider,
            initial_capital=config.trading.initial_capital,
            commission=config.trading.commission,
            slippage=getattr(config.trading, 'slippage', 0.0),
            risk_free_rate=0.02
        )
        
        # Strategy-specific parameter extraction
        strategy = parsed_args.strategy
        strategy_params = extract_strategy_parameters(parsed_args)
        results = {}

        # Run cross-validation or single backtest
        if getattr(parsed_args, 'cv', False):
            run_cross_validation(parsed_args, strategy, strategy_params, backtest_engine, output_dir, logger)
        else:
            run_single_backtest(parsed_args, strategy, strategy_params, backtest_engine, output_dir, logger, results)
        logger.info("Backtesting completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Backtesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    backtest_command() 