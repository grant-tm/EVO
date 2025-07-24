import pytest
import pandas as pd
from pathlib import Path
from evo.cli.backtest import backtest_command
from unittest.mock import patch

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cli,
    pytest.mark.backtesting
]

@pytest.fixture
def minimal_csv_file(temp_data_dir):
    """Create a minimal CSV file for backtest CLI tests."""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
        'open': [100 + i for i in range(10)],
        'high': [101 + i for i in range(10)],
        'low': [99 + i for i in range(10)],
        'close': [100 + i for i in range(10)],
        'volume': [1000 + 10 * i for i in range(10)]
    })
    csv_path = temp_data_dir / 'test_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

def test_backtest_cli_random_strategy(minimal_csv_file, config_file, temp_dir):
    """
    Test the backtest CLI with the random strategy and minimal CSV data.
    """
    args = [
        '--input', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--length', '5',
        '--log-level', 'INFO',
        '--save-results',
        'random'
    ]
    backtest_command(args)
    
    # Read the log file and check for the expected event
    log_file = Path('logs/evo.log')
    assert log_file.exists(), "Log file does not exist."
    log_content = log_file.read_text()
    assert 'backtest_result_saved' in log_content, "Expected 'backtest_result_saved' event in logs."

def test_backtest_cli_moving_average(minimal_csv_file, config_file, temp_dir):
    """
    Test the backtest CLI with the moving_average strategy.
    """
    args = [
        '--input', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--length', '5',
        '--log-level', 'INFO',
        '--save-results',
        'moving_average',
        '--short-window', '2',
        '--long-window', '5'
    ]
    backtest_command(args)
    
    log_file = Path('logs/evo.log')
    assert log_file.exists(), "Log file does not exist."
    log_content = log_file.read_text()
    assert 'backtest_result_saved' in log_content, "Expected 'backtest_result_saved' event in logs."

def test_backtest_cli_cross_validation(minimal_csv_file, config_file, temp_dir):
    """
    Test the backtest CLI with cross-validation mode for moving_average strategy.
    """
    args = [
        '--input', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--length', '5',
        '--log-level', 'INFO',
        '--save-results',
        '--cross-validation',
        '--n-splits', '2',
        '--test-size', '0.5',
        'moving_average',
        '--short-window', '2',
        '--long-window', '5'
    ]
    backtest_command(args)
    
    log_file = Path('logs/evo.log')
    assert log_file.exists(), "Log file does not exist."
    log_content = log_file.read_text()
    assert 'backtest_result_saved' in log_content, "Expected 'backtest_result_saved' event in logs."

def test_backtest_cli_mean_reversion(minimal_csv_file, config_file, temp_dir):
    """
    Test the backtest CLI with the mean_reversion strategy.
    """
    args = [
        '--input', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--length', '5',
        '--log-level', 'INFO',
        '--save-results',
        'mean_reversion',
        '--window', '3',
        '--std-dev', '1.0'
    ]
    backtest_command(args)
    
    log_file = Path('logs/evo.log')
    assert log_file.exists(), "Log file does not exist."
    log_content = log_file.read_text()
    assert 'backtest_result_saved' in log_content, "Expected 'backtest_result_saved' event in logs."

def test_backtest_cli_momentum(minimal_csv_file, config_file, temp_dir):
    """
    Test the backtest CLI with the momentum strategy.
    """
    args = [
        '--input', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--length', '5',
        '--log-level', 'INFO',
        '--save-results',
        'momentum',
        '--lookback-period', '2',
        '--momentum-threshold', '0.01'
    ]
    backtest_command(args)

    log_file = Path('logs/evo.log')
    assert log_file.exists(), "Log file does not exist."
    log_content = log_file.read_text()
    assert 'backtest_result_saved' in log_content, "Expected 'backtest_result_saved' event in logs."

def test_backtest_cli_ppo_missing_model(minimal_csv_file, config_file, temp_dir):
    """
    Test the backtest CLI with the PPO strategy and a missing model file.
    Should log an error and continue gracefully.
    """
    args = [
        '--input', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--length', '5',
        '--log-level', 'INFO',
        '--save-results',
        'ppo',
        '--model-path', str(temp_dir / 'nonexistent_model.zip')
    ]
    backtest_command(args)
    
    log_file = Path('logs/evo.log')
    assert log_file.exists(), "Log file does not exist."
    log_content = log_file.read_text()
    assert (
        "Error generating PPO signal" in log_content and "object has no attribute 'predict'" in log_content
    ), "Expected error log for missing PPO model file."

def test_backtest_cli_missing_required_argument(minimal_csv_file, config_file, temp_dir):
    """
    Test the backtest CLI with missing required argument (no --short-window for moving_average).
    Should log an error and exit.
    """
    args = [
        '--input', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--length', '5',
        '--log-level', 'INFO',
        '--save-results',
        'moving_average',
        # '--short-window' is missing
        '--long-window', '5'
    ]
    with pytest.raises(SystemExit):
        backtest_command(args)

    log_file = Path('logs/evo.log')
    assert log_file.exists(), "Log file does not exist."
    log_content = log_file.read_text().lower()
    assert (
        'the following arguments are required' in log_content or 'error' in log_content
    ), "Expected error log for missing required argument." 