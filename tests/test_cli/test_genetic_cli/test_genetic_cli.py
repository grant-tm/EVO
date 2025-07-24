import pytest
import pandas as pd
from pathlib import Path
from evo.cli.genetic import genetic_command
import json

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cli,
    pytest.mark.genetic
]


@pytest.fixture
def minimal_csv_file(temp_data_dir):
    """Create a minimal CSV file for genetic CLI tests."""
    data_length = 500
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=data_length, freq='D'),
        'open': [100 + i for i in range(data_length)],
        'high': [101 + i for i in range(data_length)],
        'low': [99 + i for i in range(data_length)],
        'close': [100 + i for i in range(data_length)],
        'volume': [1000 + 10 * i for i in range(data_length)]
    })
    csv_path = temp_data_dir / 'genetic_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def minimal_config_file(tmp_path):
    # Load the default config
    with open("config/default_config.json") as f:
        config = json.load(f)
    
    # Override training_steps
    config["training_steps"] = 1
    config["epochs"] = 1
    
    # Write to a temp file
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


def test_genetic_cli_missing_required_argument(config_file, temp_dir):
    """
    Test the genetic CLI with missing required argument (no --data-path).
    Should log an error and exit.
    """
    args = [
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--population-size', '4',
        '--max-generations', '1',
        '--num-backtests', '1',
        '--backtest-length', '10',
        '--log-level', 'INFO',
    ]
    with pytest.raises(SystemExit):
        genetic_command(args) 


@pytest.mark.slow
def test_genetic_cli_minimal_run(minimal_csv_file, minimal_config_file, temp_dir):
    """
    Test the genetic CLI with minimal required arguments.
    """
    args = [
        '--data-path', str(minimal_csv_file),
        '--config', str(minimal_config_file),
        '--output-dir', str(temp_dir),
        '--population-size', '4',
        '--max-generations', '1',
        '--num-backtests', '1',
        '--backtest-length', '10',
        '--log-level', 'INFO',
    ]
    genetic_command(args)
    # Check that results directory was created
    assert temp_dir.exists(), "Output directory was not created."


@pytest.mark.slow
def test_genetic_cli_save_history(minimal_csv_file, minimal_config_file, temp_dir):
    """
    Test the genetic CLI with search history saving enabled.
    """
    args = [
        '--data-path', str(minimal_csv_file),
        '--config', str(minimal_config_file),
        '--output-dir', str(temp_dir),
        '--population-size', '4',
        '--max-generations', '1',
        '--num-backtests', '1',
        '--backtest-length', '10',
        '--save-history',
        '--log-level', 'INFO',
    ]
    genetic_command(args)
    # Check for a file that could represent saved history (implementation-dependent)
    # This is a placeholder; adjust if a specific file is created
    assert temp_dir.exists(), "Output directory was not created."