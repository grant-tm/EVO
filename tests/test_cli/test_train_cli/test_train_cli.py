import pytest
import pandas as pd
from pathlib import Path
from evo.cli.train import train_command

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cli,
    pytest.mark.training
]

@pytest.fixture
def minimal_csv_file(temp_data_dir):
    """Create a minimal CSV file for train CLI tests."""
    data_length = 500
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=data_length, freq='D'),
        'open': [100 + i for i in range(data_length)],
        'high': [101 + i for i in range(data_length)],
        'low': [99 + i for i in range(data_length)],
        'close': [100 + i for i in range(data_length)],
        'volume': [1000 + 10 * i for i in range(data_length)]
    })
    csv_path = temp_data_dir / 'train_data.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.mark.slow
def test_train_cli_minimal_run(minimal_csv_file, config_file, temp_dir):
    """
    Test the train CLI with minimal required arguments for PPO agent.
    """
    args = [
        '--data-path', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        'ppo',
        '--env', 'TradingEnv',
        '--epochs', '1',
        '--batch-size', '4',
        '--learning-rate', '0.001',
        '--max-timesteps', '1'
    ]
    train_command(args)
    # Check that a model directory was created
    model_dir = temp_dir / 'ppo_model'
    assert model_dir.exists() or any(model_dir.parent.glob('ppo_model*')), "Model directory was not created."

@pytest.mark.slow
def test_train_cli_metrics_file(minimal_csv_file, config_file, temp_dir):
    """
    Test the train CLI with metrics saving enabled.
    """
    args = [
        '--data-path', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        '--save-metrics',
        'ppo',
        '--env', 'TradingEnv',
        '--epochs', '1',
        '--batch-size', '4',
        '--learning-rate', '0.001',
        '--max-timesteps', '1'
    ]
    train_command(args)
    metrics_files = list(temp_dir.glob('ppo_model_metrics.json'))
    assert metrics_files, "Metrics file was not created."

def test_train_cli_missing_required_argument(minimal_csv_file, config_file, temp_dir):
    """
    Test the train CLI with missing required argument (no --env for PPO).
    Should log an error and exit.
    """
    args = [
        '--data-path', str(minimal_csv_file),
        '--config', str(config_file),
        '--output-dir', str(temp_dir),
        'ppo',
        # '--env' is missing
        '--epochs', '1',
        '--batch-size', '4',
        '--learning-rate', '0.001',
        '--max-timesteps', '1'
    ]
    with pytest.raises(SystemExit):
        train_command(args)
