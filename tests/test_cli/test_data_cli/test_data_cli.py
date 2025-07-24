import pytest
import pandas as pd
from pathlib import Path
from evo.cli.data import data_command
from unittest.mock import patch, MagicMock

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cli,
    pytest.mark.data
]


def test_data_cli_missing_required_argument(temp_dir):
    """
    Test the data CLI with missing required argument (no --input for process).
    Should log an error and exit.
    """
    args = [
        'process',
        # '--input' is missing
        '--output', str(temp_dir / 'processed.csv')
    ]
    with pytest.raises(SystemExit):
        data_command(args) 


@pytest.fixture
def minimal_csv_file(temp_data_dir):
    """Create a minimal CSV file for data CLI tests."""
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


def test_data_cli_download_mocks_alpaca(temp_dir, config_file):
    """
    Test the data CLI 'download' subcommand with Alpaca provider mocked.
    """
    # Mock the data
    output_path = temp_dir / 'downloaded.csv'
    mock_df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=5, freq='D'),
        'open': [1, 2, 3, 4, 5],
        'high': [2, 3, 4, 5, 6],
        'low': [0, 1, 2, 3, 4],
        'close': [1, 2, 3, 4, 5],
        'volume': [10, 20, 30, 40, 50]
    })
    
    # Mock the data provider
    with patch('evo.cli.data.AlpacaDataProvider') as MockProvider:
        instance = MockProvider.return_value
        instance.get_historical_bars.return_value = mock_df
        args = [
            'download',
            '--provider', 'alpaca',
            '--symbol', 'AAPL',
            '--start', '2023-01-01',
            '--end', '2023-01-05',
            '--timeframe', '1Day',
            '--output', str(output_path),
            '--config', str(config_file)
        ]
        # Patch asyncio event loop to run synchronously
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_until_complete = lambda coro: mock_df
            data_command(args)
    
    # Check if the file was created
    assert output_path.exists(), "Downloaded CSV was not created."
    
    # Check if the file has the correct columns
    df = pd.read_csv(output_path)
    for feature in ['open', 'high', 'low', 'close']:
        assert feature in df.columns, f"Downloaded data missing {feature} column."


def test_data_cli_process_features_and_normalize(minimal_csv_file, temp_dir):
    """
    Test the data CLI 'process' subcommand with features and normalization.
    """
    # Run the process command
    output_path = temp_dir / 'processed.csv'
    args = [
        'process',
        '--input', str(minimal_csv_file),
        '--features', 'sma_5,ema_12',
        '--normalize', 'standard',
        '--output', str(output_path)
    ]
    data_command(args)
    
    # Check if the file was created
    assert output_path.exists(), "Processed CSV was not created."
    
    # Check if the file has the correct columns
    df = pd.read_csv(output_path)
    for feature in ['sma_5', 'ema_12']:
        assert feature in df.columns, f"Engineered feature {feature} not added."


def test_data_cli_process_no_features(minimal_csv_file, temp_dir):
    """
    Test the data CLI 'process' subcommand with no features or normalization.
    """
    # Run the process command
    output_path = temp_dir / 'processed.csv'
    args = [
        'process',
        '--input', str(minimal_csv_file),
        '--output', str(output_path)
    ]
    data_command(args)

    # Check if the file was created
    assert output_path.exists(), "Processed CSV was not created."
    
    # Check if the file has the correct columns
    df = pd.read_csv(output_path)
    for feature in ['open', 'high', 'low', 'close']:
        assert feature in df.columns, f"Processed CSV missing {feature} column."