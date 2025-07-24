"""
Data CLI for the EVO trading system.

This module provides a command-line interface for downloading and processing datasets.
"""

import sys
import pandas as pd
import asyncio
import argparse
from pathlib import Path
from typing import Optional

from datetime import datetime, timedelta
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

from ..core.config import Config, get_config
from ..core.logging import setup_logging, get_logger
from ..data import FeatureEngineer, DataNormalizer, DataStore
from ..data.providers.alpaca_provider import AlpacaDataProvider


def _load_config(parsed_args: argparse.Namespace) -> Config:
    """
    Load and override configuration from command-line arguments.
    """
    config = get_config()
    if parsed_args.config:
        config = Config(config_file=parsed_args.config, env_file=parsed_args.env_file)
    return config


def parse_flexible_date(value: str) -> datetime:
    """
    Parse a date string in flexible formats, including relative dates.
    Supports:
      - Absolute: 2023-01-01, 01/01/2023, 2023/01/01, etc.
      - Relative: -1d, -1w, -1y, etc.
    Returns a datetime object.
    Raises ValueError if parsing fails.
    """
    value = value.strip().lower()
    now = datetime.now()
    
    # Parse relative offset
    if value.startswith("-"):
        
        # Parse number and unit
        num = ''
        unit = ''
        for c in value[1:]:
            if c.isdigit():
                num += c
            else:
                unit += c
        if not num:
            raise ValueError(f"Invalid relative date: {value}")
        
        # Calculate relative date
        num = int(num)
        if unit == "d":
            dt = now - relativedelta(days=num)
        elif unit == "w":
            dt = now - relativedelta(weeks=num)
        elif unit == "m":
            dt = now - relativedelta(months=num)
        elif unit == "y":
            dt = now - relativedelta(years=num)
        else:
            raise ValueError(f"Unknown relative date unit: {unit} in {value}")
        
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Try absolute date parsing
    try:
        return date_parser.parse(value)
    except Exception:
        raise ValueError(f"Could not parse date: {value}. Try YYYY-MM-DD, MM/DD/YYYY, today, yesterday, or -7d.")


def download_data_command(parsed_args, logger):
    """Handle the 'download' subcommand."""
    
    # Load config
    config = _load_config(parsed_args)
    if parsed_args.provider == "alpaca":
        provider = AlpacaDataProvider(config)
    else:
        logger.error(f"Provider {parsed_args.provider} not supported.")
        sys.exit(1)
    
    # Parse dates
    try:
        start_dt = parse_flexible_date(parsed_args.start)
        end_dt = datetime.now()
        if not parsed_args.end == "today":
            end_dt = parse_flexible_date(parsed_args.end)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Download data
    loop = asyncio.get_event_loop()
    df = loop.run_until_complete(provider.get_historical_bars(
        symbol=parsed_args.symbol,
        start_time=start_dt,
        end_time=end_dt,
        timeframe=parsed_args.timeframe
    ))
    df.to_csv(parsed_args.output, index=False)
    logger.info(f"Downloaded data for {parsed_args.symbol} to {parsed_args.output}")


def process_data_command(parsed_args, logger):
    """Handle the 'process' subcommand."""
    df = pd.read_csv(parsed_args.input)
    
    # Feature engineering
    if parsed_args.features:
        indicators = [f.strip() for f in parsed_args.features.split(",")]
        fe = FeatureEngineer()
        df = fe.calculate_features(df, indicators=indicators)
        logger.info(f"Added features: {indicators}")
    
    # Normalization
    if parsed_args.normalize:
        normalizer = DataNormalizer({"method": parsed_args.normalize})
        normalizer.fit(df)
        df = normalizer.transform(df)
        logger.info(f"Applied normalization: {parsed_args.normalize}")
    
    # Save processed data
    df.to_csv(parsed_args.output, index=False)
    logger.info(f"Processed data saved to {parsed_args.output}")


def data_command(args: Optional[list] = None) -> None:
    """
    CLI for downloading and processing datasets.
    """
    parser = argparse.ArgumentParser(
        description="Download and process datasets for EVO trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download the previous month of AAPL data from Alpaca
  python -m evo.cli.data download --provider=alpaca --symbol=AAPL --start=-1m --timeframe=1Day --output=AAPL.csv

  # Process a CSV file with feature engineering and normalization
  python -m evo.cli.data process --input=AAPL.csv --features=sma_20,ema_10 --normalize=standard --output=AAPL_processed.csv
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Download Subcommand ---
    download_parser = subparsers.add_parser(
        "download", 
        help="Download market data from a provider"
    )
    download_parser.add_argument(
        "--output", 
        type=Path, 
        required=True, 
        help="Output file path (CSV)"
    )
    download_parser.add_argument(
        "--symbol", 
        type=str, 
        required=True, 
        help="Symbol to download (e.g., AAPL)"
    )
    download_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD or relative, e.g., 2023-01-01, -7d, today)"
    )
    download_parser.add_argument(
        "--end", 
        type=str, 
        default="today", 
        help="End date (YYYY-MM-DD or relative, e.g., 2023-06-01, -1w, yesterday)"
    )
    download_parser.add_argument(
        "--provider", 
        type=str, 
        default="alpaca", 
        help="Data provider (default: alpaca)"
    )
    download_parser.add_argument(
        "--timeframe", 
        type=str, 
        default="1Day", 
        help="Timeframe (e.g., 1Min, 1Day)"
    )
    download_parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (JSON)"
    )
    download_parser.add_argument(
        "--env-file", 
        type=Path, 
        help="Path to .env file for API keys"
    )

    # --- Process Subcommand ---
    process_parser = subparsers.add_parser(
        "process", 
        help="Process a dataset with feature engineering and normalization"
    )
    process_parser.add_argument(
        "--input", 
        type=Path, 
        required=True, 
        help="Input CSV file"
    )
    process_parser.add_argument(
        "--features",
        type=str,
        help="Comma-separated list of features/indicators to add"
    )
    process_parser.add_argument(
        "--normalize",
        type=str,
        choices=["standard", "minmax", "robust"], help="Normalization method"
    )
    process_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path (CSV)"
    )
    process_parser.add_argument(
        "--config",
        type=Path, 
        help="Path to configuration file (JSON)"
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    setup_logging()
    logger = get_logger("evo.cli.data")

    # Dispatch to subcommand
    if parsed_args.command == "download":
        download_data_command(parsed_args, logger)
    elif parsed_args.command == "process":
        process_data_command(parsed_args, logger)
    else:
        parser.print_help()
        sys.exit(1) 

if __name__ == "__main__":
    data_command()