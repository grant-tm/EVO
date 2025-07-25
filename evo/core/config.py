"""
Configuration management for the EVO trading system.

This module provides file-based configuration for trading, data, training, reward,and optimization. 
Alpaca API keys areloaded from environment variables.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from dotenv import load_dotenv

from .exceptions import ConfigurationError
from .logging import get_logger


@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    symbol: str = "SPY"
    trade_qty: int = 1
    use_simulation: bool = True
    sim_speed: float = 0.8
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% commission
    update_interval: int = 30  # seconds between updates
    close_positions_on_stop: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: str = "preprocessed_training_data.csv"
    seq_len: int = 15
    features: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume",
        "return", "volatility", "sma_5", "sma_20", "rsi", "macd"
    ])
    start_time: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365*2))
    end_time: datetime = field(default_factory=lambda: datetime.now())


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_dir: str = "trained_models"
    model_name: str = "ppo_trading_agent.zip"
    training_steps: int = 1_000_000
    
    # Default PPO hyperparameters
    learning_rate: float = 0.0003
    clip_range: float = 0.2
    batch_size: int = 64
    entropy_coef_init: float = 0.1
    entropy_coef_final: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.95


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    tp_pct: float = 1 / 100
    sl_pct: float = 1 / 100
    idle_penalty: float = 0.0003
    sl_penalty_coef: float = 10.0
    tp_reward_coef: float = 15.0
    timeout_duration: int = 3
    timeout_reward_coef: float = 2.0
    ongoing_reward_coef: float = 0.2
    reward_clip_range: tuple = (-1.0, 1.0)
    max_episode_steps: int = 5000


@dataclass
class OptimizationConfig:
    """Configuration for genetic optimization."""
    mutation_rate: float = 0.1
    elite_proportion: float = 0.2
    num_backtests: int = 500
    backtest_length: int = 360


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca API."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    paper_trading: bool = True


@dataclass
class BrokerConfig:
    """Configuration for broker connections."""
    # Alpaca-specific configuration
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    
    # Generic broker settings
    connection_timeout: int = 30
    request_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    requests_per_second: int = 10
    burst_limit: int = 50
    
    # Logging
    log_orders: bool = True
    log_positions: bool = True
    log_account: bool = True


@dataclass
class RiskLimitsConfig:
    """Configuration for risk management limits."""
    max_position_size: float = 0.1  # Max 10% of portfolio per position
    max_portfolio_exposure: float = 0.5  # Max 50% of portfolio in positions
    max_drawdown: float = 0.15  # Max 15% drawdown
    max_daily_loss: float = 0.05  # Max 5% daily loss
    max_correlation_exposure: float = 0.3  # Max 30% in correlated assets
    stop_loss_pct: float = 0.02  # 2% stop loss per position
    take_profit_pct: float = 0.04  # 4% take profit per position
    max_orders_per_day: int = 50  # Max 50 orders per day
    min_order_size: float = 100.0  # Minimum order size in USD
    max_order_size: float = 10000.0  # Maximum order size in USD


@dataclass
class PositionConfig:
    """Configuration for position management."""
    default_stop_loss_pct: float = 0.02  # 2% stop loss
    default_take_profit_pct: float = 0.04  # 4% take profit
    trailing_stop_pct: float = 0.01  # 1% trailing stop
    max_positions: int = 10  # Maximum number of concurrent positions
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility
    kelly_fraction: float = 0.25  # Fraction of Kelly criterion to use
    volatility_lookback: int = 20  # Days for volatility calculation


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and alerts."""
    monitor_interval: int = 30  # seconds between monitoring updates
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown': 0.15,
        'daily_loss': 0.05,
        'position_size': 0.2,
        'data_latency': 30
    })


@dataclass
class ExecutionConfig:
    """Configuration for execution layer."""
    risk_limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
    position_config: PositionConfig = field(default_factory=PositionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


class Config:
    """
    Main configuration class for the EVO trading system.
    
    This class manages all configuration settings with file-based configuration
    and API keys loaded from environment variables for security.
    """
    
    def __init__(self, config_file: Optional[Path] = None, env_file: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to JSON configuration file
            env_file: Path to .env file for API keys only
        """
        self.logger = get_logger(__name__)
        
        # Load API keys from environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Initialize configuration sections
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.reward = RewardConfig()
        self.optimization = OptimizationConfig()
        self.alpaca = AlpacaConfig()
        self.broker = BrokerConfig()
        self.execution = ExecutionConfig()
        
        # Load from config file if provided
        if config_file and config_file.exists():
            self._load_from_file(config_file)
        
        # Load API keys from environment variables
        self._load_api_keys()
        
        # Validate configuration
        self._validate()
        
        self.logger.info("Configuration loaded successfully")
    
    def __getitem__(self, key):
        """Allow dict-style access to configuration sections."""
        # Get the attribute by name
        try:
            attr = getattr(self, key)
        except AttributeError:
            raise KeyError(key)
        
        # If the attribute is a Config, allow further dict-style access
        if isinstance(attr, Config):
            return attr
        
        return attr

    def __setitem__(self, key, value):
        """Allow dict-style assignment to configuration sections."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(key)

    def _load_from_file(self, config_file: Path):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update each section
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    self._update_section(section, section_data)
            
            self.logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {config_file}: {str(e)}")
    
    def _update_section(self, section, section_data):
        """Recursively update a configuration section."""
        for key, value in section_data.items():
            if hasattr(section, key):
                current_value = getattr(section, key)
                
                # If the current value is a dataclass and the new value is a dict,
                # recursively update the dataclass
                if hasattr(current_value, '__dataclass_fields__') and isinstance(value, dict):
                    self._update_section(current_value, value)
                else:
                    # Otherwise, set the value directly
                    setattr(section, key, value)
    
    def _load_api_keys(self):
        """Load API keys from environment variables."""
        # Load Alpaca API keys from environment
        if os.getenv("ALPACA_API_KEY"):
            self.alpaca.api_key = os.getenv("ALPACA_API_KEY")
            self.broker.alpaca.api_key = os.getenv("ALPACA_API_KEY")
        if os.getenv("ALPACA_SECRET_KEY"):
            self.alpaca.api_secret = os.getenv("ALPACA_SECRET_KEY")
            self.broker.alpaca.api_secret = os.getenv("ALPACA_SECRET_KEY")
        
        self.logger.debug("API keys loaded from environment variables")
    
    def _validate(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate trading configuration
        if self.trading.trade_qty <= 0:
            errors.append("Trade quantity must be positive and greater than 0")
        
        if self.trading.sim_speed <= 0:
            errors.append("Simulation speed must be positive and greater than 0")
        
        # Validate data configuration
        if self.data.seq_len <= 0:
            errors.append("Sequence length must be positive and greater than 0")
        
        if not self.data.features:
            errors.append("Features list cannot be empty")
        
        # Validate training configuration
        if self.training.training_steps <= 0:
            errors.append("Training steps must be positive and greater than 0")
        
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive and greater than 0W")
        
        # Validate reward configuration
        if self.reward.tp_pct <= 0 or self.reward.sl_pct <= 0:
            errors.append("Take profit and stop loss percentages must be positive")
        
        if self.reward.max_episode_steps <= 0:
            errors.append("Max episode steps must be positive and greater than 0")
        
        # Validate optimization configuration
        if not 0 <= self.optimization.mutation_rate <= 1:
            errors.append("Mutation rate must be between 0 and 1")
        
        if not 0 <= self.optimization.elite_proportion <= 1:
            errors.append("Elite proportion must be between 0 and 1")
        
        # Validate Alpaca configuration for live trading
        if not self.trading.use_simulation:
            if not self.alpaca.api_key:
                errors.append("Alpaca API key is required for live trading")
            if not self.alpaca.api_secret:
                errors.append("Alpaca API secret is required for live trading")
        
        if errors:
            raise ConfigurationError("Configuration validation failed", details=errors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "trading": self.trading.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "reward": self.reward.__dict__,
            "optimization": self.optimization.__dict__,
            "alpaca": {
                "api_key": "***" if self.alpaca.api_key else None,
                "api_secret": "***" if self.alpaca.api_secret else None,
                "paper_trading": self.alpaca.paper_trading
            },
            "broker": {
                "alpaca": {
                    "api_key": "***" if self.broker.alpaca.api_key else None,
                    "api_secret": "***" if self.broker.alpaca.api_secret else None,
                    "paper_trading": self.broker.alpaca.paper_trading
                },
                "connection_timeout": self.broker.connection_timeout,
                "request_timeout": self.broker.request_timeout,
                "max_retries": self.broker.max_retries,
                "retry_delay": self.broker.retry_delay,
                "requests_per_second": self.broker.requests_per_second,
                "burst_limit": self.broker.burst_limit,
                "log_orders": self.broker.log_orders,
                "log_positions": self.broker.log_positions,
                "log_account": self.broker.log_account
            },
            "execution": self.execution.__dict__
        }
    
    def save(self, config_file: Path):
        """Save configuration to JSON file."""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save config file {config_file}: {str(e)}")
    
    def get_model_path(self) -> Path:
        """Get the full path to the model file."""
        return Path(self.training.model_dir) / self.training.model_name
    
    def get_data_path(self) -> Path:
        """Get the full path to the data file."""
        return Path(self.data.data_path)
    
    def __repr__(self) -> str:
        return f"Config(trading={self.trading.symbol}, simulation={self.trading.use_simulation})"


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config):
    """
    Set the global configuration instance.
    
    Args:
        config: Configuration instance to set
    """
    global _config
    _config = config 