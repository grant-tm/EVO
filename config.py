import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# ALPACA API KEYS
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

# TRADING TARGET
SYMBOL = "SPY"

# TRADING PARAMETERS
TRADE_QTY = 1

# SIMULATED DATASTREAM SETTINGS
USE_SIMULATION = True
SIM_SPEED = 0.8 # seconds

# TRAINING VARIABLES
MODEL_NAME = "ppo_trading_agent.zip"
SEQ_LEN = 15
FEATURES = [
    "open", "high", "low", "close", "volume",
    "return", "volatility", "sma_5", "sma_20", "rsi", "macd"
]
TP_PCT = 1 / 100
SL_PCT = 1 / 100
LOOKAHEAD = 3
START_TIME = datetime.now() - timedelta(days=365*2)
END_TIME = datetime.now()