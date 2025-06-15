import os
from dotenv import load_dotenv
load_dotenv()

# ALPACA API KEYS
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

# TRADING TARGET
SYMBOL = "AAPL"

# TRADING PARAMETERS
THRESHOLD = 0.55
TRADE_QTY = 1

# SIMULATED DATASTREAM SETTINGS
USE_SIMULATION = True
SIM_SPEED = 0.8 # seconds

# DATA FORMAT
SEQ_LEN = 60
FEATURES = ["open", "high", "low", "close", "volume"]