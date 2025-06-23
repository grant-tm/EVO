import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# ALPACA API KEYS
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

# TRADING TARGET
SYMBOL = "SPY"
TRADE_QTY = 1

# SIM TRADING SETTINGS
USE_SIMULATION = True
SIM_SPEED = 0.8 # seconds per minute-bar emission

# TRAINING PARAMETERS
DATA_PATH = "preprocessed_training_data.csv"
MODEL_DIR = "trained_models"
MODEL_NAME = "ppo_trading_agent.zip"
TRAINING_STEPS = 1_000_000
SEQ_LEN = 15
FEATURES = [
    "open", "high", "low", "close", "volume",
    "return", "volatility", "sma_5", "sma_20", "rsi", "macd"
]
TP_PCT = 1 / 100
SL_PCT = 1 / 100
START_TIME = datetime.now() - timedelta(days=365*2)
END_TIME = datetime.now()

# DEFAULT TRAINING PARAMETERS (GENOMIC)
LEARNING_RATE = 0.0003
CLIP_RANGE = 0.2
BATCH_SIZE = 64
ENTROPY_COEF_INIT = 0.1
ENTROPY_COEF_FINAL = 0.01
GAE_LAMBDA = 0.95
GAMMA = 0.95

# DEFAULT REWARD SHAPING PARAMETERS (GENOMIC)
IDLE_PENALTY = 0.0003
SL_PENALTY_COEF = 10.0
TP_REWARD_COEF = 15.0
TIMEOUT_DURATION = 3
TIMEOUT_REWARD_COEF = 2.0
ONGOING_REWARD_COEF = 0.2
REWARD_CLIP_RANGE = (-1.0, 1.0)
MAX_EPISODE_STEPS = 5000

# GENETIC SEARCH VARIABLES
MUTATION_RATE = 0.1
ELITE_PROPORTION = 0.2
NUM_BACKTESTS = 500
BACKTEST_LENGTH = 360