from config import SYMBOL, FEATURES, START_TIME, END_TIME, API_KEY, API_SECRET
from indicators import indicators

import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# =====================================
# DOWNLOAD DATA
# =====================================

# Initialize Alpaca Data Client
print(f"Initializing data client")
client = StockHistoricalDataClient(
    API_KEY,
    API_SECRET
)

# Fetch historical minute bars
print(f"Fetching historical minute bars for {SYMBOL} from [{START_TIME}] - [{END_TIME}]")
request_params = StockBarsRequest(
    symbol_or_symbols=[SYMBOL],
    timeframe=TimeFrame.Minute,
    start=START_TIME,
    end=END_TIME,
)
bars = client.get_stock_bars(request_params).df

# =====================================
# PREPROCESS DATA
# =====================================

# compute indicators
df = bars.copy().reset_index()
df["return"] = df["close"].pct_change()
df["volatility"] = indicators.volatility(df["close"], window=10)
df["sma_5"] = indicators.sma(df["close"], 5)
df["sma_20"] = indicators.sma(df["close"], 20)
df["rsi"] = indicators.rsi(df["close"], 14)
df["macd"] = indicators.macd(df["close"])

# scale features and save dataset
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])
df.to_csv("preprocessed_training_data.csv")