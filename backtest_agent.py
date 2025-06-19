from config import FEATURES, SEQ_LEN

import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import backtrader as bt
from numpy.lib.stride_tricks import as_strided

# =====================================
# Load and preprocess data
# =====================================
print("Loading historical data...")
df = pd.read_csv("preprocessed_training_data.csv")

# Parse and sort datetime
df["datetime"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("datetime").drop_duplicates(subset="datetime").reset_index(drop=True)

# Drop missing OHLCV
required_cols = ["datetime", "open", "high", "low", "close", "volume"]
df.dropna(subset=required_cols, inplace=True)

# Clean features
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=FEATURES, inplace=True)
df.reset_index(drop=True, inplace=True)

# Select a random slice of the data
slice_length = 10_000
max_start = len(df) - slice_length
start_idx = np.random.randint(0, max_start)
end_idx = start_idx + slice_length

print(f"Using data slice: rows {start_idx} to {end_idx}")
df = df.iloc[start_idx:end_idx].copy()
df.reset_index(drop=True, inplace=True)

print(f"Data index range: {df.index.min()} to {df.index.max()}")
print(f"Total rows: {len(df)}")

# =====================================
# Precompute PPO predictions
# =====================================
print("Precomputing PPO predictions...")

def build_sequences(array, window):
    n = array.shape[0] - window + 1
    if n <= 0:
        return np.empty((0, window, array.shape[1]), dtype=np.float32)
    s0, s1 = array.strides
    return as_strided(array, shape=(n, window, array.shape[1]), strides=(s0, s0, s1))

features = df[FEATURES].to_numpy(dtype=np.float32)
features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
X = build_sequences(features, SEQ_LEN)

print(f"Built {X.shape[0]} sequences.")

model = PPO.load("ppo_trading_agent")

# Predict in batches
actions = []
batch_size = 1024
for i in range(0, len(X), batch_size):
    batch = X[i:i+batch_size]
    preds, _ = model.predict(batch, deterministic=True)
    actions.extend(preds)

# Align actions to df
df = df.iloc[SEQ_LEN - 1:].copy()
df["ppo_action"] = actions

# Set datetime index for Backtrader
df.set_index("datetime", inplace=True)

# =====================================
# Backtrader-compatible feed
# =====================================
class FeatureData(bt.feeds.PandasData):
    lines = ('ppo_action',)
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        ('ppo_action', 'ppo_action'),
    )

# =====================================
# PPO Strategy
# =====================================
class PPOInferenceStrategy(bt.Strategy):
    def __init__(self):
        self.step = 0
        self.initial_value = self.broker.getvalue()

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        action = int(self.datas[0].ppo_action[0])

        if (self.step % 100):
            print(self.step)

        if action == 1 and not self.position:
            self.buy()
        elif action == 2 and not self.position:
            self.sell()
        elif action == 0 and self.position:
            self.close()

        self.step += 1

    def stop(self):
        final_value = self.broker.getvalue()
        pnl = final_value - self.initial_value
        print(f"Backtest complete. Final value: ${final_value:.2f}, PnL: ${pnl:.2f}")

# =====================================
# Run backtest
# =====================================
print("Running backtest...")
cerebro = bt.Cerebro()

print("\t-- Adding strategy")
cerebro.addstrategy(PPOInferenceStrategy)

print("\t-- Adding data")
cerebro.adddata(FeatureData(dataname=df))

print("\t-- Configuring broker parameters")
cerebro.broker.set_cash(100_000)
cerebro.broker.setcommission(commission=0.001)

print("\t-- Initiating backtest")
results = cerebro.run()

print("Backtest finished.")

# =====================================
# Plot results
# =====================================
print("Plotting results...")
cerebro.plot(style="candlestick")
