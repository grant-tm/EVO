from config import SYMBOL, SEQ_LEN, FEATURES, TP_PCT, SL_PCT, LOOKAHEAD, START_TIME, END_TIME, TRAINING_EPOCHS, MODEL_NAME
from indicators import indicators

import numpy as np
import pandas as pd
from collections import Counter

from datetime import datetime, timedelta

import os
from dotenv import load_dotenv
load_dotenv()

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# =====================================
# ACQUIRE DATA
# =====================================

# Initialize Alpaca Data Client
print(f"Initializing data client")
client = StockHistoricalDataClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY")
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

# scale features
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

df.to_csv("preprocessed_training_data.csv")

# =====================================
# LABEL DATA
# =====================================

# generate labels
def label_row(i):
    future_return = (df.loc[i + LOOKAHEAD, "close"] - df.loc[i, "close"]) / df.loc[i, "close"]
    if future_return > TP_PCT:
        return 1  # Buy
    elif future_return < -SL_PCT:
        return 2  # Sell
    else:
        return 0  # Hold

df["label"] = [label_row(i) if i + LOOKAHEAD < len(df) else 0 for i in range(len(df))]

# =====================================
# FORMAT DATA
# =====================================

# Create sequences for model input
features = FEATURES
X = []
y = []

print(f"Formatting data", end="\r")
for i in range(SEQ_LEN, len(df) - LOOKAHEAD):
    
    if (i % 100 == 0):
        print(f"Formatting data ({i-SEQ_LEN}/{len(df)-LOOKAHEAD})", end="\r")
    
    seq = df.loc[i - SEQ_LEN:i - 1, features].values
    label = df.loc[i, "label"]
    X.append(seq)
    y.append(label)

print(f"Formatting data ({len(df)-LOOKAHEAD}/{len(df)-LOOKAHEAD})")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(32)

# =====================================
# SPLIT DATA AND COMPUTE CLASS WEIGHTS
# =====================================

# Split chronologically
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Compute class weights: inverse of class frequencies
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weights}")

# =====================================
# DEFINE AND COMPILE MODEL
# =====================================

# Build a simple LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.SpatialDropout1D(0.1),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# TRAIN MODEL
# =====================================

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=TRAINING_EPOCHS,
    batch_size=32,
    class_weight=class_weights
)

# Save model
model.save(f"{MODEL_NAME}")
print(f"Model saved to {MODEL_NAME}")

# =====================================
# EVALUATE MODEL
# =====================================

y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
print(np.bincount(y_pred))
print(np.bincount(y_val))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))